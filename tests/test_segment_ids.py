"""Regression tests for the segment-id (token_type_ids) convention (CPU-only, no downloads).

Bug (2026-09-25): the evaluator always dropped token_type_ids, which silently mis-evaluated adapters trained with
standard BERT segment ids (hubish MNLI 0.45-0.49 instead of 0.82). Hub prateeky2806 adapters, however, were trained
WITHOUT segment ids, so the mode must be chosen per adapter.
"""
from __future__ import annotations

import json
import warnings

import pytest
import torch
import torch.nn as nn

from lora_merge_cert import eval as ev


class DummyPairTokenizer:
    """Minimal BERT-like tokenizer: [CLS]=101, [SEP]=102, word ids = len(word)+1000, pads to max_length."""

    def __call__(self, a, b=None, truncation=True, max_length=16, padding="max_length"):
        out = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for i, sa in enumerate(a):
            ids = [101] + [1000 + len(w) for w in sa.split()] + [102]
            tt = [0] * len(ids)
            if b is not None:
                sb = [1000 + len(w) for w in b[i].split()] + [102]
                ids += sb
                tt += [1] * len(sb)
            ids, tt = ids[:max_length], tt[:max_length]
            am = [1] * len(ids)
            pad = max_length - len(ids)
            out["input_ids"].append(ids + [0] * pad)
            out["token_type_ids"].append(tt + [0] * pad)
            out["attention_mask"].append(am + [0] * pad)
        return out


def _pair_ds():
    from datasets import Dataset
    return Dataset.from_dict({"premise": ["a cat sat", "dogs run fast"], "hypothesis": ["an animal", "slow"],
                              "idx": [0, 1], "label": [0, 2]})


def _loader(mode):
    from torch.utils.data import DataLoader
    ds = ev.tokenize_and_format(_pair_ds(), DummyPairTokenizer(), ("premise", "hypothesis"), max_length=16,
                                segment_ids=mode)
    return DataLoader(ds, batch_size=2)


def test_bert_mode_batch_contains_segment_ids():
    batch = next(iter(_loader("bert")))
    assert "token_type_ids" in batch
    tt = batch["token_type_ids"]
    assert tt.shape == batch["input_ids"].shape
    # second segment is marked with 1s
    assert int(tt.sum()) > 0
    assert torch.equal(tt[0, :5], torch.tensor([0, 0, 0, 0, 0]))  # [CLS] a cat sat [SEP]
    assert int(tt[0, 5]) == 1


def test_none_mode_batch_omits_segment_ids():
    batch = next(iter(_loader("none")))
    assert "token_type_ids" not in batch
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


class CaptureModel(nn.Module):
    """Records the kwargs it is called with; predicts class 0."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, **kw):
        self.calls.append(sorted(kw))
        n = kw["input_ids"].shape[0]
        return type("O", (), {"logits": torch.tensor([[1.0, 0.0, 0.0]] * n)})()


@pytest.mark.parametrize("mode,expect", [("bert", True), ("none", False)])
def test_accuracy_forwards_segment_ids_only_in_bert_mode(mode, expect):
    m = CaptureModel()
    acc = ev.accuracy_on_dataloader(m, _loader(mode), device=torch.device("cpu"))
    assert acc == 0.5
    assert ("token_type_ids" in m.calls[0]) is expect


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ev.model_input_columns("zeros")


def test_legacy_default_warns_and_keeps_none():
    with pytest.warns(UserWarning, match="segment_ids not specified"):
        assert ev._normalize_segment_ids(None) == "none"


def test_resolve_known_sources(tmp_path):
    assert ev.resolve_segment_ids("prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001") == "none"
    assert ev.resolve_segment_ids("adapters/mnli_s7_hubish") == "bert"
    assert ev.resolve_segment_ids("/abs/path/adapters/rte_s42_hubish/") == "bert"
    assert ev.resolve_segment_ids("adapters/conflict_mnli") == "none"
    # explicit override wins
    assert ev.resolve_segment_ids("adapters/mnli_s7_hubish", "none") == "none"
    assert ev.resolve_segment_ids("prateeky2806/x", "auto") == "none"


def test_resolve_metadata_file_and_provenance(tmp_path):
    d = tmp_path / "my_new_adapter"
    d.mkdir()
    ev.write_segment_meta(str(d), "none", "test")
    assert ev.resolve_segment_ids(str(d)) == "none"
    c = tmp_path / "constructed"
    c.mkdir()
    (c / "conflict_meta.json").write_text(json.dumps({"lora1_src": "prateeky2806/bert-base-uncased-sst2"}))
    assert ev.resolve_segment_ids(str(c)) == "none"


def test_resolve_unknown_warns_loudly(tmp_path):
    with pytest.warns(UserWarning, match="unknown segment-id convention"):
        assert ev.resolve_segment_ids(str(tmp_path / "mystery")) == "bert"


def test_pair_disagreement_warns():
    with pytest.warns(UserWarning, match="disagree"):
        assert ev.resolve_pair_segment_ids(["adapters/mnli_s7_hubish", "prateeky2806/x"]) == "bert"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ev.resolve_pair_segment_ids(["prateeky2806/a", "adapters/conflict_mnli"]) == "none"


def test_tiny_bert_logits_depend_on_segment_ids():
    """End-to-end sanity on a tiny random BERT: passing segment ids changes the model output."""
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=1100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=32, num_labels=3)
    model = BertForSequenceClassification(cfg).eval()
    with torch.no_grad():
        b = {k: v for k, v in next(iter(_loader("bert"))).items() if k != "labels"}
        n = {k: v for k, v in next(iter(_loader("none"))).items() if k != "labels"}
        assert not torch.allclose(model(**b).logits, model(**n).logits)
