"""Evaluation helpers for GLUE classification accuracy.

Segment-ID (token_type_ids) convention
--------------------------------------
BERT sentence-pair inputs carry segment ids (0 for sentence A, 1 for sentence B). Adapters must be
evaluated with the SAME convention they were trained with:

* ``"bert"``: pass the tokenizer's ``token_type_ids``. This is the convention of adapters trained with
  ``scripts/train_lora_glue.py`` (HF Trainer + DataCollatorWithPadding keeps token_type_ids), e.g.
  ``adapters/*_hubish``, ``adapters/mnli_seed*``.
* ``"none"``: do not pass ``token_type_ids`` (the model then uses all-zero segment ids). This is the
  convention the ``prateeky2806/*`` Hub adapters were trained with.

Diagnosis (2026-09-25, artifacts/e1_predictive/diag and artifacts/seed_fix_segid): the old evaluator always
dropped token_type_ids. That is correct for the prateeky2806 adapters (MNLI 0.807; with BERT segment ids 0.705)
but wrong for the locally trained adapters (mnli_s7_hubish 0.452 -> 0.821, mnli_s42_hubish 0.490 -> 0.824,
rte_s42_hubish 0.433 -> 0.675 once segment ids are passed). Use :func:`resolve_segment_ids` to choose the mode
per adapter. ``evaluate_glue(..., segment_ids=None)`` keeps the legacy behaviour ("none") but warns loudly.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import torch
import torch.nn as nn

SegmentIdMode = Literal["bert", "none"]
SEGMENT_ID_MODES = ("bert", "none")
#: metadata file that training scripts write next to adapter_config.json
SEGMENT_META_FILENAME = "segment_ids.json"

#: Known adapter sources -> segment-id convention (evidence in the module docstring).
#: Keys are Hub-id prefixes or local directory names (basename).
SEGMENT_ID_REGISTRY: Dict[str, str] = {
    "prateeky2806/": "none",           # Hub adapters (trained without token_type_ids)
    "conflict_mnli": "none",           # constructed from prateeky2806 mnli/sst2 adapters
    "conflict_sst2_shared": "none",
    "mnli_s7_hubish": "bert",          # scripts/train_lora_glue.py
    "mnli_s42_hubish": "bert",
    "rte_s42_hubish": "bert",
    "mnli_seed7": "bert",
    "mnli_seed42": "bert",
    "mnli_seed7_full": "bert",
    "mnli_seed42_full": "bert",
    "conflict_full_mnli": "bert",      # constructed from mnli_seed7_full / mnli_seed42_full
    "conflict_full_shared": "bert",
}

_LOUD = "\n" + "!" * 78 + "\n"


def _loud_warning(msg: str) -> None:
    warnings.warn(_LOUD + "SEGMENT-ID WARNING: " + msg + _LOUD, UserWarning, stacklevel=3)


def _check_mode(mode: str) -> str:
    if mode not in SEGMENT_ID_MODES:
        raise ValueError(f"segment_ids must be one of {SEGMENT_ID_MODES}, got {mode!r}")
    return mode


def resolve_segment_ids(adapter_source: Optional[str], explicit: Optional[str] = None,
                        *, default_unknown: str = "bert") -> str:
    """Choose the segment-id mode for one adapter.

    Priority: ``explicit`` ("bert"/"none"; "auto"/None means resolve) > ``segment_ids.json`` in the adapter
    directory > :data:`SEGMENT_ID_REGISTRY` (Hub prefix or directory basename) > ``conflict_meta.json``
    provenance (mode of ``lora1_src``) > ``default_unknown`` with a loud warning.
    """
    if explicit not in (None, "auto"):
        return _check_mode(explicit)
    if adapter_source is None:
        _loud_warning(f"no adapter source given; using default {default_unknown!r}.")
        return _check_mode(default_unknown)
    src = str(adapter_source)
    p = Path(src)
    if p.is_dir() and (p / SEGMENT_META_FILENAME).exists():
        meta = json.loads((p / SEGMENT_META_FILENAME).read_text())
        return _check_mode(meta["segment_ids"])
    for key, mode in SEGMENT_ID_REGISTRY.items():
        if key.endswith("/") and src.startswith(key):
            return mode
    base = os.path.basename(os.path.normpath(src))
    if base in SEGMENT_ID_REGISTRY:
        return SEGMENT_ID_REGISTRY[base]
    if p.is_dir() and (p / "conflict_meta.json").exists():
        meta = json.loads((p / "conflict_meta.json").read_text())
        parent = meta.get("lora1_src")
        if parent and parent != src:
            return resolve_segment_ids(parent, None, default_unknown=default_unknown)
    _loud_warning(
        f"unknown segment-id convention for adapter {src!r}; defaulting to {default_unknown!r}. "
        f"Add {SEGMENT_META_FILENAME} to the adapter dir or an entry to SEGMENT_ID_REGISTRY."
    )
    return _check_mode(default_unknown)


def resolve_pair_segment_ids(sources: Iterable[Optional[str]], explicit: Optional[str] = None) -> str:
    """Mode for a merged model built from several adapters; warns if their conventions disagree."""
    modes = [resolve_segment_ids(s, explicit) for s in sources]
    if len(set(modes)) > 1:
        _loud_warning(f"adapters disagree on segment-id convention {dict(zip(sources, modes))}; "
                      f"using that of the first adapter ({modes[0]!r}).")
    return modes[0]


def write_segment_meta(adapter_dir: str, mode: str, note: str = "") -> None:
    """Record the segment-id convention next to a saved adapter."""
    _check_mode(mode)
    Path(adapter_dir, SEGMENT_META_FILENAME).write_text(
        json.dumps({"segment_ids": mode, "note": note}, indent=2) + "\n", encoding="utf-8")


TASK_CONFIG = {
    "mnli": {"dataset": "nyu-mll/glue", "subset": "mnli", "validation_split": "validation_matched",
             "text_fields": ("premise", "hypothesis"), "num_labels": 3},
    "sst2": {"dataset": "nyu-mll/glue", "subset": "sst2", "validation_split": "validation",
             "text_fields": ("sentence",), "num_labels": 2},
    "rte": {"dataset": "nyu-mll/glue", "subset": "rte", "validation_split": "validation",
            "text_fields": ("sentence1", "sentence2"), "num_labels": 2},
    "mrpc": {"dataset": "nyu-mll/glue", "subset": "mrpc", "validation_split": "validation",
             "text_fields": ("sentence1", "sentence2"), "num_labels": 2},
    "qnli": {"dataset": "nyu-mll/glue", "subset": "qnli", "validation_split": "validation",
             "text_fields": ("question", "sentence"), "num_labels": 2},
    "qqp": {"dataset": "nyu-mll/glue", "subset": "qqp", "validation_split": "validation",
            "text_fields": ("question1", "question2"), "num_labels": 2},
    "cola": {"dataset": "nyu-mll/glue", "subset": "cola", "validation_split": "validation",
             "text_fields": ("sentence",), "num_labels": 2},
}


def load_glue_split(task: str, split: Optional[str] = None):
    """Load a GLUE validation split via HuggingFace datasets."""
    from datasets import load_dataset

    task = task.lower().replace("-", "").replace("_", "")
    if task in ("mnli", "mnlimatched"):
        cfg = TASK_CONFIG["mnli"]
    elif task in ("sst2", "sst"):
        cfg = TASK_CONFIG["sst2"]
    elif task in TASK_CONFIG:
        cfg = TASK_CONFIG[task]
    else:
        raise ValueError(f"Unsupported task: {task}")

    split = split or cfg["validation_split"]
    return load_dataset(cfg["dataset"], cfg["subset"], split=split), cfg


@torch.no_grad()
def accuracy_on_dataloader(
    model: nn.Module,
    dataloader,
    device: Optional[torch.device] = None,
) -> float:
    """Compute classification accuracy over a DataLoader of HF batches.

    Every tensor in the batch except ``labels`` is forwarded to the model, so ``token_type_ids`` is passed
    exactly when the dataloader was built with ``segment_ids="bert"``.
    """
    device = device or next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def _normalize_segment_ids(segment_ids: Optional[str]) -> str:
    if segment_ids is None:
        _loud_warning("segment_ids not specified; using legacy 'none' (token_type_ids dropped). This is only "
                      "correct for prateeky2806-style adapters. Pass segment_ids=resolve_segment_ids(adapter).")
        return "none"
    return _check_mode(segment_ids)


def model_input_columns(segment_ids: str) -> list:
    """Dataset columns handed to the model for a segment-id mode (plus 'labels')."""
    cols = ["input_ids", "attention_mask", "labels"]
    if _check_mode(segment_ids) == "bert":
        cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    return cols


def tokenize_and_format(ds, tokenizer, text_fields, *, max_length: int = 128, segment_ids: str = "none"):
    """Tokenize a GLUE split (padding to max_length) and set the torch format for ``segment_ids`` mode."""
    def tokenize_fn(examples):
        if len(text_fields) == 1:
            return tokenizer(examples[text_fields[0]], truncation=True, max_length=max_length,
                             padding="max_length")
        return tokenizer(examples[text_fields[0]], examples[text_fields[1]], truncation=True,
                         max_length=max_length, padding="max_length")

    cols_to_remove = [c for c in ds.column_names if c != "label"]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column("label", "labels")
    cols = model_input_columns(segment_ids)
    if "token_type_ids" in cols and "token_type_ids" not in ds.column_names:
        raise ValueError("segment_ids='bert' but the tokenizer returned no token_type_ids")
    ds.set_format(type="torch", columns=cols)
    return ds


def build_eval_dataloader(
    task: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    segment_ids: Optional[SegmentIdMode] = None,
):
    """Tokenize GLUE split and return a simple DataLoader (see module docstring for ``segment_ids``)."""
    from torch.utils.data import DataLoader

    mode = _normalize_segment_ids(segment_ids)
    ds, cfg = load_glue_split(task, split=split)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    ds = tokenize_and_format(ds, tokenizer, cfg["text_fields"], max_length=max_length, segment_ids=mode)
    return DataLoader(ds, batch_size=batch_size)


def evaluate_glue(
    model: nn.Module,
    tokenizer,
    task: str,
    *,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    num_samples: Optional[int] = None,
    segment_ids: Optional[SegmentIdMode] = None,
) -> Dict[str, Any]:
    """Full eval: returns {'task', 'accuracy', 'n', 'segment_ids'}.

    ``segment_ids``: "bert" | "none" (see module docstring). None = legacy "none" with a loud warning.
    """
    mode = _normalize_segment_ids(segment_ids)
    loader = build_eval_dataloader(
        task, tokenizer, batch_size=batch_size, max_length=max_length, num_samples=num_samples,
        segment_ids=mode,
    )
    if device is not None:
        model = model.to(device)
    acc = accuracy_on_dataloader(model, loader, device=device)
    n = len(loader.dataset)
    return {"task": task, "accuracy": acc, "n": n, "segment_ids": mode}


def fake_eval_from_certificates(
    certificates,
    *,
    base_mnli: float = 0.84,
    base_sst2: float = 0.92,
) -> Dict[str, Any]:
    """Dry-run stand-in: invent accuracy drops correlated with θ_min.

    Used when --dry-run so we can emit the Week-1 scatter without Hub/data.
    Heuristic: more FAIL layers / lower θ_min → larger MNLI drop under arithmetic merge;
    certificate correction recovers most of it.
    """
    import math

    fails = [c for c in certificates if c.status == "FAIL"]
    if not certificates:
        drop_arith = 0.0
        drop_cert = 0.0
    else:
        mean_theta = sum(c.theta_min_deg for c in certificates) / len(certificates)
        fail_frac = len(fails) / len(certificates)
        # Larger drop when angles are small and many FAILs
        drop_arith = 0.02 + 0.15 * fail_frac * max(0.0, (30.0 - mean_theta) / 30.0)
        drop_cert = drop_arith * 0.25  # correction recovers ~75%

    return {
        "mnli_arithmetic": base_mnli - drop_arith,
        "mnli_certificate": base_mnli - drop_cert,
        "sst2_arithmetic": base_sst2 - drop_arith * 0.5,
        "sst2_certificate": base_sst2 - drop_cert * 0.5,
        "mnli_drop_arithmetic": drop_arith,
        "mnli_drop_certificate": drop_cert,
        "fake": True,
    }
