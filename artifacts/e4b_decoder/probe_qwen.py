"""E4b pre-registration probe (CPU only, no training, no labels used): Qwen2.5-0.5B loadability as a sequence classifier,
module names, pad token, pair tokenization behaviour, token-length distribution per task (held-out + eval inputs)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1b_confirmatory"))
from tasks import ALL_SPECS, load_raw, columns, split_indices
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
REPO, REV = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"
out = {}
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
out["tokenizer"] = {"class": type(tok).__name__, "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id, "eos_token": tok.eos_token,
                    "eos_token_id": tok.eos_token_id, "bos_token": tok.bos_token, "padding_side": tok.padding_side,
                    "adds_special_tokens_example": tok("Hello world")["input_ids"]}
a, b = "A man is playing a guitar.", "A person plays music."
out["pair_check"] = {"pair_ids": tok(a + "\n", b)["input_ids"], "concat_ids": tok(a + "\n")["input_ids"] + tok(b)["input_ids"],
                     "joint_ids": tok(a + "\n" + b)["input_ids"]}
cfg = AutoConfig.from_pretrained(REPO, revision=REV)
out["config"] = {k: getattr(cfg, k, None) for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "torch_dtype", "tie_word_embeddings", "pad_token_id", "vocab_size")}
m = AutoModelForSequenceClassification.from_pretrained(REPO, revision=REV, num_labels=3, torch_dtype=torch.float32)
out["model_class"] = type(m).__name__
names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
out["linear_modules_example"] = names[:9] + names[-3:]
tgt = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
out["n_target_linear"] = sum(1 for n in names if n.split(".")[-1] in tgt)
out["score_head"] = {n: tuple(p.shape) for n, p in m.named_parameters() if n.startswith("score")}
out["n_params_total"] = sum(p.numel() for p in m.parameters())
shapes = {}
for n, mod in m.named_modules():
    if n.endswith(tuple(tgt)) and ".0." in n: shapes[n] = tuple(mod.weight.shape)
out["layer0_target_shapes"] = shapes
lens = {}
for t in ["cola","sst2","mrpc","stsb","mnli","qnli","rte","wic","snli","scitail","ag_news","imdb","trec","yelp_polarity"]:
    spec = ALL_SPECS[t]; tr = load_raw(t, spec[2]); ev = load_raw(t, spec[3])
    hold, tidx, _ = split_indices(len(tr), len(ev))
    rng = np.random.default_rng(1)
    eidx = np.sort(rng.choice(len(ev), 5000, replace=False)) if len(ev) > 5000 else np.arange(len(ev))
    res = {}
    for name, ds, idx in (("hold", tr, hold), ("eval5k", ev, eidx), ("train_sample2k", tr, np.sort(tidx[:2000]))):
        x, y, _ = columns(t, ds.select(idx.tolist()))
        txt = [p + "\n" + q for p, q in zip(x, y)] if y is not None else x
        L = np.array([len(i) for i in tok(txt)["input_ids"]])
        res[name] = {"n": int(len(L)), "mean": float(L.mean()), "p50": float(np.median(L)), "p95": float(np.percentile(L, 95)), "max": int(L.max()),
                     "frac_gt_256": float((L > 256).mean()), "mean_trunc256": float(np.minimum(L, 256).mean())}
    lens[t] = res
    print(t, res, flush=True)
out["token_lengths"] = lens
(HERE / "probe_qwen.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "token_lengths"}, indent=1, default=str))
