#!/usr/bin/env python3
"""Seed-pair (hubish) merge + certificate re-evaluation with the segment-id (token_type_ids) fix.

Reproduces scripts/run_paper_pack_4070.py (seed_hubish config) and the hubish mnli+rte certificate run
(scripts/week1_run.py) using lora_merge_cert UNCHANGED for the merge/certificate. The only code copied and
changed is the eval dataloader:
  --segment-ids none : identical to lora_merge_cert/eval.py (set_format drops token_type_ids -> all zeros)
  --segment-ids bert : keep the tokenizer's token_type_ids (the convention train_lora_glue.py trained with)
Everything else (padding=max_length 128, batch 32, head copied from adapter 1, lam=1 arithmetic sum, theta*=30,
TIES-lite densify 0.7, cert_then_trim keep 70%) is copied verbatim from run_paper_pack_4070.py.
Run from ~/Desktop/Workspace/LoRA:  PYTHONPATH=$PWD python artifacts/seed_fix_segid/seed_fix_segid.py --segment-ids bert
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path.cwd()))
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from lora_merge_cert.merge import merge_lora_models, apply_deltas_to_base, collect_lora_pairs

BASE = "bert-base-uncased"
OUT = Path("artifacts/seed_fix_segid")
TASK_CONFIG = {
    "mnli": {"subset": "mnli", "validation_split": "validation_matched", "text_fields": ("premise", "hypothesis")},
    "rte": {"subset": "rte", "validation_split": "validation", "text_fields": ("sentence1", "sentence2")},
}
LOGF = None


def log(m):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(m)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


# ---------------- eval: copy of lora_merge_cert/eval.py with a segment-id switch + per-example preds
_DS_CACHE = {}


def build_eval_dataloader(task, tokenizer, segment_ids, batch_size=32, max_length=128, num_samples=None):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    key = (task, segment_ids, num_samples)
    if key not in _DS_CACHE:
        cfg = TASK_CONFIG[task]
        ds = load_dataset("nyu-mll/glue", cfg["subset"], split=cfg["validation_split"])
        if num_samples is not None:
            ds = ds.select(range(min(num_samples, len(ds))))
        tf = cfg["text_fields"]

        def tokenize_fn(ex):
            return tokenizer(ex[tf[0]], ex[tf[1]], truncation=True, max_length=max_length, padding="max_length")
        cols_to_remove = [c for c in ds.column_names if c != "label"]
        ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        ds = ds.rename_column("label", "labels")
        cols = ["input_ids", "attention_mask", "labels"]          # original
        if segment_ids == "bert":
            cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]   # THE FIX
        ds.set_format(type="torch", columns=cols)
        _DS_CACHE[key] = ds
    return DataLoader(_DS_CACHE[key], batch_size=batch_size)


@torch.no_grad()
def evaluate(model, tok, task, device, segment_ids, num_samples=None):
    loader = build_eval_dataloader(task, tok, segment_ids, num_samples=num_samples)
    model = model.to(device).eval()
    preds, labs = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        p = model(**batch).logits.argmax(dim=-1)
        preds.append(p.cpu().numpy()); labs.append(labels.cpu().numpy())
    preds = np.concatenate(preds); labs = np.concatenate(labs)
    correct = int((preds == labs).sum())
    return {"task": task, "accuracy": correct / max(len(labs), 1), "n": int(len(labs))}, (preds == labs).astype(np.int8)


# ---------------- copied verbatim from scripts/run_paper_pack_4070.py
def ties_lite_sum(d1, d2, densify=0.7):
    out = {}
    for k in set(d1) | set(d2):
        a, b = d1.get(k), d2.get(k)
        if a is None: t = b.clone()
        elif b is None: t = a.clone()
        else:
            s = torch.sign(a) + torch.sign(b)
            s = torch.where(s == 0, torch.sign(a + b), torch.sign(s))
            t = torch.where(torch.sign(a) == s, a, torch.zeros_like(a)) + torch.where(torch.sign(b) == s, b, torch.zeros_like(b))
        flat = t.abs().flatten()
        if flat.numel() == 0:
            out[k] = t; continue
        kth = int(max(1, (1.0 - densify) * flat.numel()))
        if kth >= flat.numel():
            out[k] = t; continue
        thresh = torch.kthvalue(flat, kth).values
        out[k] = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
    return out


def build_model(base_id, nlab, clf_src, deltas, device):
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=nlab)
    src = clf_src.base_model.model.classifier
    with torch.no_grad():
        base.classifier.weight.copy_(src.weight.data)
        base.classifier.bias.copy_(src.bias.data)
    apply_deltas_to_base(base, deltas)
    return base.to(device)


def cert_trim_of(cert_d):
    cert_trim = {}
    for k, t in cert_d.items():
        flat = t.abs().flatten(); kth = int(max(1, 0.3 * flat.numel()))
        thresh = torch.kthvalue(flat, min(kth, flat.numel())).values
        cert_trim[k] = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
    return cert_trim
# ----------------


def peft_model(path, nl):
    return PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=nl), path)


def head_check(m, path):
    sd = load_file(f"{path}/adapter_model.safetensors")
    w = [v for k, v in sd.items() if "classifier" in k and k.endswith("weight")][0]
    src = m.base_model.model.classifier
    return float((src.weight.data.float().cpu() - w.float()).abs().max())


PAIRS = [
    # name, adapter1 (head source), adapter2, nlab1, nlab2, tasks evaluated with head of adapter1/adapter2
    ("seed_hubish", "adapters/mnli_s7_hubish", "adapters/mnli_s42_hubish", 3, 3, "mnli", "mnli"),
    ("mnli_rte_hubish", "adapters/mnli_s7_hubish", "adapters/rte_s42_hubish", 3, 2, "mnli", "rte"),
]


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-ids", choices=["none", "bert"], required=True)
    args = ap.parse_args()
    seg = args.segment_ids
    OUT.mkdir(parents=True, exist_ok=True)
    LOGF = OUT / "run.log"
    odir = OUT / f"seg_{seg}"; odir.mkdir(exist_ok=True)
    (odir / "preds").mkdir(exist_ok=True)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(BASE)
    log(f"=== start segment_ids={seg}")
    res = {"segment_ids": seg, "runs": [], "certs": {}}
    t0 = time.time()

    def rec(pair, method, task, head, r, corr, extra=None):
        d = {"pair": pair, "method": method, "task": task, "head": head, **r, **(extra or {})}
        res["runs"].append(d)
        np.save(odir / "preds" / f"{pair}__{method}__{task}__{head}.npy", corr)
        log(f"[{seg}] {pair} {method} task={task} head={head} acc={r['accuracy']:.5f} n={r['n']}")

    # singles (as scripts/eval_single_adapters.py: PeftModel on fresh base)
    singles = {}
    for path, nl, task in [("adapters/mnli_s7_hubish", 3, "mnli"), ("adapters/mnli_s42_hubish", 3, "mnli"), ("adapters/rte_s42_hubish", 2, "rte")]:
        m = peft_model(path, nl)
        r, c = evaluate(m, tok, task, device, seg)
        name = path.split("/")[-1]
        singles[name] = r["accuracy"]
        rec("single", name, task, name, r, c, {"head_maxdiff_vs_file": head_check(m, path)})
        if task == "mnli" and seg == "none" and name == "mnli_s7_hubish":
            r5, c5 = evaluate(m, tok, task, device, seg, num_samples=512)
            rec("single", name + "_n512", task, name, r5, c5)
        del m; torch.cuda.empty_cache()
    res["singles"] = singles

    for name, l1, l2, n1, n2, task1, task2 in PAIRS:
        m1 = peft_model(l1, n1); m2 = peft_model(l2, n2)
        h1, h2 = l1.split("/")[-1], l2.split("/")[-1]
        res.setdefault("head_copy_check", {})[name] = {h1: head_check(m1, l1), h2: head_check(m2, l2)}
        heads = [(m1, n1, task1, h1)]
        if name == "seed_hubish":
            heads.append((m2, n2, task1, h2))      # exploratory: same task, other adapter's head
        else:
            heads.append((m2, n2, task2, h2))      # rte task with rte head (new; original evaluated MNLI only)
        cert_A = None
        for subspace in ("A", "B"):
            cert_d, certs = merge_lora_models(m1, m2, subspace=subspace, theta_star_deg=30.0, use_certificate=True)
            arith_d, _ = merge_lora_models(m1, m2, subspace=subspace, theta_star_deg=30.0, use_certificate=False)
            if subspace == "A":
                cert_A = cert_d
            ths = [c.theta_min_deg for c in certs]
            res["certs"][f"{name}_{subspace}"] = {"n_fail": sum(c.status == "FAIL" for c in certs), "n_layers": len(certs),
                                                   "theta_min": min(ths), "theta_median": sorted(ths)[len(ths) // 2],
                                                   "fail_layers": [(c.layer_name, c.theta_min_deg, c.n_shared) for c in certs if c.status == "FAIL"]}
            with open(odir / f"{name}_{subspace}_th30_certificate_table.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["layer_name", "status", "theta_min_deg", "overlap", "n_shared"])
                w.writeheader()
                for c in certs:
                    w.writerow({"layer_name": c.layer_name, "status": c.status, "theta_min_deg": c.theta_min_deg, "overlap": c.overlap, "n_shared": c.n_shared})
            for mh, nl, task, hn in heads:
                for meth, dd in (("arith_sum", arith_d), (f"cert_{subspace}", cert_d)):
                    if meth == "arith_sum" and subspace == "B":
                        continue  # arithmetic delta is subspace-independent; evaluated once
                    mm = build_model(BASE, nl, mh, dd, device)
                    r, c = evaluate(mm, tok, task, device, seg)
                    rec(name, meth, task, hn, r, c)
                    if seg == "none" and name == "mnli_rte_hubish" and task == "mnli" and hn == h1 and subspace == "A":
                        r5, c5 = evaluate(mm, tok, task, device, seg, num_samples=512)
                        rec(name, meth + "_n512", task, hn, r5, c5)
                    del mm; torch.cuda.empty_cache()
        # TIES-lite and cert_then_trim exactly as the original hub-pair branch (not run for seed pair originally)
        pairs = collect_lora_pairs(m1, m2)
        dd1 = {nm: (B1 @ A1).detach().cpu() for nm, B1, A1, B2, A2 in pairs}
        dd2 = {nm: (B2 @ A2).detach().cpu() for nm, B1, A1, B2, A2 in pairs}
        ties_d = ties_lite_sum(dd1, dd2, 0.7)
        ct = cert_trim_of(cert_A)
        arith_d, _ = merge_lora_models(m1, m2, subspace="A", theta_star_deg=30.0, use_certificate=False)
        for mh, nl, task, hn in heads:
            for meth, dd in (("ties_lite", ties_d), ("cert_then_trim", ct)):
                mm = build_model(BASE, nl, mh, dd, device); r, c = evaluate(mm, tok, task, device, seg); rec(name, meth, task, hn, r, c)
                del mm; torch.cuda.empty_cache()
            for lam in (0.3, 0.5, 0.7):  # exploratory lam grid on the arithmetic sum (lam=1.0 is arith_sum)
                mm = build_model(BASE, nl, mh, {k: lam * v for k, v in arith_d.items()}, device)
                r, c = evaluate(mm, tok, task, device, seg); rec(name, f"arith_lam{lam}", task, hn, r, c)
                del mm; torch.cuda.empty_cache()
        del m1, m2; torch.cuda.empty_cache()
    res["wall_s"] = time.time() - t0
    (odir / "results.json").write_text(json.dumps(res, indent=2))
    log(f"=== done segment_ids={seg} wall={res['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
