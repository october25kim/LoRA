#!/usr/bin/env python3
"""Builds TABLE_SEED_FIXED.md + summary.json from seg_none/ and seg_bert/ (run from ~/Desktop/Workspace/LoRA)."""
import csv, json
from pathlib import Path
import numpy as np

OUT = Path("artifacts/seed_fix_segid")
R = {s: json.loads((OUT / f"seg_{s}" / "results.json").read_text()) for s in ("none", "bert")}
OLD = {  # values in the original artifacts
    ("single", "mnli_s7_hubish", "mnli"): (0.4524707080998472, "artifacts/hubish/single_mnli.json (full val)"),
    ("single", "mnli_s42_hubish", "mnli"): (0.4902699949057565, "artifacts/hubish/single_mnli.json (full val)"),
    ("seed_hubish", "arith_sum", "mnli"): (0.42903718797758533, "artifacts/paper/seed_hubish_A_th30/summary.json retain_sum"),
    ("seed_hubish", "cert_A", "mnli"): (0.42730514518593987, "artifacts/paper/TABLE_REQUIRED.md retain_corr"),
    ("seed_hubish", "cert_B", "mnli"): (0.42730514518593987, "artifacts/paper/TABLE_REQUIRED.md retain_corr"),
    ("single", "mnli_s7_hubish_n512", "mnli"): (0.4609375, "artifacts/hubish/single_mnli.json (n=512)"),
    ("mnli_rte_hubish", "arith_sum_n512", "mnli"): (0.46484375, "artifacts/hubish/mnli_rte_theta30/summary.json mnli_sum (n=512)"),
    ("mnli_rte_hubish", "cert_A_n512", "mnli"): (0.46484375, "artifacts/hubish/mnli_rte_theta30/summary.json mnli_corrected (n=512)"),
}


def get(seg, pair, method, task, head=None):
    for r in R[seg]["runs"]:
        if r["pair"] == pair and r["method"] == method and r["task"] == task and (head is None or r["head"] == head):
            return r
    return None


def preds(seg, pair, method, task, head):
    return np.load(OUT / f"seg_{seg}" / "preds" / f"{pair}__{method}__{task}__{head}.npy").astype(np.float64)


def paired_boot(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(a); d = a - b
    bs = np.array([d[rng.integers(0, N, N)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cert_table(p):
    with open(p) as f:
        return {r["layer_name"]: (r["status"], float(r["theta_min_deg"])) for r in csv.DictReader(f)}


summary = {"reproduction": {}, "cert_check": {}, "bootstrap": {}}
L = ["# Seed-pair (hubish) merge and certificate results: old eval vs segment-id-fixed eval\n",
     "Generated on ubuntu-4070 (RTX 4070 Ti SUPER) by `seed_fix_segid.py` + `build_table.py`. Everything is on the full validation set unless noted (MNLI = validation_matched, n=9815; RTE = validation, n=277).",
     "Pipeline: same as `scripts/run_paper_pack_4070.py` (lora_merge_cert unchanged; head copied from adapter 1; arithmetic sum = dW1+dW2 with lam=1; theta*=30 deg; TIES-lite densify 0.7; cert_then_trim keeps top 70% of cert_A). **The only change is that token_type_ids are passed to the model** (`--segment-ids bert`), which is the convention `train_lora_glue.py` trained with. The old eval (`lora_merge_cert/eval.py`) dropped them, so every token got segment 0.",
     "**retain** is defined as in the original TABLE_REQUIRED.md: the raw accuracy of the merged model on the task, using the head of the adapter named in the `head` column. **norm** = retain / that adapter's single accuracy (same segment convention).\n"]

# 1. reproduction check
L.append("## 1. Reproduction check (old setting = no segment ids)\n")
L.append("| quantity | original artifact | re-run, segment_ids=none | abs diff |")
L.append("|---|---:|---:|---:|")
allok = True
for (pair, meth, task), (v, src) in OLD.items():
    r = get("none", pair, meth, task, None if pair != "seed_hubish" else "mnli_s7_hubish")
    d = abs(r["accuracy"] - v); allok &= d < 1e-9
    summary["reproduction"][f"{pair}/{meth}/{task}"] = {"old": v, "rerun": r["accuracy"], "abs_diff": d, "source": src}
    L.append(f"| {pair} {meth} ({task}, n={r['n']}) | {v:.5f} | {r['accuracy']:.5f} | {d:.1e} |")
summary["reproduction_exact"] = bool(allok)
L.append(f"\nReproduction exact (bit-identical accuracies): **{allok}**. So the segment-id switch is the only difference between the old and fixed columns below.\n")

# 2. certificates
L.append("## 2. Certificates (weights only; they should not depend on the segment convention)\n")
L.append("| pair | subspace | n_fail / layers | theta_min (deg) | FAIL layers (theta_min, n_shared) | same as original artifact? | same in none vs bert runs? |")
L.append("|---|---|---:|---:|---|---|---|")
old_tabs = {"seed_hubish_A": "artifacts/paper/seed_hubish_A_th30/certificate_table.csv",
            "seed_hubish_B": "artifacts/paper/seed_hubish_B_th30/certificate_table.csv",
            "mnli_rte_hubish_A": "artifacts/hubish/mnli_rte_theta30/certificate_table.csv"}
for key, c in R["bert"]["certs"].items():
    new_b = cert_table(OUT / "seg_bert" / f"{key}_th30_certificate_table.csv")
    new_n = cert_table(OUT / "seg_none" / f"{key}_th30_certificate_table.csv")
    same_nb = all(new_b[k][0] == new_n[k][0] and abs(new_b[k][1] - new_n[k][1]) < 1e-6 for k in new_b)
    if key in old_tabs and Path(old_tabs[key]).exists():
        old = cert_table(old_tabs[key])
        common = set(old) & set(new_b)
        maxd = max(abs(old[k][1] - new_b[k][1]) for k in common)
        same_old = f"yes: {len(common)} layers, statuses identical={all(old[k][0] == new_b[k][0] for k in common)}, max |d theta|={maxd:.1e} deg"
    else:
        same_old = "no original table for this subspace"
    summary["cert_check"][key] = {**c, "same_none_vs_bert": same_nb, "vs_original": same_old}
    fl = "; ".join(f"{n.replace('base_model.model.', '')} ({t:.2f}, {s})" for n, t, s in c["fail_layers"]) or "none"
    L.append(f"| {key.rsplit('_', 1)[0]} | {key.rsplit('_', 1)[1]} | {c['n_fail']} / {c['n_layers']} | {c['theta_min']:.2f} | {fl} | {same_old} | {same_nb} |")

# 3. main table
L.append("\n## 3. Old vs fixed accuracies\n")
L.append("| pair | method | task | head | retain OLD eval (no seg ids) | norm OLD | retain FIXED (BERT seg ids) | norm FIXED |")
L.append("|---|---|---|---|---:|---:|---:|---:|")
single = {s: R[s]["singles"] for s in R}
order = ["single", "arith_sum", "cert_A", "cert_B", "ties_lite", "cert_then_trim", "arith_lam0.3", "arith_lam0.5", "arith_lam0.7"]
rows = sorted([r for r in R["bert"]["runs"] if not r["method"].endswith("_n512")],
              key=lambda r: (["single", "seed_hubish", "mnli_rte_hubish"].index(r["pair"]), r["head"] != "mnli_s7_hubish" and r["pair"] != "single",
                             r["task"], order.index(r["method"]) if r["method"] in order else 0))
for r in rows:
    o = get("none", r["pair"], r["method"], r["task"], r["head"])
    meth = r["method"]
    so, sb = single["none"][r["head"]], single["bert"][r["head"]]
    tag = ""
    if r["pair"] == "seed_hubish" and r["head"] == "mnli_s42_hubish": tag = " (exploratory head)"
    if meth in ("ties_lite", "cert_then_trim") and r["pair"] == "seed_hubish": tag += " (new: not run for seed pair originally)"
    if meth.startswith("arith_lam"): tag += " (exploratory lam)"
    if r["pair"] == "mnli_rte_hubish": tag += " (new: full val; original = MNLI n=512 only)"
    L.append(f"| {r['pair']} | {meth}{tag} | {r['task']} | {r['head']} | {o['accuracy']:.4f} | {o['accuracy'] / so:.3f} | **{r['accuracy']:.4f}** | {r['accuracy'] / sb:.3f} |")

# 4. bootstrap
L.append("\n## 4. Paired bootstrap (1000 reps over validation examples, seed 0): method minus arithmetic sum, accuracy in pp\n")
L.append("| pair | task | head | method - arith_sum | OLD eval: diff [95% CI] | FIXED eval: diff [95% CI] | FIXED verdict |")
L.append("|---|---|---|---|---|---|---|")
for pair, task, head in [("seed_hubish", "mnli", "mnli_s7_hubish"), ("seed_hubish", "mnli", "mnli_s42_hubish"),
                         ("mnli_rte_hubish", "mnli", "mnli_s7_hubish"), ("mnli_rte_hubish", "rte", "rte_s42_hubish")]:
    for meth in ("cert_A", "cert_B", "ties_lite", "cert_then_trim"):
        out = {}
        for s in ("none", "bert"):
            out[s] = paired_boot(preds(s, pair, meth, task, head), preds(s, pair, "arith_sum", task, head))
        m, lo, hi = out["bert"]
        verdict = "helps" if lo > 0 else ("hurts" if hi < 0 else "no significant difference")
        if m == 0 and lo == 0 and hi == 0: verdict = "identical (no FAIL layers -> cert == sum)" if meth.startswith("cert_") and meth != "cert_then_trim" else "identical"
        summary["bootstrap"][f"{pair}/{task}/{head}/{meth}-arith_sum"] = {s: {"diff": v[0], "ci95": [v[1], v[2]]} for s, v in out.items()}
        f = lambda v: f"{100 * v[0]:+.2f} [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}]"
        L.append(f"| {pair} | {task} | {head} | {meth} | {f(out['none'])} | {f(out['bert'])} | {verdict} |")
L.append("\nThe primary row is seed_hubish / mnli / head mnli_s7_hubish, cert_A and cert_B vs arith_sum. The original table evaluated only that row.")
(OUT / "TABLE_SEED_FIXED.md").write_text("\n".join(L) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n".join(L))
