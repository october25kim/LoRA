#!/usr/bin/env python3
"""E6 confirmatory secondary hypotheses on the lambda rule U1 and the merge-decision score M3 (PREREG_E6.md sec. 6c).
Rules, splits, statistics and thresholds are FROZEN from the post hoc analysis /workspace/lora-paper/e6_lambda (code copied read-only to
e6_lambda_code_frozen/, sha256 in e6_lambda_code_frozen/SHA256SUMS). Functions marked VERBATIM are copied unchanged from
e6_lambda/code/build_unlabeled.py (score, split), evaluate.py (snap, argmax_first, boot_weights, wmean_ci, ratio_ci, wauc, auc_ci, acc_ci)
and evaluate_addon.py (best_thr; used only to reproduce the frozen threshold). Only file paths / the table builder are new.
TIE note: agreement rates are ratios of small integers, so exact ties between lambdas occur; e6_lambda read the agreement table back
from CSV, which happened to resolve them to the smaller lambda (its stated tie rule). Here ties are made explicit by rounding the
agreement / calibration scores to 12 decimals before argmax_first (ties -> smaller lambda). This reproduces e6_lambda's U1 selection on
all 273 Qwen2.5-0.5B pairs (self-test), whereas raw floats differed in 1 pair by 1e-16.
CPU only. Inputs: pair_results_{R0,S1,P}.jsonl (with the extended-lambda fields), preds/pair_*.npz, preds/single_*.npz.
Usage: python lam_confirm_e6.py [--selftest-e4b BOXROOT]  -> lam_confirm_e6.json / lam_confirm_e6_per_pair.csv"""
import json, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
G4 = [0.3, 0.5, 0.7, 1.0]; G7 = G4 + [1.3, 1.5, 2.0]; B = 2000; SEED = 20260927
TASKS = ['cola','sst2','mrpc','stsb','mnli','qnli','rte','wic','snli','scitail','ag_news','imdb','trec','yelp_polarity']
FROZEN = {"M3_threshold_tau0.05": 0.1174999999999999, "M3_threshold_tau0.02": 0.0504428860721516, "M3_threshold_tau0.1": 0.2375,
          "rule": "predict harmful merge iff Dhat_U >= threshold (thresholds = accuracy-maximizing cut on all 780 e6_lambda pairs, evaluate_addon.best_thr, computed 2026-09-27 before any E6 merge)"}
H_THR = {"H-U1": "task-block 95% CI upper < 0 for BOTH paired differences regret(U1)-regret(lam=0.7) and regret(U1)-regret(lam=1.0)",
         "H-U1b": "U1 removes >= 80% of the avoidable loss of lam=1.0: 1 - mean regret(U1)/mean regret(lam=1) >= 0.80 (point estimate; CI reported); NOT APPLICABLE if mean regret(lam=1) <= 0 (no avoidable loss to remove)",
         "H-M3": "AUROC of Dhat_U for y = 1[D_test(lam_sel) > 0.05] >= 0.75 AND task-block CI lower bound > 0.5",
         "H-M3acc": "accuracy of the frozen-threshold M3 decision minus always-merge accuracy: task-block CI lower bound > 0"}


# ---------------- VERBATIM (build_unlabeled.py)
def score(p, y):
    if p.dtype.kind == "f": return float(spearmanr(p, y).correlation)
    return float((p == y).mean())
cal_cache = {}
def split(bb, task, n):
    k = (bb, task, n)
    if k not in cal_cache:
        rng = np.random.default_rng(20260927 + TASKS.index(task))
        nc = min(200, n // 2); idx = rng.permutation(n)
        cal = np.zeros(n, bool); cal[idx[:nc]] = True
        cal_cache[k] = cal
    return cal_cache[k]


# ---------------- VERBATIM (evaluate.py)
def snap(x, grid):
    g = np.array(grid); d = np.abs(g[None, :] - np.asarray(x)[:, None])
    return g[np.argmin(d + 1e-12 * np.arange(len(g))[None, :], axis=1)]  # ties -> smaller (first)

def argmax_first(M, grid):  # rows: pairs; ties -> smaller lambda
    return np.array(grid)[np.argmax(M, axis=1)]


def wauc(s, y, w):
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0: return np.nan
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    cmp = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
    den = wp.sum() * wn.sum()
    return float(wp @ cmp @ wn / den) if den > 0 else np.nan


class Boot:
    """evaluate.py bootstrap machinery (boot_weights / wmean_ci / ratio_ci / auc_ci / acc_ci), VERBATIM bodies, bound to a pair table."""
    def __init__(self, df):
        self.TASKNAMES = sorted(set(df.task1) | set(df.task2)); TI = {t: i for i, t in enumerate(self.TASKNAMES)}
        self.t1i = df.task1.map(TI).values; self.t2i = df.task2.map(TI).values; self.BOOTW = {}
    def boot_weights(self, mask_key, mask):
        TASKNAMES, t1i, t2i, BOOTW = self.TASKNAMES, self.t1i, self.t2i, self.BOOTW
        if mask_key in BOOTW: return BOOTW[mask_key]
        idx = np.where(mask)[0]; present = sorted(set(t1i[idx]) | set(t2i[idx]))
        r = np.random.default_rng(SEED + sum(map(ord, mask_key)))
        W = np.empty((B, len(idx)))
        for b in range(B):
            draw = r.choice(present, size=len(present), replace=True)
            c = np.bincount(draw, minlength=len(TASKNAMES))
            W[b] = c[t1i[idx]] * c[t2i[idx]]
        BOOTW[mask_key] = (idx, W); return idx, W
    def wmean_ci(self, x, key, mask):
        idx, W = self.boot_weights(key, mask); xv = x[idx]
        est = float(np.mean(xv)); bs = (W @ xv) / W.sum(1)
        return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    def ratio_ci(self, num, den, key, mask):  # 1 - mean(num)/mean(den)
        idx, W = self.boot_weights(key, mask)
        est = 1 - num[idx].mean() / den[idx].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = 1 - (W @ num[idx]) / (W @ den[idx])
        return float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    def auc_ci(self, s, y, key, mask):
        idx, W = self.boot_weights(key, mask); sv, yv = s[idx], y[idx]
        est = wauc(sv, yv, np.ones(len(idx)))
        bs = np.array([wauc(sv, yv, W[b]) for b in range(B)])
        return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    def acc_ci(self, pred, y, key, mask):
        return self.wmean_ci((pred == y).astype(float), key, mask)


# ---------------- VERBATIM (evaluate_addon.py)
def best_thr(sf, yf):
    c = np.unique(sf); c = np.r_[c, c.max() + 1]
    acc = [((sf >= t).astype(int) == yf).mean() for t in c]
    return c[int(np.argmax(acc))]


# ---------------- NEW: table builder (same quantities as build_dataset.py / build_unlabeled.py)
def jl(p):
    out, seen = [], set()
    for l in Path(p).read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if r["pair"] in seen: continue
        seen.add(r["pair"]); out.append(r)
    return out


def build(recs_by_pop, ext_by_pop, pred_dirs, single_dir, bb):
    rows = []
    for pop, recs in recs_by_pop.items():
        exd = {r["pair"]: r for r in ext_by_pop.get(pop, [])}
        for r in recs:
            e = exd.get(r["pair"], r)
            row = dict(backbone=bb, pop=pop, pair=r["pair"], a1=r["t1"], a2=r["t2"], task1=r["t1_task"], task2=r["t2_task"],
                       lam_sel=r["lam_selected"], D=r["D"], single_eval_t1=r["single_eval_t1"], single_eval_t2=r["single_eval_t2"])
            for lam in G7:
                src = r if lam in G4 else e
                row[f"Deval_{lam}"] = src[f"TA_eval_D_lam{lam}"]; row[f"hold_{lam}"] = src[f"TA_hold_norm_lam{lam}"]
                row[f"eval1_{lam}"] = src[f"TA_eval_t1_lam{lam}"]; row[f"eval2_{lam}"] = src[f"TA_eval_t2_lam{lam}"]
            Z = {}
            for d in pred_dirs:
                p = Path(d) / f"pair_{r['pair']}.npz"
                if p.exists():
                    z = np.load(p); Z.update({k: z[k] for k in z.files})
            maxdiff = 0.0
            for j, (a, t) in enumerate([(r["t1"], r["t1_task"]), (r["t2"], r["t2_task"])], 1):
                s = np.load(Path(single_dir) / f"single_{a}.npz"); se, lab = s["eval"], s["eval_labels"]
                cal = split(bb, t, len(lab)); te = ~cal
                maxdiff = max(maxdiff, abs(score(se, lab) - r[f"single_eval_t{j}"]))
                row[f"ncal{j}"] = int(cal.sum()); row[f"ntest{j}"] = int(te.sum())
                row[f"single_test{j}"] = score(se[te], lab[te]); row[f"single_cal{j}"] = score(se[cal], lab[cal])
                for lam in G7:
                    m = Z[f"TA_{a}_lam{lam}"]; assert len(m) == len(lab)
                    maxdiff = max(maxdiff, abs(score(m, lab) - row[f"eval{j}_{lam}"]))
                    row[f"agree{j}_{lam}"] = score(m[cal], se[cal]); row[f"calacc{j}_{lam}"] = score(m[cal], lab[cal]); row[f"test{j}_{lam}"] = score(m[te], lab[te])
            for lam in G7:
                row[f"agree_{lam}"] = 0.5 * (row[f"agree1_{lam}"] + row[f"agree2_{lam}"])
                row[f"calnorm_{lam}"] = 0.5 * (row[f"calacc1_{lam}"] / row["single_cal1"] + row[f"calacc2_{lam}"] / row["single_cal2"])
                row[f"Dtest_{lam}"] = 1 - 0.5 * (row[f"test1_{lam}"] / row["single_test1"] + row[f"test2_{lam}"] / row["single_test2"])
            row["recompute_maxdiff"] = maxdiff
            rows.append(row)
    return pd.DataFrame(rows)


def analyse(df, bb):
    N = len(df); BT = Boot(df)
    lam_sel4 = argmax_first(df[[f"hold_{l}" for l in G4]].values, G4)
    assert np.allclose(lam_sel4, df.lam_sel.values), "held-out selection mismatch"
    lam_sel7 = argmax_first(df[[f"hold_{l}" for l in G7]].values, G7)
    Dt = lambda lv: np.array([df.iloc[i][f"Dtest_{l}"] for i, l in enumerate(lv)])
    out = {"n_pairs": N, "recompute_maxdiff": float(df.recompute_maxdiff.max()), "frozen": FROZEN, "hypothesis_rules": H_THR, "scopes": {}}
    scopes = [(f"{bb}/all", np.ones(N, bool))] + [(f"{bb}/{p}", (df["pop"] == p).values) for p in ("P", "R0", "S1") if (df["pop"] == p).any()]
    for gname, grid, lsel in (("G4", G4, lam_sel4), ("G7", G7, lam_sel7)):
        Dsel_t = Dt(lsel)
        L = {"W1_lam1.0": np.full(N, 1.0), "fixed_lam0.7": np.full(N, 0.7), "W2_lam0.5": np.full(N, 0.5),
             "U1_agree": argmax_first(np.round(df[[f"agree_{l}" for l in grid]].values, 12), grid),       # exact ties -> smaller lambda (see TIE note)
             "Lcal_labeled_ref": argmax_first(np.round(df[[f"calnorm_{l}" for l in grid]].values, 12), grid),
             "evalopt_test": np.array(grid)[np.argmin(df[[f"Dtest_{l}" for l in grid]].values, axis=1)]}
        REG = {k: 100 * (Dt(v) - Dsel_t) for k, v in L.items()}
        for sc, mask in scopes:
            key = sc if gname == "G4" else sc      # same task-block draws for both grids within a scope
            e = out["scopes"].setdefault(sc, {}); g = e.setdefault(gname, {})
            g["regret_pp"] = {k: BT.wmean_ci(v, key, mask) for k, v in REG.items()}
            g["U1_minus_fixed0.7_pp"] = BT.wmean_ci(REG["U1_agree"] - REG["fixed_lam0.7"], key, mask)
            g["U1_minus_lam1.0_pp"] = BT.wmean_ci(REG["U1_agree"] - REG["W1_lam1.0"], key, mask)
            g["U1_minus_Lcal_pp"] = BT.wmean_ci(REG["U1_agree"] - REG["Lcal_labeled_ref"], key, mask)
            g["U1_removed_vs_lam1.0"] = BT.ratio_ci(REG["U1_agree"], REG["W1_lam1.0"], key, mask)
            g["U1_removed_vs_fixed0.7"] = BT.ratio_ci(REG["U1_agree"], REG["fixed_lam0.7"], key, mask)
            g["U1_lambda_counts"] = {str(l): int((L["U1_agree"][mask] == l).sum()) for l in grid}
            g["lam_sel_counts"] = {str(l): int((lsel[mask] == l).sum()) for l in grid}
            g["U1_match_sel"] = float((L["U1_agree"] == lsel)[mask].mean())
            if gname == "G4":
                Dhat = 1 - df[[f"agree_{l}" for l in G4]].values.max(1)
                e["M3"] = {}
                for tau in (0.02, 0.05, 0.10):
                    y = (Dsel_t > tau).astype(int); thr = FROZEN[f"M3_threshold_tau{tau:g}" if tau != 0.1 else "M3_threshold_tau0.1"]
                    pred = (Dhat >= thr).astype(int)
                    e["M3"][str(tau)] = {"prevalence": float(y[mask].mean()), "auroc": BT.auc_ci(Dhat, y, key, mask),
                                         "acc_frozen_thr": BT.acc_ci(pred, y, key, mask), "acc_always_merge": BT.acc_ci(np.zeros(N, int), y, key, mask),
                                         "acc_diff_vs_always_merge": BT.wmean_ci((pred == y).astype(float) - (y == 0).astype(float), key, mask),
                                         "acc_unfitted_Dhat_gt_tau": BT.acc_ci((Dhat > tau).astype(int), y, key, mask),
                                         "recall_bad": float(((pred == 1) & (y == 1))[mask].sum() / max(int((y[mask] == 1).sum()), 1)),
                                         "precision_bad": float(((pred == 1) & (y == 1))[mask].sum() / max(int((pred[mask] == 1).sum()), 1))}
    # decisions (primary scope = <bb>/all, G4, tau 0.05)
    P = out["scopes"][f"{bb}/all"]; g = P["G4"]; m = P["M3"]["0.05"]
    d = {"H-U1": bool(g["U1_minus_fixed0.7_pp"][2] < 0 and g["U1_minus_lam1.0_pp"][2] < 0),
         "H-U1b": (None if not g["regret_pp"]["W1_lam1.0"][0] > 0 else bool(g["U1_removed_vs_lam1.0"][0] >= 0.80)),
         "H-M3": bool(np.isfinite(m["auroc"][0]) and m["auroc"][0] >= 0.75 and m["auroc"][1] > 0.5),
         "H-M3acc": bool(m["acc_diff_vs_always_merge"][1] > 0)}
    out["decisions_primary_scope"] = {k: ("NOT APPLICABLE (mean regret of lam=1 <= 0)" if v is None else "SUPPORTED" if v else "NOT SUPPORTED") for k, v in d.items()}
    out["primary_scope"] = f"{bb}/all (R0 + S1 + P cross-task pairs), grid G4, tau 0.05"
    return out, L, REG


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest-e4b", default=None); a = ap.parse_args()
    if a.selftest_e4b:                    # reproduce the e6_lambda qwen numbers from the E4b/E5b files (box paths)
        R = Path(a.selftest_e4b)
        src = R / "e6_lambda" / "data" / "src"
        recs = {p: jl(src / f"qwen_{p}_pair_results.jsonl") for p in ("R0", "S1", "P")}
        ext = {p: jl(src / f"qwen_{p}_pair_ext_e5b.jsonl") for p in ("R0", "S1", "P")}
        df = build(recs, ext, [R / "e5/work/e4b_decoder/preds", R / "e5/e5b/preds"], R / "e5/work/e4b_decoder/preds", "qwen")
        out, _, _ = analyse(df, "qwen")
        print(json.dumps({k: out["scopes"]["qwen/all"]["G4"][k] for k in ("regret_pp", "U1_removed_vs_lam1.0", "U1_minus_fixed0.7_pp", "U1_minus_lam1.0_pp")}, indent=1))
        print(json.dumps(out["scopes"]["qwen/all"]["M3"]["0.05"], indent=1)); print("recompute maxdiff", out["recompute_maxdiff"])
        (Path.cwd() / "lam_confirm_selftest_e4b.json").write_text(json.dumps(out, indent=1, default=float)); return
    PO = json.loads((HERE / "populations_e6.json").read_text())
    recs = {}
    for p in ("R0", "S1", "P"):
        jp = HERE / f"pair_results_{p}.jsonl"
        if jp.exists():
            r = jl(jp)
            if len(r) == PO["n"][p]: recs[p] = r
    df = build(recs, {}, [HERE / "preds"], HERE / "preds", "qwen15")
    out, L, REG = analyse(df, "qwen15")
    out["populations_used"] = sorted(recs)
    (HERE / "lam_confirm_e6.json").write_text(json.dumps(out, indent=1, default=float))
    pp = df[["pop", "pair", "task1", "task2", "lam_sel", "D"]].copy()
    for k, v in L.items(): pp[f"lam_{k}"] = v
    for k, v in REG.items(): pp[f"regret_{k}"] = v
    pp["M3_Dhat_U"] = 1 - df[[f"agree_{l}" for l in G4]].values.max(1)
    pp.to_csv(HERE / "lam_confirm_e6_per_pair.csv", index=False)
    print(json.dumps(out["decisions_primary_scope"]))


if __name__ == "__main__":
    main()
