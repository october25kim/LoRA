"""E6 U-analysis data: per pair x lambda, agreement with single adapters on calibration inputs (no labels),
labeled normalized accuracy on the calibration set (reference only), and D on the test remainder (eval minus calibration)."""
import numpy as np, pandas as pd, json, os
from scipy.stats import spearmanr
def score(p, y):
    if p.dtype.kind == "f": return float(spearmanr(p, y).correlation)
    return float((p == y).mean())
R = "/workspace/lora-paper"; D = f"{R}/e6_lambda/data"
TASKS = ['cola','sst2','mrpc','stsb','mnli','qnli','rte','wic','snli','scitail','ag_news','imdb','trec','yelp_polarity']
df = pd.read_csv(f"{D}/pairs_long.csv")
def single_path(bb, a):
    t, s = a.split("@")
    if bb == "bert": return f"{R}/e1b/results/preds/single_{t}.npz" if s == "s0" else f"{R}/e1c/s1/preds/single_{t}.npz"
    return f"{R}/e5/work/{'e4a_roberta' if bb=='roberta' else 'e4b_decoder'}/preds/single_{a}.npz"
def pair_paths(bb, pop, r):
    if bb == "bert" and pop == "R0": return [f"{R}/e1b/results/preds/pair_{r.task1}-{r.task2}.npz"]
    if bb == "bert": return [f"{R}/e1c/preds/pair_{r.pair}.npz"]
    if bb == "roberta": return [f"{R}/e5/work/e4a_roberta/preds/pair_{r.pair}.npz"]
    return [f"{R}/e5/work/e4b_decoder/preds/pair_{r.pair}.npz", f"{R}/e5/e5b/preds/pair_{r.pair}.npz"]
cal_cache = {}
def split(bb, task, n):
    k = (bb, task, n)
    if k not in cal_cache:
        rng = np.random.default_rng(20260927 + TASKS.index(task))
        nc = min(200, n // 2); idx = rng.permutation(n)
        cal = np.zeros(n, bool); cal[idx[:nc]] = True
        cal_cache[k] = cal
    return cal_cache[k]
out = []; maxdiff = 0.0; nchk = 0
for _, r in df.iterrows():
    bb, pop = r["backbone"], r["pop"]
    lams = [0.3, 0.5, 0.7, 1.0] + ([1.3, 1.5, 2.0] if bb == "qwen" else [])
    Z = {}
    for p in pair_paths(bb, pop, r):
        z = np.load(p); Z.update({k: z[k] for k in z.files})
    row = dict(backbone=bb, pop=pop, pair=r.pair)
    for j, (a, t) in enumerate([(r.a1, r.task1), (r.a2, r.task2)], 1):
        s = np.load(single_path(bb, a)); se, lab = s["eval"], s["eval_labels"]
        key_a = t if (bb == "bert" and pop == "R0") else a
        cal = split(bb, t, len(lab)); te = ~cal
        acc_s_full = score(se, lab)
        maxdiff = max(maxdiff, abs(acc_s_full - r[f"single_eval_t{j}"]))
        row[f"ncal{j}"] = int(cal.sum()); row[f"ntest{j}"] = int(te.sum())
        row[f"single_test{j}"] = score(se[te], lab[te]); row[f"single_cal{j}"] = score(se[cal], lab[cal])
        for lam in lams:
            m = Z[f"TA_{key_a}_lam{lam}"]
            assert len(m) == len(lab)
            accf = score(m, lab); maxdiff = max(maxdiff, abs(accf - r[f"eval{j}_{lam}"])); nchk += 1
            row[f"agree{j}_{lam}"] = score(m[cal], se[cal])
            row[f"calacc{j}_{lam}"] = score(m[cal], lab[cal])
            row[f"test{j}_{lam}"] = score(m[te], lab[te])
    for lam in lams:
        row[f"agree_{lam}"] = 0.5 * (row[f"agree1_{lam}"] + row[f"agree2_{lam}"])
        row[f"calnorm_{lam}"] = 0.5 * (row[f"calacc1_{lam}"] / row["single_cal1"] + row[f"calacc2_{lam}"] / row["single_cal2"])
        row[f"Dtest_{lam}"] = 1 - 0.5 * (row[f"test1_{lam}"] / row["single_test1"] + row[f"test2_{lam}"] / row["single_test2"])
    out.append(row)
U = pd.DataFrame(out); U.to_csv(f"{D}/pairs_unlabeled.csv", index=False)
chk = dict(max_abs_diff_recomputed_vs_reported_acc=float(maxdiff), n_checks=nchk, n_pairs=len(U))
json.dump(chk, open(f"{D}/checks_unlabeled.json", "w"), indent=1); print(chk)
