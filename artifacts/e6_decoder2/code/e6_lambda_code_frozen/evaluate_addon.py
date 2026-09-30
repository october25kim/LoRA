"""E6 add-on analyses. ADDED AFTER RULES.md and after the first results were seen (see DEVIATIONS_E6.md A1-A3).
A1: paired task-block CIs for rule differences (operationalizes the 'practically useful' reading).
A2: M3c, a calibrated threshold for the unlabeled disagreement score (threshold fitted on other tasks' pairs).
A3: usefulness table per scope."""
import numpy as np, pandas as pd, json
R = "/workspace/lora-paper/e6_lambda"; B = 2000; SEED = 20260927
pp = pd.read_csv(f"{R}/results/per_pair_e6.csv"); U = pd.read_csv(f"{R}/data/pairs_unlabeled.csv")
N = len(pp); TAUS = [0.02, 0.05, 0.10]
TN = sorted(set(pp.task1) | set(pp.task2)); TI = {t: i for i, t in enumerate(TN)}
t1i = pp.task1.map(TI).values; t2i = pp.task2.map(TI).values
cache = {}
def bw(key, mask):
    if key in cache: return cache[key]
    idx = np.where(mask)[0]; present = sorted(set(t1i[idx]) | set(t2i[idx]))
    r = np.random.default_rng(SEED + sum(map(ord, key)))
    W = np.empty((B, len(idx)))
    for b in range(B):
        c = np.bincount(r.choice(present, len(present), replace=True), minlength=len(TN)); W[b] = c[t1i[idx]] * c[t2i[idx]]
    cache[key] = (idx, W); return cache[key]
def mci(x, key, mask):
    idx, W = bw(key, mask); v = x[idx]; bs = (W @ v) / W.sum(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def scopes():
    for bb in ["bert", "roberta", "qwen"]:
        for pop in ["R0", "S1", "P"]: yield f"{bb}/{pop}", ((pp.backbone == bb) & (pp["pop"] == pop)).values
        yield f"{bb}/all", (pp.backbone == bb).values
    yield "all/all", np.ones(N, bool)
# A1 + A3 on full-eval regrets (G4) and U-analysis regrets (test remainder)
rows = []
for src, pref, rules in [("full_eval", "regret_", ["W3", "F1_within", "F2_within", "F3_within", "F1_lobo", "F2_lobo", "F3_lobo"]),
                         ("test_remainder", "U_regret_", ["W3", "F1_within", "F1_lobo", "U1_agree", "Lcal_labeled_ref"])]:
    for sc, mask in scopes():
        r1, r2 = pp[f"{pref}W1"].values, pp[f"{pref}W2"].values
        best = "W1" if r1[mask].mean() <= r2[mask].mean() else "W2"; rb = r1 if best == "W1" else r2
        for k in rules:
            x = pp[f"{pref}{k}"].values
            d = mci(x - rb, sc, mask); m = mci(x, sc, mask)
            removed = 1 - x[mask].mean() / rb[mask].mean()
            rows.append(dict(source=src, scope=sc, rule=k, better_baseline=best, regret_pp=m[0], regret_lo=m[1], regret_hi=m[2],
                             diff_vs_better_pp=d[0], diff_lo=d[1], diff_hi=d[2], removed_vs_better=removed,
                             useful=bool(d[2] < 0 and removed >= 0.5)))
        if src == "test_remainder":
            x = pp["U_regret_U1_agree"].values - pp["U_regret_W3"].values; d = mci(x, sc, mask)
            rows.append(dict(source=src, scope=sc, rule="U1_minus_W3", diff_vs_better_pp=d[0], diff_lo=d[1], diff_hi=d[2]))
            x = pp["U_regret_U1_agree"].values - pp["U_regret_Lcal_labeled_ref"].values; d = mci(x, sc, mask)
            rows.append(dict(source=src, scope=sc, rule="U1_minus_Lcal", diff_vs_better_pp=d[0], diff_lo=d[1], diff_hi=d[2]))
pd.DataFrame(rows).to_csv(f"{R}/results/addon_paired_usefulness.csv", index=False)
# A2 calibrated threshold for M3 (both-tasks-out within backbone; and LOBO)
G4 = [0.3, 0.5, 0.7, 1.0]
Dsel_t = np.array([U.iloc[i][f"Dtest_{l}"] for i, l in enumerate(pp.lam_sel.values)])
s = pp.M3_Dhat_U.values
def best_thr(sf, yf):
    c = np.unique(sf); c = np.r_[c, c.max() + 1]
    acc = [((sf >= t).astype(int) == yf).mean() for t in c]
    return c[int(np.argmax(acc))]
out = []
for tau in TAUS:
    y = (Dsel_t > tau).astype(int)
    for scheme in ["within", "lobo"]:
        pred = np.zeros(N, int)
        for i in range(N):
            ts = {pp.task1[i], pp.task2[i]}
            ok = ~(pp.task1.isin(ts) | pp.task2.isin(ts)).values
            ok &= (pp.backbone == pp.backbone[i]).values if scheme == "within" else (pp.backbone != pp.backbone[i]).values
            pred[i] = int(s[i] >= best_thr(s[ok], y[ok]))
        for sc, mask in scopes():
            a = mci((pred == y).astype(float), sc, mask); a0 = mci((y == 0).astype(float), sc, mask)
            d = mci((pred == y).astype(float) - (y == 0).astype(float), sc, mask)
            tp = ((pred == 1) & (y == 1))[mask].sum(); fp = ((pred == 1) & (y == 0))[mask].sum(); fn = ((pred == 0) & (y == 1))[mask].sum()
            out.append(dict(tau=tau, scheme=scheme, scope=sc, prevalence=float(y[mask].mean()), acc=a[0], acc_lo=a[1], acc_hi=a[2],
                            acc_M0=a0[0], diff_vs_M0=d[0], diff_lo=d[1], diff_hi=d[2],
                            recall_bad=float(tp / max(tp + fn, 1)), precision_bad=float(tp / max(tp + fp, 1))))
pd.DataFrame(out).to_csv(f"{R}/results/addon_M3_calibrated.csv", index=False)
print("done")
