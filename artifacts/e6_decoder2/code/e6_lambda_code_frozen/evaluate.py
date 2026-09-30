"""E6 evaluation (POST HOC / EXPLORATORY). Implements RULES.md (sha256 199a9cb7...709f0c) + DEVIATIONS_E6.md C1."""
import numpy as np, pandas as pd, json, itertools
R = "/workspace/lora-paper/e6_lambda"
G4 = [0.3, 0.5, 0.7, 1.0]; G7 = G4 + [1.3, 1.5, 2.0]
FEATS = ["log_norm_ratio", "tv_cosine", "log_O_A", "mean_theta_min_A_deg", "frac_lt30", "sign_conflict_top20",
         "lam_np", "lam_pp", "layer_maxcos", "layer_dominance"]
BB = ["bert", "roberta", "qwen"]; POPS = ["R0", "S1", "P"]
TAUS = [0.02, 0.05, 0.10]; B = 2000; SEED = 20260927

df = pd.read_csv(f"{R}/data/pairs_long.csv")
U = pd.read_csv(f"{R}/data/pairs_unlabeled.csv")
assert (df[["backbone", "pop", "pair"]].values == U[["backbone", "pop", "pair"]].values).all()
N = len(df)

def snap(x, grid):
    g = np.array(grid); d = np.abs(g[None, :] - np.asarray(x)[:, None])
    return g[np.argmin(d + 1e-12 * np.arange(len(g))[None, :], axis=1)]  # ties -> smaller (first)

def argmax_first(M, grid):  # rows: pairs; ties -> smaller lambda
    return np.array(grid)[np.argmax(M, axis=1)]

# reference selections
lam_sel4 = argmax_first(df[[f"hold_{l}" for l in G4]].values, G4)
assert np.allclose(lam_sel4, df.lam_sel.values)
lam_evalopt4 = np.array(G4)[np.argmin(df[[f"Deval_{l}" for l in G4]].values, axis=1)]

def Dat(frame, lam, prefix="Deval_"):
    return np.array([frame.iloc[i][f"{prefix}{l}"] for i, l in enumerate(lam)])

X_all = df[FEATS].values.astype(float)

def ridge_fit_predict(Xf, yf, Xt, alpha=1.0):
    mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1
    Z = (Xf - mu) / sd; ym = yf.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (yf - ym))
    return ((Xt - mu) / sd) @ w + ym

def logit_fit_predict(Xf, yf, Xt, reg=1.0):
    if yf.min() == yf.max(): return np.full(len(Xt), float(yf.mean()))
    mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1
    Z = np.c_[np.ones(len(Xf)), (Xf - mu) / sd]; w = np.zeros(Z.shape[1])
    P = np.eye(Z.shape[1]) * reg; P[0, 0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - yf) + P @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + P
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-10: break
    Zt = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]
    return 1 / (1 + np.exp(-Zt @ w))

def fit_sets(scheme):
    """yield (test_indices, fit_indices) grouped by task-pair set."""
    groups = {}
    for i, r in df.iterrows(): groups.setdefault((r.backbone, frozenset([r.task1, r.task2])), []).append(i)
    for (bb, ts), idx in groups.items():
        ok = ~(df.task1.isin(ts) | df.task2.isin(ts))
        fit = ok & ((df.backbone == bb) if scheme == "within" else (df.backbone != bb))
        yield np.array(idx), np.where(fit)[0]

def fitted_rules(scheme, grid=G4, backbones=None):
    out = {k: np.full(N, np.nan) for k in ["F1", "F2", "F3"]}
    probs = {tau: np.full(N, np.nan) for tau in TAUS}
    Dg = df[[f"Deval_{l}" if f"Deval_{l}" in df else None for l in grid]].values if all(f"Deval_{l}" in df for l in grid) else None
    lam_sel_g = lam_sel4 if grid == G4 else argmax_first(df[[f"hold_{l}" for l in grid]].values, grid)
    for te, fi in fit_sets(scheme):
        if backbones and df.backbone.iloc[te[0]] not in backbones: continue
        Dfit = df.iloc[fi][[f"Deval_{l}" for l in grid]].values
        ok = ~np.isnan(Dfit).any(1); fi2 = fi[ok]; Dfit = Dfit[ok]
        out["F1"][te] = grid[int(np.argmin(Dfit.mean(0)))]
        if grid == G4:
            out["F2"][te] = snap(ridge_fit_predict(X_all[fi2], lam_sel_g[fi2], X_all[te]), grid)
            pred = np.column_stack([ridge_fit_predict(X_all[fi2], Dfit[:, k], X_all[te]) for k in range(len(grid))])
            out["F3"][te] = np.array(grid)[np.argmin(pred, axis=1)]
            Dsel_fit = df.D.values[fi2]
            for tau in TAUS:
                probs[tau][te] = logit_fit_predict(X_all[fi2], (Dsel_fit > tau).astype(float), X_all[te])
    return out, probs

# ---------------- bootstrap machinery ----------------
TASKNAMES = sorted(set(df.task1) | set(df.task2)); TI = {t: i for i, t in enumerate(TASKNAMES)}
t1i = df.task1.map(TI).values; t2i = df.task2.map(TI).values
rng = np.random.default_rng(SEED)
BOOTW = {}
def boot_weights(mask_key, mask):
    """B x n weight matrix for pairs in mask; tasks resampled among tasks present in mask."""
    if mask_key in BOOTW: return BOOTW[mask_key]
    idx = np.where(mask)[0]; present = sorted(set(t1i[idx]) | set(t2i[idx]))
    r = np.random.default_rng(SEED + sum(map(ord, mask_key)))
    W = np.empty((B, len(idx)))
    for b in range(B):
        draw = r.choice(present, size=len(present), replace=True)
        c = np.bincount(draw, minlength=len(TASKNAMES))
        W[b] = c[t1i[idx]] * c[t2i[idx]]
    BOOTW[mask_key] = (idx, W); return idx, W

def wmean_ci(x, key, mask):
    idx, W = boot_weights(key, mask); xv = x[idx]
    est = float(np.mean(xv)); bs = (W @ xv) / W.sum(1)
    return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

def ratio_ci(num, den, key, mask):  # 1 - mean(num)/mean(den)
    idx, W = boot_weights(key, mask)
    est = 1 - num[idx].mean() / den[idx].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = 1 - (W @ num[idx]) / (W @ den[idx])
    return float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

def wauc(s, y, w):
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0: return np.nan
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    cmp = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
    den = wp.sum() * wn.sum()
    return float(wp @ cmp @ wn / den) if den > 0 else np.nan

def auc_ci(s, y, key, mask):
    idx, W = boot_weights(key, mask); sv, yv = s[idx], y[idx]
    est = wauc(sv, yv, np.ones(len(idx)))
    bs = np.array([wauc(sv, yv, W[b]) for b in range(B)])
    return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

def acc_ci(pred, y, key, mask):
    return wmean_ci((pred == y).astype(float), key, mask)

def scopes():
    for bb in BB:
        for pop in POPS:
            yield f"{bb}/{pop}", ((df.backbone == bb) & (df["pop"] == pop)).values
        yield f"{bb}/all", (df.backbone == bb).values
    yield "all/all", np.ones(N, bool)

# ---------------- lambda rules on G4 (full evaluation sets) ----------------
lam = {"W1": np.full(N, 1.0), "W2": np.full(N, 0.5), "W3": snap(df.lam_np.values, G4), "W4": snap(df.lam_pp.values, G4)}
fw, pw = fitted_rules("within"); fl, pl = fitted_rules("lobo")
for k in ["F1", "F2", "F3"]: lam[k + "_within"] = fw[k]; lam[k + "_lobo"] = fl[k]
lam["evalopt"] = lam_evalopt4; lam["sel"] = lam_sel4
Dsel = df.D.values
Drule = {k: Dat(df, v) for k, v in lam.items()}
reg = {k: 100 * (v - Dsel) for k, v in Drule.items()}
RULES_ORDER = ["W1", "W2", "W3", "W4", "F1_within", "F2_within", "F3_within", "F1_lobo", "F2_lobo", "F3_lobo", "evalopt"]
rows = []
for sc, mask in scopes():
    for k in RULES_ORDER:
        m, lo, hi = wmean_ci(reg[k], sc, mask)
        r1 = ratio_ci(reg[k], reg["W1"], sc, mask); r2 = ratio_ci(reg[k], reg["W2"], sc, mask)
        rows.append(dict(scope=sc, rule=k, n=int(mask.sum()), regret_pp=m, regret_lo=lo, regret_hi=hi,
                         meanD_pct=100 * Drule[k][mask].mean(), match_sel=float((lam[k][mask] == lam_sel4[mask]).mean()),
                         removed_vs_W1=r1[0], removed_vs_W1_lo=r1[1], removed_vs_W1_hi=r1[2],
                         removed_vs_W2=r2[0], removed_vs_W2_lo=r2[1], removed_vs_W2_hi=r2[2]))
T_reg = pd.DataFrame(rows); T_reg.to_csv(f"{R}/results/lambda_regret_G4.csv", index=False)

# distribution of lambda_sel and rule outputs
dist = []
for sc, mask in scopes():
    for k in ["sel", "evalopt", "W3", "W4", "F1_within", "F2_within", "F3_within", "F1_lobo", "F2_lobo", "F3_lobo"]:
        v = lam[k][mask]; dist.append(dict(scope=sc, rule=k, **{f"frac_{g}": float((v == g).mean()) for g in G4}))
pd.DataFrame(dist).to_csv(f"{R}/results/lambda_distribution_G4.csv", index=False)
cont = df.groupby("backbone")[["lam_np", "lam_pp"]].describe().T; cont.to_csv(f"{R}/results/continuous_lambda_describe.csv")

# ---------------- Qwen G7 ----------------
q = (df.backbone == "qwen").values
lam_sel7 = np.full(N, np.nan); lam_sel7[q] = argmax_first(df.loc[q, [f"hold_{l}" for l in G7]].values, G7)
lam7 = {"W1": np.full(N, 1.0), "W2": np.full(N, 0.5), "W3": snap(df.lam_np.values, G7), "W4": snap(df.lam_pp.values, G7)}
f7, _ = fitted_rules("within", grid=G7, backbones=["qwen"]); lam7["F1_within"] = f7["F1"]
Dsel7 = np.full(N, np.nan); Dsel7[q] = Dat(df[q], lam_sel7[q])
rows7 = []
for pop in POPS + ["all"]:
    mask = q & ((df["pop"] == pop).values if pop != "all" else True)
    key = f"qwen/{pop}"
    base = {}
    for k, v in lam7.items():
        d = np.full(N, np.nan); d[mask] = Dat(df[mask], v[mask]); base[k] = 100 * (d - Dsel7)
    for k in lam7:
        m, lo, hi = wmean_ci(np.nan_to_num(base[k]), key, mask)
        r1 = ratio_ci(np.nan_to_num(base[k]), np.nan_to_num(base["W1"]), key, mask)
        r2 = ratio_ci(np.nan_to_num(base[k]), np.nan_to_num(base["W2"]), key, mask)
        rows7.append(dict(scope=key, rule=k, regret_pp=m, regret_lo=lo, regret_hi=hi,
                          removed_vs_W1=r1[0], removed_vs_W1_lo=r1[1], removed_vs_W1_hi=r1[2],
                          removed_vs_W2=r2[0], removed_vs_W2_lo=r2[1], removed_vs_W2_hi=r2[2],
                          lam_counts={str(g): int((v[mask] == g).sum()) for g in G7} if (v := lam7[k]) is not None else None))
pd.DataFrame(rows7).to_csv(f"{R}/results/lambda_regret_qwen_G7.csv", index=False)

# ---------------- merge / no-merge ----------------
M2 = {"tv_cosine": df.tv_cosine.values, "O_A": df.O_A.values, "sign_conflict_top20": df.sign_conflict_top20.values,
      "layer_maxcos": df.layer_maxcos.values, "abs_log_norm_ratio": np.abs(df.log_norm_ratio.values),
      "neg_mean_theta_min": -df.mean_theta_min_A_deg.values}
hold_ref = 1 - np.array([df.iloc[i][f"hold_{l}"] for i, l in enumerate(lam_sel4)])
mrows = []
for tau in TAUS:
    y = (Dsel > tau).astype(int)
    scores = {"M1_within": pw[tau], "M1_lobo": pl[tau], **{f"M2_{k}": v for k, v in M2.items()}, "REF_hold_1-holdnorm": hold_ref}
    for sc, mask in scopes():
        prev = float(y[mask].mean())
        for k, s in scores.items():
            a = auc_ci(s, y, sc, mask)
            row = dict(tau=tau, scope=sc, score=k, n=int(mask.sum()), prevalence=prev, auroc=a[0], auroc_lo=a[1], auroc_hi=a[2])
            if k.startswith("M1"):
                ac = acc_ci((s >= 0.5).astype(int), y, sc, mask); row.update(acc=ac[0], acc_lo=ac[1], acc_hi=ac[2])
                pred = (s >= 0.5).astype(int)
                row["bal_acc"] = float(0.5 * ((pred[mask][y[mask] == 1] == 1).mean() if (y[mask] == 1).any() else np.nan) + 0.5 * ((pred[mask][y[mask] == 0] == 0).mean() if (y[mask] == 0).any() else np.nan))
            if k.startswith("REF"):
                ac = acc_ci((s > tau).astype(int), y, sc, mask); row.update(acc=ac[0], acc_lo=ac[1], acc_hi=ac[2])
            mrows.append(row)
        m0 = acc_ci(np.zeros(N, int), y, sc, mask)
        mrows.append(dict(tau=tau, scope=sc, score="M0_always_merge", n=int(mask.sum()), prevalence=prev, acc=m0[0], acc_lo=m0[1], acc_hi=m0[2]))
T_m = pd.DataFrame(mrows); T_m.to_csv(f"{R}/results/merge_decision.csv", index=False)

# ---------------- unlabeled analysis (test remainder) ----------------
lamsU = {}
def gridof(i): return G7 if df.backbone.iloc[i] == "qwen" else G4
def U_select(prefix, grid, frame):
    return argmax_first(frame[[f"{prefix}{l}" for l in grid]].values, grid)
urows = []
for gname in ["G4", "G7"]:
    grid = G4 if gname == "G4" else G7
    msk_all = np.ones(N, bool) if gname == "G4" else q
    Ug = U[msk_all]
    lsel = lam_sel4[msk_all] if gname == "G4" else lam_sel7[msk_all]
    Dt = lambda lv: np.array([Ug.iloc[i][f"Dtest_{l}"] for i, l in enumerate(lv)])
    Dsel_t = Dt(lsel)
    L = {"W1": np.full(len(Ug), 1.0), "W2": np.full(len(Ug), 0.5),
         "W3": snap(df.lam_np.values[msk_all], grid), "W4": snap(df.lam_pp.values[msk_all], grid),
         "U1_agree": U_select("agree_", grid, Ug), "Lcal_labeled_ref": U_select("calnorm_", grid, Ug),
         "evalopt_test": np.array(grid)[np.argmin(Ug[[f"Dtest_{l}" for l in grid]].values, axis=1)]}
    if gname == "G4":
        for k in ["F1_within", "F2_within", "F3_within", "F1_lobo", "F2_lobo", "F3_lobo"]: L[k] = lam[k]
    else:
        L["F1_within"] = lam7["F1_within"][q]
    R_ = {k: 100 * (Dt(v) - Dsel_t) for k, v in L.items()}
    full = lambda v: (lambda a: (a.__setitem__(msk_all, v), a)[1])(np.zeros(N))
    for sc, mask in scopes():
        if gname == "G7" and not sc.startswith("qwen"): continue
        for k in L:
            x = full(R_[k]); m, lo, hi = wmean_ci(x, sc, mask)
            r1 = ratio_ci(x, full(R_["W1"]), sc, mask); r2 = ratio_ci(x, full(R_["W2"]), sc, mask)
            urows.append(dict(grid=gname, scope=sc, rule=k, regret_pp=m, regret_lo=lo, regret_hi=hi,
                              meanDsel_test_pct=100 * full(Dsel_t)[mask].mean(),
                              removed_vs_W1=r1[0], removed_vs_W1_lo=r1[1], removed_vs_W1_hi=r1[2],
                              removed_vs_W2=r2[0], removed_vs_W2_lo=r2[1], removed_vs_W2_hi=r2[2],
                              match_sel=float((full(L[k]) == full(lsel))[mask].mean())))
    if gname == "G4":
        lamsU = L; RegU = R_; DselT = Dsel_t
        Dhat_U = 1 - U[[f"agree_{l}" for l in G4]].values.max(1)
pd.DataFrame(urows).to_csv(f"{R}/results/unlabeled_regret.csv", index=False)
urm = []
for tau in TAUS:
    y = (DselT > tau).astype(int)
    scores = {"M3_unlabeled_Dhat": Dhat_U, "M1_within": pw[tau], "M1_lobo": pl[tau], "M2_tv_cosine": M2["tv_cosine"], "M2_O_A": M2["O_A"], "REF_hold_1-holdnorm": hold_ref}
    for sc, mask in scopes():
        for k, s in scores.items():
            a = auc_ci(s, y, sc, mask); row = dict(tau=tau, scope=sc, score=k, prevalence=float(y[mask].mean()), auroc=a[0], auroc_lo=a[1], auroc_hi=a[2])
            thr = (s > tau) if (k.startswith("M3") or k.startswith("REF")) else (s >= 0.5) if k.startswith("M1") else None
            if thr is not None:
                ac = acc_ci(thr.astype(int), y, sc, mask); row.update(acc=ac[0], acc_lo=ac[1], acc_hi=ac[2])
            urm.append(row)
        m0 = acc_ci(np.zeros(N, int), y, sc, mask)
        urm.append(dict(tau=tau, scope=sc, score="M0_always_merge", prevalence=float(y[mask].mean()), acc=m0[0], acc_lo=m0[1], acc_hi=m0[2]))
pd.DataFrame(urm).to_csv(f"{R}/results/unlabeled_merge_decision.csv", index=False)
# per-pair output
pp = df[["backbone", "pop", "pair", "task1", "task2", "lam_sel", "D", "lam_np", "lam_pp"]].copy()
for k, v in lam.items(): pp[f"lam_{k}"] = v
for k, v in reg.items(): pp[f"regret_{k}"] = v
for k, v in lamsU.items(): pp[f"U_lam_{k}"] = v
for k, v in RegU.items(): pp[f"U_regret_{k}"] = v
for tau in TAUS: pp[f"M1_within_p_tau{tau}"] = pw[tau]; pp[f"M1_lobo_p_tau{tau}"] = pl[tau]
pp["M3_Dhat_U"] = Dhat_U
pp.to_csv(f"{R}/results/per_pair_e6.csv", index=False)
print("done")
