"""E6 (post hoc, exploratory): consolidate per-pair lambda-grid results + weight features.
Outputs data/pairs_long.csv (one row per backbone x population x pair) and data/checks.json."""
import json, csv, os, math, collections
import numpy as np, pandas as pd
R = "/workspace/lora-paper"; D = f"{R}/e6_lambda/data"
G4 = [0.3, 0.5, 0.7, 1.0]; GX = [1.3, 1.5, 2.0]

def jl(p):
    out, seen = [], set()
    for l in open(p):
        r = json.loads(l)
        if r["pair"] in seen: continue  # E4a S1 duplicate (bitwise identical, DEVIATIONS_E4A N4)
        seen.add(r["pair"]); out.append(r)
    return out

rows = []
def add(bb, pop, recs, preds, ext=None):
    pr = preds.set_index("pair")
    exd = {r["pair"]: r for r in ext} if ext else {}
    for r in recs:
        pair = r["pair"]
        if bb == "bert" and pop == "R0":
            a1, a2 = r["t1"] + "@s0", r["t2"] + "@s0"; k1, k2 = r["t1"], r["t2"]
            # e1b csv uses task-name columns
            acc = lambda t, which, lam: r.get(f"TA_{which}_{t}_lam{lam}")
            t1n, t2n = r["t1"], r["t2"]
        else:
            a1, a2 = r["t1"], r["t2"]; t1n, t2n = r["t1_task"], r["t2_task"]
            acc = None
        row = dict(backbone=bb, pop=pop, pair=pair, a1=a1, a2=a2, task1=t1n, task2=t2n,
                   lam_sel=r["lam_selected"], D=r["D"], single_eval_t1=r["single_eval_t1"], single_eval_t2=r["single_eval_t2"])
        for lam in G4:
            row[f"Deval_{lam}"] = r[f"TA_eval_D_lam{lam}"]; row[f"hold_{lam}"] = r[f"TA_hold_norm_lam{lam}"]
            if bb == "bert" and pop == "R0":
                row[f"eval1_{lam}"] = r[f"TA_eval_{t1n}_lam{lam}"]; row[f"eval2_{lam}"] = r[f"TA_eval_{t2n}_lam{lam}"]
            else:
                row[f"eval1_{lam}"] = r[f"TA_eval_t1_lam{lam}"]; row[f"eval2_{lam}"] = r[f"TA_eval_t2_lam{lam}"]
        if ext:
            e = exd[pair]
            for lam in GX:
                row[f"Deval_{lam}"] = e[f"TA_eval_D_lam{lam}"]; row[f"hold_{lam}"] = e[f"TA_hold_norm_lam{lam}"]
                row[f"eval1_{lam}"] = e[f"TA_eval_t1_lam{lam}"]; row[f"eval2_{lam}"] = e[f"TA_eval_t2_lam{lam}"]
        p = pr.loc[pair]
        for c in ["O_A", "tv_cosine", "mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A",
                  "sign_conflict_top20", "sign_conflict_all", "norm_ratio", "tvnorm_t1", "tvnorm_t2", "n_layers"]:
            row[c] = p[c]
        rows.append(row)

add("bert", "R0", jl(f"{D}/src/bert_R0_pair_results.jsonl"), pd.read_csv(f"{D}/src/bert_R0_predictors.csv"))
pe = pd.read_csv(f"{D}/src/bert_e1c_predictors.csv")
for pop in ["S1", "P"]:
    add("bert", pop, jl(f"{D}/src/bert_{pop}_pair_results.jsonl"), pe)
for bb in ["roberta", "qwen"]:
    pp = pd.read_csv(f"{D}/src/{bb}_predictors.csv")
    for pop in ["R0", "S1", "P"]:
        ext = jl(f"{D}/src/qwen_{pop}_pair_ext_e5b.jsonl") if bb == "qwen" else None
        add(bb, pop, jl(f"{D}/src/{bb}_{pop}_pair_results.jsonl"), pp, ext)
df = pd.DataFrame(rows)

# ---- weight-derived per-layer features ----
checks = {}
feats = []
for bb in ["bert", "roberta", "qwen"]:
    n2 = pd.read_csv(f"{D}/layer_stats/layer_norms_{bb}.csv").pivot(index="adapter", columns="layer", values="n2")
    ip = pd.read_csv(f"{D}/layer_stats/layer_ip_{bb}.csv")
    ipd = {(a, b): g for (a, b), g in ip.groupby(["a1", "a2"])}
    sub = df[df.backbone == bb]
    for i, r in sub.iterrows():
        a1, a2 = r.a1, r.a2
        g = ipd.get((a1, a2)); sw = False
        if g is None: g = ipd[(a2, a1)]
        g = g.set_index("layer")["ip"].reindex(n2.columns).values
        x1 = n2.loc[a1].values; x2 = n2.loc[a2].values
        N1, N2 = math.sqrt(x1.sum()), math.sqrt(x2.sum()); IP = g.sum()
        cos = IP / (N1 * N2)
        S = math.sqrt(N1**2 + N2**2 + 2 * IP)
        lam_np = 0.5 * (N1 + N2) / S
        lpp1 = x1 / np.maximum(x1 + g, 1e-30); lpp2 = x2 / np.maximum(x2 + g, 1e-30)
        lpp = np.clip(0.5 * (lpp1 + lpp2), 0.25, 4.0)
        w = x1 + x2
        lam_pp = float((w * lpp).sum() / w.sum())
        cosl = g / np.sqrt(x1 * x2)
        feats.append(dict(idx=i, N1_w=N1, N2_w=N2, cos_w=cos, lam_np=lam_np, lam_pp=lam_pp,
                          layer_maxcos=float(np.max(np.abs(cosl))), layer_wcos=float((w * cosl).sum() / w.sum()),
                          layer_dominance=float((w * np.abs(0.5 * np.log(x1 / x2))).sum() / w.sum())))
F = pd.DataFrame(feats).set_index("idx")
df = df.join(F)
# validation of weight features vs frozen predictor table
df["_ratio_chk"] = np.maximum(df.N1_w, df.N2_w) / np.minimum(df.N1_w, df.N2_w)
for bb in ["bert", "roberta", "qwen"]:
    s = df[df.backbone == bb]
    checks[bb] = dict(n=int(len(s)),
        corr_cos_vs_tv_cosine=float(np.corrcoef(s.cos_w, s.tv_cosine)[0, 1]),
        max_abs_cos_diff=float(np.max(np.abs(s.cos_w - s.tv_cosine))),
        corr_ratio_vs_norm_ratio=float(np.corrcoef(s._ratio_chk, s.norm_ratio)[0, 1]),
        max_rel_norm1_diff=float(np.max(np.abs(s.N1_w / s.tvnorm_t1 - 1))),
        norm_ratio_min=float(s.norm_ratio.min()))
# sanity: D at lam_sel equals Deval at lam_sel; lam_sel on G4
chk = [abs(r.D - r[f"Deval_{r.lam_sel}"]) for _, r in df.iterrows()]
checks["max_abs_D_vs_Deval_at_lamsel"] = float(max(chk))
checks["counts"] = {f"{b}_{p}": int(n) for (b, p), n in df.groupby(["backbone", "pop"]).size().items()}
# derived scale-free features
df["log_norm_ratio"] = np.log(df.norm_ratio)
df["log_O_A"] = np.log(df.O_A)
df["frac_lt30"] = df.n_layers_theta_min_lt30_A / df.n_layers
df = df.drop(columns=["_ratio_chk"])
df.to_csv(f"{D}/pairs_long.csv", index=False)
json.dump(checks, open(f"{D}/checks.json", "w"), indent=1)
print(json.dumps(checks, indent=1))
