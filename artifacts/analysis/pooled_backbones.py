#!/usr/bin/env python3
"""EXPLORATORY per-backbone and across-backbone pooled summaries of predictor-vs-merge-loss Spearman rho.

NOT pre-registered. Written 2026-09-27 (KST) after all E1b, E1c, E4a (RoBERTa-base) and E4b (Qwen2.5-0.5B) results were known.
It changes no pre-registered verdict. Every number it prints is "derived".

Backbones and populations (three per backbone, sharing tasks, evaluation sets and, for P, the very adapters of R0/S1):
  BERT-base      : R0 = E1b (seed 0), P = E1c mixed seed, S1 = E1c seed 1;  K = 14 tasks, 91 pairs each
  RoBERTa-base   : R0, P, S1 of E4a;                                          K = 13 tasks (RTE excluded by rule b), 78 pairs each
  Qwen2.5-0.5B   : R0, P, S1 of E4b;                                          K = 14 tasks, 91 pairs each

(1) Per-backbone pooled rho:  rho_bar_b = mean of the three population rhos (equal weights), exactly as analysis/pooled_rho.py.
    CI: task-block bootstrap resampling the backbone's K tasks JOINTLY across its three populations (2,000 reps, default_rng(0),
    construction of e1b.stage3 / e1c pop_stats / e4a pop_stats; skip replicates with < 4 distinct pairs; percentile 95%).
    Joint task-label permutation (10,000, default_rng(0)) gives a descriptive one-sided p. Sanity: per-population CIs from the joint
    draws must reproduce the pre-registered analysis files (same RNG stream and task order).
(2) Across-backbone summary (exploratory): equal-weight mean of the three rho_bar_b.
    Primary uncertainty: ONE draw of task labels from the 14-task union per replicate, applied to all three backbones (pairs that
    contain a task not valid for a backbone -- RTE for RoBERTa -- are dropped for that backbone). This keeps the dependence induced by
    the shared tasks (13 of 14 tasks are common to all backbones) inside every replicate (2,000 reps, default_rng(20260927)).
    Sensitivity: independent within-backbone task draws (default_rng(101/202/303)), which ignore that dependence.
    Also a DerSimonian-Laird random-effects summary on the Fisher-z scale (variances from the within-backbone bootstrap), with
    Cochran's Q and I^2. With k = 3 backbones and dependent backbones this is descriptive only.
(3) Heterogeneity: bootstrap distribution of the differences Qwen - BERT, Qwen - RoBERTa, Qwen - mean(encoders), RoBERTa - BERT of
    rho_bar_b under the joint (primary) and independent (sensitivity) draws; percentile 95% CI and an approximate two-sided bootstrap
    p = 2 min(P(diff <= 0), P(diff >= 0)).
(4) Layer-exclusion check (Qwen; RoBERTa for comparison): O_A recomputed from predictors_layers_*.csv (layer mean of cos2_mean_A)
    after removing layer-0 k_proj, all k_proj, all d_out = 128 layers (k_proj and v_proj under GQA), the three sub-30 degree hot spots,
    every layer that is below 30 degrees in any Qwen pair, and, per pair, the layers below 30 degrees in that pair; plus a
    chance-normalized O_A (layer mean of cos2_mean_A / (r / d_out), r = 8) and O_A restricted by d_out. Per population rho with the
    pre-registered task-block CI construction, and rho_bar with the joint within-backbone CI.

Run:  /workspace/lora-paper/e1/.venv/bin/python analysis/pooled_backbones.py
Outputs: analysis/pooled_backbones.json, analysis/pooled_backbones.md
"""
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, chi2

ROOT = Path(__file__).resolve().parents[1]
E1B = ROOT / "e1b" / "results"; E1C = ROOT / "e1c"; E4A = ROOT / "e4a"; E4B = ROOT / "e4b"
OUT = ROOT / "analysis"
PREDICTORS = ["O_A", "tv_cosine", "mean_theta_min_A_deg", "min_theta_min_A_deg", "sign_conflict_top20", "sign_conflict_all", "norm_ratio"]
PRIMARY = ["O_A", "tv_cosine"]
POPS = ["R0", "P", "S1"]
BACKBONES = ["BERT", "RoBERTa", "Qwen"]
NBOOT, NPERM = 2000, 10000
R_LORA = 8


def spearman(x, y):                                    # identical to e1b.spearman
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ------------------------------------------------------------------ loading
def load_bert():
    tasks = json.loads((E1B / "stage0.json").read_text())["valid_tasks"]
    pr0 = pd.read_csv(E1B / "predictors.csv"); rr0 = pd.read_csv(E1B / "pair_results.csv")
    r0 = pr0.merge(rr0[["pair", "D"]], on="pair"); r0["t1_task"] = r0.t1; r0["t2_task"] = r0.t2
    pe = pd.read_csv(E1C / "predictors_e1c.csv"); re_ = pd.read_csv(E1C / "pair_results_e1c.csv")
    pops = {"R0": r0}
    for p in ("P", "S1"):
        pops[p] = pe.merge(re_[re_["pop"] == p][["pair", "D"]], on="pair")
    files = [E1B / "predictors.csv", E1B / "pair_results.csv", E1C / "predictors_e1c.csv", E1C / "pair_results_e1c.csv"]
    return tasks, pops, files


def load_e4(d, tag):
    PO = json.loads((d / f"populations_{tag}.json").read_text())
    tasks = PO["valid_tasks"]
    PR = pd.read_csv(d / f"predictors_{tag}.csv")
    pops = {}; files = [d / f"predictors_{tag}.csv", d / f"populations_{tag}.json"]
    for p in POPS:
        jl = d / f"pair_results_{p}.jsonl"; files.append(jl)
        recs, seen = [], set()
        for l in jl.read_text().splitlines():            # de-duplicate, keep first (as e4a/e4b load_pop)
            if not l.strip(): continue
            r = json.loads(l)
            if r["pair"] in seen: continue
            seen.add(r["pair"]); recs.append({"pair": r["pair"], "D": r["D"], "lam_selected": r["lam_selected"]})
        R = pd.DataFrame(recs)
        order = {q: i for i, q in enumerate(PR.pair)}
        df = PR.merge(R, on="pair")
        df = df.assign(_o=[order[q] for q in df.pair]).sort_values("_o").drop(columns="_o").reset_index(drop=True)
        assert len(df) == PO["n"][p], (tag, p, len(df))
        assert set(df.pair) == set(PO["populations"][p]), (tag, p)
        pops[p] = df
    return tasks, pops, files


def design(df, tasks):
    K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    I = np.array([tix[x] for x in df.t1_task]); J = np.array([tix[x] for x in df.t2_task])
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))
    assert (PI[~np.eye(K, dtype=bool)] >= 0).all()
    return I, J, PI


def draws_within(des, K, seed, npops=POPS):
    """Task-block bootstrap draws applied jointly to the populations of one backbone (e1b.stage3 construction)."""
    rng = np.random.default_rng(seed); boots = {p: [] for p in npops}; skipped = 0
    for _ in range(NBOOT):
        s = rng.integers(0, K, K); ii = [(a, b) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
        bis = {p: np.array([des[p][2][s[a], s[b]] for a, b in ii], dtype=np.int64) for p in npops}
        if len(np.unique(bis[npops[0]])) < 4: skipped += 1; continue
        for p in npops: boots[p].append(bis[p])
    return boots, skipped


def perm_rhos(x, y, maps):
    """Spearman of x with y permuted by each row of maps (maps are bijections of pairs, so ranks permute with y)."""
    rx = rankdata(x); ry = rankdata(y)
    rx = rx - rx.mean(); ryp = ry[maps]; ryp = ryp - ryp.mean(axis=1, keepdims=True)
    return (ryp @ rx) / np.sqrt((rx @ rx) * np.einsum("ij,ij->i", ryp, ryp))


# ------------------------------------------------------------------ per-backbone pooled rho
def backbone_block(name, tasks, pops, preds, extra_cols=None):
    K = len(tasks); des = {p: design(pops[p], tasks) for p in POPS}
    D = {p: pops[p]["D"].values.astype(float) for p in POPS}
    boots, skipped = draws_within(des, K, 0)
    perms = np.argsort(np.random.default_rng(0).random((NPERM, K)), axis=1)
    maps = {p: des[p][2][perms[:, des[p][0]], perms[:, des[p][1]]] for p in POPS}
    res = {}; bsbar = {}
    for c in preds:
        X = {p: pops[p][c].values.astype(float) for p in POPS}
        rho = {p: spearman(X[p], D[p]) for p in POPS}
        rbar = float(np.mean([rho[p] for p in POPS]))
        bs = {p: np.array([spearman(X[p][bi], D[p][bi]) for bi in boots[p]]) for p in POPS}
        bbar = np.nanmean(np.vstack([bs[p] for p in POPS]), axis=0)
        rp = {p: perm_rhos(X[p], D[p], maps[p]) for p in POPS}
        pbar = np.mean(np.vstack([rp[p] for p in POPS]), axis=0)
        p1 = float((1 + np.sum(pbar >= rbar - 1e-12)) / (1 + NPERM)); p2 = float((1 + np.sum(np.abs(pbar) >= abs(rbar) - 1e-12)) / (1 + NPERM))
        pp1 = {p: float((1 + np.sum(rp[p] >= rho[p] - 1e-12)) / (1 + NPERM)) for p in POPS}
        loto = {}
        for t in tasks:
            v = []
            for p in POPS:
                I, J, _ = des[p]; m = (I != tasks.index(t)) & (J != tasks.index(t)); v.append(spearman(X[p][m], D[p][m]))
            loto[t] = float(np.mean(v))
        res[c] = {"rho_per_population": rho,
                  "per_population_boot_ci95": {p: [pct(bs[p], 2.5), pct(bs[p], 97.5)] for p in POPS},
                  "per_population_perm_p_one_sided": pp1,
                  "rho_bar": rbar, "rho_bar_boot_ci95": [pct(bbar, 2.5), pct(bbar, 97.5)], "rho_bar_boot_sd": float(np.nanstd(bbar, ddof=1)),
                  "rho_bar_perm_p_one_sided": p1, "rho_bar_perm_p_two_sided": p2,
                  "fisher_z_mean_rho": float(np.tanh(np.mean([np.arctanh(rho[p]) for p in POPS]))),
                  "loto_rho_bar_range": [float(min(loto.values())), float(max(loto.values()))], "loto_rho_bar": loto}
        bsbar[c] = bbar
    return {"K": K, "tasks": tasks, "n_pairs": {p: int(len(pops[p])) for p in POPS}, "boot_reps_used": len(boots["R0"]),
            "boot_skipped": skipped, "results": res}, bsbar, des


def sanity_ci(name, block):
    ref = {}
    if name == "BERT":
        AE = json.loads((E1B / "analysis.json").read_text()); AC = json.loads((E1C / "analysis_e1c.json").read_text())
        ref[("R0", "O_A")] = AE["verdicts"]["H1"]["boot_ci95"]; ref[("R0", "tv_cosine")] = AE["verdicts"]["H2"]["boot_ci95"]
        for p in ("P", "S1"):
            for c in PRIMARY: ref[(p, c)] = AC[p]["all_predictor_stats"][c]["boot_ci95"]
        refp = {("R0", "O_A"): AE["verdicts"]["H1"]["perm_p_one_sided"], ("R0", "tv_cosine"): AE["verdicts"]["H2"]["perm_p_one_sided"]}
        for p in ("P", "S1"):
            for c in PRIMARY: refp[(p, c)] = AC[p]["all_predictor_stats"][c]["perm_p_one_sided"]
    else:
        A = json.loads(((E4A / "analysis_e4a.json") if name == "RoBERTa" else (E4B / "analysis_e4b.json")).read_text())
        for p in POPS:
            for c in PREDICTORS:
                ref[(p, c)] = A[p]["all_predictor_stats"][c]["boot_ci95"]
        refp = {(p, c): A[p]["all_predictor_stats"][c]["perm_p_one_sided"] for p in POPS for c in PREDICTORS}
    r = block["results"]
    ci = {f"{p}:{c}": max(abs(a - b) for a, b in zip(r[c]["per_population_boot_ci95"][p], v)) for (p, c), v in ref.items()}
    pp = {f"{p}:{c}": abs(r[c]["per_population_perm_p_one_sided"][p] - v) for (p, c), v in refp.items()}
    return {"max_abs_diff_ci": max(ci.values()), "max_abs_diff_perm_p": (max(pp.values()) if pp else None), "n_checked_ci": len(ci), "n_checked_p": len(pp)}


# ------------------------------------------------------------------ across-backbone
def across(data, preds):
    union = data["BERT"]["tasks"]                           # 14-task union (E1b order); Qwen has the same 14, RoBERTa 13
    Ku = len(union)
    maps_b = {}
    for b in BACKBONES:
        tix = {t: i for i, t in enumerate(data[b]["tasks"])}
        maps_b[b] = np.array([tix.get(t, -1) for t in union])
    def rbar_on(b, c, sel_idx):
        v = []
        for p in POPS:
            X = data[b]["pops"][p][c].values.astype(float); D = data[b]["pops"][p]["D"].values.astype(float)
            v.append(spearman(X[sel_idx[p]], D[sel_idx[p]]))
        return float(np.nanmean(v))
    def pair_idx(b, s_union):
        s = maps_b[b][s_union]; PI = {p: data[b]["des"][p][2] for p in POPS}
        ii = [(a, c) for a in range(len(s)) for c in range(a + 1, len(s)) if s[a] != s[c] and s[a] >= 0 and s[c] >= 0]
        return {p: np.array([PI[p][s[a], s[c]] for a, c in ii], dtype=np.int64) for p in POPS}
    # primary: joint draws over the union
    rng = np.random.default_rng(20260927); joint = []; skipped = 0
    for _ in range(NBOOT):
        s = rng.integers(0, Ku, Ku); idx = {b: pair_idx(b, s) for b in BACKBONES}
        if any(len(np.unique(idx[b]["R0"])) < 4 for b in BACKBONES): skipped += 1; continue
        joint.append(idx)
    # sensitivity: independent within-backbone draws
    indep = {}
    for b, seed in zip(BACKBONES, (101, 202, 303)):
        bo, _ = draws_within(data[b]["des"], data[b]["K"], seed); indep[b] = bo
    nind = min(len(indep[b]["R0"]) for b in BACKBONES)
    out = {"union_tasks": union, "joint_reps_used": len(joint), "joint_skipped": skipped, "indep_reps_used": nind, "results": {}}
    for c in preds:
        obs = {b: data[b]["block"]["results"][c]["rho_bar"] for b in BACKBONES}
        J = {b: np.array([rbar_on(b, c, idx[b]) for idx in joint]) for b in BACKBONES}
        Ind = {b: np.array([rbar_on(b, c, {p: indep[b][p][k] for p in POPS}) for k in range(nind)]) for b in BACKBONES}
        def summ(dist, o):
            return {"obs": float(o), "ci95": [pct(dist, 2.5), pct(dist, 97.5)],
                    "boot_p_two_sided_approx": float(min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))))}
        contr = {}
        for lab, f in (("across_mean", lambda d: (d["BERT"] + d["RoBERTa"] + d["Qwen"]) / 3),
                       ("encoder_mean", lambda d: (d["BERT"] + d["RoBERTa"]) / 2),
                       ("Qwen_minus_BERT", lambda d: d["Qwen"] - d["BERT"]),
                       ("Qwen_minus_RoBERTa", lambda d: d["Qwen"] - d["RoBERTa"]),
                       ("Qwen_minus_encoder_mean", lambda d: d["Qwen"] - (d["BERT"] + d["RoBERTa"]) / 2),
                       ("RoBERTa_minus_BERT", lambda d: d["RoBERTa"] - d["BERT"])):
            contr[lab] = {"joint": summ(f(J), f(obs)), "independent": summ(f(Ind), f(obs))}
        # DerSimonian-Laird on Fisher z, variances from within-backbone (pre-registered-stream) bootstrap of rho_bar
        z = np.array([np.arctanh(obs[b]) for b in BACKBONES])
        v = np.array([np.nanvar(np.arctanh(np.clip(data[b]["bsbar"][c], -0.999999, 0.999999)), ddof=1) for b in BACKBONES])
        w = 1 / v; zf = np.sum(w * z) / np.sum(w); Q = float(np.sum(w * (z - zf) ** 2)); dfq = len(z) - 1
        tau2 = max(0.0, (Q - dfq) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))); wr = 1 / (v + tau2)
        zr = np.sum(wr * z) / np.sum(wr); se = np.sqrt(1 / np.sum(wr))
        re_ = {"fixed_effect_rho": float(np.tanh(zf)), "random_effects_rho": float(np.tanh(zr)),
               "random_effects_ci95": [float(np.tanh(zr - 1.96 * se)), float(np.tanh(zr + 1.96 * se))],
               "tau2_z": float(tau2), "Q": Q, "Q_df": dfq, "Q_p": float(chi2.sf(Q, dfq)), "I2": float(max(0.0, (Q - dfq) / Q)) if Q > 0 else 0.0,
               "within_backbone_var_z": {b: float(x) for b, x in zip(BACKBONES, v)},
               "note": "assumes independent backbones (violated: shared tasks and evaluation sets); k = 3; descriptive only"}
        out["results"][c] = {"rho_bar_by_backbone": obs,
                             "rho_bar_by_backbone_joint_union_ci95": {b: [pct(J[b], 2.5), pct(J[b], 97.5)] for b in BACKBONES},
                             "contrasts": contr, "random_effects": re_}
    return out


# ------------------------------------------------------------------ layer-exclusion check
def layer_variants(d, tag, pops, hot_layers):
    L = pd.read_csv(d / f"predictors_layers_{tag}.csv")
    L["chance"] = R_LORA / L.d_out
    any_sub30 = set(L.loc[L.theta_min_A_deg < 30, "layer"])
    kproj = L.layer.str.contains("k_proj|attention.self.key")
    variants = {
        "O_A (all layers; recomputed)": pd.Series(True, index=L.index),
        "excl. layer-0 k_proj": ~L.layer.isin(["model.layers.0.self_attn.k_proj", "encoder.layer.0.attention.self.key"]),
        "excl. all k_proj": ~kproj,
        "excl. all d_out = 128 layers (k_proj, v_proj)": L.d_out != 128,
        "excl. hot spots (L0 k_proj, L23 v_proj, L23 down_proj)": ~L.layer.isin(hot_layers),
        "excl. every layer sub-30 deg in any pair": ~L.layer.isin(any_sub30),
        "excl. pair-specific sub-30 deg layers": L.theta_min_A_deg >= 30,
        "only d_out = 896/768 layers": L.d_out.isin([896, 768]),
        "only d_out = 4864/3072 layers": L.d_out.isin([4864, 3072]),
    }
    vals = {}
    for k, m in variants.items():
        vals[k] = L[m].groupby("pair").cos2_mean_A.mean()
    vals["chance-normalized O_A (mean cos2 / (r/d_out))"] = (L.cos2_mean_A / L.chance).groupby(L.pair).mean()
    info = {k: int(m.groupby(L.pair).sum().iloc[0]) if k != "excl. pair-specific sub-30 deg layers" else None for k, m in variants.items()}
    info["n_layer_names_sub30_any_pair"] = len(any_sub30); info["layer_names_sub30_any_pair"] = sorted(any_sub30)
    out = {p: pops[p][["pair"]].copy() for p in POPS}
    for p in POPS:
        for k, s in vals.items():
            out[p][k] = out[p].pair.map(s).values
        chk = np.max(np.abs(out[p]["O_A (all layers; recomputed)"].values - pops[p].O_A.values))
        info[f"recompute_check_max_abs_diff_{p}"] = float(chk)
        for k in vals:
            pops[p][k] = out[p][k].values
    return list(vals), info


def main():
    t0 = time.time()
    data = {}; files = []
    tb, pb, fb = load_bert(); data["BERT"] = {"tasks": tb, "pops": pb}; files += fb
    ta, pa, fa = load_e4(E4A, "e4a"); data["RoBERTa"] = {"tasks": ta, "pops": pa}; files += fa
    tq, pq, fq = load_e4(E4B, "e4b"); data["Qwen"] = {"tasks": tq, "pops": pq}; files += fq
    for b in BACKBONES:
        blk, bsbar, des = backbone_block(b, data[b]["tasks"], data[b]["pops"], PREDICTORS)
        data[b].update({"block": blk, "bsbar": bsbar, "des": des, "K": blk["K"]})
        blk["sanity_vs_prereg_files"] = sanity_ci(b, blk)
        print(b, "done", round(time.time() - t0, 1), "s", blk["sanity_vs_prereg_files"], flush=True)
    acr = across(data, PREDICTORS)
    print("across done", round(time.time() - t0, 1), flush=True)
    # layer-exclusion
    hot = ["model.layers.0.self_attn.k_proj", "model.layers.23.self_attn.v_proj", "model.layers.23.mlp.down_proj"]
    lay = {}
    for b, d, tag in (("Qwen", E4B, "e4b"), ("RoBERTa", E4A, "e4a")):
        cols, info = layer_variants(d, tag, data[b]["pops"], hot)
        blk, _, _ = backbone_block(b, data[b]["tasks"], data[b]["pops"], cols)
        lay[b] = {"variants": blk["results"], "info": info}
        print("layer variants", b, round(time.time() - t0, 1), flush=True)
    # census of Qwen sub-30 layers per population
    Lq = pd.read_csv(E4B / "predictors_layers_e4b.csv"); POq = json.loads((E4B / "populations_e4b.json").read_text())
    popof = {q: k for k, v in POq["populations"].items() for q in v}
    s30 = Lq[Lq.theta_min_A_deg < 30].assign(pop=lambda x: x.pair.map(popof))
    census = {"n_pair_layers": int(len(Lq)), "n_sub30": int(len(s30)), "min_theta": float(s30.theta_min_A_deg.min()), "n_pairs_with_sub30": int(s30.pair.nunique()),
              "by_population": {k: int(v) for k, v in s30.groupby("pop").size().items()},
              "by_layer_all": {k: int(v) for k, v in s30.layer.value_counts().items()},
              "by_layer_P": {k: int(v) for k, v in s30[s30["pop"] == "P"].layer.value_counts().items()},
              "share_layer0_kproj_all": float(np.mean(s30.layer == "model.layers.0.self_attn.k_proj")),
              "share_d_out_128_all": float(np.mean(s30.d_out == 128))}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S KST"), "status": "EXPLORATORY (not pre-registered; written after all results); all values derived",
           "inputs_sha256": {str(f.relative_to(ROOT)): sha(f) for f in files + [E4A / "predictors_layers_e4a.csv", E4B / "predictors_layers_e4b.csv"]},
           "per_backbone": {b: data[b]["block"] for b in BACKBONES}, "across_backbone": acr, "layer_exclusion": lay, "qwen_sub30_census": census,
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "pooled_backbones.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    print("wrote", round(time.time() - t0, 1), "s")


def f3(x): return f"{x:+.3f}" if np.isfinite(x) else "nan"
def ci(v): return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def write_md(o):
    L = ["# Per-backbone and across-backbone pooled Spearman ρ — EXPLORATORY (all values derived)", "",
         f"Generated {o['generated']} by `analysis/pooled_backbones.py`. Not pre-registered; written after all E1b, E1c, E4a and E4b results were known. "
         "Within a backbone the three populations (R0 seed 0, P mixed seed, S1 seed 1) share tasks, evaluation sets and adapters, so they are not independent; across backbones 13 of 14 tasks and all evaluation examples are shared, so the backbones are not independent either.", "",
         "## 1. Per-backbone pooled ρ̄ (mean of R0, P, S1; joint within-backbone task-block bootstrap, 2,000 reps)", "",
         "| backbone | K | predictor | ρ R0 | ρ P | ρ S1 | **ρ̄** | 95% CI | joint perm. p (1-sided / 2-sided) | LOTO ρ̄ range |", "|---|---:|---|---:|---:|---:|---:|---|---|---|"]
    for b in BACKBONES:
        blk = o["per_backbone"][b]
        for c in PREDICTORS:
            r = blk["results"][c]; rp = r["rho_per_population"]
            L.append(f"| {b} | {blk['K']} | {c} | {f3(rp['R0'])} | {f3(rp['P'])} | {f3(rp['S1'])} | **{f3(r['rho_bar'])}** | {ci(r['rho_bar_boot_ci95'])} | {r['rho_bar_perm_p_one_sided']:.3f} / {r['rho_bar_perm_p_two_sided']:.3f} | {ci(r['loto_rho_bar_range'])} |")
    L += ["", "Sanity (per-population CIs and one-sided permutation p from the joint draws vs the pre-registered analysis files; max |Δ|): " +
          json.dumps({b: o["per_backbone"][b]["sanity_vs_prereg_files"] for b in BACKBONES}), "",
          "## 2. Across-backbone summary and heterogeneity", "",
          f"Joint draws over the 14-task union: {o['across_backbone']['joint_reps_used']} replicates used ({o['across_backbone']['joint_skipped']} skipped). Independent within-backbone draws: {o['across_backbone']['indep_reps_used']} replicates.", "",
          "| predictor | quantity | observed | 95% CI (joint task draws) | approx. 2-sided boot p (joint) | 95% CI (independent draws) | approx. 2-sided boot p (indep.) |", "|---|---|---:|---|---:|---|---:|"]
    for c in PREDICTORS:
        R = o["across_backbone"]["results"][c]
        for lab, v in R["contrasts"].items():
            L.append(f"| {c} | {lab} | {f3(v['joint']['obs'])} | {ci(v['joint']['ci95'])} | {v['joint']['boot_p_two_sided_approx']:.3f} | {ci(v['independent']['ci95'])} | {v['independent']['boot_p_two_sided_approx']:.3f} |")
    L += ["", "Random-effects (DerSimonian–Laird, Fisher z; assumes independent backbones, k = 3; descriptive):", "",
          "| predictor | fixed-effect ρ | random-effects ρ [95% CI] | τ² (z) | Q (df 2) | Q p | I² |", "|---|---:|---|---:|---:|---:|---:|"]
    for c in PREDICTORS:
        r = o["across_backbone"]["results"][c]["random_effects"]
        L.append(f"| {c} | {f3(r['fixed_effect_rho'])} | {f3(r['random_effects_rho'])} {ci(r['random_effects_ci95'])} | {r['tau2_z']:.4f} | {r['Q']:.2f} | {r['Q_p']:.3f} | {r['I2']:.2f} |")
    L += ["", "## 3. Layer-exclusion check for O_A (Qwen; RoBERTa for comparison)", ""]
    for b in ("Qwen", "RoBERTa"):
        lay = o["layer_exclusion"][b]
        L += [f"**{b}** (recompute check max |Δ| vs frozen O_A: " + ", ".join(f"{p} {lay['info'][f'recompute_check_max_abs_diff_{p}']:.1e}" for p in POPS) + ")", "",
              "| O_A variant | layers kept | ρ R0 [CI] | ρ P [CI] (perm p) | ρ S1 [CI] | ρ̄ [joint CI] |", "|---|---:|---|---|---|---|"]
        for k, r in lay["variants"].items():
            rp = r["rho_per_population"]; cc = r["per_population_boot_ci95"]; pp = r["per_population_perm_p_one_sided"]
            kept = lay["info"].get(k); kept = "pair-specific" if kept is None and "pair-specific" in k else ("all" if kept is None else kept)
            L.append(f"| {k} | {kept} | {f3(rp['R0'])} {ci(cc['R0'])} | {f3(rp['P'])} {ci(cc['P'])} ({pp['P']:.3f}) | {f3(rp['S1'])} {ci(cc['S1'])} | {f3(r['rho_bar'])} {ci(r['rho_bar_boot_ci95'])} |")
        L.append("")
    cz = o["qwen_sub30_census"]
    L += ["Qwen sub-30° census: " + json.dumps({k: cz[k] for k in ("n_pair_layers", "n_sub30", "min_theta", "n_pairs_with_sub30", "by_population", "by_layer_P", "share_layer0_kproj_all", "share_d_out_128_all")}), ""]
    (OUT / "pooled_backbones.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
