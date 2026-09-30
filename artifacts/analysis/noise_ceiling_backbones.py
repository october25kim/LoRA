#!/usr/bin/env python3
"""EXPLORATORY reliability / noise-ceiling / detectability update for three backbones (BERT-base, RoBERTa-base, Qwen2.5-0.5B).

Not pre-registered; written 2026-09-27 (KST) after all results were known. All outputs are "derived". `analysis/noise_ceiling.py`
and its outputs (BERT only, including the evaluation-sampling bootstrap from saved per-example predictions) are left unchanged
and are read here, not recomputed.

(1) Seed test-retest reliability per backbone: Spearman over the same unordered task pairs between the three seed configurations
    (R0~S1, R0~P, P~S1) for D, O_A and tv_cosine, recomputed from the pair files and checked against the analysis files.
(2) Evaluation-sampling reliability: BERT from analysis/noise_ceiling_results.json. For RoBERTa and Qwen the per-example
    cross-task predictions were not mirrored to the box (only the S2/E3 subset), so this cannot be computed here.
(3) Attenuation ceiling sqrt(rel_D * rel_X). Primary: R0~S1 for both (the like-for-like shared-seed replicate, as in
    noise_ceiling.py, where E1b~E1c-seed1 = R0~S1). Range over the three comparisons as sensitivity.
(4) Detectability of a latent rho = 0.4 at the realized K (13 RoBERTa, 14 BERT/Qwen): Gaussian-copula pairs with target observed
    Spearman 0.4 x ceiling (as noise_ceiling.simulate_detectability), then the pre-registered statistics approximately:
    task-block bootstrap CI (500 reps), one-sided task-label permutation p (1,000 relabelings), LOTO. PASS needs rho >= 0.4, p < alpha,
    >= 75% LOTO > 0.2, with alpha = 0.05 (liberal; Holm p equals the raw p when the other hypothesis has the smaller p) and 0.025
    (conservative; Holm p = 2p when this hypothesis has the smaller p). FAIL = not PASS and CI upper < 0.3. Also latent 0.4 without
    attenuation and rho = 0 as references. Pairs are simulated as independent given the correlation (no task random effects), which
    is optimistic about how informative a K-task design is.
(5) Disattenuated pooled rho_bar (per-backbone rho_bar from analysis/pooled_backbones.json divided by the ceiling): derived,
    approximate (Pearson formula applied to Spearman coefficients; seed variation counted as noise).

Run: /workspace/lora-paper/e1/.venv/bin/python analysis/noise_ceiling_backbones.py
Outputs: analysis/NOISE_CEILING_BACKBONES.md, analysis/noise_ceiling_backbones.json
"""
import json, math, time, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pooled_backbones as pbk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "analysis"
SEED = 20260927
NSIM, B, NP = 1000, 500, 1000


def fast_rho(x, y):
    a = rankdata(x); b = rankdata(y); a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den else float("nan")


def upair(a, b): return tuple(sorted((a, b)))


def retest(pops):
    tabs = {p: pops[p].assign(u=[upair(x, y) for x, y in zip(pops[p].t1_task, pops[p].t2_task)]).set_index("u") for p in pbk.POPS}
    out = {}
    for c in ("D", "O_A", "tv_cosine"):
        for a, b in (("R0", "S1"), ("R0", "P"), ("P", "S1")):
            cm = tabs[a].index.intersection(tabs[b].index)
            out[f"{c}:{a}~{b}"] = {"rho": pbk.spearman(tabs[a].loc[cm, c], tabs[b].loc[cm, c]), "n": int(len(cm))}
    return out


def design(K):
    I, J = np.triu_indices(K, 1)
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(I)); PI[J, I] = np.arange(len(I))
    rng = np.random.default_rng(SEED + K)
    boots = []
    while len(boots) < B:
        s = rng.integers(0, K, K); ii = [(a, b) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
        bi = np.array([PI[s[a], s[b]] for a, b in ii], dtype=np.int64)
        if len(np.unique(bi)) >= 4: boots.append(bi)
    perms = np.argsort(rng.random((NP, K)), axis=1)
    maps = PI[perms[:, I], perms[:, J]]
    loto = [(I != t) & (J != t) for t in range(K)]
    return I, J, boots, maps, loto


def simulate(K, target_obs, rng, des):
    I, J, boots, maps, loto = des
    tp = 2 * math.sin(math.pi * target_obs / 6.0)                     # Spearman -> Pearson for the Gaussian copula
    n = len(I); rows = []
    for _ in range(NSIM):
        z = rng.standard_normal((n, 2)); x = z[:, 0]; y = tp * z[:, 0] + math.sqrt(max(0.0, 1 - tp * tp)) * z[:, 1]
        obs = fast_rho(x, y)
        bs = np.array([fast_rho(x[bi], y[bi]) for bi in boots]); lo, hi = np.percentile(bs, [2.5, 97.5])
        rx = rankdata(x); rx -= rx.mean(); ry = rankdata(y); ryp = ry[maps]; ryp = ryp - ryp.mean(1, keepdims=True)
        rp = (ryp @ rx) / np.sqrt((rx @ rx) * np.einsum("ij,ij->i", ryp, ryp))
        p = (1 + np.sum(rp >= obs - 1e-12)) / (1 + NP)
        lf = np.mean([fast_rho(x[m], y[m]) > 0.2 for m in loto])
        rows.append((obs, lo, hi, p, lf))
    a = np.array(rows)
    obs, lo, hi, p, lf = a.T
    pl = (obs >= 0.4) & (p < 0.05) & (lf >= 0.75); pc = (obs >= 0.4) & (p < 0.025) & (lf >= 0.75)
    fail = (~pl) & (hi < 0.3)
    return {"K": K, "target_observed_rho": target_obs, "n_sim": NSIM, "boot_B": B, "n_perm": NP,
            "mean_observed_rho": float(obs.mean()), "mean_ci_lower": float(lo.mean()), "mean_ci_upper": float(hi.mean()),
            "P_ci_lower_gt_0": float(np.mean(lo > 0)), "P_point_rho_ge_0.4": float(np.mean(obs >= 0.4)),
            "P_perm_p_lt_0.05": float(np.mean(p < 0.05)),
            "P_PASS_liberal_alpha0.05": float(pl.mean()), "P_PASS_conservative_alpha0.025": float(pc.mean()),
            "P_FAIL": float(fail.mean()), "P_INCONCLUSIVE_liberal": float(np.mean(~pl & ~fail))}


def main():
    t0 = time.time()
    data = {}
    tb, pb, _ = pbk.load_bert(); data["BERT"] = (tb, pb)
    ta, pa, _ = pbk.load_e4(pbk.E4A, "e4a"); data["RoBERTa"] = (ta, pa)
    tq, pq, _ = pbk.load_e4(pbk.E4B, "e4b"); data["Qwen"] = (tq, pq)
    NC = json.loads((OUT / "noise_ceiling_results.json").read_text())
    PB = json.loads((OUT / "pooled_backbones.json").read_text())
    AC = json.loads((pbk.E1C / "analysis_e1c.json").read_text())
    ref = {"RoBERTa": json.loads((pbk.E4A / "analysis_e4a.json").read_text())["reliability"],
           "Qwen": json.loads((pbk.E4B / "analysis_e4b.json").read_text())["reliability"]}
    res = {}
    for b, (tasks, pops) in data.items():
        rt = retest(pops)
        chk = {}
        if b in ref:
            for k, v in rt.items():
                c, pr = k.split(":"); a, bb = pr.split("~"); rk = f"{c}:{min(a, bb)}~{max(a, bb)}" if f"{c}:{a}~{bb}" not in ref[b] else f"{c}:{a}~{bb}"
                if rk not in ref[b]: rk = f"{c}:{bb}~{a}"
                chk[k] = abs(v["rho"] - ref[b][rk]["rho"])
        else:
            nc = {r["comparison"]: r["spearman_D"] for r in NC["seed_retest"]}
            chk["D:R0~S1"] = abs(rt["D:R0~S1"]["rho"] - nc["E1b~E1c-seed1"]); chk["D:R0~P"] = abs(rt["D:R0~P"]["rho"] - nc["E1b~E1c-mixed"]); chk["D:P~S1"] = abs(rt["D:P~S1"]["rho"] - nc["E1c-mixed~E1c-seed1"])
            prel = {r["predictor"]: r["spearman_e1b_vs_e1c_seed1"] for r in NC["predictor_reliability"]}
            for c in ("O_A", "tv_cosine"): chk[f"{c}:R0~S1"] = abs(rt[f"{c}:R0~S1"]["rho"] - prel[c])
        ceil = {}
        for c in ("O_A", "tv_cosine"):
            prim = math.sqrt(max(0.0, rt["D:R0~S1"]["rho"] * rt[f"{c}:R0~S1"]["rho"]))
            allc = [math.sqrt(max(0.0, rt[f"D:{k}"]["rho"] * rt[f"{c}:{k}"]["rho"])) for k in ("R0~S1", "R0~P", "P~S1")]
            rbar = PB["per_backbone"][b]["results"][c]["rho_bar"]
            ceil[c] = {"rel_D_R0S1": rt["D:R0~S1"]["rho"], "rel_X_R0S1": rt[f"{c}:R0~S1"]["rho"], "ceiling_R0S1": prim,
                       "ceiling_range_3_comparisons": [min(allc), max(allc)], "expected_observed_for_latent_0.4": 0.4 * prim,
                       "rho_bar_pooled": rbar, "rho_bar_disattenuated": rbar / prim if prim else float("nan"),
                       "rho_bar_ci_disattenuated": [v / prim for v in PB["per_backbone"][b]["results"][c]["rho_bar_boot_ci95"]]}
        samp = None
        if b == "BERT":
            samp = {r["population"]: {"mean_pair_se": r["mean_pair_se"], "split_half_r": r["split_half_r_mean"], "spearman_brown": r["split_half_sb_mean"]} for r in NC["sampling_summary"]}
        res[b] = {"K": len(tasks), "retest": rt, "retest_check_vs_files_max_abs_diff": max(chk.values()) if chk else None,
                  "ceilings": ceil, "sampling_noise": samp or "not computable on the box: per-example cross-task predictions for E4a/E4b are not mirrored (GPU machine only)"}
        print(b, "retest done; check", res[b]["retest_check_vs_files_max_abs_diff"], flush=True)
    rng = np.random.default_rng(SEED)
    des = {K: design(K) for K in (13, 14)}
    sims = []
    for b in ("BERT", "RoBERTa", "Qwen"):
        K = res[b]["K"]
        for c in ("O_A", "tv_cosine"):
            s = simulate(K, 0.4 * res[b]["ceilings"][c]["ceiling_R0S1"], rng, des[K]); s.update({"backbone": b, "predictor": c, "condition": "latent 0.4, attenuated (R0~S1 ceiling)"}); sims.append(s)
            print(b, c, round(time.time() - t0), flush=True)
    for K in (13, 14):
        for tgt, lab in ((0.4, "observed 0.4 (no attenuation)"), (0.0, "rho = 0")):
            s = simulate(K, tgt, rng, des[K]); s.update({"backbone": "any", "predictor": "-", "condition": lab}); sims.append(s)
            print(K, lab, round(time.time() - t0), flush=True)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S KST"), "status": "EXPLORATORY; derived", "per_backbone": res, "detectability_simulation": sims,
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "noise_ceiling_backbones.json").write_text(json.dumps(out, indent=1))
    L = ["# Reliability, noise ceiling and detectability across three backbones — EXPLORATORY (derived)", "",
         f"Generated {out['generated']} by `analysis/noise_ceiling_backbones.py`. Supplements `analysis/NOISE_CEILING.md` (BERT; unchanged).", "",
         "## 1. Seed test–retest reliability (Spearman over the same unordered task pairs)", "",
         "| backbone | K | D R0~S1 | D R0~P | D P~S1 | O_A R0~S1 | O_A R0~P | O_A P~S1 | cos R0~S1 | cos R0~P | cos P~S1 | check vs files (max abs diff) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b, r in res.items():
        t = r["retest"]; g = lambda k: f"{t[k]['rho']:.3f}"
        L.append(f"| {b} | {r['K']} | {g('D:R0~S1')} | {g('D:R0~P')} | {g('D:P~S1')} | {g('O_A:R0~S1')} | {g('O_A:R0~P')} | {g('O_A:P~S1')} | {g('tv_cosine:R0~S1')} | {g('tv_cosine:R0~P')} | {g('tv_cosine:P~S1')} | {r['retest_check_vs_files_max_abs_diff']:.1e} |")
    L += ["", "Evaluation-sampling reliability (BERT only; from `noise_ceiling_results.json`): " + json.dumps(res["BERT"]["sampling_noise"]) + ". RoBERTa/Qwen: " + res["Qwen"]["sampling_noise"] + ".", "",
          "## 2. Attenuation ceiling and disattenuated pooled ρ̄", "",
          "| backbone | predictor | rel_D (R0~S1) | rel_X (R0~S1) | ceiling | ceiling range (3 comparisons) | expected observed ρ for latent 0.4 | pooled ρ̄ | disattenuated ρ̄ [CI / ceiling] |", "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for b, r in res.items():
        for c, v in r["ceilings"].items():
            L.append(f"| {b} | {c} | {v['rel_D_R0S1']:.3f} | {v['rel_X_R0S1']:.3f} | {v['ceiling_R0S1']:.3f} | [{v['ceiling_range_3_comparisons'][0]:.3f}, {v['ceiling_range_3_comparisons'][1]:.3f}] | {v['expected_observed_for_latent_0.4']:.3f} | {v['rho_bar_pooled']:+.3f} | {v['rho_bar_disattenuated']:+.3f} [{v['rho_bar_ci_disattenuated'][0]:+.3f}, {v['rho_bar_ci_disattenuated'][1]:+.3f}] |")
    L += ["", "## 3. Detectability simulation (single population, pre-registered statistics approximated)", "",
          f"{NSIM} simulated studies per row; task-block bootstrap {B}; permutation {NP}. PASS (liberal) uses one-sided p < 0.05, PASS (conservative) p < 0.025.", "",
          "| condition | backbone | predictor | K | target observed ρ | mean observed ρ | mean CI [lower, upper] | P(CI lower > 0) | P(ρ ≥ 0.4) | P(p < 0.05) | P(PASS) lib. / cons. | P(FAIL) | P(INCONCLUSIVE) |", "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|"]
    for s in sims:
        L.append(f"| {s['condition']} | {s['backbone']} | {s['predictor']} | {s['K']} | {s['target_observed_rho']:.3f} | {s['mean_observed_rho']:.3f} | [{s['mean_ci_lower']:+.3f}, {s['mean_ci_upper']:+.3f}] | {s['P_ci_lower_gt_0']:.3f} | {s['P_point_rho_ge_0.4']:.3f} | {s['P_perm_p_lt_0.05']:.3f} | {s['P_PASS_liberal_alpha0.05']:.3f} / {s['P_PASS_conservative_alpha0.025']:.3f} | {s['P_FAIL']:.3f} | {s['P_INCONCLUSIVE_liberal']:.3f} |")
    L += ["", "Caveats: Pearson attenuation formula applied to Spearman coefficients; seed variation is counted as measurement noise (it is partly real variation between adapters); simulated pairs carry no task random effects; the conservative/liberal PASS rows bracket the Holm step."]
    (OUT / "NOISE_CEILING_BACKBONES.md").write_text("\n".join(L) + "\n")
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
