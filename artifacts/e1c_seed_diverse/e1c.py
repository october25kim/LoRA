#!/usr/bin/env python3
"""E1c seed-diverse replication of E1b (PREREG_E1C.md). Imports E1b e1b.py/tasks.py and E3 e3.py/merges.py UNMODIFIED (read-only).
Adapter keys: "<task>@s0" = E1b seed-0 adapter (artifacts/e1b_confirmatory/adapters/<task>), "<task>@s1" = E1c seed-1 adapter (adapters_s1/<task>).
Pair id: "<key1>__<key2>" (t1 = key1, t2 = key2; the gate projects t2).
Stages:
  equiv    equivalence gates on E1b seed-0 adapters (stage1 on the 91 R0 pairs vs E1b predictors.csv; stage2 on 2 R0 pairs vs E1b pair_results.csv)
  stage0   E1b stage0 (unmodified) on seed-1 adapters -> s1/stage0.json; re-check on seed-0 adapters -> s0_recheck/stage0.json; populations -> populations_e1c.json
  stage1   predictors for P, S1, S2 (generalized e1b.stage1, same arithmetic) -> predictors_e1c*.csv/json, frozen in predictors_e1c.sha256
  stage2   merges for one population (--pop P|S1|S2), generalized e1b.stage2 (same arithmetic) -> pair_results_<pop>.jsonl
  e3same   E3 run() (unmodified) on S2 with TA / PICO_TA / GATE / FORCEGATE, amendment-A2 grids -> e3same/
  stage3   analysis -> analysis_e1c.json, VERDICT_E1C.md, figures
"""
import argparse, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
E3C = HERE.parent / "e3_baselines" / "code"
sys.path.insert(0, str(E1B))
import e1b as bm                      # noqa: E402  (unmodified E1b pipeline; globals OUT/ADIR/LOGF are redirected below)
from tasks import TASK_TYPE           # noqa: E402

PC = json.loads((HERE / "prereg_e1c.json").read_text())
LAMS = bm.LAMS
THETA_STAR = bm.THETA_STAR
LOGF = HERE / "run_e1c.log"
bm.LOGF = LOGF
bm.OUT = HERE                          # data cache -> HERE/cache
S0_E1B = json.loads((E1B / "stage0.json").read_text())


def log(m): bm.log(m)
def task_of(k): return k.split("@")[0]
def seed_of(k): return int(k.split("@s")[1])
def pid(a, b): return f"{a}__{b}"


def adir(k):
    return (E1B / "adapters") if seed_of(k) == 0 else (HERE / "adapters_s1")


def load_ad(k):
    bm.ADIR = adir(k)
    ad = bm.load_adapter(task_of(k))
    ad["key"] = k
    return ad


def timing_add(name, k, s):
    p = HERE / f"timing_{name}.json"; d = json.loads(p.read_text()) if p.exists() else {}
    d[k] = d.get(k, 0.0) + s; bm.jdump(d, p)


# ------------------------------------------------------------------ populations
def e1b_order(tasks):
    order = {t: i for i, t in enumerate(bm.TASKS_ALL)}
    return sorted(tasks, key=lambda t: order[t])


def populations(valid):
    V = e1b_order(valid)
    P = []
    for x, y in itertools.combinations(sorted(V), 2):            # alphabetical: first -> seed 0 (t1), second -> seed 1 (t2)
        P.append((f"{x}@s0", f"{y}@s1"))
    S1 = [(f"{x}@s1", f"{y}@s1") for x, y in itertools.combinations(V, 2)]      # E1b task-list order
    S2 = [(f"{t}@s0", f"{t}@s1") for t in V]
    R0 = [(f"{x}@s0", f"{y}@s0") for x, y in itertools.combinations(V, 2)]
    return {"P": P, "S1": S1, "S2": S2, "R0": R0}


def singles(valid_s1=True):
    out = {}
    for t, r in S0_E1B["per_task"].items():
        out[f"{t}@s0"] = {"eval": r["eval"]["main"], "hold": r["hold"]["main"]}
    p = HERE / "s1" / "stage0.json"
    if valid_s1 and p.exists():
        s1 = json.loads(p.read_text())
        for t, r in s1["per_task"].items():
            out[f"{t}@s1"] = {"eval": r["eval"]["main"], "hold": r["hold"]["main"]}
    return out


# ------------------------------------------------------------------ stage 1 (generalized e1b.stage1; same arithmetic)
def compute_predictors(pairs, outdir, prefix, dev, names):
    torch = bm.setup_torch(); import pandas as pd
    keys = sorted({k for p in pairs for k in p}, key=lambda k: (seed_of(k), bm.TASKS_ALL.index(task_of(k))))
    ads = {k: load_ad(k) for k in keys}; r = bm.R["r"]
    for k in keys: assert sorted(ads[k]["layers"]) == names, k
    QA, QB = {}, {}
    for t in keys:
        QA[t], QB[t] = {}, {}
        for n in names:
            A, B = ads[t]["layers"][n]; A = A.to(dev).double(); B = B.to(dev).double()
            QA[t][n] = bm.orth(B)
            Qb, Rb = torch.linalg.qr(B); U, S, _ = torch.linalg.svd(Rb @ A, full_matrices=False); QB[t][n] = Qb @ U[:, :r]
    nrm = {t: 0.0 for t in keys}; dots = {}
    thr = bm.ties_thresholds(ads, keys, names, dev)
    conf = {}
    for n in names:
        dd = {t: bm.delta(ads[t], n, dev).flatten() for t in keys}
        for t in keys: nrm[t] += float((dd[t].double() ** 2).sum())
        for (a, b) in pairs:
            dots[(a, b)] = dots.get((a, b), 0.0) + float((dd[a].double() * dd[b].double()).sum())
            ka = dd[a].abs() >= thr[a]; kb = dd[b].abs() >= thr[b]; both = ka & kb
            opp = torch.sign(dd[a]) != torch.sign(dd[b]); nz = (dd[a] != 0) & (dd[b] != 0)
            c = conf.setdefault((a, b), [0, 0, 0, 0])
            c[0] += int((opp & both).sum()); c[1] += int(both.sum()); c[2] += int((opp & nz).sum()); c[3] += int(nz.sum())
        del dd
    g = torch.Generator(device=dev).manual_seed(0); NREP = 1000
    null_cos2 = np.zeros((NREP, len(names))); null_thmin = np.zeros((NREP, len(names))); dims = {}
    for li, n in enumerate(names):
        d = ads[keys[0]]["layers"][n][1].shape[0]; dims[n] = d
        G1 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        G2 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        s = torch.linalg.svdvals(torch.linalg.qr(G1)[0].transpose(1, 2) @ torch.linalg.qr(G2)[0]).clamp(max=1.0)
        null_cos2[:, li] = (s ** 2).mean(1).cpu().numpy(); null_thmin[:, li] = np.degrees(np.arccos(s.max(1).values.cpu().numpy()))
    null_OA = null_cos2.mean(1); mu, sd = float(null_OA.mean()), float(null_OA.std(ddof=1))
    band = {n: {"d": dims[n], "theta_min_p5": float(np.percentile(null_thmin[:, i], 5)), "theta_min_p50": float(np.percentile(null_thmin[:, i], 50)),
                "theta_min_p95": float(np.percentile(null_thmin[:, i], 95))} for i, n in enumerate(names)}
    rows, lrows = [], []
    for (a, b) in pairs:
        cA, cB, thA, thB = [], [], [], []
        for n in names:
            sa = torch.linalg.svdvals(QA[a][n].T @ QA[b][n]).clamp(max=1.0); sb = torch.linalg.svdvals(QB[a][n].T @ QB[b][n]).clamp(max=1.0)
            c2a = float((sa ** 2).mean()); c2b = float((sb ** 2).mean())
            tha = float(np.degrees(np.arccos(float(sa.max())))); thb = float(np.degrees(np.arccos(float(sb.max()))))
            n30 = int((sa > math.cos(math.radians(THETA_STAR))).sum())
            cA.append(c2a); cB.append(c2b); thA.append(tha); thB.append(thb)
            lrows.append({"pair": pid(a, b), "t1": a, "t2": b, "layer": n, "d_out": dims[n], "cos2_mean_A": c2a, "theta_min_A_deg": tha,
                          "n_angles_lt30_A": n30, "cos2_mean_B": c2b, "theta_min_B_deg": thb})
        n1, n2 = math.sqrt(nrm[a]), math.sqrt(nrm[b]); OA = float(np.mean(cA)); c = conf[(a, b)]
        rows.append({"pair": pid(a, b), "t1": a, "t2": b, "t1_task": task_of(a), "t2_task": task_of(b), "t1_seed": seed_of(a), "t2_seed": seed_of(b),
                     "O_A": OA, "O_B": float(np.mean(cB)),
                     "tv_cosine": dots[(a, b)] / (n1 * n2),
                     "mean_theta_min_A_deg": float(np.mean(thA)), "min_theta_min_A_deg": float(np.min(thA)),
                     "n_layers_theta_min_lt30_A": int(np.sum(np.array(thA) < THETA_STAR)),
                     "mean_theta_min_B_deg": float(np.mean(thB)), "null_z_O_A": (OA - mu) / sd,
                     "sign_conflict_top20": c[0] / max(c[1], 1), "sign_conflict_all": c[2] / max(c[3], 1),
                     "norm_ratio": max(n1, n2) / min(n1, n2), "tvnorm_t1": n1, "tvnorm_t2": n2,
                     "same_type": TASK_TYPE[task_of(a)] == TASK_TYPE[task_of(b)], "n_layers": len(names)})
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / f"{prefix}.csv", index=False, float_format="%.10g")
    pd.DataFrame(lrows).to_csv(outdir / f"{prefix.replace('predictors', 'predictors_layers')}.csv", index=False, float_format="%.10g")
    bm.jdump({"n_rep": NREP, "seed": 0, "r": r, "null_O_A_mean": mu, "null_O_A_sd": sd, "null_O_A_p5": float(np.percentile(null_OA, 5)),
              "null_O_A_p95": float(np.percentile(null_OA, 95)), "per_layer": band}, outdir / f"{prefix.replace('predictors', 'predictors_null')}.json")
    bm.jdump({"ties_top20_abs_threshold": thr, "tvnorm": {t: math.sqrt(nrm[t]) for t in keys}, "adapter_sha256": {k: ads[k]["sha256"] for k in keys}},
             outdir / f"{prefix.replace('predictors', 'predictors_aux')}.json")
    return rows


# ------------------------------------------------------------------ stage 2 (generalized e1b.stage2; same arithmetic)
def run_merges(pairs, jl, pdir, dev, names, single, bs, tname):
    torch = bm.setup_torch(); import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bm.BASE, revision=bm.BREV)
    keys = sorted({k for p in pairs for k in p})
    ads = {k: load_ad(k) for k in keys}
    aux = json.loads((HERE / "predictors_aux_e1c.json").read_text()) if tname != "equiv" else json.loads((HERE / "equiv" / "predictors_aux_R0.json").read_text())
    for k in keys: assert ads[k]["sha256"] == aux["adapter_sha256"][k], f"{k} adapter changed since predictors"
    thr = aux["ties_top20_abs_threshold"]
    L = pd.read_csv(HERE / "predictors_layers_e1c.csv") if tname != "equiv" else pd.read_csv(HERE / "equiv" / "predictors_layers_R0.csv")
    data = {}
    for k in keys:
        t = task_of(k)
        if t not in data: data[t] = bm.get_data(t, tok)
        assert data[t]["hold_idx_sha256"] == ads[k]["meta"]["hold_idx_sha256"], f"{k}: held-out indices differ"
    enc = bm.Enc(dev, names)
    done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()} if jl.exists() else set()
    pdir.mkdir(exist_ok=True, parents=True)
    cos_star = math.cos(math.radians(THETA_STAR))
    for (a, b) in pairs:
        key = pid(a, b)
        if key in done: continue
        tp = time.time(); rec = {"pair": key, "t1": a, "t2": b, "t1_task": task_of(a), "t2_task": task_of(b)}; preds = {}
        d1 = {n: bm.delta(ads[a], n, dev) for n in names}; d2 = {n: bm.delta(ads[b], n, dev) for n in names}

        def evaluate(merged_fn, lam, which, tag):
            enc.set_delta({n: lam * merged_fn(n) for n in names})
            out = {}
            for k in (a, b):
                t = task_of(k)
                lg = enc.predict(data[t][which], ads[k]["head"], bs)
                out[k] = bm.score(t, lg, data[t][which]["labels"])["main"]
                if which == "eval": preds[f"{tag}_{k}_lam{lam}"] = bm.compact(t, lg)
            return out

        def norm_mean(sc, which):
            return 0.5 * (sc[a] / single[a][which] + sc[b] / single[b][which])

        ta = lambda n: d1[n] + d2[n]
        sel = {}
        for lam in LAMS:
            h = evaluate(ta, lam, "hold", "TA"); e = evaluate(ta, lam, "eval", "TA")
            sel[lam] = norm_mean(h, "hold")
            rec[f"TA_hold_norm_lam{lam}"] = sel[lam]
            for i, k in ((1, a), (2, b)): rec[f"TA_hold_t{i}_lam{lam}"] = h[k]; rec[f"TA_eval_t{i}_lam{lam}"] = e[k]
            rec[f"TA_eval_D_lam{lam}"] = 1 - norm_mean(e, "eval")
        lam_ta = LAMS[int(np.argmax([sel[l] for l in LAMS]))]
        rec["lam_selected"] = lam_ta
        rec["merged_eval_t1"] = rec[f"TA_eval_t1_lam{lam_ta}"]; rec["merged_eval_t2"] = rec[f"TA_eval_t2_lam{lam_ta}"]
        rec["single_eval_t1"] = single[a]["eval"]; rec["single_eval_t2"] = single[b]["eval"]
        rec["norm_t1"] = rec["merged_eval_t1"] / single[a]["eval"]; rec["norm_t2"] = rec["merged_eval_t2"] / single[b]["eval"]
        rec["D"] = 1 - 0.5 * (rec["norm_t1"] + rec["norm_t2"])
        rec["TA_score_sel"] = 1 - rec["D"]

        def ties(n):
            x1 = d1[n] * (d1[n].abs() >= thr[a]); x2 = d2[n] * (d2[n].abs() >= thr[b])
            s = torch.sign(x1 + x2)
            m1 = (x1 != 0) & (torch.sign(x1) == s); m2 = (x2 != 0) & (torch.sign(x2) == s)
            cnt = (m1.float() + m2.float()).clamp(min=1.0)
            return (x1 * m1 + x2 * m2) / cnt
        tsel = {}
        for lam in LAMS:
            h = evaluate(ties, lam, "hold", "TIES"); tsel[lam] = norm_mean(h, "hold"); rec[f"TIES_hold_norm_lam{lam}"] = tsel[lam]
        lam_ties = LAMS[int(np.argmax([tsel[l] for l in LAMS]))]; rec["lam_TIES_own"] = lam_ties
        for tag, lam in (("atTA", lam_ta), ("own", lam_ties)):
            if tag == "own" and lam_ties == lam_ta:
                rec["TIES_own_score"] = rec["TIES_atTA_score"]; continue
            e = evaluate(ties, lam, "eval", f"TIES{tag}"); rec[f"TIES_{tag}_score"] = norm_mean(e, "eval")
            for i, k in ((1, a), (2, b)): rec[f"TIES_{tag}_eval_t{i}"] = e[k]
        lp = L[L.pair == key].set_index("layer")
        fail_layers = [n for n in names if lp.loc[n, "theta_min_A_deg"] < THETA_STAR]
        rec["gate_n_fail_layers"] = len(fail_layers); rec["gate_active"] = len(fail_layers) > 0; rec["gate_fail_layers"] = fail_layers
        if fail_layers:
            d2g = dict(d2)
            for n in fail_layers:
                U1 = bm.orth(ads[a]["layers"][n][1].to(dev).double()); U2 = bm.orth(ads[b]["layers"][n][1].to(dev).double())
                Pm, S, _ = torch.linalg.svd(U1.T @ U2)
                Sd = (U1 @ Pm[:, S > cos_star]).float()
                d2g[n] = d2[n] - Sd @ (Sd.T @ d2[n])
            gate = lambda n: d1[n] + d2g[n]
            gsel = {}
            for lam in LAMS:
                h = evaluate(gate, lam, "hold", "GATE"); gsel[lam] = norm_mean(h, "hold"); rec[f"GATE_hold_norm_lam{lam}"] = gsel[lam]
            lam_g = LAMS[int(np.argmax([gsel[l] for l in LAMS]))]; rec["lam_GATE_own"] = lam_g
            for tag, lam in (("atTA", lam_ta), ("own", lam_g)):
                if tag == "own" and lam_g == lam_ta:
                    rec["GATE_own_score"] = rec["GATE_atTA_score"]; continue
                e = evaluate(gate, lam, "eval", f"GATE{tag}"); rec[f"GATE_{tag}_score"] = norm_mean(e, "eval")
        else:
            rec["lam_GATE_own"] = lam_ta; rec["GATE_atTA_score"] = rec["TA_score_sel"]; rec["GATE_own_score"] = rec["TA_score_sel"]
        rec["secs"] = time.time() - tp
        np.savez_compressed(pdir / f"pair_{key}.npz", **preds)
        with open(jl, "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"e1c stage2[{tname}] {key}: lam*={lam_ta} D={rec['D']:.4f} TIES(atTA/own)={rec['TIES_atTA_score']:.4f}/{rec['TIES_own_score']:.4f} "
            f"GATE({len(fail_layers)} layers)={rec['GATE_atTA_score']:.4f} [{rec['secs']:.0f}s]")
        del d1, d2; torch.cuda.empty_cache()
        timing_add(tname, "stage2_gpu_wall_s", time.time() - tp)


# ------------------------------------------------------------------ stages
def st_equiv(a):
    import pandas as pd
    torch = bm.setup_torch(); dev = torch.device(a.device)
    V = S0_E1B["valid_tasks"]; names = S0_E1B["lora_layers"]
    R0 = populations(V)["R0"]
    out = HERE / "equiv"; t0 = time.time()
    compute_predictors(R0, out, "predictors_R0", dev, names)
    mine = pd.read_csv(out / "predictors_R0.csv"); ref = pd.read_csv(E1B / "predictors.csv")
    mine["pair_e1b"] = [f"{task_of(x)}-{task_of(y)}" for x, y in zip(mine.t1, mine.t2)]
    m = ref.merge(mine, left_on="pair", right_on="pair_e1b", suffixes=("_ref", "_mine"))
    res = {"n_ref": len(ref), "n_matched": len(m), "max_rel_diff": {}}
    for c in ["O_A", "O_B", "tv_cosine", "mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A", "mean_theta_min_B_deg",
              "null_z_O_A", "sign_conflict_top20", "sign_conflict_all", "norm_ratio", "tvnorm_t1", "tvnorm_t2"]:
        x = m[c + "_ref"].values.astype(float); y = m[c + "_mine"].values.astype(float)
        res["max_rel_diff"][c] = float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12)))
    res["stage1_ok"] = bool(len(m) == len(ref) == 91 and max(res["max_rel_diff"].values()) <= 1e-9)
    log(f"equiv stage1: {res}")
    timing_add("equiv", "stage1_s", time.time() - t0)
    # stage 2 on two R0 pairs
    t0 = time.time()
    sub = [p for p in R0 if f"{task_of(p[0])}-{task_of(p[1])}" in ("cola-sst2", "rte-wic")]
    jl = out / "pair_results_R0equiv.jsonl"
    run_merges(sub, jl, out / "preds", dev, names, singles(False), a.bs, "equiv")
    refp = pd.read_csv(E1B / "pair_results.csv").set_index("pair")
    diffs = {}
    for l in jl.read_text().splitlines():
        r = json.loads(l); e = f"{r['t1_task']}-{r['t2_task']}"; rr = refp.loc[e]; mx = 0.0
        t1, t2 = r["t1_task"], r["t2_task"]
        for lam in LAMS:
            for i, t in ((1, t1), (2, t2)):
                mx = max(mx, abs(r[f"TA_eval_t{i}_lam{lam}"] - rr[f"TA_eval_{t}_lam{lam}"]), abs(r[f"TA_hold_t{i}_lam{lam}"] - rr[f"TA_hold_{t}_lam{lam}"]))
            mx = max(mx, abs(r[f"TA_eval_D_lam{lam}"] - rr[f"TA_eval_D_lam{lam}"]))
        mx = max(mx, abs(r["D"] - rr["D"]), abs(r["TIES_atTA_score"] - rr["TIES_atTA_score"]), abs(r["TIES_own_score"] - rr["TIES_own_score"]))
        diffs[e] = {"max_abs_diff": mx, "lam_equal": float(r["lam_selected"]) == float(rr["lam_selected"])}
    res["stage2"] = diffs
    res["stage2_ok"] = bool(len(diffs) == 2 and all(v["max_abs_diff"] <= 1e-9 and v["lam_equal"] for v in diffs.values()))
    timing_add("equiv", "stage2_s", time.time() - t0)
    res["ok"] = res["stage1_ok"] and res["stage2_ok"]
    bm.jdump(res, out / "equivalence.json")
    log(f"equiv done: ok={res['ok']} stage2={diffs}")
    if not res["ok"]: sys.exit(9)


def st_stage0(a):
    torch = bm.setup_torch()
    V0 = PC["tasks"]; t0 = time.time()
    res = {}
    for sub, ad in (("s1", HERE / "adapters_s1"), ("s0_recheck", E1B / "adapters")):
        o = HERE / sub; o.mkdir(exist_ok=True)
        if not (o / "cache").exists(): os.symlink("../cache", o / "cache")
        bm.OUT = o; bm.ADIR = ad
        ns = argparse.Namespace(device=a.device, tasks=list(V0), force_valid=False, min_valid=10)
        res[sub] = bm.stage0(ns)
    bm.OUT = HERE
    s1 = res["s1"]; s0r = res["s0_recheck"]
    rech = {}
    for t in V0:
        r0 = S0_E1B["per_task"][t]; rr = s0r["per_task"][t]
        rech[t] = {"eval_diff": abs(rr["eval"]["main"] - r0["eval"]["main"]), "hold_diff": abs(rr["hold"]["main"] - r0["hold"]["main"]),
                   "sha_equal": rr["adapter_sha256"] == r0["adapter_sha256"], "valid_recheck": rr["valid"], "valid_e1b": r0["valid"]}
    rech_ok = all(v["eval_diff"] <= 1e-6 and v["hold_diff"] <= 1e-6 and v["sha_equal"] for v in rech.values())
    valid = [t for t in V0 if s1["per_task"][t]["valid"] and S0_E1B["per_task"][t]["valid"]]
    pops = populations(valid)
    out = {"valid_tasks": e1b_order(valid), "K": len(valid), "excluded_seed1": {t: s1["per_task"][t]["invalid_reasons"] for t in V0 if t not in valid},
           "stop": len(valid) < 10, "seed0_recheck": rech, "seed0_recheck_ok": rech_ok,
           "populations": {k: [pid(x, y) for x, y in v] for k, v in pops.items()}, "n": {k: len(v) for k, v in pops.items()},
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    bm.jdump(out, HERE / "populations_e1c.json")
    timing_add("main", "stage0_s", time.time() - t0)
    log(f"e1c stage0: valid {len(valid)} {valid}; excluded {out['excluded_seed1']}; recheck_ok={rech_ok}; n={out['n']}")
    if out["stop"]: log("STOP: fewer than 10 valid tasks"); sys.exit(2)


def st_stage1(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if (HERE / "predictors_e1c.sha256").exists(): raise SystemExit("predictors already frozen; refusing to recompute")
    if list(HERE.glob("pair_results_*.jsonl")): raise SystemExit("merge results exist before predictors -> protocol violation")
    PO = json.loads((HERE / "populations_e1c.json").read_text())
    if PO["stop"]: raise SystemExit("stage0 STOP")
    pairs = []
    for k in ("P", "S1", "S2"):
        pairs += [tuple(p.split("__")) for p in PO["populations"][k]]
    t0 = time.time()
    compute_predictors(pairs, HERE, "predictors_e1c", dev, S0_E1B["lora_layers"])
    with open(HERE / "predictors_e1c.sha256", "w") as f:
        for fn in ("predictors_e1c.csv", "predictors_layers_e1c.csv", "predictors_null_e1c.json", "predictors_aux_e1c.json"):
            f.write(f"{bm.sha256(HERE / fn)}  {fn}\n")
    timing_add("main", "stage1_s", time.time() - t0)
    log("e1c stage1 done; frozen predictors_e1c.sha256:\n" + (HERE / "predictors_e1c.sha256").read_text())


def verify_predictors():
    ok = True
    for ln in (HERE / "predictors_e1c.sha256").read_text().strip().splitlines():
        h, fn = ln.split()
        if bm.sha256(HERE / fn) != h: ok = False; log(f"PREDICTOR HASH MISMATCH {fn}")
    return ok


def st_stage2(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    PO = json.loads((HERE / "populations_e1c.json").read_text())
    pairs = [tuple(p.split("__")) for p in PO["populations"][a.pop]]
    run_merges(pairs, HERE / f"pair_results_{a.pop}.jsonl", HERE / "preds", dev, S0_E1B["lora_layers"], singles(), a.bs, a.pop)
    log(f"e1c stage2 {a.pop} done")


def st_e3same(a):
    import torch
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    sys.path.insert(0, str(E3C))
    import e3                                                     # unmodified E3 code (imports merges.py)
    out = HERE / "e3same"; out.mkdir(exist_ok=True)
    e3.OUT = out; e3.LOGF = out / "run.log"
    e3.use_grids("e1b")                                           # amendment-A2 grids
    bm.setup_torch()
    PO = json.loads((HERE / "populations_e1c.json").read_text())
    pairs = [tuple(p.split("__")) for p in PO["populations"]["S2"]]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bm.BASE, revision=bm.BREV)
    S = e3.Setting(); S.name = "e1c_S2"
    S.pairs = pairs; S.tasks = sorted({k for p in pairs for k in p}); S.names = S0_E1B["lora_layers"]; S.brev = bm.BREV; S.base = bm.BASE
    sg = singles(); S.single = {k: sg[k] for k in S.tasks}
    aux = json.loads((HERE / "predictors_aux_e1c.json").read_text())
    S.ads, S.data = {}, {}
    cache = {}
    for k in S.tasks:
        ad = load_ad(k); assert ad["sha256"] == aux["adapter_sha256"][k]
        t = task_of(k)
        if t not in cache:
            d = bm.get_data(t, tok); cache[t] = {"hold": d["hold"], "eval": d["eval"], "hold_idx_sha256": d["hold_idx_sha256"]}
        assert cache[t]["hold_idx_sha256"] == ad["meta"]["hold_idx_sha256"]
        S.ads[k] = ad; S.data[k] = cache[t]
    S.score = lambda k, lg, y: float(bm.score(task_of(k), lg, y)["main"])
    S.compact = lambda k, lg: bm.compact(task_of(k), lg)
    S.bs = 256
    ns = argparse.Namespace(device=a.device, mem_frac=a.mem_frac, methods=["TA", "PICO_TA", "GATE", "FORCEGATE"])
    t0 = time.time()
    e3.run(ns, S)
    timing_add("S2", "e3same_s", time.time() - t0)
    log("e1c e3same done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--pop", default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--mem-frac", type=float, default=0.45)
    a = ap.parse_args()
    log(f"=== e1c.py {sys.argv}")
    if a.stage == "equiv": st_equiv(a)
    elif a.stage == "stage0": st_stage0(a)
    elif a.stage == "stage1": st_stage1(a)
    elif a.stage == "stage2": st_stage2(a)
    elif a.stage == "e3same": st_e3same(a)
    elif a.stage == "stage3":
        import e1c_analysis; e1c_analysis.main()
    else: raise SystemExit("unknown stage")


if __name__ == "__main__":
    main()
