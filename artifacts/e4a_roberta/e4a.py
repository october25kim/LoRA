#!/usr/bin/env python3
"""E4a: RoBERTa-base second-backbone replication of E1b/E1c (PREREG_E4A.md). Arithmetic copied from E1b e1b.py / E1c e1c.py;
E1b e1b.py (score, compact, spearman, holm, orth, delta, ties_thresholds, jdump, sha256, setup_torch) and E3 e3.py/merges.py are
imported UNMODIFIED (read-only, no bytecode). Backbone-specific code (adapter key parsing, encoder + RoBERTa classification head,
padding with <pad>=1, no token_type_ids) is new and marked as such.
Adapter keys: "<task>@s<seed>" -> adapters_s<seed>/<task>. Pair id "<key1>__<key2>" (t1 = key1, t2 = key2; the gate projects t2).
Stages:
  pilot    evaluate pilot adapters (held-out 1,000 TRAIN examples only) and apply the pre-registered divergence rule -> pilot/
  stage0   integrity (a)(b)(c)+structural for seeds 0 and 1 -> s0/stage0.json, s1/stage0.json; populations -> populations_e4a.json
  stage1   predictors for P, R0, S1, S2 (E1b stage1 arithmetic) -> predictors_e4a*.csv/json, frozen in predictors_e4a.sha256
  stage2   merges for one population (--pop P|R0|S1|S2): E1b stage2 arithmetic -> pair_results_<pop>.jsonl
  e3same   E3 run() (unmodified; encoder swapped for the RoBERTa one) on S2: TA / PICO_TA / GATE / FORCEGATE, A2 grids -> e3same/
  stage3   analysis -> analysis_e4a.json (+ figures); VERDICT_E4A.md is written by make_verdict_e4a.py
"""
import argparse, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
E3C = HERE.parent / "e3_baselines" / "code"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(E1B))
import e1b as bm                                   # noqa: E402  unmodified E1b pipeline (helpers only)
from tasks_e4 import ALL_SPECS, TASK_TYPE, load_raw, columns, split_indices, tokenize   # noqa: E402

PA = json.loads((HERE / "prereg_e4a.json").read_text())
R = PA["recipe"]; BASE, BREV = R["base_model"], R["base_revision"]
LAMS = PA["merge"]["lambda_grid"]; THETA_STAR = 30.0
REF = PA["integrity"]["b_reference"]["reference"]
TASKS = list(PA["tasks"])                          # E1b task-list order restricted to the 14 tasks
LOGF = HERE / "run_e4a.log"
bm.LOGF = LOGF; bm.OUT = HERE
D1 = json.loads((E1B / "deviations_d1.json").read_text())
N_LAYERS = R["lora_layers_expected"]
SMOKE = os.environ.get("E4A_SMOKE") == "1"         # pipeline smoke test with throwaway 20-step adapters in a scratch copy ONLY


def log(m): bm.log(m)
def task_of(k): return k.split("@")[0]
def seed_of(k): return int(k.split("@s")[1])
def pid(a, b): return f"{a}__{b}"
def metric_name(t): return ALL_SPECS[t][8]
def nlabels(t): return ALL_SPECS[t][7]


def timing_add(name, k, s):
    p = HERE / f"timing_{name}.json"; d = json.loads(p.read_text()) if p.exists() else {}
    d[k] = d.get(k, 0.0) + s; bm.jdump(d, p)


# ------------------------------------------------------------------ adapters (NEW: RoBERTa key layout)
def adapter_dir(k):
    return HERE / f"adapters_s{seed_of(k)}" / task_of(k)


def load_adapter_dir(d):
    from safetensors.torch import load_file
    d = Path(d)
    cfg = json.loads((d / "adapter_config.json").read_text())
    sd = load_file(str(d / "adapter_model.safetensors"))
    layers, head, other = {}, {}, []
    for k, v in sd.items():
        if ".lora_A." in k:
            name = k.split("base_model.model.roberta.")[-1].split(".lora_A.")[0]
            layers[name] = (v.float(), sd[k.replace(".lora_A.", ".lora_B.")].float())
        elif ".lora_B." in k:
            pass
        elif ".classifier." in k:
            hk = k.split(".classifier.")[-1].replace("modules_to_save.default.", "")
            head[hk] = v.float()
        else:
            other.append(k)
    meta = json.loads((d / "train_meta.json").read_text())
    return dict(layers=layers, head=head, other=other, cfg=cfg, scaling=cfg["lora_alpha"] / cfg["r"], r=cfg["r"],
                sha256=bm.sha256(d / "adapter_model.safetensors"), meta=meta, dir=str(d))


def load_ad(k):
    ad = load_adapter_dir(adapter_dir(k)); ad["key"] = k
    return ad


# ------------------------------------------------------------------ data (E1b get_data with the RoBERTa tokenizer)
def train_indices(task, n_train, n_eval):
    hold, tcap, ev = split_indices(n_train, n_eval)
    if task in D1["budgets"]:
        perm = np.random.default_rng(0).permutation(n_train)
        return hold, np.sort(perm[1000:]), ev
    return hold, tcap, ev


_TOK = None
def get_tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(BASE, revision=BREV)
    return _TOK


def get_data(task):
    import torch
    cdir = HERE / "cache"; cdir.mkdir(exist_ok=True)
    cp = cdir / f"{task}.pt"
    if cp.exists():
        return torch.load(cp, weights_only=False)
    tok = get_tok(); spec = ALL_SPECS[task]
    tr = load_raw(task, spec[2]); ev = load_raw(task, spec[3])
    hold, tidx, eidx = train_indices(task, len(tr), len(ev))
    out = {}
    for name, ds, idx in (("eval", ev, eidx), ("hold", tr, hold)):
        a, b, y = columns(task, ds.select(idx.tolist()))
        ids, tt = tokenize(tok, a, b)
        out[name] = {"ids": ids, "tt": tt, "labels": y, "idx": idx}
    if nlabels(task) > 1:
        _, _, yt = columns(task, tr.select(tidx.tolist()))
        out["train_labels"] = yt
    out["train_idx_sha256"] = hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest()
    out["hold_idx_sha256"] = hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()
    torch.save(out, cp)
    return out


# ------------------------------------------------------------------ model (NEW: RoBERTa encoder + RobertaClassificationHead)
class Enc:
    """RobertaModel (no pooler) with LoRA-covered linear layers overwritten by W0 + delta; per-task RoBERTa classification head
    (x = h[<s>]; tanh(dense(x)); out_proj). fp32, TF32 off (as E1b). Length-sorted stable batches, right padding with <pad> (id 1),
    attention mask, NO token_type_ids. Provides set_delta (E1b interface) and set_scaled (E3 interface)."""

    def __init__(self, dev, layer_names, base=None, brev=None):
        from transformers import RobertaModel
        self.dev = dev
        self.m = RobertaModel.from_pretrained(BASE, revision=BREV, add_pooling_layer=False).to(dev).eval()
        mods = dict(self.m.named_modules())
        miss = [n for n in layer_names if n not in mods]
        assert not miss, miss[:5]
        self.lin = {n: mods[n] for n in layer_names}
        self.W0 = {n: l.weight.detach().clone() for n, l in self.lin.items()}
        self.pad = get_tok().pad_token_id
        self._bc = {}

    def set_delta(self, d):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_(self.W0[n] if d is None else self.W0[n] + d[n])

    def set_scaled(self, base, scale):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_(self.W0[n] + scale * base[n])

    def batches(self, data, bs):
        import torch
        key = (id(data), bs)
        if key not in self._bc:
            ids = data["ids"]
            order = np.argsort([len(x) for x in ids], kind="stable"); out = []
            for i in range(0, len(ids), bs):
                idx = order[i:i + bs]; L = max(len(ids[j]) for j in idx)
                ii = np.full((len(idx), L), self.pad, dtype=np.int64); am = np.zeros_like(ii)
                for r, j in enumerate(idx):
                    l = len(ids[j]); ii[r, :l] = ids[j]; am[r, :l] = 1
                out.append((idx, torch.from_numpy(ii).to(self.dev), torch.from_numpy(am).to(self.dev)))
            self._bc[key] = (data, out)
        return self._bc[key][1]

    def predict(self, data, head, bs=256):
        import torch
        n = len(data["ids"])
        Wd = head["dense.weight"].to(self.dev); bd = head["dense.bias"].to(self.dev)
        Wo = head["out_proj.weight"].to(self.dev); bo = head["out_proj.bias"].to(self.dev)
        out = np.empty((n, Wo.shape[0]), dtype=np.float32)
        with torch.inference_mode():
            for idx, ii, am in self.batches(data, bs):
                h = self.m(input_ids=ii, attention_mask=am).last_hidden_state[:, 0]
                x = torch.tanh(h @ Wd.T + bd)
                out[idx] = (x @ Wo.T + bo).float().cpu().numpy()
        return out


# ------------------------------------------------------------------ pilot (pre-registered divergence rule)
def st_pilot(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    PP = PA["pilot"]; pdir = HERE / "pilot"; res = {}
    names = None
    for lr in PP["lr_ladder"]:
        ld = pdir / f"lr{lr:g}"
        if not all((ld / t / "DONE").exists() for t in PP["tasks"]):
            break
        res[lr] = {}
        for t in PP["tasks"]:
            ad = load_adapter_dir(ld / t); d = get_data(t)
            if names is None: names = sorted(ad["layers"]); enc = Enc(dev, names)
            enc.set_delta({n: bm.delta(ad, n, dev) for n in names})
            lh = enc.predict(d["hold"], ad["head"]); sh = bm.score(t, lh, d["hold"]["labels"])["main"]
            yt = d["train_labels"]; cnt = np.bincount(yt, minlength=nlabels(t)); maj = int(np.argmax(cnt))
            base = float((d["hold"]["labels"] == maj).mean())
            hist = ad["meta"]["loss_history"]; losses = np.array([l for _, l in hist], float)
            tail = losses[int(math.floor(0.8 * len(losses))):] if len(losses) else losses
            ent = ad["meta"]["label_prior_entropy"]
            crit = {"nan_or_inf_loss": bool(not np.all(np.isfinite(losses))),
                    "tail_loss_ge_0.9_prior_entropy": bool(len(tail) and float(np.mean(tail)) >= 0.9 * ent),
                    "holdout_acc_lt_majority_plus_10pp": bool(sh < base + 0.10)}
            res[lr][t] = {"holdout_acc": sh, "holdout_majority_baseline": base, "tail_mean_loss": float(np.mean(tail)) if len(tail) else None,
                          "label_prior_entropy": ent, "n_lora_layers": len(ad["layers"]), "criteria": crit, "diverged": any(crit.values()),
                          "train_wall_s": ad["meta"]["train_wall_s"], "loss_history": hist}
            log(f"pilot lr={lr:g} {t}: hold={sh:.4f} (maj {base:.4f}) tail_loss={res[lr][t]['tail_mean_loss']} ent={ent:.4f} diverged={res[lr][t]['diverged']} {crit}")
        del enc; names = None; torch.cuda.empty_cache()
    dec = {"results": {f"{k:g}": v for k, v in res.items()}, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    tried = list(res)
    ok = [lr for lr in tried if not any(res[lr][t]["diverged"] for t in PP["tasks"])]
    if ok:
        dec["lr_main"] = ok[0]; dec["status"] = "decided"; dec["rule"] = "first ladder lr at which neither pilot task diverged"
    elif len(tried) == len(PP["lr_ladder"]):
        best = max(tried, key=lambda lr: np.mean([res[lr][t]["holdout_acc"] for t in PP["tasks"]]))
        dec["lr_main"] = best; dec["status"] = "decided (all diverged: best mean held-out accuracy)"; dec["rule"] = PP["rule_if_all_diverge"]
    else:
        dec["status"] = "need_next"; dec["next_lr"] = PP["lr_ladder"][len(tried)]
    bm.jdump(dec, pdir / ("pilot_decision.json" if "lr_main" in dec else "pilot_status.json"))
    log(f"pilot decision: {dec.get('status')} lr_main={dec.get('lr_main')} next={dec.get('next_lr')}")
    if "lr_main" not in dec: sys.exit(10)


# ------------------------------------------------------------------ stage 0 (E1b stage0 logic; RoBERTa head/structure)
def stage0_seed(seed, dev, out):
    torch = bm.setup_torch()
    from transformers import RobertaForSequenceClassification
    from peft import PeftModel
    import transformers, peft, datasets
    t0 = time.time()
    s0 = {"stage": 0, "seed": seed, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "gpu": torch.cuda.get_device_name(0),
          "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "datasets": datasets.__version__},
          "precision": "fp32, TF32 disabled", "tasks": TASKS, "per_task": {}}
    ads = {t: load_ad(f"{t}@s{seed}") for t in TASKS}
    names = sorted(ads[TASKS[0]]["layers"]); s0["lora_layers"] = names; s0["n_lora_layers"] = len(names)
    enc = Enc(dev, names)
    pdir = HERE / "preds"; pdir.mkdir(exist_ok=True)
    for t in TASKS:
        ad = ads[t]; d = get_data(t)
        rec = {"adapter_dir": ad["dir"], "adapter_sha256": ad["sha256"], "trained_under": ad["meta"].get("deviation", "prereg"),
               "train_steps": ad["meta"]["global_step"], "train_wall_s": ad["meta"]["train_wall_s"], "train_loss": ad["meta"]["train_loss"],
               "loss_last_logged": ad["meta"]["loss_last_logged"], "label_prior_entropy": ad["meta"]["label_prior_entropy"],
               "segment_ids": ad["meta"]["segment_ids"], "lr": ad["meta"]["recipe"]["lr"]}
        structural = []
        if sorted(ad["layers"]) != names: structural.append("LoRA layer set differs")
        if len(names) != N_LAYERS: structural.append(f"{len(names)} LoRA layers != {N_LAYERS}")
        if ad["other"]: structural.append(f"unexpected keys {ad['other'][:3]}")
        hs = {k: tuple(v.shape) for k, v in ad["head"].items()}
        if hs.get("dense.weight") != (768, 768) or hs.get("out_proj.weight") != (nlabels(t), 768) or "dense.bias" not in hs or "out_proj.bias" not in hs:
            structural.append(f"classifier head missing/wrong shape {hs}")
        if ad["r"] != R["r"] or ad["cfg"]["lora_alpha"] != R["lora_alpha"]: structural.append("r/alpha mismatch")
        if d["train_idx_sha256"] != ad["meta"]["train_idx_sha256"]: structural.append("train index sha mismatch vs train_meta")
        if d["hold_idx_sha256"] != ad["meta"]["hold_idx_sha256"]: structural.append("held-out index sha mismatch vs train_meta")
        if ad["meta"].get("mode") != ("smoke" if SMOKE else "main") or ad["meta"]["seed"] != seed: structural.append("not a main-run adapter of this seed")
        rec["structural_problems"] = structural
        # (c) PEFT equivalence on the first 64 eval examples (single, unpadded, no token_type_ids)
        sub = {"ids": d["eval"]["ids"][:64], "tt": d["eval"]["tt"][:64]}
        enc.set_delta({n: bm.delta(ad, n, dev) for n in names})
        mine = enc.predict(sub, ad["head"])
        base = RobertaForSequenceClassification.from_pretrained(BASE, revision=BREV, num_labels=nlabels(t))
        pm = PeftModel.from_pretrained(base, ad["dir"]).to(dev).eval()
        ref = []
        with torch.inference_mode():
            for i in range(64):
                ref.append(pm(input_ids=torch.tensor([sub["ids"][i]], device=dev)).logits.float().cpu().numpy()[0])
        ref = np.stack(ref); diff = float(np.abs(ref - mine).max())
        del pm, base; torch.cuda.empty_cache()
        rec["c_max_abs_logit_diff"] = diff; rec["c_ok"] = diff <= 1e-4
        le = enc.predict(d["eval"], ad["head"]); lh = enc.predict(d["hold"], ad["head"])
        se = bm.score(t, le, d["eval"]["labels"]); sh = bm.score(t, lh, d["hold"]["labels"])
        np.savez_compressed(pdir / f"single_{t}@s{seed}.npz", eval=bm.compact(t, le), hold=bm.compact(t, lh),
                            eval_labels=d["eval"]["labels"], hold_labels=d["hold"]["labels"])
        rec["eval"] = se; rec["hold"] = sh; rec["n_eval"] = se["n"]
        if metric_name(t) == "spearman":
            rec["a_threshold"] = 0.70; rec["a_ok"] = se["main"] >= 0.70; rec["majority_baseline"] = None
        else:
            yt = d["train_labels"]; cnt = np.bincount(yt, minlength=nlabels(t)); tied = np.flatnonzero(cnt == cnt.max())
            accs = {int(c): float((d["eval"]["labels"] == c).mean()) for c in tied}
            maj = max(accs, key=accs.get)
            rec["majority_label"] = maj; rec["majority_baseline"] = accs[maj]
            rec["a_threshold"] = accs[maj] + 0.10; rec["a_ok"] = se["main"] >= rec["a_threshold"]
        if t in REF:
            m = REF[t]["metric"]; ours = se["main"] if m in ("accuracy", "spearman") else se[m]
            rec["b_metric"] = m; rec["b_reference"] = REF[t]["value"]; rec["b_threshold"] = round(REF[t]["value"] - 0.05, 6)
            rec["b_ours"] = ours; rec["b_ok"] = ours >= rec["b_threshold"]
        else:
            rec["b_ok"] = None
        rec["valid"] = bool(rec["a_ok"] and rec["c_ok"] and rec["b_ok"] is not False and not structural)
        if SMOKE: rec["valid"] = bool(rec["c_ok"] and not structural); rec["SMOKE_FORCE_VALID"] = True
        reasons = [f"(a) {se['main']:.4f} < {rec['a_threshold']:.4f}"] if not rec["a_ok"] else []
        if rec["b_ok"] is False: reasons.append(f"(b) {rec['b_metric']} {rec['b_ours']:.4f} < {rec['b_threshold']:.4f}")
        if not rec["c_ok"]: reasons.append(f"(c) logit diff {diff:.2e} > 1e-4")
        reasons += structural
        rec["invalid_reasons"] = reasons
        s0["per_task"][t] = rec
        log(f"e4a stage0 s{seed} {t}: eval={se['main']:.4f} hold={sh['main']:.4f} a={rec['a_ok']} b={rec['b_ok']} c={rec['c_ok']} (diff {diff:.1e}) valid={rec['valid']} {reasons}")
    s0["valid_tasks"] = [t for t in TASKS if s0["per_task"][t]["valid"]]
    s0["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z"); s0["wall_s"] = time.time() - t0
    out.mkdir(exist_ok=True, parents=True); bm.jdump(s0, out / "stage0.json")
    del enc; torch.cuda.empty_cache()
    return s0


def populations(valid):
    V = [t for t in TASKS if t in valid]
    P = [(f"{x}@s0", f"{y}@s1") for x, y in itertools.combinations(sorted(V), 2)]      # alphabetical: first -> seed 0 (t1)
    R0 = [(f"{x}@s0", f"{y}@s0") for x, y in itertools.combinations(V, 2)]             # E1b task-list order
    S1 = [(f"{x}@s1", f"{y}@s1") for x, y in itertools.combinations(V, 2)]
    S2 = [(f"{t}@s0", f"{t}@s1") for t in V]
    return {"P": P, "R0": R0, "S1": S1, "S2": S2}


def st_stage0(a):
    torch = bm.setup_torch(); dev = torch.device(a.device); t0 = time.time()
    res = {s: stage0_seed(s, dev, HERE / f"s{s}") for s in PA["seeds"]}
    valid = [t for t in TASKS if all(res[s]["per_task"][t]["valid"] for s in PA["seeds"])]
    pops = populations(valid)
    out = {"valid_tasks": valid, "K": len(valid),
           "excluded": {t: {f"s{s}": res[s]["per_task"][t]["invalid_reasons"] for s in PA["seeds"]} for t in TASKS if t not in valid},
           "stop": len(valid) < PA["integrity"]["stop_if_valid_tasks_below"], "lora_layers": res[0]["lora_layers"],
           "populations": {k: [pid(x, y) for x, y in v] for k, v in pops.items()}, "n": {k: len(v) for k, v in pops.items()},
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    bm.jdump(out, HERE / "populations_e4a.json")
    timing_add("main", "stage0_s", time.time() - t0)
    log(f"e4a stage0: valid {len(valid)} {valid}; excluded {out['excluded']}; n={out['n']}")
    if out["stop"]: log("STOP: fewer than 10 valid tasks"); sys.exit(2)


def singles():
    out = {}
    for s in PA["seeds"]:
        s0 = json.loads((HERE / f"s{s}" / "stage0.json").read_text())
        for t, r in s0["per_task"].items():
            out[f"{t}@s{s}"] = {"eval": r["eval"]["main"], "hold": r["hold"]["main"]}
    return out


# ------------------------------------------------------------------ stage 1 (E1c compute_predictors; O_B dropped: identical to O_A)
def compute_predictors(pairs, outdir, prefix, dev, names):
    torch = bm.setup_torch(); import pandas as pd
    keys = sorted({k for p in pairs for k in p}, key=lambda k: (seed_of(k), TASKS.index(task_of(k))))
    ads = {k: load_ad(k) for k in keys}; r = R["r"]
    for k in keys: assert sorted(ads[k]["layers"]) == names, k
    QA = {}
    for t in keys:
        QA[t] = {}
        for n in names:
            B = ads[t]["layers"][n][1].to(dev).double()
            QA[t][n] = bm.orth(B)
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
        cA, thA = [], []
        for n in names:
            sa = torch.linalg.svdvals(QA[a][n].T @ QA[b][n]).clamp(max=1.0)
            c2a = float((sa ** 2).mean()); tha = float(np.degrees(np.arccos(float(sa.max()))))
            n30 = int((sa > math.cos(math.radians(THETA_STAR))).sum())
            cA.append(c2a); thA.append(tha)
            lrows.append({"pair": pid(a, b), "t1": a, "t2": b, "layer": n, "d_out": dims[n], "cos2_mean_A": c2a, "theta_min_A_deg": tha, "n_angles_lt30_A": n30})
        n1, n2 = math.sqrt(nrm[a]), math.sqrt(nrm[b]); OA = float(np.mean(cA)); c = conf[(a, b)]
        rows.append({"pair": pid(a, b), "t1": a, "t2": b, "t1_task": task_of(a), "t2_task": task_of(b), "t1_seed": seed_of(a), "t2_seed": seed_of(b),
                     "O_A": OA, "tv_cosine": dots[(a, b)] / (n1 * n2),
                     "mean_theta_min_A_deg": float(np.mean(thA)), "min_theta_min_A_deg": float(np.min(thA)),
                     "n_layers_theta_min_lt30_A": int(np.sum(np.array(thA) < THETA_STAR)), "null_z_O_A": (OA - mu) / sd,
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


def st_stage1(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if (HERE / "predictors_e4a.sha256").exists(): raise SystemExit("predictors already frozen; refusing to recompute")
    if list(HERE.glob("pair_results_*.jsonl")): raise SystemExit("merge results exist before predictors -> protocol violation")
    PO = json.loads((HERE / "populations_e4a.json").read_text())
    if PO["stop"]: raise SystemExit("stage0 STOP")
    pairs = []
    for k in ("P", "R0", "S1", "S2"):
        pairs += [tuple(p.split("__")) for p in PO["populations"][k]]
    t0 = time.time()
    compute_predictors(pairs, HERE, "predictors_e4a", dev, PO["lora_layers"])
    with open(HERE / "predictors_e4a.sha256", "w") as f:
        for fn in ("predictors_e4a.csv", "predictors_layers_e4a.csv", "predictors_null_e4a.json", "predictors_aux_e4a.json"):
            f.write(f"{bm.sha256(HERE / fn)}  {fn}\n")
    timing_add("main", "stage1_s", time.time() - t0)
    log("e4a stage1 done; frozen predictors_e4a.sha256:\n" + (HERE / "predictors_e4a.sha256").read_text())


def verify_predictors():
    ok = True
    for ln in (HERE / "predictors_e4a.sha256").read_text().strip().splitlines():
        h, fn = ln.split()
        if bm.sha256(HERE / fn) != h: ok = False; log(f"PREDICTOR HASH MISMATCH {fn}")
    return ok


# ------------------------------------------------------------------ stage 2 (E1c run_merges; same arithmetic as e1b.stage2)
def run_merges(pairs, jl, pdir, dev, names, single, bs, tname):
    torch = bm.setup_torch(); import pandas as pd
    keys = sorted({k for p in pairs for k in p})
    ads = {k: load_ad(k) for k in keys}
    aux = json.loads((HERE / "predictors_aux_e4a.json").read_text())
    for k in keys: assert ads[k]["sha256"] == aux["adapter_sha256"][k], f"{k} adapter changed since predictors"
    thr = aux["ties_top20_abs_threshold"]
    L = pd.read_csv(HERE / "predictors_layers_e4a.csv")
    data = {}
    for k in keys:
        t = task_of(k)
        if t not in data: data[t] = get_data(t)
        assert data[t]["hold_idx_sha256"] == ads[k]["meta"]["hold_idx_sha256"], f"{k}: held-out indices differ"
    enc = Enc(dev, names)
    done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()} if jl.exists() else set()
    pdir.mkdir(exist_ok=True, parents=True)
    cos_star = math.cos(math.radians(THETA_STAR))
    for (a, b) in pairs:
        key = pid(a, b)
        if jl.exists():                                            # re-read: a second process may work on the same population (reverse order)
            done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()}
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
        log(f"e4a stage2[{tname}] {key}: lam*={lam_ta} D={rec['D']:.4f} TIES(atTA/own)={rec['TIES_atTA_score']:.4f}/{rec['TIES_own_score']:.4f} "
            f"GATE({len(fail_layers)} layers)={rec['GATE_atTA_score']:.4f} [{rec['secs']:.0f}s]")
        del d1, d2; torch.cuda.empty_cache()
        timing_add(tname, "stage2_gpu_wall_s", time.time() - tp)


def st_stage2(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    PO = json.loads((HERE / "populations_e4a.json").read_text())
    pairs = [tuple(p.split("__")) for p in PO["populations"][a.pop]]
    if a.limit: pairs = pairs[:a.limit]
    if a.reverse: pairs = pairs[::-1]
    run_merges(pairs, HERE / f"pair_results_{a.pop}.jsonl", HERE / "preds", dev, PO["lora_layers"], singles(), a.bs, a.pop)
    log(f"e4a stage2 {a.pop} done")


def st_e3same(a):
    import torch
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    sys.path.insert(0, str(E3C))
    import e3                                                     # unmodified E3 code (imports merges.py)
    out = HERE / "e3same"; out.mkdir(exist_ok=True)
    e3.OUT = out; e3.LOGF = out / "run.log"
    e3.use_grids("e1b")                                           # amendment-A2 grids
    e3.Enc = Enc                                                  # RoBERTa encoder (same set_scaled/predict interface)
    bm.setup_torch()
    PO = json.loads((HERE / "populations_e4a.json").read_text())
    pairs = [tuple(p.split("__")) for p in PO["populations"]["S2"]]
    S = e3.Setting(); S.name = "e4a_S2"
    S.pairs = pairs; S.tasks = sorted({k for p in pairs for k in p}); S.names = PO["lora_layers"]; S.brev = BREV; S.base = BASE
    sg = singles(); S.single = {k: sg[k] for k in S.tasks}
    aux = json.loads((HERE / "predictors_aux_e4a.json").read_text())
    S.ads, S.data = {}, {}
    cache = {}
    for k in S.tasks:
        ad = load_ad(k); assert ad["sha256"] == aux["adapter_sha256"][k]
        t = task_of(k)
        if t not in cache:
            d = get_data(t); cache[t] = {"hold": d["hold"], "eval": d["eval"], "hold_idx_sha256": d["hold_idx_sha256"]}
        assert cache[t]["hold_idx_sha256"] == ad["meta"]["hold_idx_sha256"]
        S.ads[k] = ad; S.data[k] = cache[t]
    S.score = lambda k, lg, y: float(bm.score(task_of(k), lg, y)["main"])
    S.compact = lambda k, lg: bm.compact(task_of(k), lg)
    S.bs = 256
    ns = argparse.Namespace(device=a.device, mem_frac=a.mem_frac, methods=["TA", "PICO_TA", "GATE", "FORCEGATE"])
    t0 = time.time()
    e3.run(ns, S)
    timing_add("S2", "e3same_s", time.time() - t0)
    log("e4a e3same done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--pop", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--mem-frac", type=float, default=0.45)
    a = ap.parse_args()
    log(f"=== e4a.py {sys.argv}")
    if a.stage == "pilot": st_pilot(a)
    elif a.stage == "stage0": st_stage0(a)
    elif a.stage == "stage1": st_stage1(a)
    elif a.stage == "stage2": st_stage2(a)
    elif a.stage == "e3same": st_e3same(a)
    elif a.stage == "stage3":
        import e4a_analysis; e4a_analysis.main()
    else: raise SystemExit("unknown stage")


if __name__ == "__main__":
    main()
