#!/usr/bin/env python3
"""E6: Qwen2.5-1.5B second-DECODER replication of the E4b decoder analysis (PREREG_E6.md). Copy of E4b e4b.py; E1b e1b.py helpers imported
UNMODIFIED (read-only, no bytecode). E6 changes vs e4b.py are marked "E6:" and are either (i) backbone constants (1.5B, 196 LoRA layers,
hidden 1536), (ii) memory-only restructurings needed on a 16 GB GPU that leave every computed number bit-identical to the E4b arithmetic
(deltas streamed per layer instead of cached for all layers; W0 of the fp32 evaluator kept on CPU; stage0 runs the fp32 checks for all tasks
of a seed, then the bf16 singles; TIES top-20% threshold via an exact order statistic on CPU), or (iii) protocol items fixed in PREREG_E6.md
(two seeds fixed -> primary P; no E3 subset; deadline = pilot start + 40 h).
Adapter keys: "<task>@s<seed>" -> adapters_s<seed>/<task>. Pair id "<key1>__<key2>" (t1 = key1, t2 = key2; the gate projects t2).
Stages:
  pilot    evaluate the pilot adapters (held-out 1,000 TRAIN examples only), apply the pre-registered lr rule (seeds fixed [0, 1]) -> pilot/
  stage0   integrity (a)(c)(c2)+structural per seed -> s<seed>/stage0.json; populations -> populations_e6.json
  stage1   predictors for all populations (E1b stage1 arithmetic) -> predictors_e6*.csv/json, frozen in predictors_e6.sha256
  stage2   merges for one population (--pop P|R0|S1|S2): E1b stage2 arithmetic (bf16 merged weights) -> pair_results_<pop>.jsonl
  stage3   analysis -> analysis_e6.json (+ figures) and e6_extra.json; VERDICT_E6_DECODER2.md is written by make_verdict_e6.py
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
from tasks_e6 import ALL_SPECS, TASK_TYPE, load_raw, columns, indices_e4b, tokenize   # noqa: E402

PA = json.loads((HERE / "prereg_e6.json").read_text())
R = PA["recipe"]; BASE, BREV = R["base_model"], R["base_revision"]
LAMS = PA["merge"]["lambda_grid"]; THETA_STAR = 30.0
LAMS_EXT = PA["merge"].get("lambda_grid_extended_secondary", [])   # E6: [1.3, 1.5, 2.0] (secondary; never used for lam_selected / D)
TASKS = list(PA["tasks"])                          # E1b task-list order restricted to the 14 tasks
LOGF = HERE / "run_e6.log"
bm.LOGF = LOGF; bm.OUT = HERE
N_LAYERS = R["lora_layers_expected"]
BS = PA["merge"]["eval_batch"]
DEADLINE = float(os.environ.get("E6_DEADLINE_EPOCH", "0"))   # budget guard: no new pair/stage is started after this time (0 = off)


def seeds_used():
    p = HERE / "pilot" / "pilot_decision.json"
    return json.loads(p.read_text())["seeds"] if p.exists() else [0]


def past_deadline():
    stopf = os.environ.get("E6_STOPFILE")                        # phase control (run_pipeline_e6.sh): stop starting new pairs
    return (DEADLINE > 0 and time.time() > DEADLINE) or bool(stopf and Path(stopf).exists())
SMOKE = os.environ.get("E6_SMOKE") == "1"         # pipeline smoke test with throwaway 20-step adapters in a scratch copy ONLY
HID = R["hidden_size"]                             # E6: 1536 (E4b: 896)


def log(m): bm.log(m)
def task_of(k): return k.split("@")[0]
def seed_of(k): return int(k.split("@s")[1])
def pid(a, b): return f"{a}__{b}"
def metric_name(t): return ALL_SPECS[t][8]
def nlabels(t): return ALL_SPECS[t][7]


def timing_add(name, k, s):
    p = HERE / f"timing_{name}.json"; d = json.loads(p.read_text()) if p.exists() else {}
    d[k] = d.get(k, 0.0) + s; bm.jdump(d, p)


# ------------------------------------------------------------------ adapters (NEW: Qwen2 key layout; layer name = "model.layers.N.<module>")
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
            name = k.split("base_model.model.", 1)[-1].split(".lora_A.")[0]
            layers[name] = (v.float(), sd[k.replace(".lora_A.", ".lora_B.")].float())
        elif ".lora_B." in k:
            pass
        elif k.split("base_model.model.", 1)[-1].replace("modules_to_save.default.", "") == "score.weight":
            head["weight"] = v.float()
        else:
            other.append(k)
    meta = json.loads((d / "train_meta.json").read_text())
    return dict(layers=layers, head=head, other=other, cfg=cfg, scaling=cfg["lora_alpha"] / cfg["r"], r=cfg["r"],
                sha256=bm.sha256(d / "adapter_model.safetensors"), meta=meta, dir=str(d))


def load_ad(k):
    ad = load_adapter_dir(adapter_dir(k)); ad["key"] = k
    return ad


# ------------------------------------------------------------------ data (E1b get_data with the Qwen tokenizer and E4b indices)
def train_indices(task, n_train, n_eval):
    return indices_e4b(n_train, n_eval)


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


# ------------------------------------------------------------------ model (NEW: Qwen2 decoder + last-non-pad-token score head)
class Enc:
    """Qwen2Model (E6: 28 decoder layers + final norm) with the 196 LoRA-covered linear layers overwritten by W0 + delta; per-task head
    logits = score.weight @ h[last non-pad token] (no bias; computed in fp32 from the model-dtype hidden state).
    dtype "bf16" (merges, singles; weights = bf16(W0 + delta computed in fp32), W0 = the bf16 checkpoint, exact) or "fp32" (integrity (c), (c2)).
    Length-sorted stable batches, right padding with <|endoftext|> + attention mask (causal model: padding cannot affect the last real token).
    Interface: set_delta (E1b), set_scaled (E3), predict(data, head, bs)."""

    def __init__(self, dev, layer_names, base=None, brev=None, dtype="bf16", w0_cpu=False):
        import torch
        from transformers import Qwen2Model
        self.dev = dev
        self.dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.m = Qwen2Model.from_pretrained(BASE, revision=BREV, dtype=self.dt).to(dev).eval()
        mods = dict(self.m.named_modules())
        loc = {n: n[len("model."):] for n in layer_names}
        miss = [n for n in layer_names if loc[n] not in mods]
        assert not miss, miss[:5]
        self.lin = {n: mods[loc[n]] for n in layer_names}
        # E6 (memory only): optionally keep the W0 copies on CPU (pinned); they are moved back unchanged before use -> identical values.
        self.W0 = {n: (l.weight.detach().to("cpu").pin_memory() if w0_cpu else l.weight.detach().clone()) for n, l in self.lin.items()}
        self.w0_cpu = w0_cpu
        self.pad = get_tok().pad_token_id
        self._bc = {}

    def _w0(self, n):
        return self.W0[n].to(self.dev, non_blocking=False) if self.w0_cpu else self.W0[n]

    def set_delta(self, d):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_(self._w0(n) if d is None else (self._w0(n).float() + d[n]).to(self.dt))

    def set_delta_fn(self, fn):
        """E6 (memory only): same arithmetic as set_delta({n: fn(n)}) (W0.float() + delta, cast to the model dtype), but the fp32 delta of
        each layer is computed just before it is written, so no all-layer delta dict is held on the GPU."""
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_((self._w0(n).float() + fn(n)).to(self.dt))

    def set_scaled(self, base, scale):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_((self._w0(n).float() + scale * base[n]).to(self.dt))

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

    def predict(self, data, head, bs=128):
        import torch
        n = len(data["ids"])
        W = head["weight"].to(self.dev).float()
        out = np.empty((n, W.shape[0]), dtype=np.float32)
        with torch.inference_mode():
            for idx, ii, am in self.batches(data, bs):
                h = self.m(input_ids=ii, attention_mask=am).last_hidden_state
                last = am.sum(1) - 1
                hs = h[torch.arange(h.shape[0], device=self.dev), last].float()
                out[idx] = (hs @ W.T).cpu().numpy()
        return out


def majority(d, t, which):
    yt = d["train_labels"]; cnt = np.bincount(yt, minlength=nlabels(t)); maj = int(np.argmax(cnt))
    return float((d[which]["labels"] == maj).mean())


# ------------------------------------------------------------------ pilot (pre-registered lr rule + seed/cost rule)
def st_pilot(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    PP = PA["pilot"]; pdir = HERE / "pilot"; res = {}; names = None; enc = None; fwd = []
    for lr in PP["lr_candidates"]:
        ld = pdir / f"lr{lr:g}"
        assert all((ld / t / "DONE").exists() for t in PP["tasks"]), f"pilot adapters missing for lr {lr}"
        res[lr] = {}
        for t in PP["tasks"]:
            ad = load_adapter_dir(ld / t); d = get_data(t)
            if names is None: names = sorted(ad["layers"]); enc = Enc(dev, names, dtype="bf16")
            enc.set_delta({n: bm.delta(ad, n, dev) for n in names})
            torch.cuda.synchronize(); t0 = time.time()
            lh = enc.predict(d["hold"], ad["head"], BS)
            torch.cuda.synchronize(); dt = time.time() - t0
            fwd.append({"task": t, "lr": lr, "n_tokens": int(sum(len(x) for x in d["hold"]["ids"])), "secs": dt})
            sh = bm.score(t, lh, d["hold"]["labels"])["main"]
            base = majority(d, t, "hold")
            hist = ad["meta"]["loss_history"]; losses = np.array([l for _, l in hist], float)
            tail = losses[int(math.floor(0.8 * len(losses))):] if len(losses) else losses
            ent = ad["meta"]["label_prior_entropy"]
            crit = {"nan_or_inf_loss": bool(not np.all(np.isfinite(losses))),
                    "tail_loss_ge_0.9_prior_entropy": bool(len(tail) and float(np.mean(tail)) >= 0.9 * ent),
                    "holdout_acc_lt_majority_plus_10pp": bool(sh < base + 0.10)}
            res[lr][t] = {"holdout_acc": sh, "holdout_majority_baseline": base, "tail_mean_loss": float(np.mean(tail)) if len(tail) else None,
                          "label_prior_entropy": ent, "n_lora_layers": len(ad["layers"]), "criteria": crit, "diverged": any(crit.values()),
                          "train_wall_s": ad["meta"]["train_wall_s"], "sec_per_step": ad["meta"]["sec_per_step"], "mean_train_tokens": ad["meta"]["mean_train_tokens"],
                          "peak_mem_gb": ad["meta"]["peak_mem_gb"], "loss_history": hist}
            log(f"pilot lr={lr:g} {t}: hold={sh:.4f} (maj {base:.4f}) tail_loss={res[lr][t]['tail_mean_loss']} ent={ent} diverged={res[lr][t]['diverged']} {crit} sec/step={ad['meta']['sec_per_step']:.4f}")
    del enc; torch.cuda.empty_cache()
    ok = [lr for lr in PP["lr_candidates"] if not any(res[lr][t]["diverged"] for t in PP["tasks"])]
    pool = ok if ok else list(PP["lr_candidates"])
    score = {lr: float(np.mean([res[lr][t]["holdout_acc"] for t in PP["tasks"]])) for lr in pool}
    best = max(score.values()); lr_main = min(lr for lr in pool if score[lr] >= best - 1e-12)      # ties -> smaller lr
    # ---- cost projection (pre-registered formula, prereg_e6.json pilot.cost_rule)
    C = PP["cost_rule"]; tl = C["mean_train_tokens"]
    s_rows = [(res[lr_main][t]["mean_train_tokens"], res[lr_main][t]["sec_per_step"]) for t in PP["tasks"]]
    (x1, y1), (x2, y2) = s_rows
    beta = (y2 - y1) / (x2 - x1); alpha = y1 - beta * x1
    sps = {t: max(alpha + beta * tl[t], min(y1, y2)) for t in TASKS}
    train_seed_h = sum(PA["budgets"]["per_task"][t]["max_steps"] * sps[t] for t in TASKS) / 3600.0
    tau = sum(f["n_tokens"] for f in fwd) / sum(f["secs"] for f in fwd)        # bf16 forward tokens/s
    ht, et = C["hold_tokens"], C["eval_tokens"]
    def pair_s(i, j):
        return (4 * (ht[i] + ht[j] + et[i] + et[j]) + 4 * (ht[i] + ht[j]) + 2 * (et[i] + et[j])) / tau + C["pair_overhead_s"]
    P_s = sum(pair_s(i, j) for i, j in itertools.combinations(TASKS, 2)); S2_s = sum(pair_s(t, t) for t in TASKS)
    pilot_h = (time.time() - float((pdir / "pilot_start_epoch.txt").read_text())) / 3600.0
    merges_core_h = (P_s + S2_s * (1 + C["e3_pair_cost_factor"])) / 3600.0 / C["merge_parallel_speedup"]
    core2 = pilot_h + 2 * train_seed_h + C["stage01_h"] + merges_core_h
    full2 = core2 + 2 * P_s / 3600.0 / C["merge_parallel_speedup"]
    seeds = [0, 1]                                                     # E6: two seeds fixed by PREREG_E6.md (projection recorded only)
    dec = {"results": {f"{k:g}": v for k, v in res.items()}, "lr_main": lr_main, "non_diverged": ok, "mean_holdout_acc": {f"{k:g}": v for k, v in score.items()},
           "lr_rule": PP["decision"], "seeds": seeds, "primary_population": "P" if seeds == [0, 1] else "R0",
           "projection": {"sec_per_step_line": {"alpha": alpha, "beta": beta}, "sec_per_step_by_task": sps, "train_h_per_seed": train_seed_h,
                          "fwd_tokens_per_s_bf16": tau, "pair_s_mean_cross_task": P_s / 91.0, "pilot_h": pilot_h, "merges_core_h": merges_core_h,
                          "core_h_two_seeds": core2, "full_h_two_seeds_incl_R0_S1": full2, "threshold_h": None, "seed_rule": "E6: seeds fixed [0, 1] (not projection-dependent)", "forward_timings": fwd},
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    bm.jdump(dec, pdir / "pilot_decision.json")
    log(f"pilot decision: lr_main={lr_main} (non-diverged {ok}; mean held-out {score}); projection core2={core2:.2f}h full2={full2:.2f}h -> seeds {seeds}")


# ------------------------------------------------------------------ stage 0 (E1b stage0 logic; Qwen head/structure; (c) fp32, (c2) bf16 vs fp32)
def stage0_seed(seed, dev, out):
    """E6: E4b stage0 logic and criteria unchanged; memory-only reordering for 16 GB: pass 1 (fp32 evaluator with W0 on CPU + one PeftModel
    at a time) computes (c) and the fp32 held-out metric for every task of the seed; pass 2 (bf16 evaluator) computes the bf16 singles
    (eval + held-out) and (c2). Every number is computed by the same operations as in e4b.stage0_seed."""
    torch = bm.setup_torch()
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel
    import transformers, peft, datasets
    t0 = time.time()
    s0 = {"stage": 0, "seed": seed, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "gpu": torch.cuda.get_device_name(0),
          "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "datasets": datasets.__version__},
          "precision": "singles/merges: bf16 weights = bf16(W0 + delta), fp32 head; integrity (c) fp32, TF32 disabled", "tasks": TASKS, "per_task": {}}
    ads = {t: load_ad(f"{t}@s{seed}") for t in TASKS}
    names = sorted(ads[TASKS[0]]["layers"]); s0["lora_layers"] = names; s0["n_lora_layers"] = len(names)
    tok = get_tok()
    pdir = HERE / "preds"; pdir.mkdir(exist_ok=True)
    recs = {}
    # ---- pass 1: fp32 (c) and fp32 held-out
    enc32 = Enc(dev, names, dtype="fp32", w0_cpu=True)
    for t in TASKS:
        ad = ads[t]; d = get_data(t)
        rec = {"adapter_dir": ad["dir"], "adapter_sha256": ad["sha256"], "train_steps": ad["meta"]["global_step"], "train_wall_s": ad["meta"]["train_wall_s"],
               "train_loss": ad["meta"]["train_loss"], "loss_last_logged": ad["meta"]["loss_last_logged"], "label_prior_entropy": ad["meta"]["label_prior_entropy"],
               "lr": ad["meta"]["recipe"]["lr"], "micro_batch": ad["meta"]["micro_batch"], "peak_mem_gb": ad["meta"]["peak_mem_gb"]}
        structural = []
        if sorted(ad["layers"]) != names: structural.append("LoRA layer set differs")
        if len(names) != N_LAYERS: structural.append(f"{len(names)} LoRA layers != {N_LAYERS}")
        if ad["other"]: structural.append(f"unexpected keys {ad['other'][:3]}")
        hs = {k: tuple(v.shape) for k, v in ad["head"].items()}
        if hs != {"weight": (nlabels(t), HID)}: structural.append(f"score head missing/wrong shape {hs}")
        if ad["r"] != R["r"] or ad["cfg"]["lora_alpha"] != R["lora_alpha"]: structural.append("r/alpha mismatch")
        if d["train_idx_sha256"] != ad["meta"]["train_idx_sha256"]: structural.append("train index sha mismatch vs train_meta")
        if d["hold_idx_sha256"] != ad["meta"]["hold_idx_sha256"]: structural.append("held-out index sha mismatch vs train_meta")
        if ad["meta"].get("mode") != ("smoke" if SMOKE else "main") or ad["meta"]["seed"] != seed: structural.append("not a main-run adapter of this seed")
        rec["structural_problems"] = structural
        # (c) PEFT equivalence on the first 64 eval examples (single, unpadded), fp32, TF32 off
        sub = {"ids": d["eval"]["ids"][:64], "tt": d["eval"]["tt"][:64]}
        enc32.set_delta_fn(lambda n: bm.delta(ad, n, dev)); mine = enc32.predict(sub, ad["head"], BS)
        lh32 = enc32.predict(d["hold"], ad["head"], BS); sh32 = bm.score(t, lh32, d["hold"]["labels"])["main"]
        base = AutoModelForSequenceClassification.from_pretrained(BASE, revision=BREV, num_labels=nlabels(t), dtype=torch.float32)
        base.config.pad_token_id = tok.pad_token_id
        pm = PeftModel.from_pretrained(base, ad["dir"]).to(dev).eval()
        ref = []
        with torch.inference_mode():
            for i in range(64):
                ref.append(pm(input_ids=torch.tensor([sub["ids"][i]], device=dev)).logits.float().cpu().numpy()[0])
        ref = np.stack(ref); diff = float(np.abs(ref - mine).max()); tol = PA["integrity"]["c_tol_rel"] * max(1.0, float(np.abs(ref).max()))
        del pm, base; torch.cuda.empty_cache()
        rec["c_max_abs_logit_diff"] = diff; rec["c_tol"] = tol; rec["c_max_abs_ref_logit"] = float(np.abs(ref).max()); rec["c_ok"] = diff <= tol
        rec["c2_hold_fp32"] = sh32
        recs[t] = rec
        log(f"e6 stage0 s{seed} {t}: pass1 (c) diff {diff:.1e}/tol {tol:.1e} ok={rec['c_ok']}; hold fp32 {sh32:.4f}")
    del enc32; torch.cuda.empty_cache()
    # ---- pass 2: bf16 singles (study precision) and (c2)
    enc16 = Enc(dev, names, dtype="bf16")
    for t in TASKS:
        ad = ads[t]; d = get_data(t); rec = recs[t]; structural = rec["structural_problems"]; diff, tol = rec["c_max_abs_logit_diff"], rec["c_tol"]
        enc16.set_delta_fn(lambda n: bm.delta(ad, n, dev))
        le = enc16.predict(d["eval"], ad["head"], BS); lh = enc16.predict(d["hold"], ad["head"], BS)
        se = bm.score(t, le, d["eval"]["labels"]); sh = bm.score(t, lh, d["hold"]["labels"]); sh32 = rec["c2_hold_fp32"]
        rec["c2_hold_bf16"] = sh["main"]; rec["c2_absdiff"] = abs(sh32 - sh["main"]); rec["c2_ok"] = rec["c2_absdiff"] <= PA["integrity"]["c2_tol"]
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
        rec["b_ok"] = None; rec["b_note"] = "not applicable (no citable Qwen2.5-1.5B references for these tasks and this setup)"
        rec["valid"] = bool(rec["a_ok"] and rec["c_ok"] and rec["c2_ok"] and not structural)
        if SMOKE: rec["valid"] = bool(rec["c_ok"] and not structural); rec["SMOKE_FORCE_VALID"] = True
        reasons = [f"(a) {se['main']:.4f} < {rec['a_threshold']:.4f}"] if not rec["a_ok"] else []
        if not rec["c_ok"]: reasons.append(f"(c) logit diff {diff:.2e} > {tol:.2e}")
        if not rec["c2_ok"]: reasons.append(f"(c2) bf16 vs fp32 held-out {rec['c2_absdiff']:.4f} > {PA['integrity']['c2_tol']}")
        reasons += structural
        rec["invalid_reasons"] = reasons
        s0["per_task"][t] = rec
        log(f"e6 stage0 s{seed} {t}: eval={se['main']:.4f} hold={sh['main']:.4f} (fp32 {sh32:.4f}) a={rec['a_ok']} c={rec['c_ok']} (diff {diff:.1e}/tol {tol:.1e}) c2={rec['c2_ok']} valid={rec['valid']} {reasons}")
    s0["valid_tasks"] = [t for t in TASKS if s0["per_task"][t]["valid"]]
    s0["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z"); s0["wall_s"] = time.time() - t0
    out.mkdir(exist_ok=True, parents=True); bm.jdump(s0, out / "stage0.json")
    del enc16; torch.cuda.empty_cache()
    return s0


def populations(valid, seeds):
    V = [t for t in TASKS if t in valid]
    R0 = [(f"{x}@s0", f"{y}@s0") for x, y in itertools.combinations(V, 2)]             # E1b task-list order
    if seeds == [0]:
        return {"R0": R0}
    P = [(f"{x}@s0", f"{y}@s1") for x, y in itertools.combinations(sorted(V), 2)]      # alphabetical: first -> seed 0 (t1)
    S1 = [(f"{x}@s1", f"{y}@s1") for x, y in itertools.combinations(V, 2)]
    S2 = [(f"{t}@s0", f"{t}@s1") for t in V]
    return {"P": P, "R0": R0, "S1": S1, "S2": S2}


def st_stage0(a):
    torch = bm.setup_torch(); dev = torch.device(a.device); t0 = time.time()
    seeds = seeds_used()
    res = {s: stage0_seed(s, dev, HERE / f"s{s}") for s in seeds}
    valid = [t for t in TASKS if all(res[s]["per_task"][t]["valid"] for s in seeds)]
    pops = populations(valid, seeds)
    out = {"seeds": seeds, "primary_population": "P" if seeds == [0, 1] else "R0", "valid_tasks": valid, "K": len(valid),
           "excluded": {t: {f"s{s}": res[s]["per_task"][t]["invalid_reasons"] for s in seeds} for t in TASKS if t not in valid},
           "stop": len(valid) < PA["integrity"]["stop_if_valid_tasks_below"], "lora_layers": res[seeds[0]]["lora_layers"],
           "populations": {k: [pid(x, y) for x, y in v] for k, v in pops.items()}, "n": {k: len(v) for k, v in pops.items()},
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    bm.jdump(out, HERE / "populations_e6.json")
    timing_add("main", "stage0_s", time.time() - t0)
    log(f"e6 stage0: valid {len(valid)} {valid}; excluded {out['excluded']}; n={out['n']}")
    if out["stop"]: log("STOP: fewer than 10 valid tasks"); sys.exit(2)


def singles():
    out = {}
    for s in seeds_used():
        s0 = json.loads((HERE / f"s{s}" / "stage0.json").read_text())
        for t, r in s0["per_task"].items():
            out[f"{t}@s{s}"] = {"eval": r["eval"]["main"], "hold": r["hold"]["main"]}
    return out


# ------------------------------------------------------------------ stage 1 (E1c compute_predictors; O_B dropped: identical to O_A)
def ties_thresholds_exact_lowmem(ads, tasks, names, dev):
    """E6 (memory only): bm.ties_thresholds concatenates all |delta| of an adapter on the GPU (1.3e9 values = 5 GB for Qwen2.5-1.5B, plus the
    abs copy) and takes torch.kthvalue(v.cpu(), k + 1) = the value at sorted position k (0-based). Here each layer's delta is computed on the
    GPU exactly as in bm.delta, moved to CPU, concatenated, abs'ed in place, and the same order statistic (sorted position k) is taken with
    numpy.partition (no index array). abs() and order statistics are exact, so the threshold value is identical."""
    thr = {}
    for t in tasks:
        v = np.concatenate([bm.delta(ads[t], n, dev).flatten().cpu().numpy() for n in names])
        np.abs(v, out=v)
        k = int(round(0.8 * v.size))
        v.partition(k)
        thr[t] = float(v[k])
        del v
    return thr



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
    thr = ties_thresholds_exact_lowmem(ads, keys, names, dev)       # E6: = bm.ties_thresholds (same value), CPU order statistic
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
    if (HERE / "predictors_e6.sha256").exists(): raise SystemExit("predictors already frozen; refusing to recompute")
    if list(HERE.glob("pair_results_*.jsonl")): raise SystemExit("merge results exist before predictors -> protocol violation")
    PO = json.loads((HERE / "populations_e6.json").read_text())
    if PO["stop"]: raise SystemExit("stage0 STOP")
    pairs = []
    for k in ("P", "R0", "S1", "S2"):
        pairs += [tuple(p.split("__")) for p in PO["populations"].get(k, [])]
    t0 = time.time()
    compute_predictors(pairs, HERE, "predictors_e6", dev, PO["lora_layers"])
    with open(HERE / "predictors_e6.sha256", "w") as f:
        for fn in ("predictors_e6.csv", "predictors_layers_e6.csv", "predictors_null_e6.json", "predictors_aux_e6.json"):
            f.write(f"{bm.sha256(HERE / fn)}  {fn}\n")
    timing_add("main", "stage1_s", time.time() - t0)
    log("e6 stage1 done; frozen predictors_e6.sha256:\n" + (HERE / "predictors_e6.sha256").read_text())


def verify_predictors():
    ok = True
    for ln in (HERE / "predictors_e6.sha256").read_text().strip().splitlines():
        h, fn = ln.split()
        if bm.sha256(HERE / fn) != h: ok = False; log(f"PREDICTOR HASH MISMATCH {fn}")
    return ok


# ------------------------------------------------------------------ stage 2 (E1c run_merges; same arithmetic as e1b.stage2)
def run_merges(pairs, jl, pdir, dev, names, single, bs, tname):
    torch = bm.setup_torch(); import pandas as pd
    keys = sorted({k for p in pairs for k in p})
    ads = {k: load_ad(k) for k in keys}
    aux = json.loads((HERE / "predictors_aux_e6.json").read_text())
    for k in keys: assert ads[k]["sha256"] == aux["adapter_sha256"][k], f"{k} adapter changed since predictors"
    thr = aux["ties_top20_abs_threshold"]
    L = pd.read_csv(HERE / "predictors_layers_e6.csv")
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
        if past_deadline():
            log(f"e6 stage2[{tname}] budget deadline or phase stop-file reached before {key}: stopping this process"); return
        tp = time.time(); rec = {"pair": key, "t1": a, "t2": b, "t1_task": task_of(a), "t2_task": task_of(b)}; preds = {}
        # E6 (memory only): d1[n], d2[n] are computed on demand with bm.delta (the same op E4b cached per pair), and the merged weights are
        # written layer by layer (Enc.set_delta_fn) -> identical merged weights, without holding 2-3 all-layer fp32 delta dicts (~5 GB each).
        class _Lazy:
            def __init__(self, k): self.k = k
            def __getitem__(self, n): return bm.delta(ads[self.k], n, dev)
        d1 = _Lazy(a); d2 = _Lazy(b)

        def evaluate(merged_fn, lam, which, tag):
            enc.set_delta_fn(lambda n: lam * merged_fn(n))
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
        # E6 (PREREG_E6 sec. 5b; E5b field names): TA also at the extended lambdas {1.3, 1.5, 2.0}, held-out + eval, eval predictions saved.
        # Secondary only: lam_selected / D above stay on the E4b grid; lam_selected_G7 is recorded for the U1/G7 secondary analysis.
        for lam in LAMS_EXT:
            h = evaluate(ta, lam, "hold", "TA"); e = evaluate(ta, lam, "eval", "TA")
            sel[lam] = norm_mean(h, "hold"); rec[f"TA_hold_norm_lam{lam}"] = sel[lam]
            for i, k in ((1, a), (2, b)): rec[f"TA_hold_t{i}_lam{lam}"] = h[k]; rec[f"TA_eval_t{i}_lam{lam}"] = e[k]
            rec[f"TA_eval_D_lam{lam}"] = 1 - norm_mean(e, "eval")
        G7 = list(LAMS) + list(LAMS_EXT)
        rec["lam_selected_G7"] = G7[int(np.argmax([sel[l] for l in G7]))]
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
            Sds = {}                                                   # E6: keep only the (small) projection bases; d2g[n] computed on demand
            for n in fail_layers:
                U1 = bm.orth(ads[a]["layers"][n][1].to(dev).double()); U2 = bm.orth(ads[b]["layers"][n][1].to(dev).double())
                Pm, S, _ = torch.linalg.svd(U1.T @ U2)
                Sds[n] = (U1 @ Pm[:, S > cos_star]).float()
            def d2g_of(n):
                if n not in Sds: return d2[n]
                x = d2[n]; Sd = Sds[n]
                return x - Sd @ (Sd.T @ x)
            gate = lambda n: d1[n] + d2g_of(n)
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
        log(f"e6 stage2[{tname}] {key}: lam*={lam_ta} D={rec['D']:.4f} TIES(atTA/own)={rec['TIES_atTA_score']:.4f}/{rec['TIES_own_score']:.4f} "
            f"GATE({len(fail_layers)} layers)={rec['GATE_atTA_score']:.4f} [{rec['secs']:.0f}s]")
        del d1, d2; torch.cuda.empty_cache()
        timing_add(tname, "stage2_gpu_wall_s", time.time() - tp)


def st_stage2(a):
    torch = bm.setup_torch(); dev = torch.device(a.device)
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    PO = json.loads((HERE / "populations_e6.json").read_text())
    if a.pop not in PO["populations"]: log(f"e6 stage2: population {a.pop} not in this design (seeds {PO['seeds']}) -> skip"); return
    pairs = [tuple(p.split("__")) for p in PO["populations"][a.pop]]
    if a.limit: pairs = pairs[:a.limit]
    if a.reverse: pairs = pairs[::-1]
    run_merges(pairs, HERE / f"pair_results_{a.pop}.jsonl", HERE / "preds", dev, PO["lora_layers"], singles(), a.bs, a.pop)
    log(f"e6 stage2 {a.pop} done")


def st_e3same(a):
    raise SystemExit("E6: the E3 subset is not part of E6 (PREREG_E6.md sec. 5: unmodified e3.run() holds both all-layer fp32 deltas on the GPU, ~10.5 GB for Qwen2.5-1.5B)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--pop", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=BS); ap.add_argument("--mem-frac", type=float, default=0.45)
    a = ap.parse_args()
    log(f"=== e6.py {sys.argv}")
    if a.stage == "pilot": st_pilot(a)
    elif a.stage == "stage0": st_stage0(a)
    elif a.stage == "stage1": st_stage1(a)
    elif a.stage == "stage2": st_stage2(a)
    elif a.stage == "e3same": st_e3same(a)
    elif a.stage == "stage3":
        import e6_analysis; e6_analysis.main()
        import e6_extra; e6_extra.main()
    else: raise SystemExit("unknown stage")


if __name__ == "__main__":
    main()
