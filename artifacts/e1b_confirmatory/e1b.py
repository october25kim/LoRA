#!/usr/bin/env python3
"""E1b confirmatory pipeline (STEP 2-5). Self-contained; reads prereg.json, deviations_d1.json, tasks.py.
  stage0  integrity (a/b/c) on all 16 adapters -> stage0.json (valid task set; STOP if < 12)
  stage1  weights-only predictors over valid tasks -> predictors.csv (+ layers, null, ties thresholds), frozen in predictors.sha256
  stage2  merges: TA (lambda selected on held-out), TIES-lite, theta*=30deg hard gate -> pair_results.jsonl/.csv (resumable)
  stage3  pre-registered analysis -> analysis.json, VERDICT.md, fig_f1_scatter.png, fig_f2_heatmap.png, method_comparison.md
Usage (from artifacts/e1b_confirmatory):  python e1b.py --stage stage0|stage1|stage2|stage3 [--out DIR] [--adapters DIR]
"""
import argparse, hashlib, itertools, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tasks import ALL_SPECS, load_raw, columns, split_indices, tokenize, TASK_TYPE

P = json.loads((HERE / "prereg.json").read_text())
D1 = json.loads((HERE / "deviations_d1.json").read_text()) if (HERE / "deviations_d1.json").exists() else {"budgets": {}}
R = P["recipe"]
BASE, BREV = R["base_model"], R["base_revision"]
LAMS = P["merge"]["lambda_grid"]
REF = P["integrity"]["b_glue_reference"]["reference"]
THETA_STAR = 30.0
OUT = ADIR = LOGF = None
TASKS_ALL = list(P["tasks"])


def log(msg):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(msg)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jdump(o, p):
    Path(p).write_text(json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def timing_add(k, s):
    p = OUT / "timing.json"; d = json.loads(p.read_text()) if p.exists() else {}
    d[k] = d.get(k, 0.0) + s; jdump(d, p)


def metric_name(t): return ALL_SPECS[t][8]
def nlabels(t): return ALL_SPECS[t][7]


def pairs_of(tasks):
    order = {t: i for i, t in enumerate(TASKS_ALL)}
    ts = sorted(tasks, key=lambda t: order.get(t, 99))
    return list(itertools.combinations(ts, 2))


# ------------------------------------------------------------------ adapters
def load_adapter(task):
    from safetensors.torch import load_file
    d = ADIR / task
    cfg = json.loads((d / "adapter_config.json").read_text())
    sd = load_file(str(d / "adapter_model.safetensors"))
    layers, head, other = {}, {}, []
    for k, v in sd.items():
        if ".lora_A." in k:
            name = k.split("base_model.model.bert.")[-1].split(".lora_A.")[0]
            layers[name] = (v.float(), sd[k.replace(".lora_A.", ".lora_B.")].float())
        elif ".lora_B." in k:
            pass
        elif "classifier" in k:
            head[k.split(".")[-1]] = v.float()
        else:
            other.append(k)
    meta = json.loads((d / "train_meta.json").read_text())
    return dict(layers=layers, head=head, other=other, cfg=cfg, scaling=cfg["lora_alpha"] / cfg["r"], r=cfg["r"],
                sha256=sha256(d / "adapter_model.safetensors"), meta=meta, dir=str(d))


def delta(ad, n, dev):
    A, B = ad["layers"][n]
    return ad["scaling"] * (B.to(dev) @ A.to(dev))


# ------------------------------------------------------------------ data
def train_indices(task, n_train, n_eval):
    hold, tcap, ev = split_indices(n_train, n_eval)
    if task in D1["budgets"]:
        perm = np.random.default_rng(0).permutation(n_train)
        return hold, np.sort(perm[1000:]), ev
    return hold, tcap, ev


def get_data(task, tok):
    """returns {'eval':{ids,tt,labels,idx}, 'hold':{...}, 'train_labels': np.array} (cached)."""
    import torch
    cdir = OUT / "cache"; cdir.mkdir(exist_ok=True)
    cp = cdir / f"{task}.pt"
    if cp.exists():
        return torch.load(cp, weights_only=False)
    spec = ALL_SPECS[task]
    tr = load_raw(task, spec[2]); ev = load_raw(task, spec[3])
    hold, tidx, eidx = train_indices(task, len(tr), len(ev))
    out = {}
    for name, ds, idx in (("eval", ev, eidx), ("hold", tr, hold)):
        a, b, y = columns(task, ds.select(idx.tolist()))
        ids, tt = tokenize(tok, a, b)
        out[name] = {"ids": ids, "tt": tt, "labels": y, "idx": idx}
    kl = spec[6]
    if nlabels(task) > 1:
        if kl == "SCITAIL_LABEL":
            _, _, yt = columns(task, tr.select(tidx.tolist()))
        else:
            yt = np.array(tr.select(tidx.tolist())[kl], dtype=np.int64)
        out["train_labels"] = yt
    out["train_idx_sha256"] = hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest()
    out["hold_idx_sha256"] = hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()
    torch.save(out, cp)
    return out


# ------------------------------------------------------------------ model
class Enc:
    def __init__(self, dev, layer_names):
        from transformers import BertModel
        self.dev = dev
        self.m = BertModel.from_pretrained(BASE, revision=BREV).to(dev).eval()
        mods = dict(self.m.named_modules())
        miss = [n for n in layer_names if n not in mods]
        assert not miss, miss[:5]
        self.lin = {n: mods[n] for n in layer_names}
        self.W0 = {n: l.weight.detach().clone() for n, l in self.lin.items()}

    def set_delta(self, d):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                l.weight.copy_(self.W0[n] if d is None else self.W0[n] + d[n])

    def batches(self, data, bs):
        """length-sorted, padded GPU batches, cached per data dict (same padding for every call)."""
        import torch
        key = (id(data), bs)
        if not hasattr(self, "_bc"): self._bc = {}
        if key not in self._bc:
            ids, tt = data["ids"], data["tt"]
            order = np.argsort([len(x) for x in ids], kind="stable"); out = []
            for i in range(0, len(ids), bs):
                idx = order[i:i + bs]; L = max(len(ids[j]) for j in idx)
                ii = np.zeros((len(idx), L), dtype=np.int64); ti = np.zeros_like(ii); am = np.zeros_like(ii)
                for r, j in enumerate(idx):
                    l = len(ids[j]); ii[r, :l] = ids[j]; ti[r, :l] = tt[j]; am[r, :l] = 1
                out.append((idx, torch.from_numpy(ii).to(self.dev), torch.from_numpy(ti).to(self.dev), torch.from_numpy(am).to(self.dev)))
            self._bc[key] = (data, out)          # keep a reference so id() stays unique
        return self._bc[key][1]

    def predict(self, data, head, bs=256):
        import torch
        n = len(data["ids"])
        W = head["weight"].to(self.dev); b = head["bias"].to(self.dev)
        out = np.empty((n, W.shape[0]), dtype=np.float32)
        with torch.inference_mode():
            for idx, ii, ti, am in self.batches(data, bs):
                pooled = self.m(input_ids=ii, token_type_ids=ti, attention_mask=am).pooler_output
                out[idx] = (pooled @ W.T + b).float().cpu().numpy()
        return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def score(task, logits, labels):
    """main metric + extras. Classification: argmax accuracy (+ mcc/f1 for binary). STS-B: Spearman (+ Pearson)."""
    if metric_name(task) == "spearman":
        p = logits[:, 0]
        return {"main": spearman(p, labels), "spearman": spearman(p, labels),
                "pearson": float(np.corrcoef(p, labels)[0, 1]), "n": int(len(labels))}
    p = logits.argmax(-1)
    r = {"main": float((p == labels).mean()), "accuracy": float((p == labels).mean()), "n": int(len(labels))}
    if logits.shape[1] == 2:
        tp = int(((p == 1) & (labels == 1)).sum()); fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum()); tn = int(((p == 0) & (labels == 0)).sum())
        r["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        r["mcc"] = (tp * tn - fp * fn) / math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return r


def compact(task, logits):
    return logits[:, 0].astype(np.float32) if metric_name(task) == "spearman" else logits.argmax(-1).astype(np.int8)


def setup_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


# ------------------------------------------------------------------ stage 0
def stage0(args):
    torch = setup_torch()
    from transformers import AutoTokenizer, BertForSequenceClassification
    from peft import PeftModel
    import transformers, peft, datasets
    t0 = time.time(); dev = torch.device(args.device)
    tasks = args.tasks or TASKS_ALL
    s0 = {"stage": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "gpu": torch.cuda.get_device_name(0),
          "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                       "datasets": datasets.__version__}, "precision": "fp32, TF32 disabled", "tasks": tasks,
          "deviation_D1_tasks": sorted(D1["budgets"]), "per_task": {}, "fail_reasons": []}
    tok = AutoTokenizer.from_pretrained(BASE, revision=BREV)
    ads = {t: load_adapter(t) for t in tasks}
    names = sorted(ads[tasks[0]]["layers"])
    s0["lora_layers"] = names; s0["n_lora_layers"] = len(names)
    enc = Enc(dev, names)
    pdir = OUT / "preds"; pdir.mkdir(exist_ok=True)
    for t in tasks:
        ad = ads[t]; d = get_data(t, tok); rec = {"adapter_dir": ad["dir"], "adapter_sha256": ad["sha256"],
                                                  "trained_under": ad["meta"].get("deviation", "prereg"),
                                                  "train_steps": ad["meta"]["global_step"], "train_wall_s": ad["meta"]["train_wall_s"],
                                                  "train_loss": ad["meta"]["train_loss"], "segment_ids": ad["meta"]["segment_ids"]}
        structural = []
        if sorted(ad["layers"]) != names: structural.append("LoRA layer set differs")
        if len(names) != R["lora_layers_expected"]: structural.append(f"{len(names)} LoRA layers != 73")
        if ad["other"]: structural.append(f"unexpected keys {ad['other'][:3]}")
        if "weight" not in ad["head"] or tuple(ad["head"]["weight"].shape) != (nlabels(t), 768): structural.append("classifier head missing/wrong shape")
        if ad["r"] != R["r"] or ad["cfg"]["lora_alpha"] != R["lora_alpha"]: structural.append("r/alpha mismatch")
        if d["train_idx_sha256"] != ad["meta"]["train_idx_sha256"]: structural.append("train index sha mismatch vs train_meta")
        if d["hold_idx_sha256"] != ad["meta"]["hold_idx_sha256"]: structural.append("held-out index sha mismatch vs train_meta")
        rec["structural_problems"] = structural
        # (c) PEFT equivalence on first 64 eval examples
        sub = {"ids": d["eval"]["ids"][:64], "tt": d["eval"]["tt"][:64]}
        enc.set_delta({n: delta(ad, n, dev) for n in names})
        mine = enc.predict(sub, ad["head"])
        base = BertForSequenceClassification.from_pretrained(BASE, revision=BREV, num_labels=nlabels(t))
        pm = PeftModel.from_pretrained(base, ad["dir"]).to(dev).eval()
        ref = []
        with torch.inference_mode():
            for i in range(64):
                ref.append(pm(input_ids=torch.tensor([sub["ids"][i]], device=dev),
                              token_type_ids=torch.tensor([sub["tt"][i]], device=dev)).logits.float().cpu().numpy()[0])
        ref = np.stack(ref); diff = float(np.abs(ref - mine).max())
        del pm, base; torch.cuda.empty_cache()
        rec["c_max_abs_logit_diff"] = diff; rec["c_ok"] = diff <= 1e-4
        # single scores (eval + held-out)
        le = enc.predict(d["eval"], ad["head"]); lh = enc.predict(d["hold"], ad["head"])
        se = score(t, le, d["eval"]["labels"]); sh = score(t, lh, d["hold"]["labels"])
        np.savez_compressed(pdir / f"single_{t}.npz", eval=compact(t, le), hold=compact(t, lh),
                            eval_labels=d["eval"]["labels"], hold_labels=d["hold"]["labels"])
        rec["eval"] = se; rec["hold"] = sh; rec["n_eval"] = se["n"]
        # (a)
        if metric_name(t) == "spearman":
            rec["a_threshold"] = 0.70; rec["a_ok"] = se["main"] >= 0.70; rec["majority_baseline"] = None
        else:
            yt = d["train_labels"]; cnt = np.bincount(yt, minlength=nlabels(t)); tied = np.flatnonzero(cnt == cnt.max())
            accs = {int(c): float((d["eval"]["labels"] == c).mean()) for c in tied}
            maj = max(accs, key=accs.get)
            rec["majority_label"] = maj; rec["majority_baseline"] = accs[maj]
            rec["a_threshold"] = accs[maj] + 0.10; rec["a_ok"] = se["main"] >= rec["a_threshold"]
        # (b)
        if t in REF:
            m = REF[t]["metric"]; ours = se["main"] if m in ("accuracy", "spearman") else se[m]
            rec["b_metric"] = m; rec["b_reference"] = REF[t]["value"]; rec["b_threshold"] = REF[t]["value"] - 0.05
            rec["b_ours"] = ours; rec["b_ok"] = ours >= rec["b_threshold"]
        else:
            rec["b_ok"] = None
        rec["valid"] = bool(rec["a_ok"] and rec["c_ok"] and rec["b_ok"] is not False and not structural)
        reasons = [f"(a) {se['main']:.4f} < {rec['a_threshold']:.4f}"] if not rec["a_ok"] else []
        if rec["b_ok"] is False: reasons.append(f"(b) {rec['b_metric']} {rec['b_ours']:.4f} < {rec['b_threshold']:.4f}")
        if not rec["c_ok"]: reasons.append(f"(c) logit diff {diff:.2e} > 1e-4")
        reasons += structural
        rec["invalid_reasons"] = reasons
        s0["per_task"][t] = rec
        log(f"stage0 {t}: eval={se['main']:.4f} hold={sh['main']:.4f} a={rec['a_ok']} b={rec['b_ok']} c={rec['c_ok']} (diff {diff:.1e}) valid={rec['valid']} {reasons}")
    valid = [t for t in tasks if s0["per_task"][t]["valid"]]
    if args.force_valid:          # pipeline smoke tests with throwaway 20-step adapters ONLY
        valid = list(tasks); s0["FORCE_VALID_SMOKE_TEST"] = True
    s0["valid_tasks"] = valid; s0["excluded_tasks"] = {t: s0["per_task"][t]["invalid_reasons"] for t in tasks if t not in valid}
    s0["n_valid"] = len(valid); s0["n_pairs"] = len(pairs_of(valid))
    s0["stage0_pass"] = len(valid) >= args.min_valid
    if not s0["stage0_pass"]:
        s0["fail_reasons"].append(f"only {len(valid)} valid tasks (< {args.min_valid}) -> STOP")
    s0["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z"); s0["wall_s"] = time.time() - t0
    jdump(s0, OUT / "stage0.json"); timing_add("stage0_gpu_wall_s", time.time() - t0)
    log(f"stage0 done: {len(valid)} valid {valid}; excluded {s0['excluded_tasks']}; pass={s0['stage0_pass']}")
    return s0


# ------------------------------------------------------------------ stage 1
def orth(M):
    import torch
    return torch.linalg.qr(M)[0]


def ties_thresholds(ads, tasks, names, dev):
    import torch
    thr = {}
    for t in tasks:
        v = torch.cat([delta(ads[t], n, dev).flatten() for n in names]).abs()
        k = int(round(0.8 * v.numel()))
        thr[t] = float(torch.kthvalue(v.cpu(), k + 1).values)   # value at sorted position k (0-based), like sort(a)[k]
        del v
    return thr


def stage1(args):
    torch = setup_torch(); import pandas as pd
    t0 = time.time()
    if (OUT / "predictors.sha256").exists():
        raise SystemExit("predictors already frozen; refusing to recompute")
    if (OUT / "pair_results.jsonl").exists():
        raise SystemExit("merge results exist before predictors -> protocol violation")
    s0 = json.loads((OUT / "stage0.json").read_text())
    if not s0["stage0_pass"]: raise SystemExit("stage0 failed -> stop")
    tasks = s0["valid_tasks"]; dev = torch.device(args.device)
    ads = {t: load_adapter(t) for t in tasks}; names = s0["lora_layers"]; r = R["r"]
    QA, QB = {}, {}
    for t in tasks:
        QA[t], QB[t] = {}, {}
        for n in names:
            A, B = ads[t]["layers"][n]; A = A.to(dev).double(); B = B.to(dev).double()
            QA[t][n] = orth(B)
            Qb, Rb = torch.linalg.qr(B); U, S, _ = torch.linalg.svd(Rb @ A, full_matrices=False); QB[t][n] = Qb @ U[:, :r]
    # norms and dot products (streamed per layer to save memory)
    nrm = {t: 0.0 for t in tasks}; dots = {}
    thr = ties_thresholds(ads, tasks, names, dev)
    conf = {}
    for n in names:
        dd = {t: delta(ads[t], n, dev).flatten() for t in tasks}
        for t in tasks: nrm[t] += float((dd[t].double() ** 2).sum())
        for (a, b) in pairs_of(tasks):
            dots[(a, b)] = dots.get((a, b), 0.0) + float((dd[a].double() * dd[b].double()).sum())
            ka = dd[a].abs() >= thr[a]; kb = dd[b].abs() >= thr[b]; both = ka & kb
            opp = torch.sign(dd[a]) != torch.sign(dd[b]); nz = (dd[a] != 0) & (dd[b] != 0)
            c = conf.setdefault((a, b), [0, 0, 0, 0])
            c[0] += int((opp & both).sum()); c[1] += int(both.sum()); c[2] += int((opp & nz).sum()); c[3] += int(nz.sum())
        del dd
    g = torch.Generator(device=dev).manual_seed(0); NREP = 1000
    null_cos2 = np.zeros((NREP, len(names))); null_thmin = np.zeros((NREP, len(names))); dims = {}
    for li, n in enumerate(names):
        d = ads[tasks[0]]["layers"][n][1].shape[0]; dims[n] = d
        G1 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        G2 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        s = torch.linalg.svdvals(torch.linalg.qr(G1)[0].transpose(1, 2) @ torch.linalg.qr(G2)[0]).clamp(max=1.0)
        null_cos2[:, li] = (s ** 2).mean(1).cpu().numpy(); null_thmin[:, li] = np.degrees(np.arccos(s.max(1).values.cpu().numpy()))
    null_OA = null_cos2.mean(1); mu, sd = float(null_OA.mean()), float(null_OA.std(ddof=1))
    band = {n: {"d": dims[n], "theta_min_p5": float(np.percentile(null_thmin[:, i], 5)), "theta_min_p50": float(np.percentile(null_thmin[:, i], 50)),
                "theta_min_p95": float(np.percentile(null_thmin[:, i], 95))} for i, n in enumerate(names)}
    rows, lrows = [], []
    for (a, b) in pairs_of(tasks):
        cA, cB, thA, thB = [], [], [], []
        for n in names:
            sa = torch.linalg.svdvals(QA[a][n].T @ QA[b][n]).clamp(max=1.0); sb = torch.linalg.svdvals(QB[a][n].T @ QB[b][n]).clamp(max=1.0)
            c2a = float((sa ** 2).mean()); c2b = float((sb ** 2).mean())
            tha = float(np.degrees(np.arccos(float(sa.max())))); thb = float(np.degrees(np.arccos(float(sb.max()))))
            n30 = int((sa > math.cos(math.radians(THETA_STAR))).sum())
            cA.append(c2a); cB.append(c2b); thA.append(tha); thB.append(thb)
            lrows.append({"pair": f"{a}-{b}", "t1": a, "t2": b, "layer": n, "d_out": dims[n], "cos2_mean_A": c2a, "theta_min_A_deg": tha,
                          "n_angles_lt30_A": n30, "cos2_mean_B": c2b, "theta_min_B_deg": thb})
        n1, n2 = math.sqrt(nrm[a]), math.sqrt(nrm[b]); OA = float(np.mean(cA)); c = conf[(a, b)]
        rows.append({"pair": f"{a}-{b}", "t1": a, "t2": b, "O_A": OA, "O_B": float(np.mean(cB)),
                     "tv_cosine": dots[(a, b)] / (n1 * n2),
                     "mean_theta_min_A_deg": float(np.mean(thA)), "min_theta_min_A_deg": float(np.min(thA)),
                     "n_layers_theta_min_lt30_A": int(np.sum(np.array(thA) < THETA_STAR)),
                     "mean_theta_min_B_deg": float(np.mean(thB)), "null_z_O_A": (OA - mu) / sd,
                     "sign_conflict_top20": c[0] / max(c[1], 1), "sign_conflict_all": c[2] / max(c[3], 1),
                     "norm_ratio": max(n1, n2) / min(n1, n2), "tvnorm_t1": n1, "tvnorm_t2": n2,
                     "same_type": TASK_TYPE[a] == TASK_TYPE[b], "n_layers": len(names)})
    pd.DataFrame(rows).to_csv(OUT / "predictors.csv", index=False, float_format="%.10g")
    pd.DataFrame(lrows).to_csv(OUT / "predictors_layers.csv", index=False, float_format="%.10g")
    jdump({"n_rep": NREP, "seed": 0, "r": r, "null_O_A_mean": mu, "null_O_A_sd": sd, "null_O_A_p5": float(np.percentile(null_OA, 5)),
           "null_O_A_p95": float(np.percentile(null_OA, 95)), "per_layer": band}, OUT / "predictors_null.json")
    jdump({"ties_top20_abs_threshold": thr, "tvnorm": {t: math.sqrt(nrm[t]) for t in tasks}}, OUT / "predictors_aux.json")
    with open(OUT / "predictors.sha256", "w") as f:
        for fn in ("predictors.csv", "predictors_layers.csv", "predictors_null.json", "predictors_aux.json"):
            f.write(f"{sha256(OUT / fn)}  {fn}\n")
    timing_add("stage1_gpu_wall_s", time.time() - t0)
    log("stage1 done; frozen predictors.sha256:\n" + (OUT / "predictors.sha256").read_text())


def verify_predictors():
    ok = True
    for ln in (OUT / "predictors.sha256").read_text().strip().splitlines():
        h, fn = ln.split()
        if sha256(OUT / fn) != h: ok = False; log(f"PREDICTOR HASH MISMATCH {fn}")
    return ok


# ------------------------------------------------------------------ stage 2
def stage2(args):
    torch = setup_torch(); import pandas as pd
    from transformers import AutoTokenizer
    t0 = time.time()
    s0 = json.loads((OUT / "stage0.json").read_text())
    if not s0["stage0_pass"]: raise SystemExit("stage0 failed")
    if not verify_predictors(): raise SystemExit("predictors changed -> INVALID")
    dev = torch.device(args.device); tasks = s0["valid_tasks"]; names = s0["lora_layers"]
    tok = AutoTokenizer.from_pretrained(BASE, revision=BREV)
    ads = {t: load_adapter(t) for t in tasks}
    for t in tasks: assert ads[t]["sha256"] == s0["per_task"][t]["adapter_sha256"], f"{t} adapter changed since stage0"
    aux = json.loads((OUT / "predictors_aux.json").read_text()); thr = aux["ties_top20_abs_threshold"]
    L = pd.read_csv(OUT / "predictors_layers.csv")
    data = {t: get_data(t, tok) for t in tasks}
    single = {t: {"eval": s0["per_task"][t]["eval"]["main"], "hold": s0["per_task"][t]["hold"]["main"]} for t in tasks}
    enc = Enc(dev, names)
    jl = OUT / "pair_results.jsonl"
    done = {json.loads(l)["pair"] for l in jl.read_text().splitlines() if l.strip()} if jl.exists() else set()
    pdir = OUT / "preds"; pdir.mkdir(exist_ok=True)
    cos_star = math.cos(math.radians(THETA_STAR))

    for (a, b) in pairs_of(tasks):
        key = f"{a}-{b}"
        if key in done: continue
        tp = time.time(); rec = {"pair": key, "t1": a, "t2": b}; preds = {}
        d1 = {n: delta(ads[a], n, dev) for n in names}; d2 = {n: delta(ads[b], n, dev) for n in names}

        def evaluate(merged_fn, lam, which, tag):
            enc.set_delta({n: lam * merged_fn(n) for n in names})
            out = {}
            for t in (a, b):
                lg = enc.predict(data[t][which], ads[t]["head"], args.bs)
                out[t] = score(t, lg, data[t][which]["labels"])["main"]
                if which == "eval": preds[f"{tag}_{t}_lam{lam}"] = compact(t, lg)
            return out

        def norm_mean(sc, which):
            return 0.5 * (sc[a] / single[a][which] + sc[b] / single[b][which])

        # ---- task arithmetic
        ta = lambda n: d1[n] + d2[n]
        sel = {}
        for lam in LAMS:
            h = evaluate(ta, lam, "hold", "TA"); e = evaluate(ta, lam, "eval", "TA")
            sel[lam] = norm_mean(h, "hold")
            rec[f"TA_hold_norm_lam{lam}"] = sel[lam]
            for t in (a, b): rec[f"TA_hold_{t}_lam{lam}"] = h[t]; rec[f"TA_eval_{t}_lam{lam}"] = e[t]
            rec[f"TA_eval_D_lam{lam}"] = 1 - norm_mean(e, "eval")
        lam_ta = LAMS[int(np.argmax([sel[l] for l in LAMS]))]       # argmax -> first max -> smaller lam on ties
        rec["lam_selected"] = lam_ta
        rec["merged_eval_t1"] = rec[f"TA_eval_{a}_lam{lam_ta}"]; rec["merged_eval_t2"] = rec[f"TA_eval_{b}_lam{lam_ta}"]
        rec["single_eval_t1"] = single[a]["eval"]; rec["single_eval_t2"] = single[b]["eval"]
        rec["norm_t1"] = rec["merged_eval_t1"] / single[a]["eval"]; rec["norm_t2"] = rec["merged_eval_t2"] / single[b]["eval"]
        rec["D"] = 1 - 0.5 * (rec["norm_t1"] + rec["norm_t2"])
        rec["TA_score_sel"] = 1 - rec["D"]
        # ---- TIES-lite
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
            for t in (a, b): rec[f"TIES_{tag}_eval_{t}"] = e[t]
        # ---- theta*=30deg hard gate (project dW2 off shared directions on FAIL layers)
        lp = L[L.pair == key].set_index("layer")
        fail_layers = [n for n in names if lp.loc[n, "theta_min_A_deg"] < THETA_STAR]
        rec["gate_n_fail_layers"] = len(fail_layers); rec["gate_active"] = len(fail_layers) > 0
        if fail_layers:
            d2g = dict(d2)
            for n in fail_layers:
                U1 = orth(ads[a]["layers"][n][1].to(dev).double()); U2 = orth(ads[b]["layers"][n][1].to(dev).double())
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
        log(f"stage2 {key}: lam*={lam_ta} D={rec['D']:.4f} TIES(atTA/own)={rec['TIES_atTA_score']:.4f}/{rec['TIES_own_score']:.4f} "
            f"GATE({len(fail_layers)} layers)={rec['GATE_atTA_score']:.4f} [{rec['secs']:.0f}s]")
        del d1, d2; torch.cuda.empty_cache()
        timing_add("stage2_gpu_wall_s", time.time() - tp)
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    order = {f"{x}-{y}": i for i, (x, y) in enumerate(pairs_of(tasks))}
    recs.sort(key=lambda r: order[r["pair"]])
    pd.DataFrame(recs).to_csv(OUT / "pair_results.csv", index=False, float_format="%.10g")
    log(f"stage2 done: {len(recs)} pairs (expected {len(order)})")


# ------------------------------------------------------------------ stage 3
def holm(ps):
    keys = list(ps); order = sorted(keys, key=lambda k: ps[k]); m = len(keys); run = 0.0; adj = {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def stage3(args):
    import pandas as pd
    from scipy.stats import wilcoxon
    t0 = time.time()
    s0 = json.loads((OUT / "stage0.json").read_text())
    hash_ok = verify_predictors()
    Pd = pd.read_csv(OUT / "predictors.csv"); Rr = pd.read_csv(OUT / "pair_results.csv")
    df = Pd.merge(Rr.drop(columns=["t1", "t2"]), on="pair")
    tasks = s0["valid_tasks"]; K = len(tasks); tix = {t: i for i, t in enumerate(tasks)}
    I = np.array([tix[x] for x in df.t1]); J = np.array([tix[x] for x in df.t2])
    PI = np.full((K, K), -1); PI[I, J] = np.arange(len(df)); PI[J, I] = np.arange(len(df))
    D = df.D.values
    prim = {"H1": "O_A", "H2": "tv_cosine"}
    secondary = ["O_B", "mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A", "sign_conflict_top20",
                 "norm_ratio", "null_z_O_A", "sign_conflict_all"]
    cols = list(prim.values()) + secondary
    perms = np.argsort(np.random.default_rng(0).random((10000, K)), axis=1)    # 10000 task relabelings (seed 0)
    maps = PI[perms[:, I], perms[:, J]]
    brng = np.random.default_rng(0); boots = []; boot_skipped = 0
    for _ in range(2000):
        s = brng.integers(0, K, K); ii = [(a_, b_) for a_ in range(K) for b_ in range(a_ + 1, K) if s[a_] != s[b_]]
        bi = np.array([PI[s[a_], s[b_]] for a_, b_ in ii], dtype=np.int64)
        if len(np.unique(bi)) < 4: boot_skipped += 1; continue      # degenerate resample (same rule as pilot)
        boots.append(bi)

    def stats_for(x, y):
        obs = spearman(x, y)
        if not np.isfinite(obs):   # constant predictor (e.g. no layer below 30 deg anywhere): rho and p undefined
            return {"rho": float("nan"), "perm_p_one_sided": float("nan"), "perm_p_two_sided": float("nan"),
                    "boot_ci95": [float("nan"), float("nan")], "loto": {t: float("nan") for t in tasks}, "loto_frac_gt_0.2": float("nan"),
                    "note": f"predictor constant across pairs (value {x[0]}); rho undefined"}
        rp = np.array([spearman(x, y[m]) for m in maps])
        p1 = (1 + np.sum(rp >= obs - 1e-12)) / (1 + len(maps)); p2 = (1 + np.sum(np.abs(rp) >= abs(obs) - 1e-12)) / (1 + len(maps))
        bs = np.array([spearman(x[bi], y[bi]) for bi in boots])
        loto = {t: spearman(x[(I != tix[t]) & (J != tix[t])], y[(I != tix[t]) & (J != tix[t])]) for t in tasks}
        return {"rho": obs, "perm_p_one_sided": float(p1), "perm_p_two_sided": float(p2),
                "boot_ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "loto": loto,
                "loto_frac_gt_0.2": float(np.mean([v > 0.2 for v in loto.values()]))}
    ST = {c: stats_for(df[c].values.astype(float), D) for c in cols}
    for c in cols: log(f"stage3 {c}: rho={ST[c]['rho']:.4f} p1={ST[c]['perm_p_one_sided']:.4f} ci={ST[c]['boot_ci95']}")
    adj = holm({h: ST[c]["perm_p_one_sided"] for h, c in prim.items()})
    verdicts = {}
    for h, c in prim.items():
        s = ST[c]; crit = {"rho_ge_0.4": s["rho"] >= 0.4, "holm_p_lt_0.05": adj[h] < 0.05, "loto_ge_75pct_gt_0.2": s["loto_frac_gt_0.2"] >= 0.75}
        v = "PASS" if all(crit.values()) else ("FAIL" if s["boot_ci95"][1] < 0.3 else "INCONCLUSIVE")
        verdicts[h] = {"predictor": c, "verdict": v, "rho": s["rho"], "boot_ci95": s["boot_ci95"], "perm_p_one_sided": s["perm_p_one_sided"],
                       "holm_p": adj[h], "loto_frac_gt_0.2": s["loto_frac_gt_0.2"], "loto": s["loto"], "criteria": crit}
        log(f"stage3 {h} ({c}): {v} rho={s['rho']:.4f} holm_p={adj[h]:.4f} ci={s['boot_ci95']} loto={s['loto_frac_gt_0.2']:.2f}")
    invalid = []
    if not hash_ok: invalid.append("predictors changed after freeze")
    if not s0["stage0_pass"]: invalid.append("stage0 failed")
    fixed = {c: {f"lam{l}": spearman(df[c], df[f"TA_eval_D_lam{l}"]) for l in LAMS} for c in prim.values()}
    # method comparison (exploratory)
    meth = {"TA@lam*": df.TA_score_sel.values, "TIES-lite@lam*TA": df.TIES_atTA_score.values, "TIES-lite@own-lam": df.TIES_own_score.values,
            "gate30@lam*TA": df.GATE_atTA_score.values, "gate30@own-lam": df.GATE_own_score.values}
    prng = np.random.default_rng(0)
    MC = {}
    for k, v in meth.items():
        e = {"mean_norm_score": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v))}
        if k != "TA@lam*":
            dlt = v - meth["TA@lam*"]
            tb = [float(np.mean(dlt[bi])) for bi in boots]
            pb = [float(np.mean(dlt[prng.integers(0, len(dlt), len(dlt))])) for _ in range(2000)]
            nz = np.abs(dlt) > 1e-12
            try: wp = float(wilcoxon(dlt[nz]).pvalue) if nz.sum() >= 5 else float("nan")
            except Exception: wp = float("nan")
            e.update({"mean_diff_vs_TA": float(np.mean(dlt)), "task_block_ci95": [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
                      "pair_boot_ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))],
                      "win": int(np.sum(dlt > 1e-12)), "tie": int(np.sum(~nz)), "loss": int(np.sum(dlt < -1e-12)), "wilcoxon_p": wp})
        MC[k] = e
    gate_active = int(df.gate_active.sum())
    A = {"verdicts": verdicts, "invalid_reasons": invalid, "n_valid_tasks": K, "n_pairs": int(len(df)), "tasks": tasks,
         "excluded_tasks": s0["excluded_tasks"], "all_predictor_stats": ST, "fixed_lambda_rho": fixed, "method_comparison": MC,
         "gate_active_pairs": gate_active, "predictors_hash_ok": hash_ok, "bootstrap_reps_used": len(boots), "bootstrap_skipped_degenerate": boot_skipped,
         "D_summary": {"mean": float(D.mean()), "median": float(np.median(D)), "min": float(D.min()), "max": float(D.max()), "sd": float(D.std(ddof=1))},
         "lam_selected_counts": {str(k): int(v) for k, v in df.lam_selected.value_counts().sort_index().items()},
         "rho_OA_tvcos": spearman(df.O_A, df.tv_cosine), "min_theta_min_overall_deg": float(df.min_theta_min_A_deg.min())}
    jdump(A, OUT / "analysis.json")
    figures(df, ST, A, s0)
    method_table(A)
    timing_add("stage3_wall_s", time.time() - t0)
    write_verdict(A, df, s0)
    log(f"stage3 done: {[(h, v['verdict']) for h, v in verdicts.items()]}")


def method_table(A):
    L = ["| method | mean norm. score | median | min | mean diff vs TA | task-block 95% CI | pair-bootstrap 95% CI | win/tie/loss | Wilcoxon p |",
         "|---|---:|---:|---:|---:|---|---|---|---:|"]
    for k, e in A["method_comparison"].items():
        if "mean_diff_vs_TA" in e:
            L.append(f"| {k} | {e['mean_norm_score']:.4f} | {e['median']:.4f} | {e['min']:.4f} | {e['mean_diff_vs_TA']:+.4f} | "
                     f"[{e['task_block_ci95'][0]:+.4f}, {e['task_block_ci95'][1]:+.4f}] | [{e['pair_boot_ci95'][0]:+.4f}, {e['pair_boot_ci95'][1]:+.4f}] | "
                     f"{e['win']}/{e['tie']}/{e['loss']} | {e['wilcoxon_p']:.3g} |")
        else:
            L.append(f"| {k} | {e['mean_norm_score']:.4f} | {e['median']:.4f} | {e['min']:.4f} | — | — | — | — | — |")
    L.append(f"\nNormalized score = mean over the two tasks of merged/single eval metric (= 1 − D for TA). Gate active (≥1 layer with θ_min < 30°) in {A['gate_active_pairs']}/{A['n_pairs']} pairs; elsewhere gate ≡ TA (tie).")
    (OUT / "method_comparison.md").write_text("\n".join(L) + "\n")


def figures(df, ST, A, s0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, pandas as pd, re
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for ax, (h, c, lab) in zip(axes[:2], [("H1", "O_A", "O_A = layer-mean mean cos² principal angles (orth B)"), ("H2", "tv_cosine", "task-vector cosine (flattened ΔW)")]):
        same = df.same_type.values
        ax.scatter(df[c][~same], df.D[~same], s=22, c="tab:blue", label="cross type")
        ax.scatter(df[c][same], df.D[same], s=30, c="tab:red", label="same type group")
        for r in df.itertuples():
            if r.D > np.percentile(df.D, 90) or getattr(r, c) > np.percentile(df[c], 95):
                ax.annotate(r.pair, (getattr(r, c), r.D), fontsize=6, xytext=(2, 2), textcoords="offset points")
        v = A["verdicts"][h]
        ax.set_title(f"{h}: ρ={v['rho']:.3f}, CI [{v['boot_ci95'][0]:.2f}, {v['boot_ci95'][1]:.2f}], Holm p={v['holm_p']:.3g} → {v['verdict']}", fontsize=9)
        ax.set_xlabel(lab); ax.set_ylabel("D = 1 − mean(merged/single), TA at selected λ"); ax.legend(fontsize=8)
    ax2 = axes[2]; labs = list(ST.keys()); y = np.arange(len(labs))[::-1]
    rh = [ST[c]["rho"] for c in labs]; lo = [ST[c]["rho"] - ST[c]["boot_ci95"][0] for c in labs]; hi = [ST[c]["boot_ci95"][1] - ST[c]["rho"] for c in labs]
    ax2.barh(y, rh, xerr=[lo, hi], color=["tab:green", "tab:green"] + ["tab:gray"] * (len(labs) - 2), capsize=3)
    ax2.set_yticks(y); ax2.set_yticklabels(labs, fontsize=8); ax2.axvline(0, c="k", lw=.8); ax2.axvline(0.4, c="tab:green", ls="--", lw=.8)
    ax2.axvline(0.3, c="tab:red", ls=":", lw=.8); ax2.set_xlim(-1, 1); ax2.set_xlabel("Spearman ρ with D (task-block 95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "fig_f1_scatter.png", dpi=150); plt.close(fig)
    Lr = pd.read_csv(OUT / "predictors_layers.csv"); layers = s0["lora_layers"]
    sub = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
    def lkey(n):
        m = re.search(r"layer\.(\d+)\.", n)
        if m: rest = n.split(f"layer.{m.group(1)}.")[1]; return (int(m.group(1)), sub.index(rest) if rest in sub else 9)
        return (99, 0)
    layers = sorted(layers, key=lkey); pairs = list(df.pair)
    H = Lr.pivot(index="layer", columns="pair", values="theta_min_A_deg").loc[layers, pairs].values
    nul = json.loads((OUT / "predictors_null.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(26, 14), gridspec_kw={"width_ratios": [5, 1]}, sharey=True)
    im = a1.imshow(H, aspect="auto", cmap="viridis_r", vmin=0, vmax=90)
    yy, xx = np.where(H < THETA_STAR)
    a1.scatter(xx, yy, marker="s", s=10, facecolors="none", edgecolors="red", linewidths=0.6)
    a1.set_xticks(range(len(pairs))); a1.set_xticklabels(pairs, rotation=90, fontsize=5)
    short = [n.replace("encoder.layer.", "L").replace("attention.self.", "").replace("attention.output.dense", "attn_out").replace("intermediate.dense", "inter").replace("output.dense", "out") for n in layers]
    a1.set_yticks(range(len(layers))); a1.set_yticklabels(short, fontsize=5)
    plt.colorbar(im, ax=a1, fraction=0.02, label="θ_min (deg), orth(B)"); a1.set_title("Minimum principal angle per LoRA layer × pair (red: < 30°)")
    band = [nul["per_layer"][n] for n in layers]; yl = np.arange(len(layers))
    a2.fill_betweenx(yl, [b["theta_min_p5"] for b in band], [b["theta_min_p95"] for b in band], color="gray", alpha=.35, label="random rank-8 null 5–95%")
    a2.plot(H.min(1), yl, "r.", ms=4, label="min over pairs"); a2.plot(np.median(H, 1), yl, "b.", ms=4, label="median over pairs")
    a2.axvline(30, c="k", ls=":", lw=.8); a2.set_xlim(0, 90); a2.set_xlabel("θ_min (deg)"); a2.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig_f2_heatmap.png", dpi=130); plt.close(fig)


def write_verdict(A, df, s0):
    tm = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    tr_s = sum(v["train_wall_s"] for v in s0["per_task"].values())
    L = [f"# E1b VERDICT — H1 (O_A): **{A['verdicts']['H1']['verdict']}**; H2 (task-vector cosine): **{A['verdicts']['H2']['verdict']}**\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} on {s0['gpu']}. Pre-registration: PREREG.md / prereg.json (sha256 in prereg.sha256); deviations: deviations.md.\n"]
    if A["invalid_reasons"]: L.append(f"**INVALID flags:** {A['invalid_reasons']}\n")
    L += ["## Primary hypotheses (pre-registered; Holm over H1, H2)\n",
          f"Valid tasks: {A['n_valid_tasks']} → {A['n_pairs']} pairs. Excluded: {A['excluded_tasks'] or 'none'}.\n",
          "| | predictor | ρ | task-block 95% CI (2000) | one-sided perm p (10000) | Holm p | LOTO frac ρ>0.2 | verdict |", "|---|---|---:|---|---:|---:|---:|---|"]
    for h, v in A["verdicts"].items():
        L.append(f"| {h} | {v['predictor']} | {v['rho']:.4f} | [{v['boot_ci95'][0]:.3f}, {v['boot_ci95'][1]:.3f}] | {v['perm_p_one_sided']:.4f} | {v['holm_p']:.4f} | {v['loto_frac_gt_0.2']:.2f} | **{v['verdict']}** |")
    L.append("\nRule: PASS iff ρ ≥ 0.4 ∧ Holm p < 0.05 ∧ ≥75% LOTO ρ > 0.2; else FAIL iff bootstrap upper < 0.3; else INCONCLUSIVE.")
    for h, v in A["verdicts"].items():
        L.append(f"\n{h} criteria: {v['criteria']}. LOTO ρ: " + ", ".join(f"{t} {x:.2f}" for t, x in v["loto"].items()))
    L.append(f"\nD: mean {A['D_summary']['mean']:.4f}, median {A['D_summary']['median']:.4f}, range [{A['D_summary']['min']:.4f}, {A['D_summary']['max']:.4f}]; λ* counts {A['lam_selected_counts']}. ρ(O_A, tv_cosine) = {A['rho_OA_tvcos']:.3f}. Min θ_min over all pairs/layers = {A['min_theta_min_overall_deg']:.1f}°.\n")
    L += ["## Secondary / exploratory predictors (not used for verdicts; uncorrected)\n",
          "| predictor | ρ | 95% CI | perm p one-sided | perm p two-sided | LOTO frac>0.2 |", "|---|---:|---|---:|---:|---:|"]
    for c, s in A["all_predictor_stats"].items():
        L.append(f"| {c} | {s['rho']:.3f} | [{s['boot_ci95'][0]:.2f}, {s['boot_ci95'][1]:.2f}] | {s['perm_p_one_sided']:.4f} | {s['perm_p_two_sided']:.4f} | {s['loto_frac_gt_0.2']:.2f} |")
    L.append(f"\nFixed-λ ρ (eval D at each λ): {json.dumps({k: {l: round(x, 3) for l, x in v.items()} for k, v in A['fixed_lambda_rho'].items()})}\n")
    L += ["## Method comparison (exploratory)\n", (OUT / "method_comparison.md").read_text()]
    L += ["## Integrity (stage0)\n", "| task | trained | eval metric | score | (a) thr | (a) | (b) ref−5pp | (b) | (c) max diff | valid |", "|---|---|---|---:|---:|---|---|---|---:|---|"]
    for t, r in s0["per_task"].items():
        b = f"{r['b_metric']} {r['b_ours']:.4f} ≥ {r['b_threshold']:.4f}" if r.get("b_ok") is not None else "n/a"
        L.append(f"| {t} | {r['trained_under']} ({r['train_steps']} steps) | {metric_name(t)} | {r['eval']['main']:.4f} | {r['a_threshold']:.4f} | {r['a_ok']} | {b} | {r['b_ok']} | {r['c_max_abs_logit_diff']:.1e} | **{r['valid']}** |")
    L += ["\n## Pair results\n", "| pair | λ* | merged t1 | merged t2 | norm t1 | norm t2 | D | O_A | tv cos | TIES@λ* | gate@λ* (layers) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in df.itertuples():
        L.append(f"| {r.pair} | {r.lam_selected} | {r.merged_eval_t1:.4f} | {r.merged_eval_t2:.4f} | {r.norm_t1:.4f} | {r.norm_t2:.4f} | {r.D:.4f} | {r.O_A:.5f} | {r.tv_cosine:.4f} | {r.TIES_atTA_score:.4f} | {r.GATE_atTA_score:.4f} ({r.gate_n_fail_layers}) |")
    L += ["\n## Deviations\n", (HERE / "deviations.md").read_text()]
    L.append(f"\n## Compute\n\nTraining (final adapters): {tr_s / 3600:.2f} GPU-h; stages (s): {json.dumps({k: round(v, 1) for k, v in tm.items()})}.")
    (OUT / "VERDICT.md").write_text("\n".join(L) + "\n")


def main():
    global OUT, ADIR, LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--out", default=str(HERE)); ap.add_argument("--adapters", default=str(HERE / "adapters"))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tasks", nargs="*", default=None, help="smoke tests only"); ap.add_argument("--min-valid", type=int, default=12)
    ap.add_argument("--force-valid", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True); ADIR = Path(a.adapters).resolve(); LOGF = OUT / "run.log"
    log(f"=== e1b.py {sys.argv}")
    for st in (["stage0", "stage1", "stage2", "stage3"] if a.stage == "all" else [a.stage]):
        if st == "stage0":
            s0 = stage0(a)
            if not s0["stage0_pass"]: log("STAGE 0: fewer than 12 valid tasks -> STOP"); sys.exit(2)
        elif st == "stage1": stage1(a)
        elif st == "stage2": stage2(a)
        elif st == "stage3": stage3(a)


if __name__ == "__main__":
    main()
