#!/usr/bin/env python3
"""E1 (pre-registered): Does per-layer principal-angle overlap between two LoRA adapters predict merge loss?

Protocol: /workspace/lora-paper/review/REVIEW_2026-09-25.md, section 6 + "Claude Code / Codex prompt (E1)".
Stages:
  stage0  integrity: adapters, heads, PEFT-equivalence, single-adapter scores, card comparison, hubish inspection
  stage1  weights-only predictors -> predictors.csv (+ layer csv, null json), sha256 frozen in predictors.sha256
  stage2  task-arithmetic merges; lam selected on seed-0 1000-example TRAIN subset; final on full GLUE validation
  stage3  pre-registered analysis + figures + VERDICT.md
Usage: python e1.py --out artifacts/e1_predictive --stage {stage0,stage1,stage2,stage3,all}
"""
import argparse, hashlib, itertools, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------- constants
PREREG_TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte"]
REPOS = {
    "cola": "prateeky2806/bert-base-uncased-cola-lora-epochs-10-lr-0.0005",
    "sst2": "prateeky2806/bert-base-uncased-sst2-lora-epochs-2-lr-0.0005",
    "mrpc": "prateeky2806/bert-base-uncased-mrpc-lora-epochs-10-lr-0.0005",
    "qqp":  "prateeky2806/bert-base-uncased-qqp-lora-epochs-2-lr-0.0005",
    "mnli": "prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001",
    "qnli": "prateeky2806/bert-base-uncased-qnli-lora-epochs-2-lr-0.0005",
    "rte":  "prateeky2806/bert-base-uncased-rte-lora-epochs-10-lr-0.0005",
}
# pinned commit revisions (resolved 2026-09-25 via HF API; re-verified in stage0)
REVS = {
    "cola": "04c81291b864f34668bf3936892f7a0280c7926d",
    "sst2": "b29cf70753ccbb7135cb55c5c1358592efe92886",
    "mrpc": "519fa53ae41bc614f0138bbe22c20d08066f0653",
    "qqp":  "3055be41e42a2bbc59656ca627104521178d10f8",
    "mnli": "7b635f33ca31d5464bf31df2f8a54cf007151fee",
    "qnli": "2af1d25dab0487c29b4a3f0e585e305054a15cee",
    "rte":  "eb4723dadf546c21abbd0045a8811805238bf4c9",
}
TASKS = list(REPOS.keys())  # stsb excluded: prateeky2806 publishes no STS-B LoRA adapter
BASE = "bert-base-uncased"
NUM_LABELS = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2}
KEYS = {"cola": ("sentence", None), "sst2": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"), "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2")}
VAL_SPLIT = {t: "validation" for t in TASKS}; VAL_SPLIT["mnli"] = "validation_matched"
# model-card "evaluation set" numbers (card eval set = 100 examples, see stage0 notes)
CARD = {"cola": {"mcc": 0.6347}, "sst2": {"accuracy": 0.98}, "mrpc": {"accuracy": 0.80, "f1": 0.8611},
        "qqp": {"accuracy": 0.95, "f1": 0.9333}, "mnli": {"accuracy": 0.85}, "qnli": {"accuracy": 0.92},
        "rte": {"accuracy": 0.63}}
LAMS = [0.3, 0.5, 0.7, 1.0]
MAXLEN = 128
N_SEL = 1000
# Segment ids: the prateeky2806 adapters were evidently trained WITHOUT segment (token_type) ids, i.e. all zeros
# (stage-0 diagnostic diag/diag_tt.json: passing BERT segment ids drops MNLI 0.807->0.705, MRPC 0.860->0.716, RTE 0.614->0.560;
# single-sentence tasks unchanged). The repo's earlier pipeline (lora_merge_cert/eval.py) also fed no token_type_ids.
ZERO_SEGMENT_IDS = True
SELF_REPRO = {"mnli": {"n_first_val": 512, "accuracy": 0.811, "source": "lora-paper/hubish_report.md 'Hub MNLI control' n=512 (4070 pipeline)"}}
TASK_TYPE = {"cola": "single", "sst2": "single", "mrpc": "para", "qqp": "para", "mnli": "nli", "qnli": "nli", "rte": "nli"}

OUT = None
LOGF = None


def log(msg):
    s = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " | " + str(msg)
    print(s, flush=True)
    if LOGF:
        with open(LOGF, "a") as f:
            f.write(s + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pairs_list():
    return list(itertools.combinations(TASKS, 2))


def jdump(obj, p):
    Path(p).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def timing_add(key, secs):
    p = OUT / "timing.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = d.get(key, 0.0) + secs
    jdump(d, p)


# ----------------------------------------------------------------------------- adapters
def load_adapter(task):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = hf_hub_download(REPOS[task], "adapter_model.safetensors", revision=REVS[task])
    cp = hf_hub_download(REPOS[task], "adapter_config.json", revision=REVS[task])
    cfg = json.load(open(cp))
    sd = load_file(p)
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
    return dict(layers=layers, head=head, other=other, keys=list(sd.keys()), cfg=cfg,
                scaling=cfg["lora_alpha"] / cfg["r"], r=cfg["r"], path=p, sha256=sha256(p))


def delta_weights(ad, device):
    import torch
    return {n: (ad["scaling"] * (B.to(device) @ A.to(device))) for n, (A, B) in ad["layers"].items()}


# ----------------------------------------------------------------------------- data
def load_split(task, split):
    from datasets import load_dataset
    return load_dataset("nyu-mll/glue", task, split=split)


def sel_indices(n_total):
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_total, N_SEL, replace=False))


def tokenize(task, ds, tok):
    k1, k2 = KEYS[task]
    a = [str(x) for x in ds[k1]]; b = [str(x) for x in ds[k2]] if k2 else None
    enc = tok(a, b, truncation=True, max_length=MAXLEN) if b is not None else tok(a, truncation=True, max_length=MAXLEN)
    tt = [[0] * len(x) for x in enc["input_ids"]] if ZERO_SEGMENT_IDS else enc["token_type_ids"]
    return {"ids": enc["input_ids"], "tt": tt, "labels": np.array(list(ds["label"]), dtype=np.int64)}


def get_data(task, which, tok):
    """which in {'val','sel'}; cached on disk."""
    import torch
    cdir = OUT / "cache"; cdir.mkdir(exist_ok=True)
    cp = cdir / f"{task}_{which}.pt"
    if cp.exists():
        return torch.load(cp, weights_only=False)
    if which == "val":
        ds = load_split(task, VAL_SPLIT[task]); idx = np.arange(len(ds))
    else:
        full = load_split(task, "train"); idx = sel_indices(len(full)); ds = full.select(idx.tolist())
    d = tokenize(task, ds, tok); d["idx"] = idx
    torch.save(d, cp)
    return d


# ----------------------------------------------------------------------------- model
class Enc:
    def __init__(self, device, layer_names, base_rev):
        import torch
        from transformers import BertModel
        self.device = device
        self.m = BertModel.from_pretrained(BASE, revision=base_rev).to(device).eval()
        mods = dict(self.m.named_modules())
        missing = [n for n in layer_names if n not in mods]
        assert not missing, f"layers not found in BertModel: {missing[:5]}"
        self.lin = {n: mods[n] for n in layer_names}
        self.W0 = {n: l.weight.detach().clone() for n, l in self.lin.items()}

    def set_delta(self, delta):
        import torch
        with torch.no_grad():
            for n, l in self.lin.items():
                if delta is None:
                    l.weight.copy_(self.W0[n])
                else:
                    l.weight.copy_(self.W0[n] + delta[n])

    def predict(self, data, head, bs=128):
        import torch
        ids, tt = data["ids"], data["tt"]
        n = len(ids)
        W = head["weight"].to(self.device); b = head["bias"].to(self.device)
        out = np.empty((n, W.shape[0]), dtype=np.float32)
        order = np.argsort([len(x) for x in ids], kind="stable")
        with torch.inference_mode():
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                L = max(len(ids[j]) for j in idx)
                ii = torch.zeros((len(idx), L), dtype=torch.long); tti = torch.zeros_like(ii); am = torch.zeros_like(ii)
                for r, j in enumerate(idx):
                    l = len(ids[j]); ii[r, :l] = torch.tensor(ids[j]); tti[r, :l] = torch.tensor(tt[j]); am[r, :l] = 1
                pooled = self.m(input_ids=ii.to(self.device), token_type_ids=tti.to(self.device),
                                attention_mask=am.to(self.device)).pooler_output
                out[idx] = (pooled @ W.T + b).float().cpu().numpy()
        return out


def metrics(logits, labels):
    p = logits.argmax(-1)
    acc = float((p == labels).mean())
    res = {"accuracy": acc, "n": int(len(labels))}
    if logits.shape[1] == 2:
        tp = int(((p == 1) & (labels == 1)).sum()); fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum()); tn = int(((p == 0) & (labels == 0)).sum())
        res["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
        res["mcc"] = (tp * tn - fp * fn) / den
    return res, p.astype(np.int8)


def base_revision():
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(BASE).sha
    except Exception:
        return None


# ----------------------------------------------------------------------------- stage 0
def stage0(args):
    import torch
    from transformers import AutoTokenizer
    t0 = time.time()
    dev = torch.device(args.device)
    from huggingface_hub import HfApi
    api = HfApi()
    s0 = {"stage": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "device": str(dev),
          "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
          "versions": {"torch": torch.__version__}, "prereg_tasks": PREREG_TASKS, "checks": {}, "fail_reasons": []}
    import transformers, peft, datasets
    s0["versions"].update(transformers=transformers.__version__, peft=peft.__version__, datasets=datasets.__version__)
    # item 1: list adapters
    listing = sorted(m.id for m in api.list_models(author="prateeky2806", search="bert-base-uncased"))
    lora_listing = [m for m in listing if "-lora-" in m]
    s0["hub_listing_bert_base_uncased"] = listing
    s0["hub_listing_lora"] = lora_listing
    s0["task_availability"] = {t: [m for m in lora_listing if f"uncased-{t}-lora" in m] for t in PREREG_TASKS}
    s0["excluded_tasks"] = {t: "no prateeky2806 bert-base-uncased LoRA adapter exists on the Hub"
                            for t in PREREG_TASKS if not s0["task_availability"][t]}
    s0["final_tasks"] = TASKS
    s0["n_pairs"] = len(pairs_list())
    brev = base_revision(); s0["base_model"] = {"id": BASE, "revision": brev}
    ads, adinfo = {}, {}
    for t in TASKS:
        info = api.model_info(REPOS[t])
        ad = load_adapter(t); ads[t] = ad
        head_ok = ("weight" in ad["head"] and "bias" in ad["head"] and ad["head"]["weight"].shape == (NUM_LABELS[t], 768))
        adinfo[t] = {"repo": REPOS[t], "pinned_revision": REVS[t], "hub_head_revision_now": info.sha,
                     "revision_matches": info.sha == REVS[t], "safetensors_sha256": ad["sha256"],
                     "cfg_modules_to_save": ad["cfg"].get("modules_to_save"), "cfg_task_type": ad["cfg"].get("task_type"),
                     "r": ad["r"], "lora_alpha": ad["cfg"]["lora_alpha"], "scaling": ad["scaling"],
                     "target_modules": ad["cfg"].get("target_modules"), "n_lora_layers": len(ad["layers"]),
                     "classifier_keys": [k for k in ad["keys"] if "classifier" in k],
                     "head_shape": list(ad["head"]["weight"].shape) if "weight" in ad["head"] else None,
                     "head_weight_norm": float(ad["head"]["weight"].norm()) if "weight" in ad["head"] else None,
                     "head_bias_norm": float(ad["head"]["bias"].norm()) if "bias" in ad["head"] else None,
                     "other_keys": ad["other"], "head_present": bool(head_ok)}
        if not head_ok:
            s0["fail_reasons"].append(f"{t}: classifier head missing from adapter")
        if ad["other"]:
            s0["fail_reasons"].append(f"{t}: unexpected non-LoRA keys {ad['other'][:5]}")
    s0["adapters"] = adinfo
    names = sorted(ads[TASKS[0]]["layers"].keys())
    assert all(sorted(ads[t]["layers"].keys()) == names for t in TASKS), "LoRA layer sets differ across adapters"
    s0["lora_layers"] = names
    tok = AutoTokenizer.from_pretrained(BASE, revision=brev)
    enc = Enc(dev, names, brev)
    # item 2b: PEFT equivalence (manual merge + own head == PeftModel logits) on 64 val examples
    from transformers import BertForSequenceClassification
    from peft import PeftModel
    eq = {}
    for t in TASKS:
        d = get_data(t, "val", tok)
        sub = {"ids": d["ids"][:64], "tt": d["tt"][:64]}
        enc.set_delta(delta_weights(ads[t], dev))
        mine = enc.predict(sub, ads[t]["head"])
        base = BertForSequenceClassification.from_pretrained(BASE, revision=brev, num_labels=NUM_LABELS[t])
        pm = PeftModel.from_pretrained(base, REPOS[t], revision=REVS[t]).to(dev).eval()
        import torch as _t
        outs = []
        with _t.inference_mode():
            for i in range(64):
                o = pm(input_ids=_t.tensor([sub["ids"][i]], device=dev), token_type_ids=_t.tensor([sub["tt"][i]], device=dev)).logits
                outs.append(o.float().cpu().numpy()[0])
        ref = np.stack(outs)
        diff = float(np.abs(ref - mine).max())
        eq[t] = {"max_abs_logit_diff": diff, "argmax_agree": float((ref.argmax(-1) == mine.argmax(-1)).mean()), "ok": diff < 1e-3}
        if not eq[t]["ok"]:
            s0["fail_reasons"].append(f"{t}: manual merge != PeftModel (max diff {diff:.3g}) -> head/merge not faithful")
        del pm, base; _t.cuda.empty_cache()
        log(f"stage0 peft-equivalence {t}: maxdiff={diff:.2e}")
    s0["checks"]["peft_equivalence"] = eq
    # item 3: single-adapter scores (full validation + seed-0 train1k selection subset)
    singles = {}
    preds_dir = OUT / "preds"; preds_dir.mkdir(exist_ok=True)
    thr = {}
    tt0 = time.time(); nex = 0
    for t in TASKS:
        enc.set_delta(delta_weights(ads[t], dev))
        dv = get_data(t, "val", tok); ds_ = get_data(t, "sel", tok)
        mv, pv = metrics(enc.predict(dv, ads[t]["head"], args.bs), dv["labels"])
        ms, ps = metrics(enc.predict(ds_, ads[t]["head"], args.bs), ds_["labels"])
        nex += len(dv["labels"]) + len(ds_["labels"])
        np.savez_compressed(preds_dir / f"single_{t}.npz", val=pv, sel=ps, val_labels=dv["labels"], sel_labels=ds_["labels"], sel_idx=ds_["idx"])
        singles[t] = {"val": mv, "sel": ms, "val_split": VAL_SPLIT[t],
                      "sel_idx_sha256": hashlib.sha256(ds_["idx"].tobytes()).hexdigest()}
        lim = 0.60 if t == "mnli" else 0.55
        thr[t] = {"val_accuracy": mv["accuracy"], "threshold": lim, "ok": mv["accuracy"] >= lim}
        if not thr[t]["ok"]:
            s0["fail_reasons"].append(f"{t}: single accuracy {mv['accuracy']:.4f} < {lim}")
        log(f"stage0 single {t}: val={mv} sel={ms}")
    if dev.type == "cuda": torch.cuda.synchronize()
    s0["throughput_examples_per_s"] = nex / (time.time() - tt0)
    s0["singles"] = singles
    s0["zero_segment_ids"] = ZERO_SEGMENT_IDS
    sr = {}
    for t, ref in SELF_REPRO.items():
        pv = np.load(preds_dir / f"single_{t}.npz")
        acc = float((pv["val"][:ref["n_first_val"]] == pv["val_labels"][:ref["n_first_val"]]).mean())
        sr[t] = {**ref, "ours": acc, "shortfall_pp": 100 * (ref["accuracy"] - acc), "flag_gt2pp": 100 * (ref["accuracy"] - acc) > 2}
        if sr[t]["flag_gt2pp"]:
            s0["fail_reasons"].append(f"{t}: >2pp below own earlier reproduction ({acc:.4f} vs {ref['accuracy']})")
    s0["self_reproduction"] = sr
    log(f"stage0 self-reproduction: {sr}")
    s0["checks"]["accuracy_thresholds"] = thr
    # base model (no adapter) reference is uninformative here (random heads) -> skipped
    # card comparison: card numbers are on a 100-example eval set carved from TRAIN (inferred from
    # steps/epoch = ceil((n_train-100)/32)); compare full-val and also probe candidate 100-example train slices
    card = {}
    for t in TASKS:
        full = load_split(t, "train"); n = len(full)
        cands = {"train[:100]": np.arange(100), "train[-100:]": np.arange(n - 100, n)}
        for seed in (28, 42, 0):
            sp = full.train_test_split(test_size=100, seed=seed)
            # recover indices via the private indices mapping when available
            try:
                cands[f"train_test_split(test_size=100,seed={seed})"] = np.array(sp["test"]._indices.column(0).to_pylist())
            except Exception:
                pass
        enc.set_delta(delta_weights(ads[t], dev))
        probe = {}
        for cname, idx in cands.items():
            dsx = tokenize(t, full.select(idx.tolist()), tok)
            m, _ = metrics(enc.predict(dsx, ads[t]["head"], args.bs), dsx["labels"])
            probe[cname] = m
        entry = {"card": CARD[t], "full_val": {k: singles[t]["val"].get(k) for k in CARD[t]}, "probe_train_slices": probe}
        cm = list(CARD[t].keys())[0]
        entry["card_metric"] = cm
        entry["shortfall_fullval_pp"] = 100 * (CARD[t][cm] - singles[t]["val"][cm])
        entry["flag_gt2pp_fullval"] = entry["shortfall_fullval_pp"] > 2.0
        entry["exact_match_slices"] = [c for c, m in probe.items() if all(abs(m.get(k, -9) - v) < 0.00501 for k, v in CARD[t].items())]
        card[t] = entry
        log(f"stage0 card {t}: card={CARD[t]} fullval_short={entry['shortfall_fullval_pp']:.2f}pp exact={entry['exact_match_slices']}")
    s0["card_comparison"] = card
    s0["card_note"] = ("Card 'evaluation set' metrics have 2-decimal granularity and steps/epoch equal ceil((n_train-100)/32) for "
                       "cola/mrpc/rte/mnli, i.e. the card eval set is a 100-example hold-out from GLUE TRAIN, not GLUE validation.")
    # item 4: local hubish adapters
    from safetensors.torch import load_file
    hub = {}
    for a in ["mnli_s7_hubish", "mnli_s42_hubish", "rte_s42_hubish"]:
        p = Path(args.repo_root) / "adapters" / a
        if not (p / "adapter_model.safetensors").exists():
            hub[a] = "not found"; continue
        cfg = json.load(open(p / "adapter_config.json"))
        sd = load_file(str(p / "adapter_model.safetensors"))
        hk = [k for k in sd if "classifier" in k or "modules_to_save" in k or "score" in k]
        W = [sd[k] for k in hk if k.endswith("weight")]
        bvec = [sd[k] for k in hk if k.endswith("bias")]
        C = W[0].shape[0] if W else None
        lb = [v.float().norm().item() for k, v in sd.items() if ".lora_B." in k]
        hub[a] = {"cfg_modules_to_save": cfg.get("modules_to_save"), "task_type": cfg.get("task_type"),
                  "head_keys": hk, "head_saved": bool(hk),
                  "head_weight_norm": float(W[0].norm()) if W else None,
                  "expected_norm_at_init_std0.02": 0.02 * math.sqrt(C * 768) if C else None,
                  "head_bias_norm": float(bvec[0].norm()) if bvec else None,
                  "bias_nonzero_(init=0)": bool(bvec and float(bvec[0].abs().max()) > 0),
                  "mean_lora_B_norm": float(np.mean(lb)) if lb else None, "n_keys": len(sd)}
    hubref = {t: {"head_weight_norm": adinfo[t]["head_weight_norm"], "head_bias_norm": adinfo[t]["head_bias_norm"],
                  "mean_lora_B_norm": float(np.mean([B.norm().item() for (_, B) in ads[t]["layers"].values()]))} for t in ("mnli", "rte")}
    s0["hubish_item4"] = {"adapters": hub, "hub_reference": hubref}
    s0["stage0_pass"] = len(s0["fail_reasons"]) == 0
    s0["card_flags_gt2pp"] = [t for t in TASKS if card[t]["flag_gt2pp_fullval"]]
    s0["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    s0["wall_s"] = time.time() - t0
    jdump(s0, OUT / "stage0.json")
    timing_add("stage0_gpu_wall_s", time.time() - t0)
    log(f"stage0 done pass={s0['stage0_pass']} fail={s0['fail_reasons']} card_flags={s0['card_flags_gt2pp']}")
    return s0


# ----------------------------------------------------------------------------- stage 1
def orth(M):
    import torch
    Q, _ = torch.linalg.qr(M)
    return Q


def pa_cos(Q1, Q2):
    import torch
    return torch.linalg.svdvals(Q1.T @ Q2).clamp(max=1.0)


def stage1(args):
    import torch, pandas as pd
    t0 = time.time()
    if (OUT / "predictors.csv").exists() or (OUT / "predictors.sha256").exists():
        raise SystemExit("predictors already frozen; refusing to recompute (protocol: never edit after freeze)")
    if (OUT / "pair_results.jsonl").exists():
        raise SystemExit("merge results exist before predictors -> protocol violation")
    dev = torch.device(args.device)
    ads = {t: load_adapter(t) for t in TASKS}
    names = sorted(ads[TASKS[0]]["layers"].keys())
    r = ads[TASKS[0]]["r"]
    QA, QB, dW = {}, {}, {}
    for t in TASKS:
        QA[t], QB[t] = {}, {}
        for n in names:
            A, B = ads[t]["layers"][n]
            A = A.to(dev).double(); B = B.to(dev).double()
            QA[t][n] = orth(B)                                       # def A: orth(B)
            # def B: top-r left singular vectors of B@A (via QR of B: BA = Q_B R_B A)
            Qb, Rb = torch.linalg.qr(B)
            U, S, _ = torch.linalg.svd(Rb @ A, full_matrices=False)
            QB[t][n] = Qb @ U[:, :r]
        dW[t] = torch.cat([(ads[t]["scaling"] * (ads[t]["layers"][n][1].to(dev) @ ads[t]["layers"][n][0].to(dev))).flatten() for n in names])
    # random-subspace null (seed 0): 1000 random rank-r pairs per layer (same out-dim as that layer's B)
    g = torch.Generator(device=dev).manual_seed(0)
    NREP = 1000
    null_cos2 = np.zeros((NREP, len(names))); null_thmin = np.zeros((NREP, len(names)))
    dims = {}
    for li, n in enumerate(names):
        d = ads[TASKS[0]]["layers"][n][1].shape[0]; dims[n] = d
        G1 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        G2 = torch.randn((NREP, d, r), generator=g, device=dev, dtype=torch.float64)
        Q1, _ = torch.linalg.qr(G1); Q2, _ = torch.linalg.qr(G2)
        s = torch.linalg.svdvals(Q1.transpose(1, 2) @ Q2).clamp(max=1.0)
        null_cos2[:, li] = (s ** 2).mean(1).cpu().numpy()
        null_thmin[:, li] = np.degrees(np.arccos(s.max(1).values.cpu().numpy()))
    null_OA = null_cos2.mean(1)
    mu, sd = float(null_OA.mean()), float(null_OA.std(ddof=1))
    band = {n: {"d": dims[n], "theta_min_p5": float(np.percentile(null_thmin[:, i], 5)),
                "theta_min_p50": float(np.percentile(null_thmin[:, i], 50)),
                "theta_min_p95": float(np.percentile(null_thmin[:, i], 95)),
                "cos2_mean_p5": float(np.percentile(null_cos2[:, i], 5)), "cos2_mean_p95": float(np.percentile(null_cos2[:, i], 95))}
            for i, n in enumerate(names)}
    # TIES trim masks (top 20% magnitude per task vector)
    top = {}
    for t in TASKS:
        a = dW[t].abs()
        k = int(round(0.8 * a.numel()))
        thr = torch.sort(a).values[k]
        top[t] = a >= thr
    rows, lrows = [], []
    for (t1, t2) in pairs_list():
        cA, cB, thA, thB = [], [], [], []
        for n in names:
            sa = pa_cos(QA[t1][n], QA[t2][n]); sb = pa_cos(QB[t1][n], QB[t2][n])
            c2a = float((sa ** 2).mean()); c2b = float((sb ** 2).mean())
            tha = float(np.degrees(np.arccos(float(sa.max())))); thb = float(np.degrees(np.arccos(float(sb.max()))))
            cA.append(c2a); cB.append(c2b); thA.append(tha); thB.append(thb)
            lrows.append({"pair": f"{t1}-{t2}", "t1": t1, "t2": t2, "layer": n, "d_out": dims[n],
                          "cos2_mean_A": c2a, "theta_min_A_deg": tha, "cos2_mean_B": c2b, "theta_min_B_deg": thb})
        v1, v2 = dW[t1], dW[t2]
        n1, n2 = float(v1.norm()), float(v2.norm())
        both = top[t1] & top[t2]
        conf_top = float(((torch.sign(v1) != torch.sign(v2)) & both).sum() / both.sum().clamp(min=1))
        nz = (v1 != 0) & (v2 != 0)
        conf_all = float(((torch.sign(v1) != torch.sign(v2)) & nz).sum() / nz.sum().clamp(min=1))
        OA = float(np.mean(cA))
        rows.append({"pair": f"{t1}-{t2}", "t1": t1, "t2": t2, "O_A": OA, "O_B": float(np.mean(cB)),
                     "mean_theta_min_A_deg": float(np.mean(thA)), "min_theta_min_A_deg": float(np.min(thA)),
                     "n_layers_theta_min_lt30_A": int(np.sum(np.array(thA) < 30)),
                     "mean_theta_min_B_deg": float(np.mean(thB)), "n_layers_theta_min_lt30_B": int(np.sum(np.array(thB) < 30)),
                     "null_z_O_A": (OA - mu) / sd, "tv_cosine": float((v1 @ v2) / (n1 * n2)),
                     "sign_conflict_top20": conf_top, "sign_conflict_all": conf_all,
                     "norm_ratio": max(n1, n2) / min(n1, n2), "tvnorm_t1": n1, "tvnorm_t2": n2, "n_layers": len(names)})
    pd.DataFrame(rows).to_csv(OUT / "predictors.csv", index=False, float_format="%.10g")
    pd.DataFrame(lrows).to_csv(OUT / "predictors_layers.csv", index=False, float_format="%.10g")
    jdump({"n_rep": NREP, "seed": 0, "r": r, "null_O_A_mean": mu, "null_O_A_sd": sd,
           "null_O_A_p5": float(np.percentile(null_OA, 5)), "null_O_A_p95": float(np.percentile(null_OA, 95)),
           "per_layer": band, "note": "random rank-r subspaces (Gaussian->QR) of the layer's output dim; z = (O_A - mu)/sd"},
          OUT / "predictors_null.json")
    with open(OUT / "predictors.sha256", "w") as f:
        for fn in ("predictors.csv", "predictors_layers.csv", "predictors_null.json"):
            f.write(f"{sha256(OUT / fn)}  {fn}\n")
    timing_add("stage1_gpu_wall_s", time.time() - t0)
    log("stage1 done; frozen predictors.sha256:\n" + (OUT / "predictors.sha256").read_text())


def verify_predictors():
    ok = True; lines = (OUT / "predictors.sha256").read_text().strip().splitlines()
    for ln in lines:
        h, fn = ln.split()
        if sha256(OUT / fn) != h:
            ok = False; log(f"PREDICTOR HASH MISMATCH: {fn}")
    return ok


# ----------------------------------------------------------------------------- stage 2
def stage2(args):
    import torch, pandas as pd
    from transformers import AutoTokenizer
    t0 = time.time()
    s0 = json.loads((OUT / "stage0.json").read_text())
    if not s0["stage0_pass"]:
        raise SystemExit("stage0 failed -> stop")
    if not (OUT / "protocol_deviations.md").exists():
        raise SystemExit("protocol_deviations.md must be written before stage2")
    if not verify_predictors():
        raise SystemExit("predictors changed -> INVALID")
    dev = torch.device(args.device)
    brev = s0["base_model"]["revision"]
    tok = AutoTokenizer.from_pretrained(BASE, revision=brev)
    ads = {t: load_adapter(t) for t in TASKS}
    names = s0["lora_layers"]
    dW = {t: delta_weights(ads[t], dev) for t in TASKS}
    enc = Enc(dev, names, brev)
    data = {t: {"val": get_data(t, "val", tok), "sel": get_data(t, "sel", tok)} for t in TASKS}
    single = {t: {"val": s0["singles"][t]["val"]["accuracy"], "sel": s0["singles"][t]["sel"]["accuracy"]} for t in TASKS}
    jl = OUT / "pair_results.jsonl"
    done = set()
    if jl.exists():
        for ln in jl.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["pair"])
    fails = json.loads((OUT / "pair_failures.json").read_text()) if (OUT / "pair_failures.json").exists() else {}
    pdir = OUT / "preds"; pdir.mkdir(exist_ok=True)
    for (t1, t2) in pairs_list():
        key = f"{t1}-{t2}"
        if key in done or fails.get(key, 0) >= 2:
            continue
        tp = time.time()
        try:
            rec = {"pair": key, "t1": t1, "t2": t2}
            sel_scores, val_scores, preds = {}, {}, {}
            for lam in LAMS:
                enc.set_delta({n: lam * (dW[t1][n] + dW[t2][n]) for n in names})
                for t in (t1, t2):
                    m, _ = metrics(enc.predict(data[t]["sel"], ads[t]["head"], args.bs), data[t]["sel"]["labels"])
                    rec[f"sel_acc_{t}_lam{lam}"] = m["accuracy"]
                    if args.explore_val_all_lams or True:
                        mv, pv = metrics(enc.predict(data[t]["val"], ads[t]["head"], args.bs), data[t]["val"]["labels"])
                        rec[f"val_acc_{t}_lam{lam}"] = mv["accuracy"]; preds[f"{t}_lam{lam}"] = pv
                sel_scores[lam] = 0.5 * (rec[f"sel_acc_{t1}_lam{lam}"] / single[t1]["sel"] + rec[f"sel_acc_{t2}_lam{lam}"] / single[t2]["sel"])
                rec[f"sel_norm_mean_lam{lam}"] = sel_scores[lam]
                rec[f"val_D_lam{lam}"] = 1 - 0.5 * (rec[f"val_acc_{t1}_lam{lam}"] / single[t1]["val"] + rec[f"val_acc_{t2}_lam{lam}"] / single[t2]["val"])
            lam_star = LAMS[int(np.argmax([sel_scores[l] for l in LAMS]))]  # ties -> smallest lam
            rec["lam_selected"] = lam_star
            rec["single_val_" + t1] = single[t1]["val"]; rec["single_val_" + t2] = single[t2]["val"]
            rec["merged_val_t1"] = rec[f"val_acc_{t1}_lam{lam_star}"]; rec["merged_val_t2"] = rec[f"val_acc_{t2}_lam{lam_star}"]
            rec["norm_t1"] = rec["merged_val_t1"] / single[t1]["val"]; rec["norm_t2"] = rec["merged_val_t2"] / single[t2]["val"]
            rec["D"] = 1 - 0.5 * (rec["norm_t1"] + rec["norm_t2"])
            rec["secs"] = time.time() - tp
            np.savez_compressed(pdir / f"pair_{key}.npz", **preds)
            with open(jl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"stage2 {key}: lam*={lam_star} merged=({rec['merged_val_t1']:.4f},{rec['merged_val_t2']:.4f}) D={rec['D']:.4f} [{rec['secs']:.0f}s]")
        except Exception as e:
            fails[key] = fails.get(key, 0) + 1
            jdump(fails, OUT / "pair_failures.json")
            log(f"stage2 {key} CRASH #{fails[key]}: {e}\n{traceback.format_exc()}")
            torch.cuda.empty_cache()
            if fails[key] < 2:
                return stage2(args)
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    order = {f"{a}-{b}": i for i, (a, b) in enumerate(pairs_list())}
    recs.sort(key=lambda r: order[r["pair"]])
    pd.DataFrame(recs).to_csv(OUT / "pair_results.csv", index=False, float_format="%.10g")
    timing_add("stage2_gpu_wall_s", time.time() - t0)
    log(f"stage2 done: {len(recs)} pairs; excluded={[k for k, v in fails.items() if v >= 2]}")


# ----------------------------------------------------------------------------- stage 3
def spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic)


def stage3(args):
    import pandas as pd
    t0 = time.time()
    s0 = json.loads((OUT / "stage0.json").read_text())
    hash_ok = verify_predictors()
    P = pd.read_csv(OUT / "predictors.csv")
    R = pd.read_csv(OUT / "pair_results.csv")
    decisions = json.loads((OUT / "protocol_decisions.json").read_text())
    df = P.merge(R[["pair", "D", "lam_selected", "merged_val_t1", "merged_val_t2", "norm_t1", "norm_t2"] +
                   [c for c in R.columns if c.startswith("val_D_lam")]], on="pair")
    tasks = TASKS
    tix = {t: i for i, t in enumerate(tasks)}
    K = len(tasks)
    Dm = np.full((K, K), np.nan); Xm = {}
    for _, r in df.iterrows():
        i, j = tix[r.t1], tix[r.t2]; Dm[i, j] = Dm[j, i] = r.D
    pairs_ij = [(tix[a], tix[b]) for a, b in zip(df.t1, df.t2)]
    D = df.D.values
    primary = "O_A"
    secondary = ["O_B", "mean_theta_min_A_deg", "null_z_O_A", "tv_cosine", "sign_conflict_top20"]
    extra = ["n_layers_theta_min_lt30_A", "min_theta_min_A_deg", "sign_conflict_all", "norm_ratio"]
    rng = np.random.default_rng(0)

    def mat(col):
        M = np.full((K, K), np.nan)
        for (i, j), v in zip(pairs_ij, df[col].values):
            M[i, j] = M[j, i] = v
        return M
    mats = {c: mat(c) for c in [primary] + secondary + extra}

    def rho_perm_p(col, nrep=10000):
        obs = spearman(df[col], D)
        X = df[col].values
        cnt_g = cnt_a = 0
        for _ in range(nrep):
            pi = rng.permutation(K)
            Dp = np.array([Dm[pi[i], pi[j]] for i, j in pairs_ij])
            rp = spearman(X, Dp)
            cnt_g += rp >= obs - 1e-12; cnt_a += abs(rp) >= abs(obs) - 1e-12
        # exact enumeration over all K! relabelings
        eg = ea = 0; tot = 0
        for pi in itertools.permutations(range(K)):
            Dp = np.array([Dm[pi[i], pi[j]] for i, j in pairs_ij]); rp = spearman(X, Dp)
            eg += rp >= obs - 1e-12; ea += abs(rp) >= abs(obs) - 1e-12; tot += 1
        if not np.isfinite(obs):  # constant predictor -> rho undefined -> p undefined
            return obs, float("nan"), float("nan"), float("nan"), float("nan")
        return obs, (1 + cnt_g) / (1 + nrep), (1 + cnt_a) / (1 + nrep), eg / tot, ea / tot

    def boot(cols, nrep=2000):
        out = {c: [] for c in cols}; diff = []; skipped = 0
        for _ in range(nrep):
            s = rng.integers(0, K, K)
            ii = [(s[a], s[b]) for a in range(K) for b in range(a + 1, K) if s[a] != s[b]]
            if len(set(tuple(sorted(p)) for p in ii)) < 4:
                skipped += 1; continue
            y = np.array([Dm[i, j] for i, j in ii])
            vals = {}
            for c in cols:
                x = np.array([mats[c][i, j] for i, j in ii]); vals[c] = spearman(x, y); out[c].append(vals[c])
            diff.append(vals["O_A"] - vals["tv_cosine"])
        ci = {c: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for c, v in out.items()}
        return ci, [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))], skipped

    def loto(col):
        res = {}
        for t in tasks:
            m = (df.t1 != t) & (df.t2 != t)
            res[t] = spearman(df[col][m], df.D[m])
        return res

    allcols = [primary] + secondary + extra
    ci, diff_ci, skipped = boot(allcols)
    stats = {}
    for c in allcols:
        obs, pg, pa, eg, ea = rho_perm_p(c)
        stats[c] = {"rho": obs, "perm_p_one_sided_greater_10000": pg, "perm_p_two_sided_10000": pa,
                    "perm_p_one_sided_exact_5040": eg, "perm_p_two_sided_exact_5040": ea,
                    "task_block_bootstrap_95ci": ci[c], "loto": loto(c)}
        log(f"stage3 {c}: rho={obs:.4f} p1={pg:.4f} p2={pa:.4f} ci={ci[c]}")
    # Holm over the 5 pre-listed secondary predictors (two-sided permutation p)
    ps = [(c, stats[c]["perm_p_two_sided_10000"]) for c in secondary]
    order = sorted(range(len(ps)), key=lambda i: ps[i][1]); m = len(ps); running = 0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * ps[i][1]); running = max(running, adj)
        stats[ps[i][0]]["holm_p"] = running
    # example-level bootstrap (1000 reps) of primary rho, resampling validation examples per task
    ex_rhos = []
    try:
        sp = {t: np.load(OUT / "preds" / f"single_{t}.npz") for t in tasks}
        pp = {r.pair: np.load(OUT / "preds" / f"pair_{r.pair}.npz") for r in df.itertuples()}
        corr_s = {t: (sp[t]["val"] == sp[t]["val_labels"]).astype(np.float32) for t in tasks}
        corr_m = {}
        for r in df.itertuples():
            lam = r.lam_selected
            for t in (r.t1, r.t2):
                corr_m[(r.pair, t)] = (pp[r.pair][f"{t}_lam{lam}"] == sp[t]["val_labels"]).astype(np.float32)
        for _ in range(1000):
            idx = {t: rng.integers(0, len(corr_s[t]), len(corr_s[t])) for t in tasks}
            s_acc = {t: corr_s[t][idx[t]].mean() for t in tasks}
            Db = [1 - 0.5 * (corr_m[(r.pair, r.t1)][idx[r.t1]].mean() / s_acc[r.t1] + corr_m[(r.pair, r.t2)][idx[r.t2]].mean() / s_acc[r.t2]) for r in df.itertuples()]
            ex_rhos.append(spearman(df.O_A, Db))
        ex_ci = [float(np.percentile(ex_rhos, 2.5)), float(np.percentile(ex_rhos, 97.5))]
    except Exception as e:
        ex_ci = f"failed: {e}"
    # exploratory: rho(O_A, D) at each fixed lam (validation)
    fixed = {f"lam{l}": spearman(df.O_A, df[f"val_D_lam{l}"]) for l in LAMS if f"val_D_lam{l}" in df}
    fixed_tv = {f"lam{l}": spearman(df.tv_cosine, df[f"val_D_lam{l}"]) for l in LAMS if f"val_D_lam{l}" in df}
    pr = stats[primary]
    n_loto_ok = int(sum(1 for v in pr["loto"].values() if v > 0.3))
    need = decisions["loto_required"]
    crit = {"rho_ge_0.5": pr["rho"] >= 0.5, "perm_p_lt_0.05": pr[decisions["primary_p"]] < 0.05,
            f"loto_ge_{need}of{K}": n_loto_ok >= need}
    if pr["rho"] >= 0.5 and pr[decisions["primary_p"]] < 0.05 and n_loto_ok >= need:
        stat_verdict = "PASS"
    elif pr["task_block_bootstrap_95ci"][1] < 0.3:
        stat_verdict = "FAIL"
    else:
        stat_verdict = "INCONCLUSIVE"
    invalid = []
    if not s0["stage0_pass"]: invalid.append("stage0 failed")
    if not hash_ok: invalid.append("predictors changed after freeze")
    if decisions.get("card_rule_invalidates") and s0["card_flags_gt2pp"]:
        invalid.append(f"single-adapter >2pp below card: {s0['card_flags_gt2pp']}")
    verdict = "INVALID" if invalid else stat_verdict
    A = {"verdict": verdict, "statistical_verdict_ignoring_invalid": stat_verdict, "invalid_reasons": invalid,
         "criteria": crit, "n_loto_rho_gt_0.3": n_loto_ok, "n_pairs": int(len(df)), "tasks": tasks,
         "primary": {"predictor": primary, **pr, "example_bootstrap_95ci_1000": ex_ci},
         "secondary_holm": {c: stats[c] for c in secondary},
         "extra_exploratory_uncorrected": {c: stats[c] for c in extra},
         "delta_rho_OA_minus_tvcos": {"value": pr["rho"] - stats["tv_cosine"]["rho"], "task_block_bootstrap_95ci": diff_ci},
         "exploratory_rho_OA_vs_D_at_fixed_lam_val": fixed, "exploratory_rho_tvcos_vs_D_at_fixed_lam_val": fixed_tv,
         "bootstrap_skipped_degenerate": skipped, "predictors_hash_ok": hash_ok, "decisions": decisions,
         "D_summary": {"mean": float(D.mean()), "min": float(D.min()), "max": float(D.max()), "sd": float(D.std(ddof=1))},
         "lam_selected_counts": {str(k): int(v) for k, v in df.lam_selected.value_counts().items()}}
    jdump(A, OUT / "analysis.json")
    figures(df, stats, allcols, s0)
    timing_add("stage3_wall_s", time.time() - t0)
    write_verdict(A, df, s0, stats, secondary, extra)
    log(f"stage3 done verdict={verdict} (stat={stat_verdict})")


def figures(df, stats, cols, s0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    nul = json.loads((OUT / "predictors_null.json").read_text())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    cat = lambda r: "same type" if TASK_TYPE[r.t1] == TASK_TYPE[r.t2] else "cross type"
    colors = {"same type": "tab:red", "cross type": "tab:blue"}
    for r in df.itertuples():
        c = cat(r)
        ax.scatter(r.O_A, r.D, color=colors[c], s=40)
        ax.annotate(r.pair, (r.O_A, r.D), fontsize=7, xytext=(3, 2), textcoords="offset points")
    for k, v in colors.items(): ax.scatter([], [], color=v, label=k)
    ax.axvline(nul["null_O_A_mean"], color="gray", ls=":", lw=1, label="random-subspace null mean")
    pr = stats["O_A"]
    ax.set_title(f"Spearman rho = {pr['rho']:.3f}  (task-block 95% CI [{pr['task_block_bootstrap_95ci'][0]:.2f}, {pr['task_block_bootstrap_95ci'][1]:.2f}])", fontsize=10)
    ax.set_xlabel("O_A = layer-mean of mean cos^2(principal angles), orth(B)")
    ax.set_ylabel("pair loss D = 1 - mean(merged/single)")
    mu, sd = nul["null_O_A_mean"], nul["null_O_A_sd"]
    sec = ax.secondary_xaxis("top", functions=(lambda x: (x - mu) / sd, lambda z: z * sd + mu)); sec.set_xlabel("null z-score")
    ax.legend(fontsize=8)
    labs = [c for c in cols]
    rh = [stats[c]["rho"] for c in labs]
    lo = [stats[c]["rho"] - stats[c]["task_block_bootstrap_95ci"][0] for c in labs]
    hi = [stats[c]["task_block_bootstrap_95ci"][1] - stats[c]["rho"] for c in labs]
    y = np.arange(len(labs))[::-1]
    ax2.barh(y, rh, xerr=[lo, hi], color=["tab:green"] + ["tab:gray"] * (len(labs) - 1), capsize=3)
    ax2.set_yticks(y); ax2.set_yticklabels(labs, fontsize=8); ax2.axvline(0, color="k", lw=0.8)
    ax2.axvline(0.5, color="tab:green", ls="--", lw=0.8); ax2.set_xlabel("Spearman rho with D (task-block 95% CI)")
    ax2.set_xlim(-1, 1)
    fig.tight_layout(); fig.savefig(OUT / "fig_f1_scatter.png", dpi=160); plt.close(fig)
    L = pd.read_csv(OUT / "predictors_layers.csv")
    layers = s0["lora_layers"]
    def lkey(n):
        import re
        m = re.search(r"layer\.(\d+)\.", n); sub = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
        if m:
            rest = n.split(f"layer.{m.group(1)}.")[1]
            return (int(m.group(1)), sub.index(rest) if rest in sub else 9)
        return (99, 0)
    layers = sorted(layers, key=lkey)
    pairs = list(df.pair)
    H = L.pivot(index="layer", columns="pair", values="theta_min_A_deg").loc[layers, pairs].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 13), gridspec_kw={"width_ratios": [3.2, 1]}, sharey=True)
    im = a1.imshow(H, aspect="auto", cmap="viridis_r", vmin=0, vmax=90)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] < 30: a1.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red", lw=1.2))
    a1.set_xticks(range(len(pairs))); a1.set_xticklabels(pairs, rotation=90, fontsize=7)
    short = [n.replace("encoder.layer.", "L").replace("attention.self.", "").replace("attention.output.dense", "attn_out").replace("intermediate.dense", "inter").replace("output.dense", "out") for n in layers]
    a1.set_yticks(range(len(layers))); a1.set_yticklabels(short, fontsize=5)
    plt.colorbar(im, ax=a1, fraction=0.03, label="theta_min (deg), def A")
    a1.set_title("Minimum principal angle per layer x pair (red: < 30 deg)")
    band = [nul["per_layer"][n] for n in layers]
    yy = np.arange(len(layers))
    a2.fill_betweenx(yy, [b["theta_min_p5"] for b in band], [b["theta_min_p95"] for b in band], color="gray", alpha=0.35, label="random rank-8 null 5-95%")
    a2.plot(H.min(1), yy, "r.", ms=4, label="min over pairs"); a2.plot(np.median(H, 1), yy, "b.", ms=4, label="median over pairs")
    a2.axvline(30, color="k", ls=":", lw=0.8); a2.set_xlim(0, 90); a2.set_xlabel("theta_min (deg)"); a2.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig_f2_heatmap.png", dpi=150); plt.close(fig)


def write_verdict(A, df, s0, stats, secondary, extra):
    tm = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    pr = A["primary"]
    L = []
    L.append(f"# E1 VERDICT: **{A['verdict']}**\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} on {s0.get('gpu')}. Pre-registered question: does per-layer principal-angle overlap (O_A) predict task-arithmetic merge loss D?\n")
    if A["invalid_reasons"]:
        L.append(f"INVALID reasons: {A['invalid_reasons']}. Statistical verdict that would otherwise apply: **{A['statistical_verdict_ignoring_invalid']}**.\n")
    L.append("## Primary (pre-registered)\n")
    L.append(f"- Pairs: {A['n_pairs']} (tasks: {', '.join(A['tasks'])}; STS-B excluded: no public adapter)")
    L.append(f"- Spearman rho(O_A, D) = **{pr['rho']:.4f}**")
    L.append(f"- Task-block bootstrap 95% CI (2000 reps): [{pr['task_block_bootstrap_95ci'][0]:.4f}, {pr['task_block_bootstrap_95ci'][1]:.4f}]")
    L.append(f"- Task-label permutation p (10000 reps): one-sided (rho>=obs) = {pr['perm_p_one_sided_greater_10000']:.4f}; two-sided = {pr['perm_p_two_sided_10000']:.4f}; exact over 7!=5040 relabelings: one-sided {pr['perm_p_one_sided_exact_5040']:.4f}, two-sided {pr['perm_p_two_sided_exact_5040']:.4f}")
    L.append(f"- Example-level bootstrap 95% CI (1000 reps): {pr['example_bootstrap_95ci_1000']}")
    L.append(f"- Leave-one-task-out rho: " + ", ".join(f"{t}: {v:.3f}" for t, v in pr["loto"].items()) + f"  -> {A['n_loto_rho_gt_0.3']}/{len(A['tasks'])} > 0.3 (required {A['decisions']['loto_required']})")
    L.append(f"- Criteria: {A['criteria']}")
    L.append(f"- D: mean {A['D_summary']['mean']:.4f}, range [{A['D_summary']['min']:.4f}, {A['D_summary']['max']:.4f}]; lam selected counts {A['lam_selected_counts']}\n")
    L.append("## Secondary predictors (exploratory; Holm over the 5 pre-listed; two-sided task-permutation p)\n")
    L.append("| predictor | rho | 95% task-block CI | perm p (2-sided) | Holm p | LOTO rho range |")
    L.append("|---|---:|---|---:|---:|---|")
    for c in ["O_A"] + secondary + extra:
        s = stats[c]; lv = list(s["loto"].values())
        L.append(f"| {c}{' (primary)' if c == 'O_A' else ''}{' (extra, uncorrected)' if c in extra else ''} | {s['rho']:.3f} | [{s['task_block_bootstrap_95ci'][0]:.2f}, {s['task_block_bootstrap_95ci'][1]:.2f}] | {s['perm_p_two_sided_10000']:.4f} | {s.get('holm_p', float('nan')):.4f} | [{np.nanmin(lv):.2f}, {np.nanmax(lv):.2f}] |")
    d = A["delta_rho_OA_minus_tvcos"]
    L.append(f"\nDelta rho (O_A - task-vector cosine) = {d['value']:.3f}, task-block 95% CI [{d['task_block_bootstrap_95ci'][0]:.3f}, {d['task_block_bootstrap_95ci'][1]:.3f}]")
    L.append(f"\nExploratory (not pre-registered) rho(O_A, D) at fixed lam on validation: {A['exploratory_rho_OA_vs_D_at_fixed_lam_val']}; tv-cosine: {A['exploratory_rho_tvcos_vs_D_at_fixed_lam_val']}\n")
    L.append("## Single-adapter scores (full GLUE validation; accuracy is the pre-registered metric)\n")
    L.append("| task | repo@rev | head saved | val acc | extra | seed-0 train1k acc | card (100-ex train hold-out) | full-val shortfall vs card |")
    L.append("|---|---|---|---:|---|---:|---|---:|")
    for t in TASKS:
        sv = s0["singles"][t]; c = s0["card_comparison"][t]; ad = s0["adapters"][t]
        ex = ", ".join(f"{k}={sv['val'][k]:.4f}" for k in ("mcc", "f1") if k in sv["val"])
        L.append(f"| {t} | {ad['repo'].split('/')[-1]}@{ad['pinned_revision'][:8]} | {ad['head_present']} | {sv['val']['accuracy']:.4f} | {ex} | {sv['sel']['accuracy']:.4f} | {c['card']} | {c['shortfall_fullval_pp']:.2f} pp{' FLAG' if c['flag_gt2pp_fullval'] else ''} |")
    eqv = s0["checks"]["peft_equivalence"]
    L.append(f"\nIntegrity: manual merge + own head vs PeftModel max |logit diff| = {max(v['max_abs_logit_diff'] for v in eqv.values()):.1e} (all 7 adapters; heads loaded from adapter files). "
             f"Self-reproduction MNLI first-512 val: {s0['self_reproduction']['mnli']['ours']:.4f} vs 0.811 earlier. Segment ids: all-zero (see deviations #5).")
    L.append("\n## Stage 0 item 4: local hubish adapters (ubuntu-4070 adapters/)\n")
    hb = s0["hubish_item4"]["adapters"]
    dg = OUT / "diag" / "hubish_segment_ids.json"
    dgd = json.loads(dg.read_text()) if dg.exists() else {}
    L.append("| adapter | cfg modules_to_save | head saved | head W norm (init~) | bias norm (init 0) | val acc, no segment ids (old eval) | val acc, BERT segment ids |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for a, h in hb.items():
        if isinstance(h, str):
            L.append(f"| {a} | {h} | | | | | |"); continue
        d = dgd.get(a, {})
        L.append(f"| {a} | {h['cfg_modules_to_save']} | {h['head_saved']} | {h['head_weight_norm']:.3f} ({h['expected_norm_at_init_std0.02']:.3f}) | {h['head_bias_norm']:.4f} | {d.get('zero_segment_ids', float('nan')):.4f} | {d.get('bert_segment_ids', float('nan')):.4f} |")
    L.append("\nThe heads are saved and were trained (bias moved away from its zero init). The earlier 0.45-0.49 comes from a **train/eval segment-id mismatch**: `scripts/train_lora_glue.py` trains WITH token_type_ids, but `lora_merge_cert/eval.py` evaluates WITHOUT them. The Hub adapters have the opposite convention (trained without segment ids).")
    L.append("\n## Pair results\n")
    L.append("| pair | lam* | merged t1 | merged t2 | norm t1 | norm t2 | D | O_A | tv cos |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in df.itertuples():
        L.append(f"| {r.pair} | {r.lam_selected} | {r.merged_val_t1:.4f} | {r.merged_val_t2:.4f} | {r.norm_t1:.4f} | {r.norm_t2:.4f} | {r.D:.4f} | {r.O_A:.5f} | {r.tv_cosine:.4f} |")
    L.append("\n## Deviations from protocol\n")
    L.append((OUT / "protocol_deviations.md").read_text())
    pn = OUT / "post_run_notes.md"
    if pn.exists():
        L.append("\n## Post-run notes (written after results; no effect on predictors, selection, or verdict rules)\n")
        L.append(pn.read_text())
    L.append("\n## Compute\n")
    L.append(f"timing (s): {json.dumps({k: round(v, 1) for k, v in tm.items()})}; total GPU wall ~ {sum(v for k, v in tm.items() if 'gpu' in k) / 3600:.2f} h")
    (OUT / "VERDICT.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    global OUT, LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/e1_predictive")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--explore-val-all-lams", action="store_true", default=True)
    args = ap.parse_args()
    OUT = Path(args.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    LOGF = OUT / "run.log"
    log(f"=== e1.py stage={args.stage} argv={sys.argv}")
    st = args.stage
    if st in ("stage0", "all"):
        s0 = stage0(args)
        if not s0["stage0_pass"]:
            log("STAGE 0 FAILED -> stop"); sys.exit(2)
    if st in ("stage1", "all"):
        stage1(args)
    if st in ("stage2", "all"):
        stage2(args)
    if st in ("stage3", "all"):
        stage3(args)
        print((OUT / "VERDICT.md").read_text())


if __name__ == "__main__":
    main()
