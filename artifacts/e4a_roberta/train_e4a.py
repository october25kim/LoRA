#!/usr/bin/env python3
"""E4a STEP 1: train one RoBERTa-base LoRA adapter per (task, seed). Copy of E1c train_e1c.py (itself a copy of E1b train_e1b.py /
train_e1b_d1.py) with these changes only (PREREG_E4A.md sec. 1): backbone roberta-base; explicit target_modules (72 layers, no pooler
exists); no token_type_ids anywhere; seed from --seed (0 or 1); lr from the pre-registered pilot decision (pilot/pilot_decision.json);
output adapters_s<seed>/<task>. Budgets = E1b prereg.json, mnli = E1b deviation D1 (full pool). E1b tasks.py imported unmodified.
Pilot mode (--pilot-lr X): trains into pilot/lr<X>/<task> (seed 0) and is never used by the main study.
Smoke mode (E4A_SMOKE=1): 20 steps into smoke/, deleted afterwards, never used.
No eval-set scoring here."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(HERE))
from tasks_e4 import ALL_SPECS, load_raw, columns, split_indices, tokenize   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-lr", type=float, default=None); a = ap.parse_args()
    task, seed = a.task, a.seed
    PA = json.loads((HERE / "prereg_e4a.json").read_text())
    assert task in PA["tasks"], f"{task} not an E4a task"
    assert seed in PA["seeds"], seed
    P = json.loads((E1B / "prereg.json").read_text())
    D1 = json.loads((E1B / "deviations_d1.json").read_text())
    is_d1 = task in D1["budgets"]
    B = dict(D1["budgets"][task] if is_d1 else P["budgets"][task])
    R = dict(PA["recipe"]); R["seed"] = seed
    smoke = os.environ.get("E4A_SMOKE") == "1"
    if a.pilot_lr is not None:
        assert seed == 0 and task in PA["pilot"]["tasks"] and a.pilot_lr in PA["pilot"]["lr_ladder"]
        R["lr"] = a.pilot_lr
        out = HERE / "pilot" / f"lr{a.pilot_lr:g}" / task
    elif smoke:
        out = HERE / "smoke" / f"s{seed}" / task
        B = dict(B, max_steps=20, warmup_steps=2)
    else:
        dec = json.loads((HERE / "pilot" / "pilot_decision.json").read_text())
        assert dec["lr_main"] in PA["pilot"]["lr_ladder"]
        R["lr"] = dec["lr_main"]
        out = HERE / f"adapters_s{seed}" / task
    if (out / "DONE").exists():
        print(f"{task} s{seed}: DONE exists, skipping"); return
    out.mkdir(parents=True, exist_ok=True)
    import torch, transformers, peft, datasets
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorWithPadding, set_seed)
    spec = ALL_SPECS[task]
    tr = load_raw(task, spec[2]); ev_n = len(load_raw(task, spec[3]))
    hold, tidx, eidx = split_indices(len(tr), ev_n)
    if is_d1:                                                      # as train_e1b_d1.py
        perm = np.random.default_rng(0).permutation(len(tr))
        assert np.array_equal(np.sort(perm[:1000]), hold)
        tidx = np.sort(perm[1000:])
    assert len(tidx) == B["n_train_used"], (len(tidx), B)
    assert len(np.intersect1d(hold, tidx)) == 0
    m0 = json.loads((E1B / "adapters" / task / "train_meta.json").read_text())      # identical splits to E1b (BERT) adapters
    t_sha = hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest(); h_sha = hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()
    assert t_sha == m0["train_idx_sha256"] and h_sha == m0["hold_idx_sha256"], "split differs from E1b adapter"
    if not smoke:
        assert m0["budget"]["max_steps"] == B["max_steps"], (m0["budget"], B)
    ta, tb, y = columns(task, tr.select(tidx.tolist()))
    tok = AutoTokenizer.from_pretrained(R["base_model"], revision=R["base_revision"])
    ids, _ = tokenize(tok, ta, tb)
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids], "labels": y.tolist()})
    set_seed(R["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(R["base_model"], revision=R["base_revision"], num_labels=spec[7])
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=R["r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"],
                      target_modules=R["target_modules"], modules_to_save=R["modules_to_save"])
    model = get_peft_model(model, lcfg)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    assert n_lora == R["lora_layers_expected"], f"{n_lora} LoRA layers != {R['lora_layers_expected']}"
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warm = B["warmup_steps"]
    targs = TrainingArguments(output_dir=str(out / "_trainer"), per_device_train_batch_size=R["batch_size"],
                              max_steps=B["max_steps"], learning_rate=R["lr"], warmup_steps=warm,
                              weight_decay=R["weight_decay"], lr_scheduler_type="linear", logging_steps=50,
                              save_strategy="no", eval_strategy="no", report_to=[], seed=R["seed"], data_seed=R["seed"],
                              fp16=True, remove_unused_columns=False, dataloader_pin_memory=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    model.save_pretrained(str(out))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    cnt = np.bincount(y.astype(np.int64), minlength=spec[7]) if spec[7] > 1 else None
    prior_entropy = float(-(cnt / cnt.sum() * np.log(np.maximum(cnt / cnt.sum(), 1e-12))).sum()) if cnt is not None else None
    meta = {"task": task, "seed": seed, "spec": list(map(str, spec)), "budget": B, "recipe": R,
            "budget_source": "E1b deviations_d1.json (D1, full data)" if is_d1 else "E1b prereg.json",
            "mode": "pilot" if a.pilot_lr is not None else ("smoke" if smoke else "main"),
            "n_trainable_params": int(ntr), "n_lora_layers": int(n_lora),
            "train_idx_sha256": t_sha, "hold_idx_sha256": h_sha,
            "segment_ids": "none: RoBERTa (type_vocab_size=1); tokenizer returns no token_type_ids; model never receives token_type_ids",
            "train_wall_s": wall, "global_step": int(trainer.state.global_step), "train_loss": float(res.training_loss),
            "label_prior_entropy": prior_entropy,
            "loss_first_logged": hist[0]["loss"] if hist else None, "loss_last_logged": hist[-1]["loss"] if hist else None,
            "loss_history": [(h["step"], h["loss"]) for h in hist],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                         "datasets": datasets.__version__}, "gpu": torch.cuda.get_device_name(0),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t0)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if is_d1:
        meta["deviation"] = "E1b D1 budget (full data), as used for the E1b adapter"
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    import shutil; shutil.rmtree(out / "_trainer", ignore_errors=True)
    (out / "DONE").write_text(meta["finished"] + "\n")
    print(f"{task} s{seed}: done steps={meta['global_step']} wall={wall:.0f}s loss={meta['train_loss']:.4f} last={meta['loss_last_logged']}")


if __name__ == "__main__":
    main()
