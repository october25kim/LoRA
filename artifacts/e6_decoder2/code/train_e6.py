#!/usr/bin/env python3
"""E6 STEP 1: train one Qwen2.5-1.5B LoRA adapter (+ classification head `score`) per (task, seed). Copy of E4b train_e4b.py with the
E6 changes of PREREG_E6.md sec. 1 ONLY: backbone Qwen/Qwen2.5-1.5B (28 layers -> 196 LoRA layers, head 1536 -> num_labels) and
gradient checkpointing (non-reentrant; numerically neutral: activations are recomputed, RNG state preserved), plus the E6 base micro-batch table.
Everything else is E4b: Qwen2ForSequenceClassification (last non-pad token pooling, pad = <|endoftext|>), fp32 master weights + bf16 autocast,
LoRA r=8 a=16 dropout 0.1 on q/k/v/o/gate/up/down_proj, modules_to_save ["score"], max_len 256, E4b budgets (tasks_e6.budget_e4b), lr from the
E6 pilot decision (pilot/pilot_decision.json). Pilot mode (--pilot-lr X): trains into pilot/lr<X>/<task> (seed 0), never used by the main study.
Smoke mode (E6_SMOKE=1): 20 steps into smoke/ (scratch copy only). No eval-set scoring here.
--micro-div k: micro-batch = base_micro / k with gradient accumulation so that the effective batch stays 32 (pre-registered OOM fallback)."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(HERE))
from tasks_e6 import ALL_SPECS, load_raw, columns, indices_e4b, tokenize, budget_e4b   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-lr", type=float, default=None); ap.add_argument("--micro-div", type=int, default=1); a = ap.parse_args()
    task, seed = a.task, a.seed
    PA = json.loads((HERE / "prereg_e6.json").read_text())
    assert task in PA["tasks"], f"{task} not an E6 task"
    R = dict(PA["recipe"]); R["seed"] = seed
    smoke = os.environ.get("E6_SMOKE") == "1"
    spec = ALL_SPECS[task]
    tr = load_raw(task, spec[2]); ev_n = len(load_raw(task, spec[3]))
    hold, tidx, eidx = indices_e4b(len(tr), ev_n)
    B = budget_e4b(len(tr) - len(hold), len(tidx))
    assert B == {k: PA["budgets"]["per_task"][task][k] for k in B}, (B, PA["budgets"]["per_task"][task])
    if a.pilot_lr is not None:
        assert seed == 0 and task in PA["pilot"]["tasks"] and a.pilot_lr in PA["pilot"]["lr_candidates"]
        R["lr"] = a.pilot_lr
        out = HERE / "pilot" / f"lr{a.pilot_lr:g}" / task
    elif smoke:
        out = HERE / "smoke" / f"s{seed}" / task
        B = dict(B, max_steps=20, warmup_steps=2); R["lr"] = 3e-4          # smoke only (throwaway adapters)
    else:
        assert seed in PA["seeds_max"], seed
        dec = json.loads((HERE / "pilot" / "pilot_decision.json").read_text())
        assert dec["lr_main"] in PA["pilot"]["lr_candidates"] and seed in dec["seeds"]
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
    assert len(np.intersect1d(hold, tidx)) == 0
    t_sha = hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest(); h_sha = hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()
    m0 = json.loads((E1B / "adapters" / task / "train_meta.json").read_text())
    assert h_sha == m0["hold_idx_sha256"], "held-out split differs from E1b"
    ta, tb, y = columns(task, tr.select(tidx.tolist()))
    tok = AutoTokenizer.from_pretrained(R["base_model"], revision=R["base_revision"])
    assert tok.pad_token_id == R["pad_token_id"] and tok.padding_side == "right", (tok.pad_token_id, tok.padding_side)
    ids, _ = tokenize(tok, ta, tb)
    ntok = int(sum(len(x) for x in ids))
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids], "labels": y.tolist()})
    set_seed(R["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(R["base_model"], revision=R["base_revision"], num_labels=spec[7], dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=R["r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"],
                      target_modules=R["target_modules"], modules_to_save=R["modules_to_save"])
    model = get_peft_model(model, lcfg)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))
    assert n_lora == R["lora_layers_expected"], f"{n_lora} LoRA layers != {R['lora_layers_expected']}"
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_micro = R["micro_batch"].get(task, R["micro_batch"]["default"])
    micro = max(1, base_micro // a.micro_div); accum = R["batch_size"] // micro
    assert micro * accum == R["batch_size"], (micro, accum)
    targs = TrainingArguments(output_dir=str(out / "_trainer"), per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
                              max_steps=B["max_steps"], learning_rate=R["lr"], warmup_steps=B["warmup_steps"],
                              weight_decay=R["weight_decay"], lr_scheduler_type="linear", logging_steps=R["logging_steps"],
                              save_strategy="no", eval_strategy="no", report_to=[], seed=R["seed"], data_seed=R["seed"],
                              bf16=True, remove_unused_columns=False, dataloader_pin_memory=True,
                              gradient_checkpointing=R["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False})   # E6: GC
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=DataCollatorWithPadding(tok))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    model.save_pretrained(str(out))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    cnt = np.bincount(y.astype(np.int64), minlength=spec[7]) if spec[7] > 1 else None
    prior_entropy = float(-(cnt / cnt.sum() * np.log(np.maximum(cnt / cnt.sum(), 1e-12))).sum()) if cnt is not None else None
    meta = {"task": task, "seed": seed, "spec": list(map(str, spec)), "budget": B, "recipe": R,
            "mode": "pilot" if a.pilot_lr is not None else ("smoke" if smoke else "main"),
            "micro_batch": micro, "grad_accum": accum, "gradient_checkpointing": bool(R["gradient_checkpointing"]), "micro_div": a.micro_div,
            "n_trainable_params": int(ntr), "n_lora_layers": int(n_lora), "n_train_tokens": ntok, "mean_train_tokens": ntok / len(ids),
            "train_idx_sha256": t_sha, "hold_idx_sha256": h_sha,
            "train_wall_s": wall, "sec_per_step": wall / max(1, int(trainer.state.global_step)), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "global_step": int(trainer.state.global_step), "train_loss": float(res.training_loss),
            "label_prior_entropy": prior_entropy,
            "loss_first_logged": hist[0]["loss"] if hist else None, "loss_last_logged": hist[-1]["loss"] if hist else None,
            "loss_history": [(h["step"], h["loss"]) for h in hist],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                         "datasets": datasets.__version__}, "gpu": torch.cuda.get_device_name(0),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t0)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    import shutil; shutil.rmtree(out / "_trainer", ignore_errors=True)
    (out / "DONE").write_text(meta["finished"] + "\n")
    print(f"{task} s{seed}: done steps={meta['global_step']} wall={wall:.0f}s loss={meta['train_loss']:.4f} last={meta['loss_last_logged']}")


if __name__ == "__main__":
    main()
