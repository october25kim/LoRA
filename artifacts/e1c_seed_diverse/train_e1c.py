#!/usr/bin/env python3
"""E1c STEP 1: train one seed-1 LoRA adapter per task. Copy of E1b train_e1b.py / train_e1b_d1.py (PREREG_E1C.md sec. 1).
Differences from E1b: (i) recipe seed overridden 0 -> 1 (set_seed, TrainingArguments seed/data_seed); (ii) output dir adapters_s1/<task>;
(iii) budgets: E1b prereg.json, except tasks covered by E1b deviation D1 (mnli), which use deviations_d1.json and the uncapped pool
(exactly as train_e1b_d1.py). E1b tasks.py is imported unmodified (read-only). No eval-set scoring here.
Usage: python train_e1c.py --task cola   (run from artifacts/e1c_seed_diverse). Resumable: skips if adapters_s1/<task>/DONE."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(E1B))
from tasks import ALL_SPECS, load_raw, columns, split_indices, tokenize   # E1b, unmodified

E1C_SEED = 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
    task = a.task
    PC = json.loads((HERE / "prereg_e1c.json").read_text())
    assert task in PC["tasks"], f"{task} not an E1c task"
    P = json.loads((E1B / "prereg.json").read_text())
    D1 = json.loads((E1B / "deviations_d1.json").read_text())
    is_d1 = task in D1["budgets"]
    B = D1["budgets"][task] if is_d1 else P["budgets"][task]
    R = dict(P["recipe"]); R["seed_e1b"] = R["seed"]; R["seed"] = E1C_SEED
    out = HERE / "adapters_s1" / task
    if (out / "DONE").exists():
        print(f"{task}: DONE exists, skipping"); return
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
    # identical splits to the E1b seed-0 adapter of this task
    m0 = json.loads((E1B / "adapters" / task / "train_meta.json").read_text())
    t_sha = hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest(); h_sha = hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()
    assert t_sha == m0["train_idx_sha256"] and h_sha == m0["hold_idx_sha256"], "split differs from E1b seed-0 adapter"
    assert m0["budget"]["max_steps"] == B["max_steps"], (m0["budget"], B)
    ta, tb, y = columns(task, tr.select(tidx.tolist()))
    tok = AutoTokenizer.from_pretrained(R["base_model"], revision=R["base_revision"])
    ids, tts = tokenize(tok, ta, tb)
    ds = Dataset.from_dict({"input_ids": ids, "token_type_ids": tts, "attention_mask": [[1] * len(x) for x in ids],
                            "labels": y.tolist()})
    set_seed(R["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(R["base_model"], revision=R["base_revision"],
                                                               num_labels=spec[7])
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=R["r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"],
                      target_modules=R["target_modules"], modules_to_save=R["modules_to_save"])
    model = get_peft_model(model, lcfg)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    targs = TrainingArguments(output_dir=str(out / "_trainer"), per_device_train_batch_size=R["batch_size"],
                              max_steps=B["max_steps"], learning_rate=R["lr"], warmup_steps=B["warmup_steps"],
                              weight_decay=R["weight_decay"], lr_scheduler_type="linear", logging_steps=50,
                              save_strategy="no", eval_strategy="no", report_to=[], seed=R["seed"], data_seed=R["seed"],
                              fp16=True, remove_unused_columns=False, dataloader_pin_memory=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    model.save_pretrained(str(out))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"task": task, "spec": list(map(str, spec)), "budget": B, "recipe": R, "e1c_seed": E1C_SEED,
            "budget_source": "E1b deviations_d1.json (D1, full data)" if is_d1 else "E1b prereg.json",
            "n_trainable_params": int(ntr),
            "train_idx_sha256": t_sha, "hold_idx_sha256": h_sha,
            "segment_ids": "standard BERT token_type_ids from the tokenizer (0 = segment A incl. [CLS] and first [SEP], 1 = segment B); single-sentence tasks all 0",
            "train_wall_s": wall, "global_step": int(trainer.state.global_step), "train_loss": float(res.training_loss),
            "loss_first_logged": hist[0]["loss"] if hist else None, "loss_last_logged": hist[-1]["loss"] if hist else None,
            "loss_history": [(h["step"], h["loss"]) for h in hist],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                         "datasets": datasets.__version__}, "gpu": torch.cuda.get_device_name(0),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t0)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if is_d1:
        meta["deviation"] = "E1b D1 budget (full data), as used for the E1b seed-0 adapter"
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    import shutil; shutil.rmtree(out / "_trainer", ignore_errors=True)
    (out / "DONE").write_text(meta["finished"] + "\n")
    print(f"{task}: done steps={meta['global_step']} wall={wall:.0f}s loss={meta['train_loss']:.4f}")


if __name__ == "__main__":
    main()
