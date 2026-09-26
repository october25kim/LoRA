#!/usr/bin/env python3
"""E1b: train one LoRA adapter per task exactly as pre-registered in prereg.json (budgets read from there).
Usage: python train_e1b.py --task cola   (run from artifacts/e1b_confirmatory). Resumable: skips if adapters/<task>/DONE.
NO evaluation on eval sets happens here (single-adapter scores are computed only in stage0)."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tasks import ALL_SPECS, load_raw, columns, split_indices, tokenize


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
    task = a.task
    P = json.loads((HERE / "prereg.json").read_text())
    B = P["budgets"][task]; R = P["recipe"]
    smoke = os.environ.get("E1B_SMOKE") == "1"   # pipeline smoke test: 20 steps into smoke/, deleted afterwards, never used
    out = HERE / ("smoke" if smoke else "adapters") / task
    if smoke:
        B = dict(B, max_steps=20, warmup_steps=2)
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
    assert len(tidx) == B["n_train_used"], (len(tidx), B)
    assert len(np.intersect1d(hold, tidx)) == 0
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
    meta = {"task": task, "spec": list(map(str, spec)), "budget": B, "recipe": R, "n_trainable_params": int(ntr),
            "train_idx_sha256": hashlib.sha256(tidx.astype(np.int64).tobytes()).hexdigest(),
            "hold_idx_sha256": hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest(),
            "segment_ids": "standard BERT token_type_ids from the tokenizer (0 = segment A incl. [CLS] and first [SEP], 1 = segment B); single-sentence tasks all 0",
            "train_wall_s": wall, "global_step": int(trainer.state.global_step), "train_loss": float(res.training_loss),
            "loss_first_logged": hist[0]["loss"] if hist else None, "loss_last_logged": hist[-1]["loss"] if hist else None,
            "loss_history": [(h["step"], h["loss"]) for h in hist],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                         "datasets": datasets.__version__}, "gpu": torch.cuda.get_device_name(0),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    import shutil; shutil.rmtree(out / "_trainer", ignore_errors=True)
    (out / "DONE").write_text(meta["finished"] + "\n")
    print(f"{task}: done steps={meta['global_step']} wall={wall:.0f}s loss={meta['train_loss']:.4f}")


if __name__ == "__main__":
    main()
