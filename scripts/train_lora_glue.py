#!/usr/bin/env python3
"""Minimal stub: train a PEFT LoRA on GLUE MNLI or SST-2.

This script is intentionally thin — Week-1 does not require training in-box.
If no public Hub adapters are available, run this on a GPU machine:

  python scripts/train_lora_glue.py --task mnli --output-dir adapters/mnli-lora
  python scripts/train_lora_glue.py --task sst2 --output-dir adapters/sst2-lora

Then:
  python scripts/week1_run.py \\
    --lora-mnli adapters/mnli-lora \\
    --lora-sst2 adapters/sst2-lora \\
    --subspace A --output-dir artifacts
"""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA on GLUE (stub / minimal)")
    p.add_argument("--task", choices=["mnli", "sst2"], required=True)
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=-1, help="Optional early stop for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    task_map = {
        "mnli": ("glue", "mnli", 3, ("premise", "hypothesis"), "validation_matched"),
        "sst2": ("glue", "sst2", 2, ("sentence",), "validation"),
    }
    ds_name, subset, num_labels, fields, _ = task_map[args.task]

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=num_labels
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["query", "value", "key", "dense"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw = load_dataset(ds_name, subset)

    def tokenize(batch):
        if len(fields) == 1:
            return tok(batch[fields[0]], truncation=True, max_length=args.max_length)
        return tok(
            batch[fields[0]], batch[fields[1]], truncation=True, max_length=args.max_length
        )

    # MNLI has train + validation_matched; SST-2 has train + validation
    train_split = raw["train"]
    cols = [c for c in train_split.column_names if c not in ("label",)]
    train_ds = train_split.map(tokenize, batched=True, remove_columns=cols)
    train_ds = train_ds.rename_column("label", "labels")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
