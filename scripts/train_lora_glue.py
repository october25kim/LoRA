#!/usr/bin/env python3
"""Minimal: train a PEFT LoRA on GLUE tasks (MNLI/SST-2/RTE/QNLI/QQP)."""

from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA on GLUE")
    p.add_argument(
        "--task",
        choices=["mnli", "sst2", "rte", "qnli", "qqp", "cola"],
        required=True,
    )
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=-1, help="Early stop for smoke/CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=-1, help="Subsample train set")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        set_seed,
    )

    set_seed(args.seed)

    task_map = {
        "mnli": ("nyu-mll/glue", "mnli", 3, ("premise", "hypothesis")),
        "sst2": ("nyu-mll/glue", "sst2", 2, ("sentence",)),
        "rte": ("nyu-mll/glue", "rte", 2, ("sentence1", "sentence2")),
        "qnli": ("nyu-mll/glue", "qnli", 2, ("question", "sentence")),
        "qqp": ("nyu-mll/glue", "qqp", 2, ("question1", "question2")),
        "cola": ("nyu-mll/glue", "cola", 2, ("sentence",)),
    }
    ds_name, subset, num_labels, fields = task_map[args.task]

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
    train_split = raw["train"]
    if args.train_samples > 0:
        n = min(args.train_samples, len(train_split))
        train_split = train_split.shuffle(seed=args.seed).select(range(n))

    def tokenize(batch):
        if len(fields) == 1:
            return tok(batch[fields[0]], truncation=True, max_length=args.max_length)
        return tok(
            batch[fields[0]], batch[fields[1]], truncation=True, max_length=args.max_length
        )

    cols = [c for c in train_split.column_names if c not in ("label",)]
    train_ds = train_split.map(tokenize, batched=True, remove_columns=cols)
    train_ds = train_ds.rename_column("label", "labels")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=args.seed,
        remove_unused_columns=False,
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
