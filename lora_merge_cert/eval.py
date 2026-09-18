"""Evaluation helpers for GLUE MNLI / SST-2 accuracy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


TASK_CONFIG = {
    "mnli": {
        "dataset": "glue",
        "subset": "mnli",
        "validation_split": "validation_matched",
        "text_fields": ("premise", "hypothesis"),
        "num_labels": 3,
    },
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "validation_split": "validation",
        "text_fields": ("sentence",),
        "num_labels": 2,
    },
}


def load_glue_split(task: str, split: Optional[str] = None):
    """Load a GLUE validation split via HuggingFace datasets."""
    from datasets import load_dataset

    task = task.lower().replace("-", "").replace("_", "")
    if task in ("mnli", "mnlimatched"):
        cfg = TASK_CONFIG["mnli"]
    elif task in ("sst2", "sst"):
        cfg = TASK_CONFIG["sst2"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    split = split or cfg["validation_split"]
    return load_dataset(cfg["dataset"], cfg["subset"], split=split), cfg


@torch.no_grad()
def accuracy_on_dataloader(
    model: nn.Module,
    dataloader,
    device: Optional[torch.device] = None,
) -> float:
    """Compute classification accuracy over a DataLoader of HF batches."""
    device = device or next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def build_eval_dataloader(
    task: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
):
    """Tokenize GLUE split and return a simple DataLoader."""
    from torch.utils.data import DataLoader

    ds, cfg = load_glue_split(task, split=split)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    text_fields = cfg["text_fields"]

    def tokenize_fn(examples):
        if len(text_fields) == 1:
            return tokenizer(
                examples[text_fields[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        return tokenizer(
            examples[text_fields[0]],
            examples[text_fields[1]],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    cols_to_remove = [c for c in ds.column_names if c != "label"]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(ds, batch_size=batch_size)


def evaluate_glue(
    model: nn.Module,
    tokenizer,
    task: str,
    *,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Full eval: returns {'task', 'accuracy', 'n'}."""
    loader = build_eval_dataloader(
        task, tokenizer, batch_size=batch_size, max_length=max_length, num_samples=num_samples
    )
    if device is not None:
        model = model.to(device)
    acc = accuracy_on_dataloader(model, loader, device=device)
    n = len(loader.dataset)
    return {"task": task, "accuracy": acc, "n": n}


def fake_eval_from_certificates(
    certificates,
    *,
    base_mnli: float = 0.84,
    base_sst2: float = 0.92,
) -> Dict[str, Any]:
    """Dry-run stand-in: invent accuracy drops correlated with θ_min.

    Used when --dry-run so we can emit the Week-1 scatter without Hub/data.
    Heuristic: more FAIL layers / lower θ_min → larger MNLI drop under arithmetic merge;
    certificate correction recovers most of it.
    """
    import math

    fails = [c for c in certificates if c.status == "FAIL"]
    if not certificates:
        drop_arith = 0.0
        drop_cert = 0.0
    else:
        mean_theta = sum(c.theta_min_deg for c in certificates) / len(certificates)
        fail_frac = len(fails) / len(certificates)
        # Larger drop when angles are small and many FAILs
        drop_arith = 0.02 + 0.15 * fail_frac * max(0.0, (30.0 - mean_theta) / 30.0)
        drop_cert = drop_arith * 0.25  # correction recovers ~75%

    return {
        "mnli_arithmetic": base_mnli - drop_arith,
        "mnli_certificate": base_mnli - drop_cert,
        "sst2_arithmetic": base_sst2 - drop_arith * 0.5,
        "sst2_certificate": base_sst2 - drop_cert * 0.5,
        "mnli_drop_arithmetic": drop_arith,
        "mnli_drop_certificate": drop_cert,
        "fake": True,
    }
