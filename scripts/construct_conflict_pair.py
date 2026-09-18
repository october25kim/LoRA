#!/usr/bin/env python3
"""Construct a high-interference LoRA pair by sharing k column directions of B.

Starts from Hub MNLI LoRA (task1) and a second Hub LoRA (default: SST-2), then
overwrites task2's LoRA B (and optionally A) so that the first `n_shared`
orthonormal columns of orth(B2) align with orth(B1). Guarantees θ_min ≈ 0 on
every LoRA layer → many FAILs at θ★=30°.

This is a controlled stress-test when independently trained Hub adapters are
nearly orthogonal (θ_min ≳ 50°).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


def _lora_modules(model):
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            out[name] = mod
    return out


def _get_AB(mod):
    A_dict, B_dict = mod.lora_A, mod.lora_B
    name = list(A_dict.keys())[0]
    return A_dict[name].weight, B_dict[name].weight, name


def share_subspace(B1: torch.Tensor, B2: torch.Tensor, n_shared: int, gen: torch.Generator):
    """Return new B2 (out, r) whose first n_shared cols span same as B1's top cols."""
    out, r = B2.shape
    n_shared = min(n_shared, r, B1.shape[1])
    # Orthonormal basis for col(B1)
    Q1, _ = torch.linalg.qr(B1.float(), mode="reduced")  # (out, r1)
    shared = Q1[:, :n_shared]  # (out, n_shared)
    # Random unique directions orthogonal to shared
    noise = torch.randn(out, r - n_shared, generator=gen, dtype=torch.float32)
    if n_shared > 0:
        noise = noise - shared @ (shared.T @ noise)
    Qn, _ = torch.linalg.qr(noise, mode="reduced")
    # Scale to match original B2 Frobenius norm roughly
    scales = B2.float().norm(dim=0).clamp_min(1e-6)
    B2_new = torch.zeros_like(B2, dtype=torch.float32)
    for i in range(n_shared):
        B2_new[:, i] = shared[:, i] * scales[i]
    for j in range(r - n_shared):
        B2_new[:, n_shared + j] = Qn[:, j] * scales[n_shared + j]
    return B2_new.to(dtype=B2.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--lora1", default="prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001")
    p.add_argument("--lora2-src", default="prateeky2806/bert-base-uncased-sst2-lora-epochs-2-lr-0.0005")
    p.add_argument("--n-labels1", type=int, default=3)
    p.add_argument("--n-labels2", type=int, default=2)
    p.add_argument("--n-shared", type=int, default=3, help="Shared B-column directions per layer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out1", default="adapters/conflict_mnli")
    p.add_argument("--out2", default="adapters/conflict_sst2_shared")
    args = p.parse_args()

    out1, out2 = Path(args.out1), Path(args.out2)
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.lora1} / {args.lora2_src} ...")
    m1 = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=args.n_labels1),
        args.lora1,
    )
    m2 = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=args.n_labels2),
        args.lora2_src,
    )

    mods1, mods2 = _lora_modules(m1), _lora_modules(m2)
    common = sorted(set(mods1) & set(mods2))
    gen = torch.Generator().manual_seed(args.seed)
    n_touched = 0
    with torch.no_grad():
        for name in common:
            A1, B1, _ = _get_AB(mods1[name])
            A2, B2, aname = _get_AB(mods2[name])
            B2_new = share_subspace(B1.data, B2.data, args.n_shared, gen)
            mods2[name].lora_B[aname].weight.copy_(B2_new)
            n_touched += 1

    print(f"Rewrote B on {n_touched} layers with n_shared={args.n_shared}")
    m1.save_pretrained(str(out1))
    m2.save_pretrained(str(out2))
    # tokenizers optional for week1
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(out1))
    tok.save_pretrained(str(out2))

    meta = {
        "method": "construct_share_B_columns",
        "lora1_src": args.lora1,
        "lora2_src": args.lora2_src,
        "n_shared": args.n_shared,
        "n_layers_rewritten": n_touched,
        "out1": str(out1),
        "out2": str(out2),
        "note": "Task2 B columns 0..n_shared-1 aligned to orth(B1); guarantees θ_min~0 (subspace A).",
    }
    (out2 / "conflict_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Saved {out1} and {out2}")


if __name__ == "__main__":
    main()
