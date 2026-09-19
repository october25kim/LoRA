#!/usr/bin/env python3
"""Eval single PEFT adapters / base on MNLI (n=512 and full val)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from lora_merge_cert.eval import evaluate_glue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--also-full-val", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    out = {"device": str(device), "base_model": args.base_model, "runs": []}

    base = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3).to(device)
    r = evaluate_glue(base, tok, "mnli", device=device, num_samples=args.num_samples)
    entry = {"name": "base_no_adapter", "n": args.num_samples, **r}
    if args.also_full_val:
        entry["full_val"] = evaluate_glue(base, tok, "mnli", device=device, num_samples=None)
    out["runs"].append(entry)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for path in args.adapters:
        base = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
        model = PeftModel.from_pretrained(base, path).to(device)
        r = evaluate_glue(model, tok, "mnli", device=device, num_samples=args.num_samples)
        entry = {"name": path, "n": args.num_samples, **r}
        if args.also_full_val:
            entry["full_val"] = evaluate_glue(model, tok, "mnli", device=device, num_samples=None)
        out["runs"].append(entry)
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
