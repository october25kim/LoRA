#!/usr/bin/env python3
"""Week-1 CLI: LoRA Merge Certificate dry-run / real merge + eval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running without install: repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from lora_merge_cert.certificate import (
    DEFAULT_THETA_STAR_DEG,
    certify_and_merge_layer,
)
from lora_merge_cert.eval import fake_eval_from_certificates
from lora_merge_cert.merge import synthetic_lora_pair


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LoRA Merge Certificate — Week-1 runner")
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--lora-mnli", default=None, help="HF Hub ID or local path for MNLI LoRA")
    p.add_argument("--lora-sst2", default=None, help="HF Hub ID or local path for SST-2 LoRA")
    p.add_argument("--subspace", choices=["A", "B"], default="A")
    p.add_argument("--theta-star-deg", type=float, default=DEFAULT_THETA_STAR_DEG)
    p.add_argument("--dry-run", action="store_true", help="Synthetic LoRAs; no Hub downloads")
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=12, help="Synthetic layer count for dry-run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _write_certificate_table(certs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer_name",
        "status",
        "theta_min_deg",
        "overlap",
        "n_shared",
        "rank1",
        "rank2",
        "thetas_deg",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in certs:
            row = c.to_dict()
            row["thetas_deg"] = json.dumps(row["thetas_deg"])
            w.writerow(row)

    # Also markdown for quick viewing
    md = path.with_suffix(".md")
    lines = [
        "| layer | status | θ_min (°) | overlap | n_shared |",
        "|---|---|---:|---:|---:|",
    ]
    for c in certs:
        lines.append(
            f"| {c.layer_name} | {c.status} | {c.theta_min_deg:.2f} | {c.overlap:.4f} | {c.n_shared} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scatter(certs, eval_summary: dict, path: Path):
    """x=θ_min, y=MNLI acc drop (fake or real). One point per layer (proxy)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        path.with_suffix(".txt").write_text(
            "matplotlib unavailable; skip scatter\n", encoding="utf-8"
        )
        return

    xs = [c.theta_min_deg for c in certs]
    # Proxy per-layer drop: FAIL → larger drop contribution
    drop_arith = eval_summary.get("mnli_drop_arithmetic", 0.05)
    ys = []
    for c in certs:
        if c.status == "FAIL":
            ys.append(drop_arith * (1.0 + (30.0 - c.theta_min_deg) / 60.0))
        else:
            ys.append(drop_arith * 0.15)

    colors = ["#d62728" if c.status == "FAIL" else "#2ca02c" for c in certs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=colors, s=60, edgecolors="k", linewidths=0.4)
    ax.axvline(30.0, color="gray", linestyle="--", label="θ★ = 30°")
    ax.set_xlabel("θ_min (degrees)")
    ax.set_ylabel("MNLI accuracy drop (proxy)")
    ax.set_title("Week-1: θ_min vs MNLI drop")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_dry(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # BERT-base style layer names
    layer_names = []
    for i in range(args.n_layers):
        for proj in ("query", "key", "value", "dense"):
            layer_names.append(f"encoder.layer.{i}.attention.self.{proj}")

    # Vary shared_dirs so some PASS / some FAIL
    certs = []
    rng = np.random.default_rng(args.seed)
    for li, name in enumerate(layer_names):
        shared = int(rng.integers(0, 4))  # 0 → likely PASS; 1–3 → FAIL
        B1, A1, B2, A2 = synthetic_lora_pair(
            m=64, n=64, r=args.rank, shared_dirs=shared, seed=args.seed + li
        )
        # Orthogonalize unique case for clean PASS when shared=0
        if shared == 0:
            # Make B2 nearly orthogonal to B1
            g = torch.Generator().manual_seed(args.seed + li + 999)
            Q1, _ = torch.linalg.qr(B1)
            noise = torch.randn(64, args.rank, generator=g)
            # Project out Q1
            noise = noise - Q1 @ (Q1.T @ noise)
            B2 = noise
            A2 = torch.randn(args.rank, 64, generator=g) * 0.1

        _dW, cert = certify_and_merge_layer(
            B1, A1, B2, A2,
            theta_star_deg=args.theta_star_deg,
            subspace=args.subspace,
            layer_name=name,
        )
        certs.append(cert)

    _write_certificate_table(certs, out / "certificate_table.csv")
    eval_summary = fake_eval_from_certificates(certs)
    _write_scatter(certs, eval_summary, out / "theta_vs_mnli_drop.png")

    n_pass = sum(1 for c in certs if c.status == "PASS")
    n_fail = sum(1 for c in certs if c.status == "FAIL")
    summary = {
        "mode": "dry-run",
        "base_model": args.base_model,
        "subspace": args.subspace,
        "theta_star_deg": args.theta_star_deg,
        "rank": args.rank,
        "n_layers_certified": len(certs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "eval": eval_summary,
        "arithmetic_mnli": eval_summary["mnli_arithmetic"],
        "certificate_mnli": eval_summary["mnli_certificate"],
        "note": "Synthetic LoRAs; accuracies are heuristic proxies (fake=True).",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_real(args) -> dict:
    """Load HF base + two PEFT adapters, merge, optionally eval."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.lora_mnli or not args.lora_sst2:
        raise SystemExit(
            "Real mode requires --lora-mnli and --lora-sst2 "
            "(Hub IDs or local paths). Use --dry-run for synthetic demo, "
            "or see scripts/train_lora_glue.py to train adapters."
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    from lora_merge_cert.merge import merge_lora_models

    device = torch.device(args.device)

    # Load two separately adapted copies (PEFT attaches adapters in-place)
    print(f"Loading base {args.base_model} + MNLI adapter {args.lora_mnli} ...")
    base1 = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
    model_mnli = PeftModel.from_pretrained(base1, args.lora_mnli)

    print(f"Loading base {args.base_model} + SST-2 adapter {args.lora_sst2} ...")
    base2 = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    # For merge we only need LoRA A/B weights; label count mismatch is OK for weight extract
    model_sst2 = PeftModel.from_pretrained(base2, args.lora_sst2)

    merged_deltas, certs = merge_lora_models(
        model_mnli,
        model_sst2,
        subspace=args.subspace,
        theta_star_deg=args.theta_star_deg,
        resvd_rank=2 * args.rank,
        use_certificate=True,
    )
    _write_certificate_table(certs, out / "certificate_table.csv")

    # Arithmetic-only deltas for comparison
    arith_deltas, _ = merge_lora_models(
        model_mnli, model_sst2,
        subspace=args.subspace,
        theta_star_deg=args.theta_star_deg,
        resvd_rank=2 * args.rank,
        use_certificate=False,
    )

    eval_results = {"fake": False, "available": False}
    try:
        from lora_merge_cert.eval import evaluate_glue
        from lora_merge_cert.merge import apply_deltas_to_base
        import copy

        tok = AutoTokenizer.from_pretrained(args.base_model)
        n_eval = 512  # CPU-friendly subset; still real GLUE metrics

        def _copy_classifier(src_peft, dst_base):
            src_clf = src_peft.base_model.model.classifier
            with torch.no_grad():
                dst_base.classifier.weight.copy_(src_clf.weight.data)
                dst_base.classifier.bias.copy_(src_clf.bias.data)

        def _build_merged(num_labels, clf_src, deltas):
            base = AutoModelForSequenceClassification.from_pretrained(
                args.base_model, num_labels=num_labels
            )
            _copy_classifier(clf_src, base)
            apply_deltas_to_base(base, deltas)
            return base.to(device)

        mnli_arith = _build_merged(3, model_mnli, arith_deltas)
        mnli_cert_m = _build_merged(3, model_mnli, merged_deltas)
        sst_arith = _build_merged(2, model_sst2, arith_deltas)
        sst_cert_m = _build_merged(2, model_sst2, merged_deltas)

        r_mnli_a = evaluate_glue(mnli_arith, tok, "mnli", device=device, num_samples=n_eval)
        r_mnli_c = evaluate_glue(mnli_cert_m, tok, "mnli", device=device, num_samples=n_eval)
        r_sst_a = evaluate_glue(sst_arith, tok, "sst2", device=device, num_samples=n_eval)
        r_sst_c = evaluate_glue(sst_cert_m, tok, "sst2", device=device, num_samples=n_eval)

        eval_results = {
            "fake": False,
            "available": True,
            "num_samples": n_eval,
            "mnli_arithmetic": r_mnli_a,
            "mnli_certificate": r_mnli_c,
            "sst2_arithmetic": r_sst_a,
            "sst2_certificate": r_sst_c,
            "headline": {
                "mnli_sum": r_mnli_a["accuracy"],
                "mnli_corrected": r_mnli_c["accuracy"],
                "sst2_sum": r_sst_a["accuracy"],
                "sst2_corrected": r_sst_c["accuracy"],
            },
        }
    except Exception as e:
        import traceback
        eval_results = {
            "fake": False,
            "available": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "note": "Datasets/eval skipped; certificate table still written.",
        }

    n_pass = sum(1 for c in certs if c.status == "PASS")
    n_fail = sum(1 for c in certs if c.status == "FAIL")
    summary = {
        "mode": "real",
        "base_model": args.base_model,
        "lora_mnli": args.lora_mnli,
        "lora_sst2": args.lora_sst2,
        "subspace": args.subspace,
        "theta_star_deg": args.theta_star_deg,
        "n_layers_certified": len(certs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_merged_deltas": len(merged_deltas),
        "eval": eval_results,
        "device": str(device),
    }
    if eval_results.get("headline"):
        summary.update({
            "mnli_sum": eval_results["headline"]["mnli_sum"],
            "mnli_corrected": eval_results["headline"]["mnli_corrected"],
            "sst2_sum": eval_results["headline"]["sst2_sum"],
            "sst2_corrected": eval_results["headline"]["sst2_corrected"],
        })
    # Scatter with fake proxy if no real per-layer drops
    fake = fake_eval_from_certificates(certs)
    _write_scatter(certs, fake, out / "theta_vs_mnli_drop.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    args = parse_args(argv)
    # Resolve output-dir relative to CWD (user expectation)
    args.output_dir = str(Path(args.output_dir).resolve())

    if args.dry_run:
        summary = run_dry(args)
    else:
        summary = run_real(args)

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
