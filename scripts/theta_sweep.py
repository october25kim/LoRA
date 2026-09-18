#!/usr/bin/env python3
"""Global θ★ sensitivity sweep for one or more LoRA pairs (validation only)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from lora_merge_cert.merge import merge_lora_models, apply_deltas_to_base
from lora_merge_cert.eval import evaluate_glue


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--lora1", required=True)
    p.add_argument("--lora2", required=True)
    p.add_argument("--n-labels1", type=int, default=3)
    p.add_argument("--n-labels2", type=int, default=2)
    p.add_argument("--task1", default="mnli")
    p.add_argument("--task2", default="sst2")
    p.add_argument("--pair-name", required=True)
    p.add_argument("--subspace", default="A", choices=["A", "B"])
    p.add_argument("--thetas", default="20,30,45,60,75")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="artifacts/theta_sweep")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def _copy_classifier(src_peft, dst_base):
    src_clf = src_peft.base_model.model.classifier
    with torch.no_grad():
        dst_base.classifier.weight.copy_(src_clf.weight.data)
        dst_base.classifier.bias.copy_(src_clf.bias.data)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    thetas = [float(x) for x in args.thetas.split(",")]

    print(f"Loading pair {args.pair_name}: {args.lora1} + {args.lora2}")
    m1 = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=args.n_labels1),
        args.lora1,
    )
    m2 = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=args.n_labels2),
        args.lora2,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)

    # Arithmetic deltas once (independent of θ★)
    arith_deltas, _ = merge_lora_models(
        m1, m2, subspace=args.subspace, theta_star_deg=30.0,
        resvd_rank=2 * args.rank, use_certificate=False,
    )

    rows = []
    per_theta_certs = {}

    # Eval arithmetic once
    t1_sum = t2_sum = None
    if not args.skip_eval:
        def build(num_labels, clf_src, deltas):
            base = AutoModelForSequenceClassification.from_pretrained(
                args.base_model, num_labels=num_labels
            )
            _copy_classifier(clf_src, base)
            apply_deltas_to_base(base, deltas)
            return base.to(device)

        print("Evaluating arithmetic sum ...")
        m_t1_a = build(args.n_labels1, m1, arith_deltas)
        m_t2_a = build(args.n_labels2, m2, arith_deltas)
        r1a = evaluate_glue(m_t1_a, tok, args.task1, device=device, num_samples=args.num_samples)
        r2a = evaluate_glue(m_t2_a, tok, args.task2, device=device, num_samples=args.num_samples)
        t1_sum, t2_sum = r1a["accuracy"], r2a["accuracy"]
        del m_t1_a, m_t2_a
        print(f"  {args.task1}_sum={t1_sum:.4f} {args.task2}_sum={t2_sum:.4f}")

    for th in thetas:
        print(f"\n=== θ★={th}° ===")
        merged, certs = merge_lora_models(
            m1, m2, subspace=args.subspace, theta_star_deg=th,
            resvd_rank=2 * args.rank, use_certificate=True,
        )
        thetas_min = [c.theta_min_deg for c in certs]
        n_pass = sum(1 for c in certs if c.status == "PASS")
        n_fail = sum(1 for c in certs if c.status == "FAIL")
        amin = float(min(thetas_min)) if thetas_min else float("nan")
        amean = float(np.mean(thetas_min)) if thetas_min else float("nan")

        t1_c = t2_c = None
        if not args.skip_eval:
            m_t1_c = build(args.n_labels1, m1, merged)
            m_t2_c = build(args.n_labels2, m2, merged)
            r1c = evaluate_glue(m_t1_c, tok, args.task1, device=device, num_samples=args.num_samples)
            r2c = evaluate_glue(m_t2_c, tok, args.task2, device=device, num_samples=args.num_samples)
            t1_c, t2_c = r1c["accuracy"], r2c["accuracy"]
            del m_t1_c, m_t2_c
            print(f"  pass={n_pass} fail={n_fail} minθ={amin:.2f} "
                  f"{args.task1}_corr={t1_c:.4f} {args.task2}_corr={t2_c:.4f}")
        else:
            print(f"  pass={n_pass} fail={n_fail} minθ={amin:.2f} (eval skipped)")

        gap = None
        if t1_sum is not None and t1_c is not None:
            gap = abs(t1_c - t1_sum) > 1e-9 or abs(t2_c - t2_sum) > 1e-9

        row = {
            "pair": args.pair_name,
            "subspace": args.subspace,
            "theta_star_deg": th,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "min_theta_min": amin,
            "mean_theta_min": amean,
            f"{args.task1}_sum": t1_sum,
            f"{args.task1}_corrected": t1_c,
            f"{args.task2}_sum": t2_sum,
            f"{args.task2}_corrected": t2_c,
            "angle_metric_gap": gap,
            "lora1": args.lora1,
            "lora2": args.lora2,
            "num_samples": args.num_samples,
        }
        rows.append(row)
        per_theta_certs[str(th)] = {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "min_theta_min": amin,
            "lowest5": sorted(thetas_min)[:5],
        }

        # Write cert table for this θ
        pair_dir = out / args.pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        cert_path = pair_dir / f"certificate_theta{int(th)}.csv"
        with cert_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["layer_name", "status", "theta_min_deg", "overlap", "n_shared"],
            )
            w.writeheader()
            for c in certs:
                w.writerow({
                    "layer_name": c.layer_name,
                    "status": c.status,
                    "theta_min_deg": c.theta_min_deg,
                    "overlap": c.overlap,
                    "n_shared": c.n_shared,
                })

    # Append/write CSV
    csv_path = out / "summary.csv"
    fieldnames = list(rows[0].keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    js_path = out / f"{args.pair_name}_sweep.json"
    payload = {
        "pair": args.pair_name,
        "lora1": args.lora1,
        "lora2": args.lora2,
        "subspace": args.subspace,
        "rows": rows,
        "per_theta": per_theta_certs,
        "any_angle_metric_gap": any(r.get("angle_metric_gap") for r in rows),
    }
    js_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {csv_path} and {js_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
