#!/usr/bin/env python3
from __future__ import annotations
import json, csv
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from lora_merge_cert.merge import merge_lora_models, apply_deltas_to_base, collect_lora_pairs
from lora_merge_cert.eval import evaluate_glue

def ties_lite_sum(d1, d2, densify=0.7):
    out = {}
    for k in set(d1) | set(d2):
        a, b = d1.get(k), d2.get(k)
        if a is None: t = b.clone()
        elif b is None: t = a.clone()
        else:
            s = torch.sign(a) + torch.sign(b)
            s = torch.where(s == 0, torch.sign(a + b), torch.sign(s))
            t = torch.where(torch.sign(a) == s, a, torch.zeros_like(a)) + torch.where(torch.sign(b) == s, b, torch.zeros_like(b))
        flat = t.abs().flatten()
        if flat.numel() == 0:
            out[k] = t; continue
        kth = int(max(1, (1.0 - densify) * flat.numel()))
        if kth >= flat.numel():
            out[k] = t; continue
        thresh = torch.kthvalue(flat, kth).values
        out[k] = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
    return out

def build_model(base_id, nlab, clf_src, deltas, device):
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=nlab)
    src = clf_src.base_model.model.classifier
    with torch.no_grad():
        base.classifier.weight.copy_(src.weight.data)
        base.classifier.bias.copy_(src.bias.data)
    apply_deltas_to_base(base, deltas)
    return base.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_id = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(base_id)
    out_dir = Path("artifacts/paper"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [
        ("hub_mnli_sst2", "prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001",
         "prateeky2806/bert-base-uncased-sst2-lora-epochs-2-lr-0.0005", 3, 2, "mnli", True),
        ("seed_hubish", "adapters/mnli_s7_hubish", "adapters/mnli_s42_hubish", 3, 3, "mnli", False),
    ]
    for name, l1, l2, n1, n2, task, do_ties in configs:
        print("LOAD", name, flush=True)
        m1 = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=n1), l1)
        m2 = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=n2), l2)
        for subspace in ("A", "B"):
            print("CERT", name, subspace, flush=True)
            cert_d, certs = merge_lora_models(m1, m2, subspace=subspace, theta_star_deg=30.0, use_certificate=True)
            arith_d, _ = merge_lora_models(m1, m2, subspace=subspace, theta_star_deg=30.0, use_certificate=False)
            ma = build_model(base_id, n1, m1, arith_d, device)
            mc = build_model(base_id, n1, m1, cert_d, device)
            ra = evaluate_glue(ma, tok, task, device=device, num_samples=None)
            rc = evaluate_glue(mc, tok, task, device=device, num_samples=None)
            ths = [c.theta_min_deg for c in certs]
            n_fail = sum(1 for c in certs if c.status == "FAIL")
            rec = {"pair": name, "method": f"cert_{subspace}", "subspace": subspace,
                   "n_fail": n_fail, "theta_min": min(ths), "theta_median": sorted(ths)[len(ths)//2],
                   "retain_sum": ra["accuracy"], "retain_corr": rc["accuracy"], "n": ra.get("n")}
            print(rec, flush=True); rows.append(rec)
            dest = out_dir / f"{name}_{subspace}_th30"; dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "certificate_table.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["layer_name","status","theta_min_deg","overlap","n_shared"])
                w.writeheader()
                for c in certs:
                    w.writerow({"layer_name": c.layer_name, "status": c.status,
                                "theta_min_deg": c.theta_min_deg, "overlap": c.overlap, "n_shared": c.n_shared})
            (dest / "summary.json").write_text(json.dumps({"arith": ra, "cert": rc, **rec}, indent=2))
            del ma, mc; torch.cuda.empty_cache()
            if do_ties and subspace == "A":
                print("TIES-LITE", name, flush=True)
                pairs = collect_lora_pairs(m1, m2)
                dd1 = {nm: (B1 @ A1).detach().cpu() for nm, B1, A1, B2, A2 in pairs}
                dd2 = {nm: (B2 @ A2).detach().cpu() for nm, B1, A1, B2, A2 in pairs}
                ties_d = ties_lite_sum(dd1, dd2, 0.7)
                mt = build_model(base_id, n1, m1, ties_d, device)
                try: rt = evaluate_glue(mt, tok, task, device=device, num_samples=None)
                except Exception as e: rt = {"accuracy": None, "error": str(e)}
                cert_trim = {}
                for k, t in cert_d.items():
                    flat = t.abs().flatten(); kth = int(max(1, 0.3 * flat.numel()))
                    thresh = torch.kthvalue(flat, min(kth, flat.numel())).values
                    cert_trim[k] = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
                mct = build_model(base_id, n1, m1, cert_trim, device)
                rct = evaluate_glue(mct, tok, task, device=device, num_samples=None)
                for method, rr in [("ties_lite", rt), ("cert_then_trim", rct), ("arithmetic", ra)]:
                    rows.append({"pair": name, "method": method, "subspace": "A",
                                 "retain_acc": rr.get("accuracy") if isinstance(rr, dict) else None,
                                 "n": rr.get("n") if isinstance(rr, dict) else None})
                    print({"method": method, "acc": rr}, flush=True)
                del mt, mct; torch.cuda.empty_cache()
        del m1, m2; torch.cuda.empty_cache()
    (out_dir / "fullval_summary.json").write_text(json.dumps(rows, indent=2))
    lines = ["| pair | method | subspace | n_fail | retain | n |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.get('pair')} | {r.get('method')} | {r.get('subspace')} | {r.get('n_fail')} | {r.get('retain_corr', r.get('retain_acc', r.get('retain_sum')))} | {r.get('n')} |")
    (out_dir / "TABLE_REQUIRED.md").write_text("\n".join(lines) + "\n")
    print("DONE", len(rows), flush=True)

if __name__ == "__main__":
    main()
