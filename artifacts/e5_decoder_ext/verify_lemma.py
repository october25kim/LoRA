#!/usr/bin/env python3
"""Numerical check of the hard-gate lemma (paper/LEMMA_PROJECTION.md) on a pair of PEFT LoRA adapters.

For every LoRA layer it rebuilds the certificate exactly as lora_merge_cert does
(U_i = orth(B_i) [def A] or top-r left singular vectors of B_i A_i [def B]; principal angles from svd(U1^T U2);
shared basis U = QR(U1 P[:, theta < theta*])), and on FAIL layers it reports

  X_i = U^T dW_i                  (k x n; dW_i = (alpha/r) B_i A_i)
  c   = <X1, X2>_F / ||X1||_F^2   (matched coefficient; lemma: gate == TA with coefficient mu = 1/(1+c) on span(U) iff R = 0)
  cos = <X1, X2>_F / (||X1|| ||X2||)   (collinearity of the shared-direction blocks; exact equivalence iff |cos| = 1)
  eps = ||R||_F / ||(1+c) X1||_F, R = X2 - c X1   (relative deviation from a pure coefficient change on span(U))
  per-direction c_j, cos_j for each shared direction u_j (rows of X_i)
  energy: ||X2||^2/||dW2||^2 (share of dW2 removed), ||X2|| / ||dW1+dW2|| (size of the edit relative to the merged update)
  identity residuals: ||M_gate - (M_sum - U U^T dW2)||, ||U U^T M_gate - U X1||, ||(I-UU^T)(M_gate - M_sum)||
  signs: fraction of entries of the merged layer whose sign the gate flips; TIES-style entrywise sign-conflict rate
         between dW1 and dW2 before vs. between dW1 and (I-UU^T)dW2 after the gate.

Usage (on ubuntu-4070, repo root):
  python verify_lemma.py adapters/mnli_s7_hubish adapters/mnli_s42_hubish --out seed_pair_lemma.json
  python verify_lemma.py A_DIR B_DIR --subspace B --theta-star 30 --all-layers
  python verify_lemma.py --selftest          # synthetic check of the algebra, no adapters needed
Requires torch and safetensors (or a PyTorch adapter_model.bin). CPU is fine (73 layers of 768x768/3072).
"""
from __future__ import annotations

import argparse, json, math, re, sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
KEY_RE = re.compile(r"^(?P<layer>.+)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")


def load_adapter(d: Path):
    cfg = json.loads((d / "adapter_config.json").read_text())
    if (d / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        sd = load_file(str(d / "adapter_model.safetensors"))
    else:
        sd = torch.load(d / "adapter_model.bin", map_location="cpu")
    layers = {}
    for k, v in sd.items():
        m = KEY_RE.match(k)
        if m:
            layers.setdefault(m.group("layer"), {})[m.group("ab")] = v.to(torch.float64)
    r = int(cfg["r"]); alpha = float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    return layers, scale, cfg


def orth(B):                                   # identical rule to lora_merge_cert.subspace.orth
    Q, R = torch.linalg.qr(B, mode="reduced")
    diag = torch.abs(torch.diag(R))
    tol = diag.max() * B.shape[0] * torch.finfo(B.dtype).eps if diag.numel() else 0.0
    keep = diag > tol
    return Q[:, keep] if keep.any() else Q


def left_sv(dW, r):
    U, S, _ = torch.linalg.svd(dW, full_matrices=False)
    return U[:, :r]


def lemma_stats(dW1, dW2, U):
    """All quantities of the lemma for one layer with shared orthonormal basis U (m x k)."""
    X1 = U.T @ dW1; X2 = U.T @ dW2
    n1, n2 = X1.norm(), X2.norm()
    ip = (X1 * X2).sum()
    c = (ip / n1 ** 2).item() if n1 > 0 else float("nan")
    cos = (ip / (n1 * n2)).item() if n1 > 0 and n2 > 0 else float("nan")
    R = X2 - c * X1 if n1 > 0 else X2
    eps = (R.norm() / ((1 + c) * X1).norm()).item() if n1 > 0 and abs(1 + c) > 0 else float("nan")
    Msum = dW1 + dW2; PU = U @ U.T
    Mgate = dW1 + dW2 - PU @ dW2
    id1 = (Mgate - (Msum - U @ X2)).norm().item()
    id2 = (PU @ Mgate - U @ X1).norm().item()
    id3 = ((Mgate - Msum) - PU @ (Mgate - Msum)).norm().item()
    per_dir = []
    for j in range(U.shape[1]):
        a, b = X1[j], X2[j]
        na, nb = a.norm(), b.norm()
        per_dir.append({"c_j": ((a @ b) / na ** 2).item() if na > 0 else None,
                        "cos_j": ((a @ b) / (na * nb)).item() if na > 0 and nb > 0 else None,
                        "norm_ratio_X2_over_X1": (nb / na).item() if na > 0 else None})
    nz = (Msum != 0) & (Mgate != 0)
    flip = ((torch.sign(Msum) != torch.sign(Mgate)) & nz).double().mean().item()
    dW2g = dW2 - PU @ dW2
    def conflict(a, b):
        m = (a != 0) & (b != 0)
        return ((torch.sign(a) != torch.sign(b)) & m).double().sum().item() / max(1, m.sum().item())
    mu = 1.0 / (1.0 + c) if abs(1 + c) > 1e-12 else float("nan")
    regime = ("aligned (c>0): coefficient reduction mu in (0,1)" if c > 0 else
              "partially cancelling (-1<c<0): gate AMPLIFIES the shared block (mu>1)" if -1 < c < 0 else
              "reversing (c<-1): gate flips the merged shared block toward task 1 (mu<0)" if c < -1 else "degenerate")
    return {"k": int(U.shape[1]), "c": c, "cos_X1_X2": cos, "mu_equiv": mu, "eps_rel_residual": eps, "regime": regime,
            "norm_X1": n1.item(), "norm_X2": n2.item(),
            "share_dW2_energy_removed": (n2 ** 2 / dW2.norm() ** 2).item(),
            "share_dW1_energy_in_U": (n1 ** 2 / dW1.norm() ** 2).item(),
            "edit_size_rel_to_merge": (n2 / Msum.norm()).item(),
            "identity_residuals": [id1, id2, id3], "per_direction": per_dir,
            "frac_entries_sign_flipped_by_gate": flip,
            "entrywise_sign_conflict_dW1_vs_dW2": conflict(dW1, dW2),
            "entrywise_sign_conflict_dW1_vs_gated_dW2": conflict(dW1, dW2g)}


def run(a_dir, b_dir, theta_star, subspace, all_layers):
    L1, s1, _ = load_adapter(Path(a_dir)); L2, s2, _ = load_adapter(Path(b_dir))
    names = [n for n in L1 if n in L2 and "A" in L1[n] and "B" in L1[n]]
    cstar = math.cos(math.radians(theta_star))
    out = {"adapter_1": str(a_dir), "adapter_2": str(b_dir), "theta_star_deg": theta_star, "subspace": subspace,
           "n_layers": len(names), "layers": []}
    for n in sorted(names):
        B1, A1, B2, A2 = L1[n]["B"], L1[n]["A"], L2[n]["B"], L2[n]["A"]
        dW1 = s1 * B1 @ A1; dW2 = s2 * B2 @ A2
        U1 = orth(B1) if subspace == "A" else left_sv(dW1, B1.shape[1])
        U2 = orth(B2) if subspace == "A" else left_sv(dW2, B2.shape[1])
        P, S, _ = torch.linalg.svd(U1.T @ U2, full_matrices=False)
        S = S.clamp(0, 1); th = torch.rad2deg(torch.arccos(S))
        idx = S > cstar
        rec = {"layer": n, "theta_deg": [round(x, 3) for x in th.tolist()], "theta_min_deg": th.min().item(),
               "status": "FAIL" if bool(idx.any()) else "PASS"}
        if idx.any():
            U, _ = torch.linalg.qr(U1 @ P[:, idx], mode="reduced")
            rec.update(lemma_stats(dW1, dW2, U))
        elif all_layers:                           # context: top principal direction even though the layer passes
            U, _ = torch.linalg.qr(U1 @ P[:, :1], mode="reduced")
            rec.update({"top_direction_only": True, **lemma_stats(dW1, dW2, U)})
        out["layers"].append(rec)
    fails = [r for r in out["layers"] if r["status"] == "FAIL"]
    out["n_fail"] = len(fails)
    out["summary_fail_layers"] = [{k: r[k] for k in ("layer", "theta_min_deg", "k", "c", "cos_X1_X2", "mu_equiv", "eps_rel_residual",
                                                     "regime", "edit_size_rel_to_merge", "frac_entries_sign_flipped_by_gate",
                                                     "entrywise_sign_conflict_dW1_vs_dW2", "entrywise_sign_conflict_dW1_vs_gated_dW2")}
                                  for r in fails]
    return out


def selftest():
    g = torch.Generator().manual_seed(0)
    m, n, k = 64, 48, 2
    U, _ = torch.linalg.qr(torch.randn(m, k, generator=g))
    P = torch.eye(m) - U @ U.T
    dW1 = torch.randn(m, n, generator=g)
    res = {}
    for c in (0.8, -0.4, -2.5):                    # collinear shared blocks: exact equivalence with mu = 1/(1+c)
        dW2 = P @ torch.randn(m, n, generator=g) + c * (U @ (U.T @ dW1))
        Mgate = dW1 + P @ dW2; Msum = dW1 + dW2
        mu = 1 / (1 + c)
        res[f"collinear_c={c}"] = {"max|gate - [P Msum + mu UU^T Msum]|": (Mgate - (P @ Msum + mu * U @ U.T @ Msum)).abs().max().item(),
                                   **{k_: v for k_, v in lemma_stats(dW1, dW2, U).items() if k_ in ("c", "cos_X1_X2", "eps_rel_residual", "regime")}}
    dW2 = torch.randn(m, n, generator=g)           # generic case: residual term -U R/(1+c) is exactly the gap
    st = lemma_stats(dW1, dW2, U); c = st["c"]
    X1 = U.T @ dW1; X2 = U.T @ dW2; R = X2 - c * X1
    Mgate = dW1 + P @ dW2; Msum = dW1 + dW2
    gap = Mgate - (P @ Msum + U @ U.T @ Msum / (1 + c))
    res["generic"] = {"c": c, "cos": st["cos_X1_X2"], "max|gap + U R/(1+c)|": (gap + U @ R / (1 + c)).abs().max().item(),
                      "eps": st["eps_rel_residual"]}
    # sign-blindness: flipping dW2 leaves the gate decision/U unchanged but flips c
    res["sign_blind_c_pair"] = [lemma_stats(dW1, dW1 * 0.7, U)["c"], lemma_stats(dW1, -dW1 * 0.7, U)["c"]]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter1", nargs="?"); ap.add_argument("adapter2", nargs="?")
    ap.add_argument("--theta-star", type=float, default=30.0); ap.add_argument("--subspace", choices=["A", "B"], default="A")
    ap.add_argument("--all-layers", action="store_true", help="also report the top principal direction of PASS layers")
    ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=1)); return
    if not (a.adapter1 and a.adapter2):
        ap.error("two adapter directories required (or --selftest)")
    res = run(a.adapter1, a.adapter2, a.theta_star, a.subspace, a.all_layers)
    s = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(json.dumps({"n_layers": res["n_layers"], "n_fail": res["n_fail"], "fail_layers": res["summary_fail_layers"]}, indent=1))


if __name__ == "__main__":
    main()
