"""Per-layer LoRA merge certificate: PASS/FAIL and closed-form correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import math
import torch

from .angles import build_shared_basis, overlap_frobenius, principal_angles
from .subspace import SubspaceDef, extract_basis


DEFAULT_THETA_STAR_DEG = 30.0


@dataclass
class LayerCertificate:
    """Certificate record for one linear layer."""

    layer_name: str
    theta_min_deg: float
    overlap: float
    n_shared: int
    status: str  # "PASS" or "FAIL"
    rank1: int
    rank2: int
    thetas_deg: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(rad, torch.Tensor):
        return rad * (180.0 / math.pi)
    return rad * (180.0 / math.pi)


def project_out(dW: torch.Tensor, Ustar: torch.Tensor) -> torch.Tensor:
    """ΔW_corr = (I - P) ΔW = ΔW - U★ (U★^T ΔW)."""
    return dW - Ustar @ (Ustar.T @ dW)


def certify_and_merge_layer(
    B1: torch.Tensor,
    A1: torch.Tensor,
    B2: torch.Tensor,
    A2: torch.Tensor,
    *,
    theta_star_deg: float = DEFAULT_THETA_STAR_DEG,
    subspace: SubspaceDef = "A",
    layer_name: str = "",
) -> tuple[torch.Tensor, LayerCertificate]:
    """Compute certificate and merged ΔW for one layer.

    METHOD §5–§7:
      PASS (θ_min ≥ θ★): ΔW = ΔW1 + ΔW2
      FAIL: ΔW2_corr = (I - P_S★) ΔW2,  ΔW = ΔW1 + ΔW2_corr
    """
    dW1 = B1 @ A1
    dW2 = B2 @ A2

    U1 = extract_basis(B1, A1, definition=subspace)
    U2 = extract_basis(B2, A2, definition=subspace)

    thetas, _sigmas, P, _Q = principal_angles(U1, U2)
    overlap = float(overlap_frobenius(U1, U2).item())
    theta_star = deg2rad(theta_star_deg)

    Ustar, idx = build_shared_basis(U1, P, thetas, theta_star)
    n_shared = int(idx.sum().item()) if idx.numel() else 0
    theta_min = float(thetas[0].item()) if thetas.numel() else math.pi / 2
    status = "FAIL" if n_shared > 0 else "PASS"

    if status == "FAIL":
        assert Ustar is not None
        dW2_corr = project_out(dW2, Ustar)
        dW_merge = dW1 + dW2_corr
    else:
        dW_merge = dW1 + dW2

    cert = LayerCertificate(
        layer_name=layer_name,
        theta_min_deg=float(rad2deg(theta_min)),
        overlap=overlap,
        n_shared=n_shared,
        status=status,
        rank1=int(U1.shape[1]),
        rank2=int(U2.shape[1]),
        thetas_deg=[float(x) for x in rad2deg(thetas).tolist()],
    )
    return dW_merge, cert


def arithmetic_sum(
    B1: torch.Tensor, A1: torch.Tensor, B2: torch.Tensor, A2: torch.Tensor
) -> torch.Tensor:
    """Plain arithmetic merge ΔW1 + ΔW2 (no certificate correction)."""
    return B1 @ A1 + B2 @ A2
