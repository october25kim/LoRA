"""Walk PEFT LoRA modules, apply certificate merge, optional re-SVD to rank 2r."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .certificate import (
    DEFAULT_THETA_STAR_DEG,
    LayerCertificate,
    arithmetic_sum,
    certify_and_merge_layer,
)
from .subspace import SubspaceDef


def _get_lora_AB(module: nn.Module) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract (B, A) from a PEFT LoRA linear module if present.

    PEFT stores:
      lora_A: (r, in)   — our A
      lora_B: (out, r)  — our B
    Scaling: ΔW = (B @ A) * (alpha / r)  — we fold scale into B for merge math.
    """
    # Common PEFT attribute patterns
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None

    # Adapter name: default "default"
    lora_A_dict = module.lora_A
    lora_B_dict = module.lora_B
    if isinstance(lora_A_dict, nn.ModuleDict):
        names = list(lora_A_dict.keys())
        if not names:
            return None
        name = names[0]
        A_mod = lora_A_dict[name]
        B_mod = lora_B_dict[name]
    else:
        A_mod = lora_A_dict
        B_mod = lora_B_dict

    A = A_mod.weight.detach().float()  # (r, in)
    B = B_mod.weight.detach().float()  # (out, r)

    # Apply PEFT scaling if available
    scaling = 1.0
    if hasattr(module, "scaling"):
        sc = module.scaling
        if isinstance(sc, dict):
            scaling = float(next(iter(sc.values())))
        else:
            scaling = float(sc)
    elif hasattr(module, "lora_alpha") and hasattr(module, "r"):
        r = module.r
        if isinstance(r, dict):
            r = next(iter(r.values()))
        alpha = module.lora_alpha
        if isinstance(alpha, dict):
            alpha = next(iter(alpha.values()))
        scaling = float(alpha) / float(r)

    # Fold sqrt(scaling) into both, or scaling into B: B_eff = B * scaling
    B = B * scaling
    return B, A


def collect_lora_pairs(
    model1: nn.Module,
    model2: nn.Module,
) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect matching LoRA (B1,A1,B2,A2) pairs by module name."""
    map1: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    map2: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for name, mod in model1.named_modules():
        ab = _get_lora_AB(mod)
        if ab is not None:
            map1[name] = ab

    for name, mod in model2.named_modules():
        ab = _get_lora_AB(mod)
        if ab is not None:
            map2[name] = ab

    common = sorted(set(map1) & set(map2))
    pairs = []
    for name in common:
        B1, A1 = map1[name]
        B2, A2 = map2[name]
        pairs.append((name, B1, A1, B2, A2))
    return pairs


def resvd_to_rank(dW: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factor ΔW ≈ B @ A with rank ≤ `rank` via truncated SVD.

    Returns B (m, k), A (k, n) with k = min(rank, rank(dW)).
    """
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    k = min(rank, S.numel())
    # Drop near-zero singular values
    tol = S.max() * max(dW.shape) * torch.finfo(dW.dtype).eps if S.numel() else 0.0
    k_eff = min(k, int((S > tol).sum().item()) or 1)
    S_sqrt = torch.sqrt(S[:k_eff])
    B = U[:, :k_eff] * S_sqrt.unsqueeze(0)
    A = S_sqrt.unsqueeze(1) * Vh[:k_eff, :]
    return B, A


def merge_lora_models(
    model1: nn.Module,
    model2: nn.Module,
    *,
    subspace: SubspaceDef = "A",
    theta_star_deg: float = DEFAULT_THETA_STAR_DEG,
    resvd_rank: Optional[int] = None,
    use_certificate: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[LayerCertificate]]:
    """Merge two PEFT-wrapped models layer-wise.

    Returns:
        merged_deltas: {layer_name: ΔW_merged}
        certificates: list of LayerCertificate
    """
    pairs = collect_lora_pairs(model1, model2)
    merged: Dict[str, torch.Tensor] = {}
    certs: List[LayerCertificate] = []

    for name, B1, A1, B2, A2 in pairs:
        if use_certificate:
            dW, cert = certify_and_merge_layer(
                B1, A1, B2, A2,
                theta_star_deg=theta_star_deg,
                subspace=subspace,
                layer_name=name,
            )
        else:
            dW = arithmetic_sum(B1, A1, B2, A2)
            from .certificate import LayerCertificate as LC
            cert = LC(
                layer_name=name,
                theta_min_deg=float("nan"),
                overlap=float("nan"),
                n_shared=0,
                status="ARITH",
                rank1=B1.shape[1],
                rank2=B2.shape[1],
                thetas_deg=[],
            )

        if resvd_rank is not None:
            B_m, A_m = resvd_to_rank(dW, resvd_rank)
            dW = B_m @ A_m

        merged[name] = dW
        certs.append(cert)

    return merged, certs


def apply_deltas_to_base(
    base_model: nn.Module,
    deltas: Dict[str, torch.Tensor],
    *,
    name_map: Optional[Dict[str, str]] = None,
) -> nn.Module:
    """Add merged ΔW onto matching linear weight modules of a base model.

    `deltas` keys are PEFT module names (e.g. '...query'). We try to find
    the corresponding `.weight` leaf on the base model.
    """
    name_map = name_map or {}
    named = dict(base_model.named_modules())

    for lora_name, dW in deltas.items():
        target_name = name_map.get(lora_name, lora_name)
        # Strip peft wrapper suffixes if present
        candidates = [
            target_name,
            target_name.replace(".base_layer", ""),
            target_name.replace("base_model.model.", ""),
            target_name.replace("base_model.", ""),
        ]
        # Also try without trailing adapter path quirks
        mod = None
        for c in candidates:
            if c in named:
                mod = named[c]
                break
        if mod is None or not hasattr(mod, "weight"):
            continue
        w = mod.weight
        if w.shape != dW.shape:
            # Might be transposed convention — skip safely
            continue
        with torch.no_grad():
            w.add_(dW.to(device=w.device, dtype=w.dtype))
    return base_model


def synthetic_lora_pair(
    m: int = 64,
    n: int = 64,
    r: int = 8,
    shared_dirs: int = 2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic LoRA A/B with controllable shared column directions.

    Used by dry-run and unit tests. Task-1 and task-2 share `shared_dirs`
    columns in B (near-identical), plus independent directions.
    """
    g = torch.Generator().manual_seed(seed)
    # Shared columns
    B_shared = torch.randn(m, shared_dirs, generator=g)
    B1_unique = torch.randn(m, r - shared_dirs, generator=g)
    B2_unique = torch.randn(m, r - shared_dirs, generator=g)
    # Small perturbation on shared for task 2 so angles are small but nonzero
    noise = 0.05 * torch.randn(m, shared_dirs, generator=g)
    B1 = torch.cat([B_shared, B1_unique], dim=1)
    B2 = torch.cat([B_shared + noise, B2_unique], dim=1)
    A1 = torch.randn(r, n, generator=g) * 0.1
    A2 = torch.randn(r, n, generator=g) * 0.1
    return B1, A1, B2, A2
