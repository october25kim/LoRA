"""Subspace extraction for LoRA adapters (definitions A and B)."""

from __future__ import annotations

from typing import Literal

import torch

SubspaceDef = Literal["A", "B"]


def orth(B: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for column space of B via thin QR (definition A).

    METHOD §2 def A: U_i = orth(B_i)
    """
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D, got shape {tuple(B.shape)}")
    # Drop near-zero columns after QR if rank-deficient
    Q, R = torch.linalg.qr(B, mode="reduced")
    # Keep columns whose diagonal R entries are significant
    diag = torch.abs(torch.diag(R))
    tol = diag.max() * B.shape[0] * torch.finfo(B.dtype).eps if diag.numel() else 0.0
    keep = diag > tol
    if keep.any():
        return Q[:, keep]
    return Q


def left_singular_vectors(dW: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    """Left singular vectors of ΔW = B @ A (definition B).

    METHOD §2 def B: ΔW_i = Ũ_i Σ_i Ṽ_i^T, U_i = Ũ_i
    """
    if dW.ndim != 2:
        raise ValueError(f"dW must be 2-D, got shape {tuple(dW.shape)}")
    U, S, _ = torch.linalg.svd(dW, full_matrices=False)
    if rank is None:
        # Numerical rank
        tol = S.max() * max(dW.shape) * torch.finfo(dW.dtype).eps if S.numel() else 0.0
        rank = int((S > tol).sum().item())
        rank = max(rank, 1) if S.numel() else 0
    rank = min(rank, U.shape[1])
    return U[:, :rank]


def extract_basis(
    B: torch.Tensor,
    A: torch.Tensor | None = None,
    definition: SubspaceDef = "A",
) -> torch.Tensor:
    """Extract orthonormal subspace U for a LoRA (B, A) pair.

    Args:
        B: (m, r) LoRA B matrix
        A: (r, n) LoRA A matrix (required for definition B)
        definition: "A" → orth(B); "B" → left SVD of B@A
    """
    definition = definition.upper()  # type: ignore[assignment]
    if definition == "A":
        return orth(B)
    if definition == "B":
        if A is None:
            raise ValueError("definition B requires A matrix")
        dW = B @ A
        return left_singular_vectors(dW, rank=B.shape[1])
    raise ValueError(f"Unknown subspace definition: {definition!r}; use 'A' or 'B'")
