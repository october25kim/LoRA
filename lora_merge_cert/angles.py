"""Principal angles between LoRA subspaces and shared-direction construction."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def principal_angles(
    U1: torch.Tensor,
    U2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute principal angles between column spaces of U1 and U2.

    From METHOD §3:
        U1^T U2 = P Σ Q^T,  cos θ_k = σ_k,  θ_k = arccos(σ_k) ∈ [0, π/2]
        overlap = ||U1^T U2||_F^2

    Args:
        U1, U2: orthonormal bases, shape (m, r).

    Returns:
        thetas: (min(r1,r2),) angles in radians, ascending (θ_min first)
        sigmas: singular values clipped to [0, 1]
        P: left singular vectors of U1.T @ U2
        Q: right singular vectors of U1.T @ U2
    """
    M = U1.T @ U2  # (r1, r2)
    # Full SVD for canonical vectors
    P, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Q = Vh.T
    sigmas = torch.clamp(S, 0.0, 1.0)
    thetas = torch.arccos(sigmas)
    return thetas, sigmas, P, Q


def overlap_frobenius(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """overlap = ||U1^T U2||_F^2 (METHOD §3)."""
    M = U1.T @ U2
    return torch.sum(M * M)


def build_shared_basis(
    U1: torch.Tensor,
    P: torch.Tensor,
    thetas: torch.Tensor,
    theta_star: float,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """Build U★ = QR(U1 @ P[:, idx]) for directions with θ_k < θ★.

    From METHOD §4 / §7:
        S★ = span{ u_k^(1) : θ_k < θ★ }
        u_k^(1) = U1 p_k
        U★ from selected canonical vectors then QR

    Returns:
        Ustar: (m, k) orthonormal basis, or None if no shared directions
        idx: boolean mask over principal angles (True = shared / FAIL direction)
    """
    idx = thetas < theta_star
    if not bool(idx.any()):
        return None, idx

    # Canonical vectors in task-1 basis: U1 @ P[:, idx]
    U_cand = U1 @ P[:, idx]
    # Re-orthonormalize via thin QR
    Q, _ = torch.linalg.qr(U_cand, mode="reduced")
    return Q, idx


def select_shared_indices(thetas: torch.Tensor, theta_star: float) -> torch.Tensor:
    """Boolean mask: True where θ_k < θ★ (shared / interference directions)."""
    return thetas < theta_star
