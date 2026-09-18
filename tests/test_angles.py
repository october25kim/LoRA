"""Unit tests for principal angles and shared-basis construction."""

from __future__ import annotations

import math

import torch

from lora_merge_cert.angles import (
    build_shared_basis,
    overlap_frobenius,
    principal_angles,
)
from lora_merge_cert.subspace import orth


def test_identical_subspaces_zero_angles():
    g = torch.Generator().manual_seed(0)
    B = torch.randn(32, 4, generator=g)
    U = orth(B)
    thetas, sigmas, P, Q = principal_angles(U, U)
    assert torch.allclose(sigmas, torch.ones_like(sigmas), atol=1e-5)
    assert torch.allclose(thetas, torch.zeros_like(thetas), atol=1e-3)
    ov = overlap_frobenius(U, U)
    assert abs(float(ov) - 4.0) < 1e-4


def test_orthogonal_subspaces_right_angle():
    g = torch.Generator().manual_seed(1)
    M = torch.randn(64, 8, generator=g)
    Qfull, _ = torch.linalg.qr(M)
    U1 = Qfull[:, :4]
    U2 = Qfull[:, 4:8]
    thetas, sigmas, _, _ = principal_angles(U1, U2)
    assert torch.allclose(sigmas, torch.zeros_like(sigmas), atol=1e-5)
    expected = math.pi / 2
    assert torch.allclose(thetas, torch.full_like(thetas, expected), atol=1e-4)
    ov = overlap_frobenius(U1, U2)
    assert float(ov) < 1e-8


def test_build_shared_basis_selects_small_angles():
    g = torch.Generator().manual_seed(2)
    M = torch.randn(40, 6, generator=g)
    Qfull, _ = torch.linalg.qr(M)
    # Two shared + two unique each → small angles for shared
    shared = Qfull[:, :2]
    U1 = torch.cat([shared, Qfull[:, 2:4]], dim=1)
    # Slightly perturb shared for U2
    shared2 = orth(shared + 0.01 * torch.randn_like(shared))
    U2 = torch.cat([shared2, Qfull[:, 4:6]], dim=1)
    # Re-orthonormalize
    U1 = orth(U1)
    U2 = orth(U2)

    thetas, _, P, _ = principal_angles(U1, U2)
    theta_star = math.radians(30.0)
    Ustar, idx = build_shared_basis(U1, P, thetas, theta_star)
    assert Ustar is not None
    assert int(idx.sum()) >= 1
    # Ustar should be orthonormal
    Gram = Ustar.T @ Ustar
    assert torch.allclose(Gram, torch.eye(Ustar.shape[1]), atol=1e-5)


def test_no_shared_returns_none():
    g = torch.Generator().manual_seed(3)
    M = torch.randn(40, 8, generator=g)
    Qfull, _ = torch.linalg.qr(M)
    U1, U2 = Qfull[:, :4], Qfull[:, 4:8]
    thetas, _, P, _ = principal_angles(U1, U2)
    Ustar, idx = build_shared_basis(U1, P, thetas, math.radians(30.0))
    assert Ustar is None
    assert not bool(idx.any())
