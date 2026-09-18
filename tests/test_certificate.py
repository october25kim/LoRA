"""Unit tests for PASS identity and FAIL shared-direction removal."""

from __future__ import annotations

import math

import torch

from lora_merge_cert.certificate import (
    arithmetic_sum,
    certify_and_merge_layer,
    project_out,
)
from lora_merge_cert.merge import synthetic_lora_pair
from lora_merge_cert.subspace import extract_basis, orth
from lora_merge_cert.angles import principal_angles, build_shared_basis, overlap_frobenius


def test_project_out_removes_component():
    g = torch.Generator().manual_seed(10)
    U = orth(torch.randn(20, 3, generator=g))
    dW = torch.randn(20, 15, generator=g)
    # Inject a known shared component
    coeff = torch.randn(3, 15, generator=g)
    dW = dW + U @ coeff
    corr = project_out(dW, U)
    # U^T corr should be ~0
    residual = U.T @ corr
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_pass_identity_equals_arithmetic_sum():
    """When subspaces are orthogonal (PASS), merge == ΔW1+ΔW2."""
    g = torch.Generator().manual_seed(11)
    M = torch.randn(48, 16, generator=g)
    Q, _ = torch.linalg.qr(M)
    r = 4
    B1 = Q[:, :r]
    B2 = Q[:, r : 2 * r]
    A1 = torch.randn(r, 32, generator=g) * 0.1
    A2 = torch.randn(r, 32, generator=g) * 0.1

    dW_cert, cert = certify_and_merge_layer(
        B1, A1, B2, A2, theta_star_deg=30.0, subspace="A", layer_name="pass_layer"
    )
    dW_arith = arithmetic_sum(B1, A1, B2, A2)

    assert cert.status == "PASS", f"expected PASS, got {cert.status} θ={cert.theta_min_deg}"
    assert cert.n_shared == 0
    assert torch.allclose(dW_cert, dW_arith, atol=1e-5)


def test_fail_removes_shared_direction():
    """On FAIL, shared column-space component of ΔW2 is removed."""
    B1, A1, B2, A2 = synthetic_lora_pair(m=40, n=30, r=6, shared_dirs=3, seed=12)

    dW_cert, cert = certify_and_merge_layer(
        B1, A1, B2, A2, theta_star_deg=30.0, subspace="A", layer_name="fail_layer"
    )
    assert cert.status == "FAIL", f"expected FAIL, got PASS θ={cert.theta_min_deg}"
    assert cert.n_shared >= 1

    dW1 = B1 @ A1
    dW2 = B2 @ A2
    dW_arith = dW1 + dW2

    # Corrected merge must differ from plain sum
    assert not torch.allclose(dW_cert, dW_arith, atol=1e-4)

    # Reconstruct U★ and verify U★^T (dW_cert - dW1) ≈ 0
    # (i.e. the task-2 contribution after correction is orthogonal to S★)
    U1 = extract_basis(B1, A1, definition="A")
    U2 = extract_basis(B2, A2, definition="A")
    thetas, _, P, _ = principal_angles(U1, U2)
    Ustar, idx = build_shared_basis(U1, P, thetas, math.radians(30.0))
    assert Ustar is not None
    dW2_corr_part = dW_cert - dW1
    residual = Ustar.T @ dW2_corr_part
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)


def test_subspace_B_runs():
    B1, A1, B2, A2 = synthetic_lora_pair(m=32, n=24, r=4, shared_dirs=1, seed=13)
    dW, cert = certify_and_merge_layer(
        B1, A1, B2, A2, theta_star_deg=30.0, subspace="B", layer_name="b_def"
    )
    assert dW.shape == (32, 24)
    assert cert.status in ("PASS", "FAIL")
    assert cert.theta_min_deg >= 0.0


def test_overlap_bounded_by_rank():
    g = torch.Generator().manual_seed(14)
    U1 = orth(torch.randn(20, 5, generator=g))
    U2 = orth(torch.randn(20, 5, generator=g))
    ov = float(overlap_frobenius(U1, U2))
    assert 0.0 <= ov <= 5.0 + 1e-5
