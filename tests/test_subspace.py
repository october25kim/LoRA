"""Unit tests for subspace definitions A and B."""

from __future__ import annotations

import torch

from lora_merge_cert.subspace import extract_basis, left_singular_vectors, orth


def test_orth_is_orthonormal():
    g = torch.Generator().manual_seed(20)
    B = torch.randn(50, 7, generator=g)
    U = orth(B)
    assert U.shape[0] == 50
    assert U.shape[1] <= 7
    Gram = U.T @ U
    assert torch.allclose(Gram, torch.eye(U.shape[1]), atol=1e-5)


def test_left_singular_matches_extract_B():
    g = torch.Generator().manual_seed(21)
    B = torch.randn(30, 5, generator=g)
    A = torch.randn(5, 20, generator=g)
    dW = B @ A
    U_svd = left_singular_vectors(dW, rank=5)
    U_ext = extract_basis(B, A, definition="B")
    # Spans should match (up to sign/permutation): check projectors
    P1 = U_svd @ U_svd.T
    P2 = U_ext @ U_ext.T
    assert torch.allclose(P1, P2, atol=1e-4)


def test_definition_A_ignores_A():
    g = torch.Generator().manual_seed(22)
    B = torch.randn(25, 4, generator=g)
    A = torch.randn(4, 10, generator=g)
    U_a = extract_basis(B, A, definition="A")
    U_only = orth(B)
    assert torch.allclose(U_a, U_only, atol=1e-6)
