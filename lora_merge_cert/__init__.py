"""LoRA Merge Certificate — Week-1 research package.

Per-layer principal-angle certificate for merging two LoRA adapters.
See METHOD.md / README for formulas (θ★=30°, subspace A/B, PASS/FAIL correction).
"""

from .angles import (
    build_shared_basis,
    overlap_frobenius,
    principal_angles,
    select_shared_indices,
)
from .certificate import (
    DEFAULT_THETA_STAR_DEG,
    LayerCertificate,
    arithmetic_sum,
    certify_and_merge_layer,
    project_out,
)
from .eval import evaluate_glue, fake_eval_from_certificates
from .merge import (
    apply_deltas_to_base,
    collect_lora_pairs,
    merge_lora_models,
    resvd_to_rank,
    synthetic_lora_pair,
)
from .subspace import extract_basis, left_singular_vectors, orth

__all__ = [
    "DEFAULT_THETA_STAR_DEG",
    "LayerCertificate",
    "arithmetic_sum",
    "apply_deltas_to_base",
    "build_shared_basis",
    "certify_and_merge_layer",
    "collect_lora_pairs",
    "evaluate_glue",
    "extract_basis",
    "fake_eval_from_certificates",
    "left_singular_vectors",
    "merge_lora_models",
    "orth",
    "overlap_frobenius",
    "principal_angles",
    "project_out",
    "resvd_to_rank",
    "select_shared_indices",
    "synthetic_lora_pair",
]

__version__ = "0.1.0"
