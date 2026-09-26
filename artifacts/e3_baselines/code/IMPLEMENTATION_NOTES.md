# E3 implementation notes and fidelity caveats (written 2026-09-25 before the pilot; referenced by PREREG_E3.md)

Common interface (`merges.py`): `f(dWs, hp, fac=None, ctx=None) -> merged ΔW` per layer. `dWs` = dense ΔW_t = 2·B_t A_t
(float32). `fac` = (2·B_t, A_t). `ctx` = model-level extras (global thresholds, RNG). `merge_model(method, hp, names, dW, fac)` handles the
cross-layer parts. Every method is exactly `scale × base`: TA/TIES/DARE/TSV-M/KnOTS/gate scale by λ, Pico by c (unit-tested). The runner
therefore builds `base` once per non-scale config and evaluates all scales from it. For TA this reproduces E1's `W0 + λ(ΔW1+ΔW2)` bitwise.
Scope: the 73 LoRA-covered matrices. Heads are never merged; each task uses its own head on the shared merged encoder.

## TA
λ·ΣΔW_t. Bitwise equal to E1/E1b stage-2 TA (unit test and pilot sanity check).

## TIES (Yadav et al. 2023; prateeky2806/ties-merging `ties_minimal.ipynb`)
- Trim: keep |x| ≥ kthvalue(|x|, d − int(d·k/100)), computed per task over the concatenation of its 73 LoRA matrices. Official TIES
  flattens the whole task vector; for LoRA the task vector is zero elsewhere, so we use the LoRA-covered matrices as the trim domain,
  which matches E1b TIES-lite (E1b differs by one element in the kth index).
- Sign election: sign(Σ trimmed values). Zero signs are set to the majority sign: official uses the global majority, we use the
  per-layer majority. This only matters for coordinates whose trimmed sum is exactly 0, where either nothing is selected or there is an
  exact cancellation (measure zero). Unit test: identical to the vendored official function on random multi-layer data, and a
  hand-computed toy example.
- Disjoint mean over entries that agree with the elected sign, then × λ (official `flat_ptm + lamda * merged_tv`).
- TIES is applied to the dense ΔW (not to B and A separately), as in LoRA-TIES practice (PEFT, KnOTS baselines).

## DARE (Yu et al. 2024; MergeLM `mask_input_with_mask_rate`)
- `mask ~ Bernoulli(p)`; x·(1−mask)/(1−p). One `torch.Generator` seeded 0 per merge call, drawn in layer-major, task-minor order
  (deterministic per pair). Masks depend on the pair; they are not per-task fixed.
- DARE_TA = DARE then TA. DARE_TIES = DARE then full TIES with k = 20 (MergeLM `mask_apply_method="ties_merging"`, default
  `param_value_mask_rate=0.8`). The trim threshold is recomputed on the DARE'd updates. For p = 0.9 the k = 20 trim is a no-op, because
  ≥ 80% of entries are already zero and the threshold is 0. This is NOT MergeKit's `dare_ties`, which skips the magnitude trim.

## TSV-M (Gargiulo et al. CVPR 2025; task_singular_vectors `compute_and_sum_svd_mem_reduction`)
- Official: per task SVD (fp32), keep k = int(min_dim · 1/T) components, place them in blocks of `sum_u` (d_out × min_dim),
  `sum_s`, `sum_v`, then merged = U_u V_uᵀ diag(sum_s) U_v V_vᵀ. The polar factors come from the SVDs of sum_u and sum_v. Unit test:
  equal to the vendored official loop on random full-rank matrices.
- **LoRA caveat.** A rank-8 ΔW has only 8 non-null singular directions. The official k = min_dim/2 (384 for 768-dim layers) keeps
  376 numerically-null directions per task. Their basis is arbitrary (defined by the SVD implementation, cuSOLVER vs LAPACK), yet
  they enter the polar orthogonalization and can rotate the meaningful directions. We therefore include two variants in the grid:
  `official` (exact rule) and `lora` (k = r = 8; sum_u has T·r columns). With T = 1, `lora` reconstructs ΔW exactly. With disjoint
  supports, TSV-M(`lora`) = TA (unit-tested).
- Non-matrix parameters (biases, LayerNorm) are not in scope: LoRA does not change them.

## KnOTS (Stoica et al. ICLR 2025; gstoica27/KnOTS `task_merger.py::SVDMerger`, `masking_ops.py`)
- Alignment: SVD of the column concatenation [ΔW_1 | … | ΔW_T] (`concat_across_output=True`), float64, keep s > 1e-5, cast
  to float32. Task rep = diag(s) V_t (their `apply_svd` folds s into V and sets task_Ss = 1).
- Merge: `ties_masking` on the stacked flattened reps of ALL layers, i.e. a global top-K% threshold per task (official `kthvalue`
  rule; K = 100 means no trim) and sign election by summed values. Then `masked_merge`: multiply by scaling_coeffs[0] = λ BEFORE
  the disjoint mean (== λ × disjoint mean). Reconstruct U·merged_rep.
- The official LoRA handler uses ΔW = B@A without the α/r factor (their configs have α = r). We use the true ΔW = 2BA. SVD and TIES are
  scale-equivariant except for the absolute 1e-5 singular-value cut-off, which is irrelevant here (all 16 kept singular values ≫ 1e-5).
- Speed: we compute the identical SVD via the rank-2r factorization [B1 B2]·blockdiag(A1, A2) (two thin QRs plus a 16 × 16 SVD in
  fp64). Unit-tested against the dense fp64 path and against a vendored official pipeline (same merged update to 1e-4, sign-equivariance).
- The KnOTS-TA variant (tv mask, no trim, sum) reduces exactly to TA (unit-tested), so only KnOTS-TIES is a distinct method.
  KnOTS-DARE-TIES is not run (DARE is covered separately).

## Pico (Tang & Yang, arXiv 2604.16826 "Crowded in B-Space", §4.1–4.2). No public code was found; this is a reimplementation from the text.
Implemented exactly as written:
B_all = [B_1 … B_T] = UΣVᵀ (thin, m = min(d_out, T·r) columns); s_j = σ_j²/Σ_k σ_k²; α_j = 1/(1 + (T−1)s_j);
S = I + U diag(α − 1) Uᵀ; B̃_t = S B_t; ΔW̃_t = B̃_t A_t; ΔW_calib = M(ΔW̃_1..T); γ = (1/T)Σ‖ΔW_t‖_F / ‖ΔW_calib‖_F; ΔW_Pico = γ ΔW_calib.
Ambiguities and our choices:
1. **LoRA scaling** is "absorbed into B" (paper). We use B·(α/r). s_j is invariant to a common scale, so this has no effect.
2. **Which ΔW_t in γ:** the paper writes ΔW_t without a tilde. We use the ORIGINAL (uncalibrated) updates. Using calibrated norms would
   change the overall scale only, which the tuned c absorbs.
3. **Granularity:** "for each target module in each layer", so per LoRA matrix (73 per model), including the pooler.
4. **Base merger scale:** with TA or TIES as the base merger, any λ inside M cancels in γ. The paper's Pico has no free global scale;
   its output norm is the mean source norm per layer. The paper does not say whether an extra coefficient was tuned. To give Pico the
   same 8-config budget, we add an outer multiplier c; c = 1 is the literal method, and it is additionally reported untuned. For two
   near-orthogonal adapters, c = 1 corresponds roughly to TA with λ ≈ 0.7.
5. **"Sharing score" semantics:** s_j is the energy fraction of the joint spectrum. It does not check that a direction is used by more
   than one task, so a dominant direction used by only one task is also shrunk (towards 1/(1 + (T−1)s_j) ≥ 1/T). For T = 2 the
   shrink factor is ≥ 1/2 and usually mild. We implement the formula as written and do not add a cross-task test.
6. **Which B-directions:** all m columns of U get α_j. S acts as the identity on the orthogonal complement of span(B_all), so
   ΔW̃_t differs from ΔW_t only inside the joint column space.
7. **Pico with TIES:** TIES (k = 20, global threshold recomputed on the calibrated updates), then γ-rescale (exploratory PICO_TIES).
   The paper's TSV-M + Pico combination is not run.
8. **Rank / module scope:** the paper uses rank 16 on Llama q/v only; here r = 8, BERT, and all 73 LoRA matrices.
Unit tests: T = 1 → identity (α = 1) → returns ΔW; α ∈ [1/T, 1]; ‖output‖ = c·mean‖ΔW_t‖; hand example with a single shared
direction (α = 1/2 for both tasks); two orthogonal equal-energy directions → uniform α = 2/3 → after rescale, parallel to TA.

## Hard gate (θ★ = 30°) and soft gate
- E1b definition, copied: U_i = orth(B_i) via QR (float64). Pm, S = svd(U1ᵀU2). The layer FAILs iff max S > cos 30°.
  Sd = U1 Pm[:, S > cos 30°]. ΔW2 ← ΔW2 − β Sd Sdᵀ ΔW2 (β = 1: hard gate). The second task is t2 in pair order. Non-FAIL layers are
  bitwise TA. When a pair has no FAIL layer, the gate equals TA on every layer, so the runner reuses TA's evaluations
  (bitwise-identical merged weights) instead of recomputing them.
- The soft gate keeps the hard gate's FAIL-layer criterion and only weakens the projection. On E1 Hub pairs (min θ_min = 44.3°) both
  gates equal TA by construction.

## Tuning protocol
- Each method's grid has ≤ 8 configs (TA and all primary methods: exactly 8). Selection: argmax of the held-out mean normalized score,
  first max in grid order (TA grid ascending → smaller λ on ties, as E1). Then one eval-set evaluation.
- E1 pilot caveat: E1's "held-out" 1,000 examples are a seed-0 subset of GLUE TRAIN. The Hub adapters were trained on (most of) GLUE
  train (their model cards show a 100-example hold-out only), so pilot selection happens on in-distribution, mostly seen data. E1b fixes this.
