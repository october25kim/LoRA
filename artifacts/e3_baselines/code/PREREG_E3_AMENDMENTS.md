# PREREG_E3 amendments (append-only, dated KST)

## A1 — 2026-09-25 22:50 KST (implementation only; no change to methods, grids, selection, or analysis rules)
Written after a 1-pair smoke test on E1 (cola-sst2) and before the pilot run and any E1b run.
- `merges.tsvm`, kmode `lora`: the per-task SVD of the rank-8 ΔW = (2B)A is now computed exactly from the factors
  (`lowrank_svd`: thin QRs of B and Aᵀ plus an 8 × 8 core SVD, in fp64) instead of by a dense fp32 SVD. This is mathematically the same
  operation (unit test `test_tsvm_lora_fast_equals_dense`). The reason is speed: dense cuSOLVER SVDs made TSV-M take about 100 s per
  pair. kmode `official` still uses the dense SVD, as in the official code.
- `e3.py`: the base of the selected config is kept in memory, so it is not rebuilt for the eval-set evaluation. The TA-λ and Pico-c = 1
  extras reuse the same base. Results are unchanged; only compute is saved.

## A2 — 2026-09-26 00:45 KST (before any E3 computation on E1b adapters; supersedes the corresponding parts of PREREG_E3.md for the E1b run)
**Disclosure.** A2 was written after E1b had finished. The E1b verdict and the E1b stage-2 results were known when it was
written: the TA λ-selection counts (0.5: 22, 0.7: 60, 1.0: 9), TIES-lite selecting λ = 1.0 (its grid maximum) in 82/91 pairs, the
E1b minimum θ_min of 45.1° over all layers, and the exclusion of qqp and boolq. The E1 pilot (`pilot_e1/`, 21 pairs, v1 grids) had also
finished; its selected-config/boundary rates were seen. No E3 method had been run on any E1b adapter, and no E3 eval-set number on E1b existed.
The changes were requested by the study owner for the reasons given below.

1. **Units.** The 14 valid E1b tasks (cola, sst2, mrpc, stsb, mnli, qnli, rte, wic, snli, scitail, ag_news, imdb, trec, yelp_polarity)
   give **91 pairs**. qqp (full-data D1 adapter collapsed, eval acc 0.624) and boolq (never learned) are excluded, following
   E1b `stage0.json` / VERDICT. This is exactly the §1 rule; it is now stated explicitly.
2. **The θ★ = 30° hard gate cannot fire on E1b** (min θ_min = 45.1°), so GATE ≡ TA on all 91 pairs (gain exactly 0). S2 is
   "not estimable (n = 0)". GATE stays in the family, as registered.
3. **New primary arm FORCEGATE (threshold-free gate).** On every layer, ΔW2 ← ΔW2 − SSᵀΔW2, where S = the k = 1 most-aligned
   principal vector of orth(B1) with respect to orth(B2) (largest principal cosine), regardless of its angle. Then λ(ΔW1 + ΔW2) with λ = TA grid (8).
   (`merges.gate` with `force_k=1`; `gate_forced_basis`; unit test `test_forced_gate`.) The **primary Holm family becomes 8 methods**:
   TIES, DARE_TA, DARE_TIES, TSVM, KNOTS, PICO_TA, GATE, FORCEGATE. The decision rule is unchanged (mean gain ≥ +0.5 pp AND Holm p < 0.05).
4. **Gate vs soft, threshold-free (secondary).** S1 (GATE − PICO_TA) is kept. It is now also computed as **S1b = FORCEGATE − PICO_TA**
   (mean, task-block CI, two-sided sign-flip p, W/T/L). Together with the primary PICO_TA-vs-TA and FORCEGATE-vs-TA tests, this answers
   hard-projection vs soft-shrinkage without depending on θ★.
   The exploratory SOFTGATE now uses the forced basis (β ∈ {0.25, 0.5, 0.75} × λ ∈ {0.7, 1.0}); the threshold version is ≡ TA on E1b.
5. **Wider coefficient grids** (still 8 configs for TA and every primary method; order = tie-break order):
   - TA, GATE, FORCEGATE: λ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3}. This contains the E1b grid {0.3, 0.5, 0.7, 1.0}; E1b TA
     chose 0.5 in 22 pairs and never went above 1.0.
   - TIES: k=10: λ ∈ {1.5, 2.5}; k=20: λ ∈ {0.7, 1.0, 1.5, 2.0}; k=30: λ ∈ {1.0, 1.5}.
   - DARE_TA: p ∈ {0.5, 0.9} × λ ∈ {0.5, 0.7, 1.0, 1.4}.
   - DARE_TIES: p ∈ {0.5, 0.9} × λ ∈ {0.7, 1.0, 1.5, 2.0}.
   - TSVM: kmode ∈ {lora, official} × λ ∈ {0.5, 0.7, 1.0, 1.5}.
   - KNOTS: topK ∈ {20, 100} × λ ∈ {0.7, 1.0, 1.5, 2.0}.
   - PICO_TA: c ∈ {0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5} (c = 1 ≈ TA λ 0.7 for near-orthogonal pairs).
   - Exploratory PICO_TIES: k = 20, c ∈ {0.75, 1.0, 1.25, 1.5}.
   Rationale: E1b TIES-lite was grid-limited at λ = 1.0. In the pilot, TIES / DARE_TIES / KnOTS often chose the LOWEST λ of their v1 grid,
   so the v2 grids extend in both directions at equal size.
6. Everything else in PREREG_E3.md is unchanged: data, selection rule, statistics, sanity checks, and exploratory items.
   `e3.py --setting e1b` uses these v2 grids (`GRIDS_V2`, `PRIMARY_V2`) and refuses to run unless both PREREG_E3.md and this file
   match their hashes. The pilot keeps the registered v1 grids.
