# E1c deviations and implementation notes (append-only; KST)

## N1 — 2026-09-26 05:43 KST (implementation note, not a change of method)
The stage-3 analysis code is in a separate file, `e1c_analysis.py`, which `e1c.py --stage stage3` calls. It was written while seed-1 training
was running, before any seed-1 adapter had been evaluated, and before integrity, predictors, or merges existed for E1c. Its sha256 is recorded in
`code_analysis_e1c.sha256` at deployment. The statistics are copied verbatim from `e1b.stage3` (perms, bootstrap, LOTO, Holm, rule) and from
`e3.signflip`. It also reruns the same statistics on the E1b R0 data and checks that they reproduce E1b `analysis.json` exactly (analysis-code equivalence).

## N2 — equivalence gates (PREREG_E1C §4–5), run 05:39:47–05:41:09 KST, before training (05:41:26)
stage1 on the 91 R0 pairs reproduced E1b `predictors.csv` exactly: max relative difference 0 on all 13 numeric columns.
stage2 on cola-sst2 and rte-wic reproduced E1b `pair_results.csv`: max |Δ| 4.9e-11 and 4.3e-11, which is CSV %.10g rounding, with λ\* equal.
Both gates passed (`equiv/equivalence.json`).

## N3 — post-run notes, 2026-09-26 ~09:20 KST (written AFTER all E1c results; they change no predictor, selection, exclusion, or verdict)
1. **O_B is identical to O_A.** This holds in E1b as well as E1c (max |O_A − O_B| = 0 in both `predictors.csv` files). In `e1b.stage1`,
   QB = Qb·U, where U comes from the SVD of Rb·A (8 × 8, orthogonal), so QB spans col(B), the same subspace as orth(B). "O_B" and
   `mean_theta_min_B_deg` therefore duplicate O_A and `mean_theta_min_A_deg`. They are not an independent A-side predictor, and this also
   applies to the E1b secondary table. It does not affect H1/H2/H1c/H2c.
2. **Shared initialization explains little of the inflated O_A.** Mixed-seed O_A is only about 5% lower than shared-seed O_A for the same task pair
   (P/R0 ratio of means 0.954; P lower in 85% of pairs; Wilcoxon p ≈ 2e-12). The null z is still 15–75, and 100% of P pairs lie above the null p95.
   Task-vector cosine roughly halves (mean 0.0008 vs 0.0017). The O_A range is not wider under mixed seeds (IQR [0.0147, 0.0174] vs [0.0153, 0.0186]).
3. **E1b's H1 FAIL does not replicate as FAIL.** P (mixed seed): ρ = −0.025, CI [−0.51, 0.43] → INCONCLUSIVE. S1 (E1b recipe, seed 1, shared init):
   ρ = −0.030, CI [−0.44, 0.39] → INCONCLUSIVE (rule applied descriptively). All three point estimates are ≤ 0 (−0.19, −0.03, −0.02), and none comes close to PASS.
   What changes is the width of the task-block CI. So the FAIL-vs-INCONCLUSIVE boundary for H1 is seed-sensitive at K = 14.
   The per-pair D reliability across seed configurations is Spearman 0.67 (R0~P), 0.67 (R0~S1), and 0.81 (P~S1).
4. **Same-task seed pairs never reach 30°.** Minimum θ_min = 39.8° (MNLI, pooler.dense). In the hubish MNLI s7 × s42 pair it was 20.6°, under a different recipe
   (dropout 0.05, different seeds, 500-step warmup). The θ★ = 30° gate therefore never fires in E1c (0/14 S2, 0/91 P, 0/91 S1), and the E3-S2 analogue is not estimable (n = 0).
5. `make_verdict_e1c.py` (a report writer that reads `analysis_e1c.json`), `pair_results_e1c.csv` (the three jsonl files concatenated), and
   `gpu_time_summary.json` were created after the results. They are for reporting only.
