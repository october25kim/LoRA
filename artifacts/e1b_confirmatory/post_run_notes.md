# E1b post-run notes (written 2026-09-26 ~00:35 KST, AFTER all results; no effect on predictors, selection, exclusions, or verdicts)

1. **QQP (D1 full-data adapter) collapsed during training.** Training loss fell to about 0.42 by step 1,000. It then rose to about 0.65–0.67
   around steps 5k–8k and stayed there. That value is the entropy of the label prior, so the model had degenerated to the majority class:
   eval accuracy 0.6240, versus a majority baseline of 0.6240 on this subset. This is an optimization instability over the long high-lr phase
   (lr 1e-3, 34k steps, fp16), not a data problem. The pre-registered integrity rule excluded it. Per instructions, it was not retrained.
2. **BoolQ never learned.** Loss was flat at about 0.67 from the first steps. Eval accuracy 0.6217 equals the majority baseline. Excluded by rule (a),
   as pre-registered.
3. **Exploratory sensitivity (post-verdict, `sensitivity_cap60k.json`).** The kept 60k-capped adapters scored MNLI 0.7922, which passes (a) and (b)
   (threshold 0.7891), and QQP 0.8551, which passes (a) but misses (b) by 0.2pp (threshold 0.8571). So under the original 60k budget, QQP would
   also have been excluded, by a narrow margin, and MNLI would have been kept. The capped adapters were not merged.
4. **Stage-3 re-run (reporting fix only).** `n_layers_theta_min_lt30_A` is 0 for every pair, so its ρ is undefined. The first stage-3 run
   printed a spurious permutation p = 0.0001 for this predictor, because NaN comparisons counted as "not exceeding". stage3 was re-run with a
   guard that reports NaN. Verdicts and all other numbers are identical (the first-run outputs are in `diag/stage3_first_run/`; code hash in
   `code_stage3_rerun.sha256`). No predictor or merge was recomputed.
5. **Hard gate θ★ = 30° never activates.** The minimum θ_min over all 91 pairs × 73 layers is 45.1° (pilot: 44.3°). So the gate is identical
   to TA for every pair, and the method comparison for the gate is 91 exact ties.
6. **TIES-lite caveat.** A disjoint-mean TIES vector has roughly the magnitude of a single task vector, while TA sums two. At TA's λ* it is
   therefore under-scaled. With its own held-out λ, it picked the grid maximum λ = 1.0 in 82 of 91 pairs. So its own-λ score is probably
   limited by the pre-registered grid, and TIES-lite results should be read as "within this grid".
7. **All overlaps are far above the random-subspace null** (O_A null z = 16 to 95), yet no layer comes near 30°. Every adapter shares seed 0
   (identical LoRA-A initialization), a known property declared in PREREG.md. This probably inflates overlap for all pairs uniformly.
