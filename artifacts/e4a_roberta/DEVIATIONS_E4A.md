# E4a — deviations and notes (append-only; KST)

**No deviation from PREREG_E4A.md / prereg_e4a.json affected training, integrity, predictors, merges or the analysis.** Notes:

- **N1 (2026-09-26 09:54, pilot outcome; pre-registered rule, not a deviation).** The pilot ran 09:44:18–09:54:11.
  - lr 1e-3: sst2 held-out 0.953 (OK); rte **diverged** (held-out 0.512 vs majority 0.488; tail loss 0.695 ≈ ln 2 = prior entropy; the model never learned).
  - lr 5e-4: sst2 0.958, rte 0.742, neither diverged.
  - → **lr_main = 5e-4** for all tasks and both seeds (`pilot/pilot_decision.json`, sha256 in `pilot/pilot_decision.sha256`). lr 2e-4 was not piloted, as the ladder requires.
- **N2 (09:55, code; before any main training and before code hashing).** The throwaway 3-task smoke test (scratch copy outside `artifacts/`, 20-step adapters) found a crash in
  `e4a_analysis.method_comp`: with 3 tasks every task-block bootstrap resample is degenerate, so `boots == []`. The test `boots is not None` became `if boots`, so the pair
  bootstrap is used only when there are no task-block resamples. With 13 tasks, 2,000/2,000 task-block resamples were used in P, R0 and S1 (0 skipped), so the fix has no effect on the
  reported numbers. Code was hashed afterwards (`code_e4a.sha256`, `code_analysis_e4a.sha256`, 09:56). The smoke copy was deleted.
- **N3 (11:39, integrity; pre-registered rule and pre-registered risk).** **RTE failed rule (b) in both seeds**: eval accuracy 0.7004 (s0) and 0.7040 (s1), below the threshold 0.737
  (fairseq roberta.base 78.7 − 5 pp). (a) and (c) passed. As §2 requires, RTE is removed from every population. **K = 13 valid tasks** (≥ 10, no STOP);
  n = P 78, R0 78, S1 78, S2 13. CoLA (MCC), MRPC, SST-2, STS-B, MNLI and QNLI passed (b) in both seeds. No training collapse: every adapter's final logged loss was far below the prior entropy.
- **N4 (15:00, merges).** The two S1 processes (forward and reverse) met in the middle, and both computed `snli@s1__imdb@s1`. The two records are bitwise-identical
  (D = 0.015116, λ* = 0.7). The pre-written de-duplication in `load_pop` keeps the first one. S1 has 78 unique pairs.
- **N5 (compute).** Pilot start 09:44:18 → training 09:56:25–11:36:55 → stage0/1 → merges end 15:01:14 → stage3 end 15:02:09.
  **5.28 h wall-clock** (cap 10 h). Summed process time: training 4.82 h, pilot 0.15 h, stage-2 merges 6.11 h, E3 subset 0.53 h (`gpu_time_summary.json`).
- **N6 (timing of the E4b pre-registration).** `PREREG_E4B.md` / `prereg_e4b.json` were hashed at 10:02, while E4a was training and before any E4a integrity or merge result existed.
