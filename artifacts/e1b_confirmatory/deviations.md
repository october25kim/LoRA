# E1b deviations from PREREG.md (append-only; each entry timestamped KST)

## D1 — full-data MNLI and QQP (written 2026-09-25 21:03 KST, before any evaluation-set scoring)
MNLI and QQP retrained on full train minus the same 1,000 held-out examples, 3 epochs, same recipe otherwise, because the 60k
cap made the pre-registered published-reference check (−5pp vs HF run_glue bert-base dev) structurally unreachable; decided
before any evaluation-set scoring; the capped adapters are kept under a separate name and not used.

- Decision by the study owner at 2026-09-25 21:02 KST, after reading the pre-registered risk note (PREREG.md §2). At that point
  only training losses existed. No adapter had been scored on any evaluation set or held-out set.
- Integrity rule, thresholds, λ grid, analysis, and all other tasks' budgets are **unchanged**. BoolQ and RTE keep their
  pre-registered budgets and are excluded if they fail the check.
- New budgets (machine-readable: `deviations_d1.json`; code: `train_e1b_d1.py`, which differs from `train_e1b.py` only in
  reading these budgets and using the uncapped pool; `run_train_d1.sh`):
  - MNLI: 391,702 training examples (392,702 − 1,000 held-out), 3 epochs, 12,241 steps/epoch → **36,723 steps**, warmup 500
    (pre-registered: 60,000 examples, 5,625 steps).
  - QQP: 362,846 training examples (363,846 − 1,000 held-out), 3 epochs, 11,339 steps/epoch → **34,017 steps**, warmup 500
    (pre-registered: 60,000 examples, 5,625 steps).
  - The held-out 1,000 examples are identical to the pre-registered ones (asserted in code). The pre-registered step cap of 5,625 does not apply to these two tasks.
  - Total training steps: 54,288 → 113,403. Projected training GPU time is still far below the 12 h cap.
- The capped adapters from the original run are renamed `adapters/mnli_cap60k` and `adapters/qqp_cap60k` and are not used for
  integrity, predictors, merges, or verdicts. After the verdict they may be scored as a clearly labelled exploratory sensitivity check.
