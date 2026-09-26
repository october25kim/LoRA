# Changelog / Notices

## 2026-09-26 — Notice: withdrawn result

The constructed-conflict result stated in the message of commit `80f5ec7`
("Add baselines and full-seed constructed conflict certificates": *"Forced shared-B conflict yields 73 FAIL at 30° with MNLI sum
0.398 → corrected 0.510"*) is **WITHDRAWN** and should not be cited.

- Cause: evaluation bug. Before the segment-id fix (`463cb6a`), `evaluate_glue` always dropped `token_type_ids`, which mis-evaluates
  adapters trained with BERT segment ids (e.g. `adapters/conflict_full_*`, `adapters/mnli_seed*`). The number was also computed on a
  small subset (n = 512).
- The artifacts committed in `80f5ec7` (`artifacts/conflict_full_theta*/`, `artifacts/baselines/single_adapters.json`,
  `artifacts/seed_pair*/`) are retained for provenance only; their accuracy numbers predate the fix. Git history is not rewritten.

## 2026-09-26 — release/pipelines

- Merged `fix/segment-ids` (segment-id / `token_type_ids` evaluation fix with per-adapter `segment_ids` mode, registry and
  `segment_ids.json`; hubish certificates; paper_pack table) with `main` (baselines, constructed-conflict and seed-pair certificates).
- Added pre-registered pipelines: `artifacts/e1b_confirmatory/`, `artifacts/e3_baselines/`, `artifacts/e1c_seed_diverse/`,
  `artifacts/lemma_check/` (E1 `artifacts/e1_predictive/` and `artifacts/seed_fix_segid/` were already tracked). See README → Experiments.
- Software: Python 3.11.15, torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.0, datasets 5.0.1; NVIDIA driver 570.211.01,
  RTX 4070 Ti SUPER 16 GB, Ubuntu 24.04.4.
