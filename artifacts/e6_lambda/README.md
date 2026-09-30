# E6: pre-merge λ rules and merge/no-merge decisions (post hoc, exploratory)

- `RULES.md` + `RULES.sha256` + `RULES_timestamp.txt`: the rule list, fixed before any scoring. `DEVIATIONS_E6.md`: clarification C1 and the post-results additions A1–A3.
- `data/`: `src/` holds copies of the per-pair result files and predictor tables (with SHA256SUMS). `layer_stats/` holds per-layer norms and inner products (computed on the 4070 CPU by `code/layer_stats_cpu.py`; tarball sha256 cc67570d…). `pairs_long.csv` and `pairs_unlabeled.csv` are the consolidated tables; `checks*.json` are the integrity checks.
- `code/`: `build_dataset.py` → `build_unlabeled.py` → `evaluate.py` → `evaluate_addon.py` → `make_verdict_e6.py` (run from `e6_lambda/` with `../e1/.venv/bin/python`).
- `results/`: all tables, including `per_pair_e6.csv`.
- `VERDICT_E6.md` (the report) and `MANUSCRIPT_INSERT.md` (draft §4.8 and Box 1 update; the manuscript itself is untouched).
