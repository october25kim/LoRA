# E5 — deviations and notes (append-only; times KST)

Pre-registration: `PREREG_E5.md` + `prereg_e5.json`, hashed 2026-09-27 14:56:21 KST (`prereg_e5.sha256`, `PREREG_RECORD_E5.txt`).
Committed locally on ubuntu-4070 as commit `d0975c481cdc8d1ca41bd6d7dd5710225afe0e4f` (2026-09-27 14:56:27 +0900). **Not pushed.**
E5 pipeline launched 14:56:31 KST (`launch_time.txt`); E5c (box, CPU) started 14:57 KST.

## Notes (no protocol change)
- **N1 (14:56, git).** The prereg commit is on a new local branch `e5-prereg` (parent = `463cb6a`, the HEAD of `fix/segment-ids`). It was created with
  plumbing (`git read-tree HEAD` into a temporary index → `git add -f` of the 10 E5 files → `git write-tree` → `git commit-tree -p HEAD` →
  `git update-ref refs/heads/e5-prereg`). This way the checked-out branch `fix/segment-ids` (waiting for a manual push/PR) and the working tree
  were not changed. `git show e5-prereg:artifacts/e5_accept/PREREG_E5.md` returns the hashed file.
- **N2 (14:53–14:55, smoke).** The pre-registered pipeline smoke test ran on the throwaway E4b pilot adapters (sst2 lr 1e-4 × rte lr 1e-3, singles set to 1.0).
  The outputs are kept, clearly labelled, in `smoke_pilot_adapters_NOT_STUDY/`. They contain no study numbers, and no code change followed the smoke test.
  E5b/E5d were not smoke-tested on throwaway adapters; their first real outputs are checked by the pre-registered sanity checks.

## Post-run notes (added 2026-09-27 ~20:20 KST, after all runs completed)

- N3. `make_verdict_e5.py` was written after the prereg hash, while runs were in progress. It only formats the analysis JSONs produced by the pre-registered, hashed code; it computes no new statistics. The hand-written "Interpretation" section in VERDICT_E5.md was added afterwards. Its numbers are copied from the tables.
- N4. The E5c outputs are on the box at `e5/e5c/` (`e5c_reliability.json`, `e5c_pair_bootstrap.csv`), because E5c ran on the box CPU as pre-registered. A copy was placed in the 4070 `artifacts/e5_accept/e5c/`.
- N5. `e5_analysis.py` (E5a part) and the E5d analysis were also run once by hand before the runner's final analysis step. The code was the same hashed code, and the runner's final run overwrote those outputs with identical results. The runner's final analysis JSONs are the ones reported.
- N6. The E5b sanity check "λ = 1.0 recomputation" exists only for P (reported as None for R0/S1). Only P recomputes λ = 1.0 inside E5b, because the E5a/E5b cross-check covers P. This is not a protocol change.
- No protocol deviations: every pre-registered run completed on its full population (E5a 91/91 within the 7 h deadline; E5b P/R0/S1 91/91 each), with no retries.
