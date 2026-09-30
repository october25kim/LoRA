# E4b — deviations and notes (append-only; KST)

Pre-registration: `PREREG_E4B.md` / `prereg_e4b.json`, hashed 2026-09-26 10:02:03 KST (`prereg_e4b.sha256`). No E4b GPU work happened before E4a finished (E4a merges ended 15:01:14).

## Before the pilot (smoke test in a scratch copy `~/e4_smoke/e4b_decoder`, 20-step throwaway adapters, 15:02–15:30; deleted afterwards; no study numbers)
- **D1 (15:05, code: smoke-only path).** `train_e4b.py` smoke mode used `recipe.lr = null` (the lr comes from the pilot) and crashed. Smoke mode now sets lr = 3e-4 for the throwaway
  adapters. The pilot and main paths are unchanged.
- **D2 (15:30, execution; within the prereg).** At micro-batch 32, RTE ran out of memory (15.5 GB in use at ≤ 256 tokens). Qwen activations with fp32 master weights take ≈ 2.3 MB/token.
  The **pre-registered OOM fallback** (halve the micro-batch, double the accumulation; effective batch 32) handled it automatically: rte ran at micro-batch 16, peak 9.1 GB.
  Measured 20-step peaks: sst2 5.9 GB, stsb 8.0 GB, trec 4.4 GB, imdb (micro-batch 16) 9.7 GB.
  Two concurrent training processes would not fit reliably, so **main training runs as a single GPU process** (the prereg allows at most 2). The fallback may therefore apply to other long-sequence tasks
  (e.g. qnli, mnli, ag_news, scitail). It is logged per adapter (`train_meta.json: micro_batch, micro_div`; failed attempts in `logs/*_fail_div*.log`).
- **D3 (15:30, pipeline orchestration only; no computation changed).** The E3 subset (`e3.py run()`, unmodified) needs ≈ 12 GB for Qwen. With `--mem-frac 0.5` it ran out of memory. A stage-2
  process peaks at ≈ 6.7 GB. So `run_pipeline_e4b.sh` now uses a phased schedule that keeps the pre-registered priority P → S2+E3 → R0 → S1:
  - phase 1: P ∥ (S2, then R0 in reverse order, stopped through a stop-file once P is done);
  - phase 2: E3 subset alone on the GPU (`--mem-frac 0.9`);
  - phase 3: R0 → S1 in both directions.
  `e4b.past_deadline()` also honors the stop-file (`E4B_STOPFILE`). All code was re-hashed afterwards in `code_e4b.sha256` (before the pilot).
- Smoke results (pipeline only): (c) max logit diffs 2e-5–8e-5, far below tolerance; (c2) within 0.5 pp. All stages ran: pilot rule, stage0, stage1, stage2 P/S2/R0/S1, forced-gate test with θ★ = 89° (168 layers gated),
  E3 subset, stage3, verdict. The analysis-code equivalence check on E1b reproduced E1b exactly.

## Pilot (pre-registered pre-step; 15:29:24–15:44:53 KST)
- Outcome, applied under the pre-registered rule (`pilot/pilot_decision.json`, sha256 in `pilot/pilot_decision.sha256`, written 15:44:52):
  - Held-out accuracy (sst2 / rte) by lr:
    - 1e-4: 0.942 / 0.735
    - 3e-4: 0.956 / 0.783
    - 1e-3: 0.920 / 0.746
  - None of the lrs diverged.
  - **lr_main = 3e-4.**
  - Projected core compute for 2 seeds = 5.52 h (≤ 11 h threshold), so we use **seeds [0, 1]**, and the **primary population is P** (mixed seed).
- rte hit the expected OOM at micro-batch 32 in every pilot run (see D2). Each time it fell back to micro-batch 16 (rc=1 then rc=0 in `pilot.log`). Its peak was 9.7 GB.
- **N1 (15:46:56, operational; no computation).** The first main-training launch passed the 28 queue items as a single argument, because the remote login shell (zsh) does not word-split.
  `train_e4b.py` exited at argument parsing with rc=1. No adapter or file was created. The line stays in `train_e4b.log`.
  We relaunched correctly at **15:47:19** (`train_launch_time.txt`), with one queue Q1 containing the 14 tasks @s0 followed by the 14 tasks @s1. `run_pipeline_e4b.sh` started at 15:46:58 and is waiting for the adapters.

## Main run
- Training ran 15:47:19–18:00:30 KST as a single queue (2.22 h, both seeds).
  - Every adapter finished on its first successful attempt. The pre-registered micro-batch fallback (div 2) was used for mnli, qnli, rte, scitail and ag_news in both seeds; imdb and yelp use micro-batch 16 by design.
  - No collapse. The sst2@s0 main run reproduced the pilot lr=3e-4 run exactly (same loss trajectory, held-out 0.956).
- Stage0 (18:05:43): **all 14 tasks valid in both seeds** under (a), (c) and (c2). There were no exclusions. Population sizes: P 91, R0 91, S1 91, S2 14.
- Stage1 froze the predictors at 18:09:32 (`predictors_e4b.sha256`).
- **D4 (18:21, orchestration only; no computation changed).** In phase 1, the stage-2 P process reached ≈ 13 GB with the real adapters and evaluation sets (the smoke test had measured ≈ 6.7 GB).
  - Process B (S2, then R0 in reverse) therefore ran out of memory 3×3 times between 18:09 and 18:13. The first S2 pair (`cola@s0__cola@s1`) completed and is kept (stage 2 is resumable per pair). Every other attempt failed while loading or merging and wrote nothing.
  - Unchanged, the original runner would never run S2 and would start two concurrent processes in phase 3. So its parent (pid 2026925) was stopped at 18:21:22. The P subshell kept running untouched.
  - `run_pipeline_e4b_serial.sh` (sha256 `89e2ffa0…`, in `run_pipeline_e4b_serial.sha256`) takes over once P finishes (stop-file). It re-checks the prereg, code and predictor hashes, then runs **one GPU process at a time**: S2 → E3 subset → R0 → S1 → stage3.
  - This keeps the pre-registered priority P → S2+E3 → R0 → S1 and the same budget deadline (pilot start + 13.6 h). All merge, evaluation and analysis code is unchanged.
- Observation (reported in secondary results): unlike BERT and RoBERTa, Qwen2.5-0.5B cross-task adapter pairs have layers with θ_min < 30°. Across 48,216 pair-layers, 351 have θ_min < 30° (minimum 13.9°), in 235 of 287 pairs. They are concentrated in `layers.0.self_attn.k_proj` (d_out = 128 under GQA) and in layer-23 v_proj/down_proj. So the pre-registered 30° gate fires in 1 layer for most pairs.
