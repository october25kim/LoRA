# E7 deviations log (PLAN_E7.md is kept unchanged; its hash is in PLAN_E7.stamp)

- 2026-09-29 19:27:47 KST: first launch (19:25:49 KST) stopped after 3 pairs of stage a1: the timed "singles" segment included adapter
  loading from disk (+ sha256), which is shared by every rule and not part of any rule's cost. Patched e7.py to load adapters and prebuild the
  padded cal / held-out batches BEFORE the timers (no change to any computed prediction/entropy). The 3 discarded records are kept in
  discarded_timing_bug/ (entropies there are identical in method). Relaunched with e7.py sha256 5bd0b99cc685c86081fb8bf6cd9d277f472c0a0c187d5396933241f3a74869f5.
- 2026-09-29 19:30:23 KST: second launch stopped after 5 pairs of a1: the adapter cache helper used dict.setdefault(k, load_ad(k)), which evaluates
  load_ad eagerly, so the timed single-adapter forward reloaded the adapter from disk for every layer (singles ~9 s instead of ~1-2 s). Timing-only
  bug (predictions unaffected). Fixed; the 5 records moved to discarded_timing_bug2/; relaunched with e7.py sha256 1eb5931be6ae18404b3f6cd583aaec3baa9af293cf8c856edc6a8470612105e0.
- 2026-09-29 19:35:09 KST: ADDED (exploratory, "0.5B if cheap" in the request): e7_q05.py = stage a1 on the Qwen2.5-0.5B E4b pairs (P/R0/S1, 273;
  G4 from e4b pair_results, G7 from E5b extended results), identical computation to e7.py a1. Order change: STOP_AFTER_P created so run_e7.sh ends
  after a2 on P; run_e7_followup.sh then runs e7_q05.py -> a2 on R0,S1 -> e7_analysis.py. e7_q05.py sha256 9e6eeca3550a0776b05f5912166cd18402fc991f7ba3973398925d2709556485.
- 2026-09-29 21:08:56 KST: e7_analysis.py (analysis code, written after PLAN but before looking at any E7 result; first run on the complete A1 data at 21:08 KST)
  additionally reports U1 recomputed from the E7 cal logits (U1_recomputed_E7_logits; sensitivity for the bf16 batch-composition differences:
  recomputed merged cal argmax matches E6 cached predictions in 99.8% of cal examples, min 97.5% per pair/lambda) and the Qwen2.5-0.5B scope.
- 20:46-21:05 KST: a CPU-only smoke test of the a2 code path (2 steps, one pair, output in /tmp only, not in this directory) was run to check the
  LoRA-branch hook + gradient-checkpointing backward before the GPU a2 stage; it passed the backward step and was stopped during its (slow CPU)
  test-set eval. It competed for CPU with stage c (no timings are taken in stage c), so c pairs 3-7 ran slower.
