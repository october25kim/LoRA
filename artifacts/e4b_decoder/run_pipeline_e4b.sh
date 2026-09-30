#!/usr/bin/env bash
# E4b STEP 2-4 runner: waits for all adapters of the seeds chosen by the pilot, then stage0 -> stage1 (freeze) -> merges in 2 GPU processes.
#   two seeds: phase 1  A: P (primary) || B: S2, then R0 (reverse) until P is done;  phase 2: E3 subset alone (~12 GB);
#              phase 3  A: R0 -> S1 || B: R0 (reverse) -> S1 (reverse)                (priority P -> S2+E3 -> R0 -> S1)
#   one seed:  proc A: R0;  proc B: R0 (reverse)
# Budget guard: E4B_DEADLINE_EPOCH = pilot start + 13.6 h (no new pair / E3 run after it). Resumable.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export E4B_DEADLINE_EPOCH=$(( $(cat pilot/pilot_start_epoch.txt) + 48960 ))
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg_e4b.json'))['tasks']))")
SEEDS=$("$PY" -c "import json;print(' '.join(map(str, json.load(open('pilot/pilot_decision.json'))['seeds'])))")
for s in $SEEDS; do for t in $TASKS; do until [ -f "adapters_s$s/$t/DONE" ]; do sleep 30; done; done; done
sha256sum -c prereg_e4b.sha256 >> pipeline_e4b.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e4b.log; exit 3; }
sha256sum -c code_e4b.sha256 >> pipeline_e4b.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e4b.log; exit 3; }
sha256sum -c pilot/pilot_decision.sha256 >> pipeline_e4b.log 2>&1 || { echo "pilot decision changed -> abort" >> pipeline_e4b.log; exit 3; }
echo "$(date '+%F %T %Z') pipeline start (seeds $SEEDS; deadline $(date -d @$E4B_DEADLINE_EPOCH '+%F %T %Z'))" >> pipeline_e4b.log
if [ ! -f populations_e4b.json ]; then
  $PY e4b.py --stage stage0 >> pipeline_e4b_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage0 rc=$rc" >> pipeline_e4b.log
  [ $rc -eq 0 ] || { echo "STOP after stage0 (rc=$rc)" >> pipeline_e4b.log; exit $rc; }
fi
if [ ! -f predictors_e4b.sha256 ]; then
  $PY e4b.py --stage stage1 >> pipeline_e4b_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage1 rc=$rc" >> pipeline_e4b.log; [ $rc -eq 0 ] || exit $rc
fi
sha256sum -c predictors_e4b.sha256 >> pipeline_e4b.log 2>&1 || exit 8
runpop() { for attempt in 1 2 3; do $PY e4b.py --stage stage2 --pop $1 $2 >> pipeline_e4b_stdout_$1$2.log 2>&1; rc=$?;
           echo "$(date '+%F %T %Z') stage2 $1 $2 attempt $attempt rc=$rc" >> pipeline_e4b.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
rune3() { for attempt in 1 2; do $PY e4b.py --stage e3same --mem-frac 0.9 >> pipeline_e4b_stdout_e3.log 2>&1; rc=$?;
          echo "$(date '+%F %T %Z') e3same attempt $attempt rc=$rc" >> pipeline_e4b.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
if [ "$SEEDS" = "0 1" ]; then
  # phase 1: P (proc A) || S2 then R0-reverse (proc B, stopped via stop-file when P is done)
  rm -f STOP_PHASE1
  ( runpop P; touch STOP_PHASE1 ) &
  PA=$!
  ( runpop S2; export E4B_STOPFILE=$PWD/STOP_PHASE1; runpop R0 --reverse ) &
  PB=$!
  wait $PA; wait $PB; rm -f STOP_PHASE1
  echo "$(date '+%F %T %Z') phase 1 (P, S2) finished" >> pipeline_e4b.log
  # phase 2: E3 subset alone on the GPU (needs ~12 GB)
  rune3
  # phase 3: R0 (both directions), then S1 (both directions)
  ( runpop R0; runpop S1 ) &
  PA=$!
  ( runpop R0 --reverse; runpop S1 --reverse ) &
  PB=$!
else
  ( runpop R0 ) &
  PA=$!
  ( sleep 60; runpop R0 --reverse ) &
  PB=$!
fi
wait $PA; wait $PB
echo "$(date '+%F %T %Z') === merges finished" >> pipeline_e4b.log
$PY e4b.py --stage stage3 >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') stage3 rc=$?" >> pipeline_e4b.log
echo "$(date '+%F %T %Z') === pipeline finished" >> pipeline_e4b.log
