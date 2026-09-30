#!/usr/bin/env bash
# E6 STEP 2-4 runner (serial, ONE GPU process at a time; = the E4b serial continuation D4 applied from the start):
#   waits for all 28 adapters -> stage0 (integrity) -> stage1 (predictors, frozen) -> merges P -> S2 -> R0 -> S1 -> stage3 -> lam_confirm (CPU) -> verdict.
# Budget guard: E6_DEADLINE_EPOCH = pilot start + 39 h (no new merge pair after it). Resumable (per pair).
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export E6_DEADLINE_EPOCH=$(( $(cat pilot/pilot_start_epoch.txt) + 140400 ))
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg_e6.json'))['tasks']))")
SEEDS=$("$PY" -c "import json;print(' '.join(map(str, json.load(open('pilot/pilot_decision.json'))['seeds'])))")
echo "$(date '+%F %T %Z') pipeline waiting for adapters (seeds $SEEDS)" >> pipeline_e6.log
for s in $SEEDS; do for t in $TASKS; do until [ -f "adapters_s$s/$t/DONE" ]; do sleep 60; done; done; done
sha256sum -c prereg_e6.sha256 >> pipeline_e6.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e6.log; exit 3; }
sha256sum -c code_e6.sha256 >> pipeline_e6.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e6.log; exit 3; }
sha256sum -c pilot/pilot_decision.sha256 >> pipeline_e6.log 2>&1 || { echo "pilot decision changed -> abort" >> pipeline_e6.log; exit 3; }
echo "$(date '+%F %T %Z') pipeline start (seeds $SEEDS; deadline $(date -d @$E6_DEADLINE_EPOCH '+%F %T %Z'))" >> pipeline_e6.log
if [ ! -f populations_e6.json ]; then
  $PY e6.py --stage stage0 >> pipeline_e6_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage0 rc=$rc" >> pipeline_e6.log
  [ $rc -eq 0 ] || { echo "STOP after stage0 (rc=$rc)" >> pipeline_e6.log; exit $rc; }
fi
if [ ! -f predictors_e6.sha256 ]; then
  $PY e6.py --stage stage1 >> pipeline_e6_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage1 rc=$rc" >> pipeline_e6.log; [ $rc -eq 0 ] || exit $rc
fi
sha256sum -c predictors_e6.sha256 >> pipeline_e6.log 2>&1 || exit 8
runpop() { for attempt in 1 2 3; do $PY e6.py --stage stage2 --pop $1 >> pipeline_e6_stdout_$1.log 2>&1; rc=$?;
           echo "$(date '+%F %T %Z') stage2 $1 attempt $attempt rc=$rc" >> pipeline_e6.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
runpop P
runpop S2
runpop R0
runpop S1
echo "$(date '+%F %T %Z') === merges finished" >> pipeline_e6.log
$PY e6.py --stage stage3 >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') stage3 rc=$?" >> pipeline_e6.log
$PY lam_confirm_e6.py >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') lam_confirm rc=$?" >> pipeline_e6.log
$PY make_verdict_e6.py >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') verdict rc=$?" >> pipeline_e6.log
echo "$(date '+%F %T %Z') === pipeline finished" >> pipeline_e6.log
