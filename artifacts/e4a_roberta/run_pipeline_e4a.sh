#!/usr/bin/env bash
# E4a STEP 2-4 runner: waits for all 28 adapters, then stage0 -> stage1 (freeze) -> merges in 2 concurrent GPU processes:
#   proc A: P (primary) -> S1 (forward);  proc B: R0 -> S2 -> e3same -> S1 (reverse). Resumable (stage2 skips finished pairs).
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg_e4a.json'))['tasks']))")
for s in 0 1; do for t in $TASKS; do until [ -f "adapters_s$s/$t/DONE" ]; do sleep 30; done; done; done
sha256sum -c prereg_e4a.sha256 >> pipeline_e4a.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e4a.log; exit 3; }
sha256sum -c code_e4a.sha256 >> pipeline_e4a.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e4a.log; exit 3; }
sha256sum -c code_analysis_e4a.sha256 >> pipeline_e4a.log 2>&1 || { echo "analysis code hash mismatch -> abort" >> pipeline_e4a.log; exit 3; }
echo "$(date '+%F %T %Z') pipeline start" >> pipeline_e4a.log
if [ ! -f populations_e4a.json ]; then
  $PY e4a.py --stage stage0 >> pipeline_e4a_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage0 rc=$rc" >> pipeline_e4a.log
  [ $rc -eq 0 ] || { echo "STOP after stage0 (rc=$rc)" >> pipeline_e4a.log; exit $rc; }
fi
if [ ! -f predictors_e4a.sha256 ]; then
  $PY e4a.py --stage stage1 >> pipeline_e4a_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage1 rc=$rc" >> pipeline_e4a.log; [ $rc -eq 0 ] || exit $rc
fi
sha256sum -c predictors_e4a.sha256 >> pipeline_e4a.log 2>&1 || exit 8
runpop() { for attempt in 1 2 3; do $PY e4a.py --stage stage2 --pop $1 $2 >> pipeline_e4a_stdout_$1$2.log 2>&1; rc=$?;
           echo "$(date '+%F %T %Z') stage2 $1 $2 attempt $attempt rc=$rc" >> pipeline_e4a.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
( runpop P; runpop S1 ) &
PA=$!
( runpop R0; runpop S2; $PY e4a.py --stage e3same --mem-frac 0.45 >> pipeline_e4a_stdout_e3.log 2>&1; echo "$(date '+%F %T %Z') e3same rc=$?" >> pipeline_e4a.log; runpop S1 --reverse ) &
PB=$!
wait $PA; wait $PB
echo "$(date '+%F %T %Z') === merges finished" >> pipeline_e4a.log
$PY e4a.py --stage stage3 >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') stage3 rc=$?" >> pipeline_e4a.log
echo "$(date '+%F %T %Z') === pipeline finished" >> pipeline_e4a.log
