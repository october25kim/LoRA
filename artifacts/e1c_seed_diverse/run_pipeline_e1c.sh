#!/usr/bin/env bash
# E1c STEP 2-4 runner: waits for all 14 seed-1 adapters, then stage0 -> stage1 (freeze) -> merges in 2 concurrent GPU processes:
#   proc A: P (primary);  proc B: S2 -> e3same -> S1.   Resumable (stage2 skips finished pairs).
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg_e1c.json'))['tasks']))")
for t in $TASKS; do until [ -f "adapters_s1/$t/DONE" ]; do sleep 30; done; done
sha256sum -c prereg_e1c.sha256 >> pipeline_e1c.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e1c.log; exit 3; }
sha256sum -c code_e1c.sha256 >> pipeline_e1c.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e1c.log; exit 3; }
echo "$(date '+%F %T %Z') pipeline start" >> pipeline_e1c.log
if [ ! -f populations_e1c.json ]; then
  $PY e1c.py --stage stage0 >> pipeline_e1c_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage0 rc=$rc" >> pipeline_e1c.log
  [ $rc -eq 0 ] || { echo "STOP after stage0 (rc=$rc)" >> pipeline_e1c.log; exit $rc; }
fi
if [ ! -f predictors_e1c.sha256 ]; then
  $PY e1c.py --stage stage1 >> pipeline_e1c_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') stage1 rc=$rc" >> pipeline_e1c.log; [ $rc -eq 0 ] || exit $rc
fi
sha256sum -c predictors_e1c.sha256 >> pipeline_e1c.log 2>&1 || exit 8
runpop() { for attempt in 1 2 3; do $PY e1c.py --stage stage2 --pop $1 >> pipeline_e1c_stdout_$1.log 2>&1; rc=$?;
           echo "$(date '+%F %T %Z') stage2 $1 attempt $attempt rc=$rc" >> pipeline_e1c.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
( runpop P ) &
PA=$!
( runpop S2; $PY e1c.py --stage e3same --mem-frac 0.45 >> pipeline_e1c_stdout_e3.log 2>&1; echo "$(date '+%F %T %Z') e3same rc=$?" >> pipeline_e1c.log; runpop S1 ) &
PB=$!
wait $PA; wait $PB
echo "$(date '+%F %T %Z') === merges finished" >> pipeline_e1c.log
