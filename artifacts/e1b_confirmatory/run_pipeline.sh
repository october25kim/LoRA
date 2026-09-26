#!/usr/bin/env bash
# E1b STEP 2-5 runner. Waits for the D1 full-data MNLI/QQP training to finish, then stage0 -> stage1 -> stage2 -> stage3.
# Resumable: stage0/stage1 are skipped if their outputs exist; stage2 skips finished pairs.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
until grep -q "=== D1 run finished" train.log; do sleep 60; done
for t in $($PY -c "import json;print(\" \".join(json.load(open(\"prereg.json\"))[\"tasks\"]))"); do
  [ -f adapters/$t/DONE ] || { echo "$(date "+%F %T %Z") missing adapter $t -> abort" >> pipeline.log; exit 6; }
done
grep -q deviation adapters/mnli/train_meta.json && grep -q deviation adapters/qqp/train_meta.json || { echo "$(date "+%F %T %Z") mnli/qqp not D1 adapters -> abort" >> pipeline.log; exit 7; }
sha256sum -c prereg.sha256 deviations.sha256 >> pipeline.log 2>&1 || { echo "hash mismatch -> abort" >> pipeline.log; exit 3; }
sha256sum e1b.py tasks.py run_pipeline.sh > code_pipeline.sha256
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> pipeline.log
echo "$(date "+%F %T %Z") pipeline start" >> pipeline.log
if [ ! -f stage0.json ]; then
  $PY e1b.py --stage stage0 >> pipeline_stdout.log 2>&1; rc=$?
  echo "$(date "+%F %T %Z") stage0 rc=$rc" >> pipeline.log
  [ $rc -eq 0 ] || { echo "STOP after stage0 (rc=$rc)" >> pipeline.log; exit $rc; }
fi
if [ ! -f predictors.sha256 ]; then
  $PY e1b.py --stage stage1 >> pipeline_stdout.log 2>&1; rc=$?; echo "$(date "+%F %T %Z") stage1 rc=$rc" >> pipeline.log; [ $rc -eq 0 ] || exit $rc
fi
sha256sum -c predictors.sha256 >> pipeline.log 2>&1 || exit 8
for attempt in 1 2 3; do
  $PY e1b.py --stage stage2 >> pipeline_stdout.log 2>&1; rc=$?; echo "$(date "+%F %T %Z") stage2 attempt $attempt rc=$rc" >> pipeline.log
  [ $rc -eq 0 ] && break; sleep 30
done
[ $rc -eq 0 ] || exit $rc
$PY e1b.py --stage stage3 >> pipeline_stdout.log 2>&1; rc=$?; echo "$(date "+%F %T %Z") stage3 rc=$rc" >> pipeline.log
echo "$(date "+%F %T %Z") === pipeline finished" >> pipeline.log
