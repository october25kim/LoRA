#!/usr/bin/env bash
# E5 runner (PREREG_E5.md sec. 6): GPU jobs strictly serial (E5a -> E5b P,R0,S1); E5d on CPU concurrently; then analysis.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
L=$(date +%s); echo $L > launch_epoch.txt; date '+%F %T %Z' > launch_time.txt
echo "$(date '+%F %T %Z') E5 launch" >> pipeline_e5.log
sha256sum -c prereg_e5.sha256 >> pipeline_e5.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e5.log; exit 3; }
sha256sum -c code_e5.sha256 >> pipeline_e5.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e5.log; exit 3; }
nohup env CUDA_VISIBLE_DEVICES= $PY e5.py --stage e5d --device cpu --threads ${E5D_THREADS:-6} > e5d_stdout.log 2>&1 &
echo "$(date '+%F %T %Z') e5d (CPU) started pid $!" >> pipeline_e5.log
for attempt in 1 2 3; do
  E5_DEADLINE_EPOCH=$((L + 25200)) $PY e5.py --stage e5a --mem-frac 0.9 >> e5a_stdout.log 2>&1; rc=$?
  echo "$(date '+%F %T %Z') e5a attempt $attempt rc=$rc" >> pipeline_e5.log; [ $rc -eq 0 ] && break; sleep 30
done
for attempt in 1 2 3; do
  E5_DEADLINE_EPOCH=$((L + 41400)) $PY e5.py --stage e5b --mem-frac 0.9 >> e5b_stdout.log 2>&1; rc=$?
  echo "$(date '+%F %T %Z') e5b attempt $attempt rc=$rc" >> pipeline_e5.log; [ $rc -eq 0 ] && break; sleep 30
done
echo "$(date '+%F %T %Z') === GPU work finished" >> pipeline_e5.log
wait
$PY e5_analysis.py e5a e5b e5d >> analysis_stdout.log 2>&1; echo "$(date '+%F %T %Z') analysis rc=$?" >> pipeline_e5.log
echo "$(date '+%F %T %Z') === E5 pipeline finished" >> pipeline_e5.log
