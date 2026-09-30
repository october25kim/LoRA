#!/usr/bin/env bash
# E4b deviation D4 (orchestration only; no computation changed): continuation of run_pipeline_e4b.sh after phase-1 process B
# (S2, R0-reverse) ran out of memory next to the P process (~13 GB peak with real adapters/eval sets). After P finishes,
# the remaining work runs SERIALLY in one GPU process at a time, keeping the pre-registered priority P -> S2+E3 -> R0 -> S1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export E4B_DEADLINE_EPOCH=$(( $(cat pilot/pilot_start_epoch.txt) + 48960 ))
echo "$(date '+%F %T %Z') serial continuation started (D4); waiting for P (STOP_PHASE1)" >> pipeline_e4b.log
until [ -f STOP_PHASE1 ]; do sleep 30; done; rm -f STOP_PHASE1
echo "$(date '+%F %T %Z') phase 1 (P) finished" >> pipeline_e4b.log
sha256sum -c prereg_e4b.sha256 >> pipeline_e4b.log 2>&1 || { echo "prereg hash mismatch -> abort" >> pipeline_e4b.log; exit 3; }
sha256sum -c code_e4b.sha256 >> pipeline_e4b.log 2>&1 || { echo "code hash mismatch -> abort" >> pipeline_e4b.log; exit 3; }
sha256sum -c predictors_e4b.sha256 >> pipeline_e4b.log 2>&1 || exit 8
runpop() { for attempt in 1 2 3; do $PY e4b.py --stage stage2 --pop $1 $2 >> pipeline_e4b_stdout_$1$2.log 2>&1; rc=$?;
           echo "$(date '+%F %T %Z') stage2 $1 $2 attempt $attempt rc=$rc" >> pipeline_e4b.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
rune3() { for attempt in 1 2; do $PY e4b.py --stage e3same --mem-frac 0.9 >> pipeline_e4b_stdout_e3.log 2>&1; rc=$?;
          echo "$(date '+%F %T %Z') e3same attempt $attempt rc=$rc" >> pipeline_e4b.log; [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
runpop S2
rune3
runpop R0
runpop S1
echo "$(date '+%F %T %Z') === merges finished" >> pipeline_e4b.log
$PY e4b.py --stage stage3 >> stage3_stdout.log 2>&1; echo "$(date '+%F %T %Z') stage3 rc=$?" >> pipeline_e4b.log
echo "$(date '+%F %T %Z') === pipeline finished" >> pipeline_e4b.log
