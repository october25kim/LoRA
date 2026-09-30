#!/usr/bin/env bash
# E7 runner (serial, one GPU process). Resumable: each stage skips finished items. Usage: nohup bash run_e7.sh > run_e7_stdout.log 2>&1 &
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
[ -f PLAN_E7.stamp ] || { echo "plan_written_before_evals: $(date '+%F %T %Z')"; sha256sum PLAN_E7.md e7.py run_e7.sh; } > PLAN_E7.stamp
echo $$ > run_e7.pid
run() { for attempt in 1 2 3; do $PY e7.py "$@" >> run_e7_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') $* attempt $attempt rc=$rc" >> run_e7_driver.log
        [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
run --stage a1
run --stage holdtime --n 30
run --stage c
run --stage a2 --pops P
[ -f STOP_AFTER_P ] || run --stage a2 --pops R0,S1
echo "$(date '+%F %T %Z') ALL DONE" >> run_e7_driver.log; echo "$(date '+%F %T %Z') ALL DONE" > STATUS.txt
