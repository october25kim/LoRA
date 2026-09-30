#!/usr/bin/env bash
# waits for run_e7.sh (which stops after a2/P because STOP_AFTER_P exists), then: 0.5B entropy (a1_q05) -> a2 R0,S1 -> analysis.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
echo $$ > run_e7_followup.pid
while kill -0 $(cat run_e7.pid) 2>/dev/null; do sleep 60; done
run() { for attempt in 1 2 3; do $PY "$@" >> run_e7_stdout.log 2>&1; rc=$?; echo "$(date '+%F %T %Z') $* attempt $attempt rc=$rc" >> run_e7_driver.log
        [ $rc -eq 0 ] && return 0; sleep 30; done; return 1; }
run e7_q05.py
run e7.py --stage a2 --pops R0,S1
run e7_analysis.py
echo "$(date '+%F %T %Z') FOLLOWUP DONE (all stages)" >> run_e7_driver.log; echo "$(date '+%F %T %Z') ALL DONE (incl. followup)" > STATUS.txt
