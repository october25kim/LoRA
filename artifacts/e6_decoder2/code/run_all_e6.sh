#!/usr/bin/env bash
# E6 master runner (serial, ONE GPU process at a time): pilot -> training (14 tasks x seed 0, then 14 tasks x seed 1) -> pipeline
# (stage0 -> stage1 -> merges P, S2, R0, S1 -> stage3 -> lam_confirm -> verdict). Every step verifies prereg_e6.sha256 / code_e6.sha256.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
echo "$(date '+%F %T %Z') run_all start pid $$" >> run_all_e6.log
bash run_pilot_e6.sh; rc=$?
echo "$(date '+%F %T %Z') pilot rc=$rc" >> run_all_e6.log
[ -f pilot/pilot_decision.sha256 ] || { echo "$(date '+%F %T %Z') STOP: no pilot decision" >> run_all_e6.log; exit 4; }
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg_e6.json'))['tasks']))")
ITEMS=()
for s in 0 1; do for t in $TASKS; do ITEMS+=("$t@s$s"); done; done
bash run_train_e6.sh Q1 "${ITEMS[@]}"; rc=$?
echo "$(date '+%F %T %Z') training rc=$rc" >> run_all_e6.log
for it in "${ITEMS[@]}"; do t=${it%@s*}; s=${it#*@s}; [ -f "adapters_s$s/$t/DONE" ] || { echo "$(date '+%F %T %Z') STOP: adapter $it missing after training (see train_e6.log)" >> run_all_e6.log; exit 5; }; done
bash run_pipeline_e6.sh; rc=$?
echo "$(date '+%F %T %Z') pipeline rc=$rc" >> run_all_e6.log
nvidia-smi --query-gpu=name --format=csv,noheader >> run_all_e6.log 2>&1
echo "$(date '+%F %T %Z') === run_all finished" >> run_all_e6.log
