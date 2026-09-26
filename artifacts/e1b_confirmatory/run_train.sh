#!/usr/bin/env bash
# E1b STEP 1 runner: trains all pre-registered tasks sequentially (resumable: tasks with adapters/<task>/DONE are skipped).
# Launch from anywhere:  nohup bash run_train.sh >> train_nohup.out 2>&1 &
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
(cd . && sha256sum -c prereg.sha256) || { echo "$(date '+%F %T %Z') prereg hash mismatch -> abort" >> train.log; exit 3; }
TASKS=$("$PY" -c "import json;print(' '.join(json.load(open('prereg.json'))['tasks']))")
echo "$(date '+%F %T %Z') === run_train start pid $$ tasks: $TASKS" >> train.log
for t in $TASKS; do
  if [ -f "adapters/$t/DONE" ]; then echo "$(date '+%F %T %Z') skip $t (DONE)" >> train.log; continue; fi
  for attempt in 1 2; do
    echo "$(date '+%F %T %Z') start $t attempt $attempt" >> train.log
    "$PY" train_e1b.py --task "$t" > "logs/train_$t.log" 2>&1
    rc=$?
    echo "$(date '+%F %T %Z') end $t rc=$rc $(tail -1 logs/train_$t.log | cut -c1-200)" >> train.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
done
echo "$(date '+%F %T %Z') === run_train finished" >> train.log
