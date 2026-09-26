#!/usr/bin/env bash
# E1c STEP 1 runner. Usage: bash run_train_e1c.sh <queue-name> <task> [<task> ...]   (resumable via adapters_s1/<task>/DONE)
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
Q=$1; shift
sha256sum -c prereg_e1c.sha256 >/dev/null || { echo "$(date '+%F %T %Z') [$Q] prereg hash mismatch -> abort" >> train_e1c.log; exit 3; }
mkdir -p logs adapters_s1
echo "$(date '+%F %T %Z') [$Q] === start pid $$ tasks: $*" >> train_e1c.log
for t in "$@"; do
  if [ -f "adapters_s1/$t/DONE" ]; then echo "$(date '+%F %T %Z') [$Q] skip $t (DONE)" >> train_e1c.log; continue; fi
  for attempt in 1 2; do
    echo "$(date '+%F %T %Z') [$Q] start $t attempt $attempt" >> train_e1c.log
    "$PY" train_e1c.py --task "$t" > "logs/train_s1_$t.log" 2>&1
    rc=$?
    echo "$(date '+%F %T %Z') [$Q] end $t rc=$rc $(tail -1 logs/train_s1_$t.log | cut -c1-200)" >> train_e1c.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
done
echo "$(date '+%F %T %Z') [$Q] === finished" >> train_e1c.log
