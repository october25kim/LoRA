#!/usr/bin/env bash
# Deviation D1 runner: waits for the original run_train.sh (PID in $1) to finish, renames the capped MNLI/QQP adapters to
# *_cap60k (kept, not used), then trains full-data MNLI and QQP with train_e1b_d1.py. Resumable (DONE markers).
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done; fi
grep -q "run_train finished" train.log || { echo "$(date '+%F %T %Z') D1: original run not finished cleanly -> abort" >> train.log; exit 4; }
sha256sum -c deviations.sha256 || { echo "$(date '+%F %T %Z') D1: deviations hash mismatch -> abort" >> train.log; exit 3; }
for t in mnli qqp; do
  if [ -f "adapters/$t/DONE" ] && ! grep -q '"deviation"' "adapters/$t/train_meta.json"; then
    [ -e "adapters/${t}_cap60k" ] && { echo "$(date '+%F %T %Z') D1: adapters/${t}_cap60k exists -> abort" >> train.log; exit 5; }
    mv "adapters/$t" "adapters/${t}_cap60k" && echo "$(date '+%F %T %Z') D1: renamed adapters/$t -> adapters/${t}_cap60k" >> train.log
  fi
done
for t in mnli qqp; do
  if [ -f "adapters/$t/DONE" ]; then echo "$(date '+%F %T %Z') D1 skip $t (DONE)" >> train.log; continue; fi
  for attempt in 1 2; do
    echo "$(date '+%F %T %Z') D1 start $t (full data) attempt $attempt" >> train.log
    "$PY" train_e1b_d1.py --task "$t" > "logs/train_${t}_full.log" 2>&1
    rc=$?
    echo "$(date '+%F %T %Z') D1 end $t rc=$rc $(tail -1 logs/train_${t}_full.log | cut -c1-200)" >> train.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
done
echo "$(date '+%F %T %Z') === D1 run finished" >> train.log
