#!/usr/bin/env bash
# E6 STEP 1 runner (= E4b runner). Usage: bash run_train_e6.sh <queue-name> <task@sSEED> [...]  (resumable via adapters_s<seed>/<task>/DONE)
# Pre-registered OOM fallback: on failure, retry with the micro-batch halved (--micro-div 2, then 4); effective batch stays 32.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
Q=$1; shift
sha256sum -c prereg_e6.sha256 >/dev/null || { echo "$(date '+%F %T %Z') [$Q] prereg hash mismatch -> abort" >> train_e6.log; exit 3; }
sha256sum -c code_e6.sha256 >/dev/null || { echo "$(date '+%F %T %Z') [$Q] code hash mismatch -> abort" >> train_e6.log; exit 3; }
sha256sum -c pilot/pilot_decision.sha256 >/dev/null || { echo "$(date '+%F %T %Z') [$Q] pilot decision missing/changed -> abort" >> train_e6.log; exit 3; }
mkdir -p logs
echo "$(date '+%F %T %Z') [$Q] === start pid $$ items: $*" >> train_e6.log
for it in "$@"; do
  t=${it%@s*}; s=${it#*@s}
  if [ -f "adapters_s$s/$t/DONE" ]; then echo "$(date '+%F %T %Z') [$Q] skip $it (DONE)" >> train_e6.log; continue; fi
  for div in 1 2 4; do
    echo "$(date '+%F %T %Z') [$Q] start $it micro-div $div" >> train_e6.log
    "$PY" train_e6.py --task "$t" --seed "$s" --micro-div $div > "logs/train_s${s}_$t.log" 2>&1
    rc=$?
    echo "$(date '+%F %T %Z') [$Q] end $it rc=$rc $(tail -1 logs/train_s${s}_$t.log | cut -c1-200)" >> train_e6.log
    [ $rc -eq 0 ] && break
    cp "logs/train_s${s}_$t.log" "logs/train_s${s}_${t}_fail_div$div.log"; rm -rf "adapters_s$s/$t"; sleep 20
  done
done
echo "$(date '+%F %T %Z') [$Q] === finished" >> train_e6.log
