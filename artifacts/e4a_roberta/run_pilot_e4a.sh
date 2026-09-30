#!/usr/bin/env bash
# E4a pre-registered pilot (PREREG_E4A.md 1a): sst2 + rte, seed 0, lr ladder 1e-3 -> 5e-4 -> 2e-4; held-out-only evaluation.
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
sha256sum -c prereg_e4a.sha256 >/dev/null || { echo "$(date '+%F %T %Z') prereg hash mismatch -> abort" >> pilot.log; exit 3; }
mkdir -p logs pilot
for lr in 0.001 0.0005 0.0002; do
  for t in sst2 rte; do
    echo "$(date '+%F %T %Z') pilot start $t lr=$lr" >> pilot.log
    "$PY" train_e4a.py --task $t --seed 0 --pilot-lr $lr > logs/pilot_${t}_lr${lr}.log 2>&1; rc=$?
    echo "$(date '+%F %T %Z') pilot end $t lr=$lr rc=$rc $(tail -1 logs/pilot_${t}_lr${lr}.log | cut -c1-200)" >> pilot.log
    [ $rc -eq 0 ] || { echo "pilot training failed -> abort" >> pilot.log; exit 4; }
  done
  "$PY" e4a.py --stage pilot >> pilot.log 2>&1; rc=$?
  echo "$(date '+%F %T %Z') pilot eval after lr=$lr rc=$rc" >> pilot.log
  [ $rc -eq 0 ] && break
  [ $rc -eq 10 ] || { echo "pilot eval error -> abort" >> pilot.log; exit 5; }
done
[ -f pilot/pilot_decision.json ] && sha256sum pilot/pilot_decision.json > pilot/pilot_decision.sha256
echo "$(date '+%F %T %Z') === pilot finished" >> pilot.log
