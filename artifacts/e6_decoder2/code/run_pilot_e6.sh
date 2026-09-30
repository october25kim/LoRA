#!/usr/bin/env bash
# E6 pre-registered pilot (PREREG_E6.md 1a; = E4b pilot): sst2 + rte, seed 0, lr 1e-4 / 3e-4 / 1e-3 (all trained); held-out-only evaluation; seeds fixed [0,1].
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
sha256sum -c prereg_e6.sha256 >/dev/null || { echo "$(date '+%F %T %Z') prereg hash mismatch -> abort" >> pilot.log; exit 3; }
sha256sum -c code_e6.sha256 >/dev/null || { echo "$(date '+%F %T %Z') code hash mismatch -> abort" >> pilot.log; exit 3; }
mkdir -p logs pilot
[ -f pilot/pilot_start_epoch.txt ] || { date +%s > pilot/pilot_start_epoch.txt; date '+%F %T %Z' > pilot/pilot_start_time.txt; }
for lr in 0.0001 0.0003 0.001; do
  for t in sst2 rte; do
    [ -f pilot/lr$lr/$t/DONE ] && continue
    for div in 1 2; do
      echo "$(date '+%F %T %Z') pilot start $t lr=$lr micro-div=$div" >> pilot.log
      "$PY" train_e6.py --task $t --seed 0 --pilot-lr $lr --micro-div $div > logs/pilot_${t}_lr${lr}.log 2>&1; rc=$?
      echo "$(date '+%F %T %Z') pilot end $t lr=$lr rc=$rc $(tail -1 logs/pilot_${t}_lr${lr}.log | cut -c1-200)" >> pilot.log
      [ $rc -eq 0 ] && break
      rm -rf pilot/lr$lr/$t
    done
    [ -f pilot/lr$lr/$t/DONE ] || { echo "pilot training failed -> abort" >> pilot.log; exit 4; }
  done
done
"$PY" e6.py --stage pilot >> pilot.log 2>&1; rc=$?
echo "$(date '+%F %T %Z') pilot eval rc=$rc" >> pilot.log
[ $rc -eq 0 ] && [ -f pilot/pilot_decision.json ] && sha256sum pilot/pilot_decision.json > pilot/pilot_decision.sha256
echo "$(date '+%F %T %Z') === pilot finished" >> pilot.log
