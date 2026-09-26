#!/usr/bin/env bash
# E3 PILOT (exploratory) on the 21 E1 Hub pairs. ubuntu-4070.  bash .../e3_baselines/code/run_pilot_e1.sh
set -euo pipefail
REPO=$HOME/Desktop/Workspace/LoRA
CODE=$REPO/artifacts/e3_baselines/code
OUT=${OUT:-$REPO/artifacts/e3_baselines/pilot_e1}
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
cd "$CODE"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
mkdir -p "$OUT"
sha256sum merges.py e3.py test_merges.py PREREG_E3.md IMPLEMENTATION_NOTES.md > "$OUT/code_at_launch.sha256"
nohup nice -n "${NICE:-10}" "$PY" e3.py --setting e1 --stage all --out "$OUT" --mem-frac "${MEM_FRAC:-0.22}" ${EXTRA:-} \
  >> "$OUT/nohup.out" 2>&1 &
echo "E3 pilot launched: PID $!  log: $OUT/run.log"
