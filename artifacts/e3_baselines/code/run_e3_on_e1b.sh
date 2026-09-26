#!/usr/bin/env bash
# E3 on the E1b adapters (PREREG_E3.md). Run on ubuntu-4070 AFTER E1b has finished:
#   bash ~/Desktop/Workspace/LoRA/artifacts/e3_baselines/code/run_e3_on_e1b.sh
# Env overrides: FORCE=1 (skip the "E1b finished" checks), MEM_FRAC (default 0.22 ~ 3.5 GB), NICE (default 10).
# Resumable: re-running continues after the last finished pair (results.jsonl). Analysis is re-run at the end.
set -euo pipefail
REPO=$HOME/Desktop/Workspace/LoRA
CODE=$REPO/artifacts/e3_baselines/code
E1B=$REPO/artifacts/e1b_confirmatory
OUT=$REPO/artifacts/e3_baselines/e1b_run
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
cd "$CODE"
sha256sum -c PREREG_E3.sha256 PREREG_E3_AMENDMENTS.sha256   # prereg, notes, amendments unchanged
[ -f "$E1B/stage0.json" ] || { echo "E1b stage0.json missing -> E1b not ready"; exit 1; }
if [ "${FORCE:-0}" != 1 ]; then
  [ -f "$E1B/VERDICT.md" ] || { echo "E1b VERDICT.md missing -> E1b not finished (FORCE=1 to override)"; exit 1; }
  if pgrep -f "e1b.py|train_e1b" > /dev/null; then echo "an E1b process is still running (FORCE=1 to override)"; exit 1; fi
fi
[ -f "$E1B/pair_results.csv" ] || echo "WARNING: E1b pair_results.csv missing -> TA sanity check vs E1b will be skipped"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
"$PY" -m pytest -q -p no:cacheprovider test_merges.py
mkdir -p "$OUT"
sha256sum merges.py e3.py test_merges.py PREREG_E3.md IMPLEMENTATION_NOTES.md PREREG_E3_AMENDMENTS.md > "$OUT/code_at_launch.sha256"
nohup nice -n "${NICE:-10}" "$PY" e3.py --setting e1b --stage all --out "$OUT" --mem-frac "${MEM_FRAC:-0.22}" \
  >> "$OUT/nohup.out" 2>&1 &
echo "E3-E1b launched: PID $!  log: $OUT/run.log  report (when done): $OUT/E3_E1B_REPORT.md"
