#!/usr/bin/env bash
# E1 launcher for ubuntu-4070 (GPU, full validation). Usage (from ~/Desktop/Workspace/LoRA):
#   bash artifacts/e1_predictive/run_e1_4070.sh [OUT_DIR]            # default OUT_DIR=artifacts/e1_predictive
# Confirmatory re-run that reuses the frozen predictors:
#   mkdir -p artifacts/e1_confirm && cp artifacts/e1_predictive/{e1.py,predictors.csv,predictors_layers.csv,predictors_null.json,predictors.sha256,protocol_decisions.json,protocol_deviations.md} artifacts/e1_confirm/
#   bash artifacts/e1_predictive/run_e1_4070.sh artifacts/e1_confirm
# Stage 1 is skipped when predictors.sha256 already exists; e1.py checks all predictor hashes before Stage 2 and Stage 3.
# Stage 2 can be resumed: finished pairs are appended to pair_results.jsonl and skipped on restart.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root (~/Desktop/Workspace/LoRA)
OUT=${1:-artifacts/e1_predictive}
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
CODE=artifacts/e1_predictive/e1.py
[ -f "$OUT/e1.py" ] && CODE="$OUT/e1.py"
mkdir -p "$OUT"
run() {
  set -e
  [ -f "$OUT/stage0.json" ] || "$PY" "$CODE" --out "$OUT" --stage stage0 --repo-root .
  [ -f "$OUT/predictors.sha256" ] || "$PY" "$CODE" --out "$OUT" --stage stage1 --repo-root .
  (cd "$OUT" && sha256sum -c predictors.sha256)
  "$PY" "$CODE" --out "$OUT" --stage stage2 --repo-root .
  "$PY" "$CODE" --out "$OUT" --stage stage3 --repo-root .
  echo "E1 FINISHED $(date)"
}
export -f run; export OUT PY CODE
nohup bash -c run >> "$OUT/nohup_e1.out" 2>&1 &
echo "E1 launched: PID $!  log: $OUT/run.log (stdout/stderr: $OUT/nohup_e1.out)"
