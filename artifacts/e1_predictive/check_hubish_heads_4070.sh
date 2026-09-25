#!/usr/bin/env bash
# Stage 0 item 4: does each local hubish adapter save its classifier head? (run on ubuntu-4070)
cd ~/Desktop/Workspace/LoRA || exit 1
PY=${PY:-$HOME/miniconda3/envs/torch/bin/python}
for a in adapters/mnli_s7_hubish adapters/mnli_s42_hubish adapters/rte_s42_hubish; do
  [ -f "$a/adapter_model.safetensors" ] || { echo "$a: missing"; continue; }
  "$PY" - "$a" <<'PY'
import json, sys
from safetensors.torch import load_file
a = sys.argv[1]
cfg = json.load(open(f"{a}/adapter_config.json"))
sd = load_file(f"{a}/adapter_model.safetensors")
hk = [k for k in sd if "classifier" in k or "modules_to_save" in k or "score" in k]
print(f"{a}: modules_to_save(cfg)={cfg.get('modules_to_save')} task_type={cfg.get('task_type')} n_keys={len(sd)}")
for k in hk:
    v = sd[k].float()
    print(f"   {k} {tuple(v.shape)} norm={v.norm():.4f} max|.|={v.abs().max():.4f}")
if not hk:
    print("   NO classifier / modules_to_save keys -> head would be randomly re-initialised at load")
PY
done
