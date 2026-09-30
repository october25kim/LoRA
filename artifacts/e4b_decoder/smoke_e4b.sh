#!/usr/bin/env bash
# E4b pipeline smoke test in a scratch copy (~/e4_smoke/e4b_decoder); 20-step throwaway adapters; nothing here is study evidence.
set -u
R=~/Desktop/Workspace/LoRA/artifacts
S=~/e4_smoke; rm -rf $S; mkdir -p $S/e4b_decoder
ln -s $R/e1b_confirmatory $S/e1b_confirmatory; ln -s $R/e3_baselines $S/e3_baselines; ln -s $R/e4a_roberta $S/e4a_roberta; ln -s $R/e1c_seed_diverse $S/e1c_seed_diverse
cd $R/e4b_decoder && cp tasks_e4b.py train_e4b.py e4b.py e4b_analysis.py make_verdict_e4b.py prereg_e4b.json $S/e4b_decoder/
cd $S/e4b_decoder
export E4B_SMOKE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
PY=~/miniconda3/envs/torch/bin/python
$PY - <<'P'
import json; p = json.load(open("prereg_e4b.json")); p["tasks"] = ["sst2", "stsb", "rte", "trec", "imdb"]; p["integrity"]["stop_if_valid_tasks_below"] = 2
json.dump(p, open("prereg_e4b.json", "w"), indent=1)
P
for s in 0 1; do for t in sst2 stsb rte trec; do for div in 1 2 4; do rm -rf smoke/s$s/$t; $PY train_e4b.py --task $t --seed $s --micro-div $div > smoke_train_${t}_s$s.log 2>&1 && break; echo "TRAIN FAIL $t $s div $div"; done; tail -1 smoke_train_${t}_s$s.log; done; done
for div in 1 2 4; do rm -rf smoke/s0/imdb; $PY train_e4b.py --task imdb --seed 0 --micro-div $div > smoke_train_imdb_s0.log 2>&1 && break; echo "TRAIN FAIL imdb div $div"; done; tail -1 smoke_train_imdb_s0.log
for t in sst2 stsb rte trec; do $PY -c "import json;m=json.load(open('smoke/s0/$t/train_meta.json'));print('MEM $t', round(m['peak_mem_gb'],2), 'sec/step', round(m['sec_per_step'],3), 'micro', m['micro_batch'])"; done
$PY -c "import json;m=json.load(open('smoke/s0/imdb/train_meta.json'));print('IMDB peak mem GB', m['peak_mem_gb'], 'sec/step', m['sec_per_step'], 'micro', m['micro_batch'])"
$PY - <<'P'
import json; p = json.load(open("prereg_e4b.json")); p["tasks"] = ["sst2", "stsb", "rte", "trec"]
json.dump(p, open("prereg_e4b.json", "w"), indent=1)
P
ln -s smoke/s0 adapters_s0; ln -s smoke/s1 adapters_s1
mkdir -p pilot; for lr in 0.0001 0.0003 0.001; do mkdir -p pilot/lr$lr; ln -s ../../smoke/s0/sst2 pilot/lr$lr/sst2; ln -s ../../smoke/s0/rte pilot/lr$lr/rte; done
date +%s > pilot/pilot_start_epoch.txt
$PY e4b.py --stage pilot || echo "PILOT STAGE FAIL"
$PY - <<'P'
import json; d = json.load(open("pilot/pilot_decision.json")); print("SMOKE pilot decision", d["lr_main"], d["seeds"], json.dumps(d["projection"])[:600])
d["seeds"] = [0, 1]; d["primary_population"] = "P"; json.dump(d, open("pilot/pilot_decision.json", "w"), indent=1)
P
$PY e4b.py --stage stage0 || echo "STAGE0 FAIL"
$PY e4b.py --stage stage1 || echo "STAGE1 FAIL"
for pop in P S2 R0 S1; do $PY e4b.py --stage stage2 --pop $pop || echo "STAGE2 $pop FAIL"; done
$PY e4b.py --stage e3same --mem-frac 0.5 || echo "E3 FAIL"
$PY e4b.py --stage stage3 || echo "STAGE3 FAIL"
$PY make_verdict_e4b.py || echo "VERDICT FAIL"
$PY - <<'P'
import sys, json; sys.path.insert(0, ".")
import e4b, torch
from pathlib import Path
e4b.THETA_STAR = 89.0
PO = json.loads(Path("populations_e4b.json").read_text())
e4b.run_merges([("trec@s0", "trec@s1")], Path("gate_test.jsonl"), Path("preds_gt"), torch.device("cuda"), PO["lora_layers"], e4b.singles(), 128, "gatetest")
r = json.loads(Path("gate_test.jsonl").read_text().splitlines()[0]); print("GATE_TEST", r["gate_n_fail_layers"], r["GATE_atTA_score"], r["GATE_own_score"], r["TA_score_sel"])
P
echo SMOKE_DONE
