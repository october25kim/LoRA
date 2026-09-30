"""Writes gpu_time_summary.json (E4a): wall-clock from pilot start to the end of the last GPU job + summed process time. Reporting only."""
import json, glob, re, datetime as dt
from pathlib import Path
H = Path(__file__).resolve().parent
def p(s): return dt.datetime.strptime(s.strip()[:19], "%Y-%m-%d %H:%M:%S")
start = p((H / "pilot_launch_time.txt").read_text())
ends = [l for l in (H / "pipeline_e4a.log").read_text().splitlines() if "stage3 rc=" in l or "merges finished" in l]
end_merges = p([l for l in ends if "merges finished" in l][-1][:19]); end_all = p([l for l in ends if "stage3 rc=" in l][-1][:19])
train = {}
for f in glob.glob(str(H / "adapters_s*/*/train_meta.json")):
    m = json.loads(Path(f).read_text()); train[f"{m['task']}@s{m['seed']}"] = m["train_wall_s"]
pilot = {}
for f in glob.glob(str(H / "pilot/lr*/*/train_meta.json")):
    m = json.loads(Path(f).read_text()); pilot[f"{m['task']}@lr{m['recipe']['lr']}"] = m["train_wall_s"]
tm = {Path(f).stem: json.loads(Path(f).read_text()) for f in glob.glob(str(H / "timing_*.json"))}
tl = (H / "train_e4a.log").read_text().splitlines()
t_first = p(re.search(r"^(\S+ \S+)", tl[0]).group(1)); t_last = max(p(l[:19]) for l in tl if "=== finished" in l)
out = {"pilot_start_kst": str(start), "training_start_kst": str(t_first), "training_end_kst": str(t_last), "merges_end_kst": str(end_merges), "stage3_end_kst": str(end_all),
       "wall_clock_h_pilot_to_last_gpu_job": (end_merges - start).total_seconds() / 3600, "wall_clock_h_incl_stage3": (end_all - start).total_seconds() / 3600,
       "cap_h": 10, "training_process_h": sum(train.values()) / 3600, "pilot_process_h": sum(pilot.values()) / 3600,
       "stage2_process_h": sum(v.get("stage2_gpu_wall_s", 0) for v in tm.values()) / 3600, "e3same_h": tm.get("timing_S2", {}).get("e3same_s", 0) / 3600,
       "stage0_h": tm.get("timing_main", {}).get("stage0_s", 0) / 3600, "stage1_h": tm.get("timing_main", {}).get("stage1_s", 0) / 3600,
       "train_wall_s_by_adapter": train, "pilot_wall_s": pilot}
(H / "gpu_time_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
