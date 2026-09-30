#!/usr/bin/env python3
"""Derived descriptive quantities from the final E7 per-pair/per-tuple CSVs and jsonl records
(e7_baselines/final/e7_baselines). Exploratory/post hoc; no new inference. Writes analysis/e7_extract.md."""
import json
from pathlib import Path
import csv
import numpy as np

class Frame:
    """Minimal column store (pandas is not installed in the analysis venv)."""
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, k):
        if isinstance(k, str):
            v = [r[k] for r in self.rows]
            try: return np.array([float(x) if x != "" else np.nan for x in v])
            except ValueError: return np.array(v)
        return Frame([r for r, m in zip(self.rows, k) if m])

def read_csv(p): return Frame(list(csv.DictReader(open(p))))
R = Path(__file__).resolve().parents[1]
D = R / "e7_baselines/final/e7_baselines"
A = read_csv(D / "e7_A_per_pair.csv"); Q = read_csv(D / "e7_A_q05_per_pair.csv"); C = read_csv(D / "e7_C_per_tuple.csv")
L = []
w = L.append
w("# E7 derived extract (exploratory / post hoc; computed from final E7 CSVs by analysis/e7_extract.py)\n")
ada = A["A2_regret_ADA"]; u1 = A["G4_regret_U1_agree"]
w("## ADA (AdaMerging-style gradient), Qwen2.5-1.5B, n = %d" % int(np.isfinite(ada).sum()))
w(f"- mean regret {np.mean(ada):.2f} pp; median {np.median(ada):.2f} pp; max {np.max(ada):.2f} pp; 90th percentile {np.quantile(ada,0.9):.2f} pp")
w(f"- fraction of pairs with ADA regret > U1 regret: {(ada > u1).mean():.3f}; < U1: {(ada < u1).mean():.3f}")
w(f"- pairs with ADA regret > 5 pp: {(ada > 5).sum()}; > 10 pp: {(ada > 10).sum()}")
for p in ["P", "R0", "S1"]:
    s = A[A["pop"] == p]
    w(f"- population {p}: ADA mean regret {np.mean(s['A2_regret_ADA']):.2f} pp, median {np.median(s['A2_regret_ADA']):.2f} pp (n = {len(s)})")
w("")
w("## ENT lambda choices")
g7 = A["G7_lam_ENT_min_entropy"]; w(f"- Qwen2.5-1.5B, G7: ENT chose lambda > 1 in {(g7 > 1).sum()} of {len(g7)} pairs; U1 in {(A['G7_lam_U1_agree'] > 1).sum()}")
w(f"- Qwen2.5-1.5B, G4: ENT chose lambda = 1.0 in {(A['G4_lam_ENT_min_entropy'] == 1.0).sum()} of {len(A)} pairs")
w(f"- Qwen2.5-0.5B, G4: ENT chose lambda = 0.3 in {(Q['G4_lam_ENT_min_entropy'] == 0.3).sum()} and lambda = 1.0 in {(Q['G4_lam_ENT_min_entropy'] == 1.0).sum()} of {len(Q)} pairs")
w("")
w("## Cost ratios (from RESULTS_E7.md section B; Qwen2.5-1.5B, G4)")
hold = [json.loads(l) for l in open(D / "holdtime.jsonl")]
w(f"- held-out tuning timed pairs: {len(hold)}")
u1_med, hold_med, u1_mean, hold_mean, ada_med = 6.35, 16.81, 7.65, 22.88, 31.70
w(f"- median wall-clock ratio held-out tuning / U1: {hold_med/u1_med:.1f}; mean ratio {hold_mean/u1_mean:.1f}; ADA / U1 median ratio {ada_med/u1_med:.1f}")
w("")
w("## Multi-task tuples")
for m in [3, 4]:
    s = C[C["m"] == m]
    w(f"- m = {m}: n = {len(s)}; mean held-out-tuned test loss {100*np.mean(s['G4_Dtest_sel']):.2f} pp; U1 regret median {np.median(s['G4_regret_U1_agree']):.2f} pp; "
      f"U1 chose lambda_sel in {(s['G4_lam_U1_agree'] == s['G4_lam_sel']).mean():.3f} of tuples; ENT chose lambda = 1.0 in {(s['G4_lam_ENT_min_entropy'] == 1.0).sum()}")
w(f"- pairs (Qwen2.5-1.5B, G4): U1 chose lambda_sel in {(A['G4_lam_U1_agree'] == A['lam_sel']).mean():.3f} of pairs")
(R / "analysis/e7_extract.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
