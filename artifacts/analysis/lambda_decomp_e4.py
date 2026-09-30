"""Derived (post hoc, exploratory): lambda decomposition of TA pair loss for E4a/E4b populations.
D_selected = pre-registered D (held-out-selected lambda); D_oracle = min over the 4-point grid of eval D;
overshoot = D_selected - D_oracle; D(lambda=1) and its excess over D_oracle. Means only (no inference).
Input: e4a/pair_results_{P,R0,S1}.jsonl, e4b/pair_results_{P,R0,S1}.jsonl (dedup by pair, first record kept)."""
import json, numpy as np
GRID = ['0.3', '0.5', '0.7', '1.0']
rows = []
for bb, name in [('e4a', 'RoBERTa'), ('e4b', 'Qwen')]:
    for pop in ['P', 'R0', 'S1']:
        R = {}
        for l in open(f'{bb}/pair_results_{pop}.jsonl'):
            r = json.loads(l); R.setdefault(r['pair'], r)
        R = list(R.values())
        Dsel = np.array([r['D'] for r in R])
        Dgrid = np.array([[r[f'TA_eval_D_lam{g}'] for g in GRID] for r in R])
        Dor = Dgrid.min(1); D1 = Dgrid[:, 3]
        lam_or = np.array(GRID)[Dgrid.argmin(1)]
        rows.append(dict(backbone=name, pop=pop, n=len(R), D_selected=Dsel.mean(), D_oracle=Dor.mean(),
                         overshoot=(Dsel - Dor).mean(), D_lam1=D1.mean(), excess_lam1=(D1 - Dor).mean(),
                         frac_oracle_at_1=float((lam_or == '1.0').mean())))
json.dump(rows, open('analysis/lambda_decomp_e4.json', 'w'), indent=1)
with open('analysis/lambda_decomp_e4.md', 'w') as f:
    f.write('# Lambda decomposition, E4 populations (derived; exploratory; means over pairs, no inference)\n\n')
    f.write('| backbone | pop | n | D_selected | D_oracle | overshoot | D(λ=1) | D(λ=1) − D_oracle | share of pairs with oracle λ = 1.0 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:
        f.write(f"| {r['backbone']} | {r['pop']} | {r['n']} | {r['D_selected']:.4f} | {r['D_oracle']:.4f} | {r['overshoot']:.4f} | {r['D_lam1']:.4f} | {r['excess_lam1']:.4f} | {r['frac_oracle_at_1']:.2f} |\n")
print(open('analysis/lambda_decomp_e4.md').read())
