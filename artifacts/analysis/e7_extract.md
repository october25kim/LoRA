# E7 derived extract (exploratory / post hoc; computed from final E7 CSVs by analysis/e7_extract.py)

## ADA (AdaMerging-style gradient), Qwen2.5-1.5B, n = 273
- mean regret 7.35 pp; median 0.27 pp; max 67.55 pp; 90th percentile 26.78 pp
- fraction of pairs with ADA regret > U1 regret: 0.626; < U1: 0.370
- pairs with ADA regret > 5 pp: 76; > 10 pp: 67
- population P: ADA mean regret 5.96 pp, median 0.24 pp (n = 91)
- population R0: ADA mean regret 7.51 pp, median 0.30 pp (n = 91)
- population S1: ADA mean regret 8.58 pp, median 0.30 pp (n = 91)

## ENT lambda choices
- Qwen2.5-1.5B, G7: ENT chose lambda > 1 in 175 of 273 pairs; U1 in 11
- Qwen2.5-1.5B, G4: ENT chose lambda = 1.0 in 237 of 273 pairs
- Qwen2.5-0.5B, G4: ENT chose lambda = 0.3 in 68 and lambda = 1.0 in 158 of 273 pairs

## Cost ratios (from RESULTS_E7.md section B; Qwen2.5-1.5B, G4)
- held-out tuning timed pairs: 30
- median wall-clock ratio held-out tuning / U1: 2.6; mean ratio 3.0; ADA / U1 median ratio 5.0

## Multi-task tuples
- m = 3: n = 30; mean held-out-tuned test loss 5.18 pp; U1 regret median 0.00 pp; U1 chose lambda_sel in 0.800 of tuples; ENT chose lambda = 1.0 in 20
- m = 4: n = 15; mean held-out-tuned test loss 5.25 pp; U1 regret median 0.00 pp; U1 chose lambda_sel in 0.667 of tuples; ENT chose lambda = 1.0 in 6
- pairs (Qwen2.5-1.5B, G4): U1 chose lambda_sel in 0.601 of pairs
