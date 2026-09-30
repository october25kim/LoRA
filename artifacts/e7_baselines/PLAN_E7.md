# E7 — PLAN / DEVIATIONS note (EXPLORATORY, POST HOC; NOT PREREGISTERED)

Written BEFORE any E7 evaluation was run. Timestamp and code hash are in PLAN_E7.stamp (written by the launcher
immediately before the first GPU job; sha256 of this file and of code/e7.py recorded there).

Status: everything in E7 is an exploratory / post hoc addition requested after E6 was completed and its verdict written.
Nothing here changes any E6 confirmatory result. All E6 inputs (adapters, cached per-lambda eval results pair_results_*.jsonl,
cached per-example eval predictions preds/*.npz, calibration split) are read-only and reused.

## Setting
Qwen2.5-1.5B E6 pairs, populations P (91), R0 (91), S1 (91) = 273 cross-task pairs (same scope as lam_confirm_e6.py; S2 excluded as there).
Grids: G4 = {0.3,0.5,0.7,1.0} (primary, as in E6), G7 = G4 + {1.3,1.5,2.0} (secondary).
Unlabeled inputs: the same <=200 calibration eval inputs per task used by U1 (lam_confirm_e6.split, verbatim). Test = remaining eval examples.
Regret (pp) = 100 * (Dtest(lambda_rule) - Dtest(lambda_sel)), lambda_sel = held-out-tuned lambda (E6 lam_selected; G7: held-out argmax over G7),
exactly as in lam_confirm_e6.py. Oracle-on-test (evalopt_test) also reported.
CIs: task-block bootstrap (B=2000, lam_confirm_e6.Boot, verbatim), paired (same draws) for differences U1 - baseline. A pair-level iid
bootstrap is reported as secondary.

## A. label-free lambda-selection baselines (all computed on the same <=200 unlabeled cal inputs per task)
A1 ENT (entropy minimisation over the grid, AdaMerging-style objective, task-wise scalar lambda shared by both adapters, as in TA grid):
   lambda_ENT = argmin_lambda  mean_j H_j(lambda), H_j = mean over task-j cal inputs of Shannon entropy (nats) of softmax(head_j(merged)).
   Regression task (stsb) has no predictive distribution: pairs containing stsb use the classification task's entropy only.
   Ties (after rounding to 12 decimals) -> smaller lambda. Sensitivity: ENTn (entropy divided by log C_j); and all analyses excluding stsb pairs.
   Requires a new GPU forward (E6 cached argmax predictions only): merged model at every lambda of G7 on cal inputs; recomputed argmax is checked
   against the cached E6 predictions (match rate reported). Regret uses the cached E6 test-split results (unchanged).
A2 ADA (AdaMerging-style gradient optimisation, optional): task-wise coefficients (lam1, lam2) for the two adapters, init 0.3, Adam lr 0.02,
   60 steps, minibatch 16 cal inputs per classification task per step, loss = mean entropy, lambdas clamped to [0, 2.5]; adapters applied as
   bf16 LoRA branches during optimisation; the final (lam1, lam2) merged model (bf16 W0 + lam1 dW1 + lam2 dW2, as E6) is evaluated on the test split.
   Population P first (primary); R0/S1 only if time. Continuous lambda, so regret is vs lambda_sel as above.
A3 RAND: lambda uniformly random on the grid; reported as the exact expectation (mean regret over grid points) per pair.
A4 fixed lambda = 0.7 and lambda = 1.0 (already in E6), U1 (E6 rule, verbatim), labeled Lcal reference, oracle.
Report: mean regret (pp) with CI for each rule; U1 minus each baseline with paired CI; scopes all/P/R0/S1; G4 primary, G7 secondary.

## B. compute cost (measured on the RTX 4070 Ti SUPER 16GB, torch.cuda.synchronize wall-clock, peak torch.cuda.max_memory_allocated)
Per pair: U1 = 2 single-adapter forwards on cal + merged forwards on cal at each G4 lambda (+ weight writes); M3 uses exactly the U1
quantities (Dhat_U = 1 - max agreement) -> no extra cost beyond U1; ENT = merged forwards on cal at each G4 lambda (no singles);
held-out tuning = merged forwards on the 2x1000 held-out labeled train examples at each G4 lambda, measured on 30 random pairs (seed 20260929),
also verifying that lam_selected reproduces; ADA = measured optimisation + (for comparison) not counting the final test eval.
Forward counts reported as number of examples forwarded and number of weight writes. Base model load time excluded (shared by all).

## C. small multi-task merging check (exploratory)
30 random task triples and 15 random 4-tuples (distinct tasks, adapter seed per slot uniform {0,1}, numpy seed 20260930), TA
W0 + lambda * sum dW_j on G7. For each: held-out norm score (lambda_sel = argmax on G4, ties smaller), test-split scores, cal agreement,
cal entropy. Rules: U1 (argmax mean agreement), ENT, lambda=1, 0.7, RAND, oracle. Regret vs lambda_sel with task-block bootstrap
(weights = product of task counts). M3: Dhat_U = 1 - max_G4 agreement, AUROC for y = 1[Dtest(lambda_sel) > 0.05] and 0.02
(reported only if both classes present), plus accuracy of the frozen E6 threshold vs always-merge.

## Budget / order
A1 (all 273) -> holdtime (30) -> C (45 tuples) -> A2 (P, then R0,S1 if time). Target < 12 h GPU. nohup, resumable per item, STATUS.txt.
Deviations from this plan will be appended below with timestamps.

## Deviations log
(none yet)
