# Seed-pair (hubish) merge and certificate results: old eval vs segment-id-fixed eval

Generated on ubuntu-4070 (RTX 4070 Ti SUPER) by `seed_fix_segid.py` + `build_table.py`. Everything is on the full validation set unless noted (MNLI = validation_matched, n=9815; RTE = validation, n=277).
Pipeline: same as `scripts/run_paper_pack_4070.py` (lora_merge_cert unchanged; head copied from adapter 1; arithmetic sum = dW1+dW2 with lam=1; theta*=30 deg; TIES-lite densify 0.7; cert_then_trim keeps top 70% of cert_A). **The only change is that token_type_ids are passed to the model** (`--segment-ids bert`), which is the convention `train_lora_glue.py` trained with. The old eval (`lora_merge_cert/eval.py`) dropped them, so every token got segment 0.
**retain** is defined as in the original TABLE_REQUIRED.md: the raw accuracy of the merged model on the task, using the head of the adapter named in the `head` column. **norm** = retain / that adapter's single accuracy (same segment convention).

## 1. Reproduction check (old setting = no segment ids)

| quantity | original artifact | re-run, segment_ids=none | abs diff |
|---|---:|---:|---:|
| single mnli_s7_hubish (mnli, n=9815) | 0.45247 | 0.45247 | 0.0e+00 |
| single mnli_s42_hubish (mnli, n=9815) | 0.49027 | 0.49027 | 0.0e+00 |
| seed_hubish arith_sum (mnli, n=9815) | 0.42904 | 0.42904 | 0.0e+00 |
| seed_hubish cert_A (mnli, n=9815) | 0.42731 | 0.42731 | 0.0e+00 |
| seed_hubish cert_B (mnli, n=9815) | 0.42731 | 0.42731 | 0.0e+00 |
| single mnli_s7_hubish_n512 (mnli, n=512) | 0.46094 | 0.46094 | 0.0e+00 |
| mnli_rte_hubish arith_sum_n512 (mnli, n=512) | 0.46484 | 0.46484 | 0.0e+00 |
| mnli_rte_hubish cert_A_n512 (mnli, n=512) | 0.46484 | 0.46484 | 0.0e+00 |

Reproduction exact (bit-identical accuracies): **True**. So the segment-id switch is the only difference between the old and fixed columns below.

## 2. Certificates (weights only; they should not depend on the segment convention)

| pair | subspace | n_fail / layers | theta_min (deg) | FAIL layers (theta_min, n_shared) | same as original artifact? | same in none vs bert runs? |
|---|---|---:|---:|---|---|---|
| seed_hubish | A | 2 / 73 | 20.60 | bert.encoder.layer.11.output.dense (26.46, 1); bert.pooler.dense (20.60, 1) | yes: 73 layers, statuses identical=True, max |d theta|=0.0e+00 deg | True |
| seed_hubish | B | 2 / 73 | 20.60 | bert.encoder.layer.11.output.dense (26.46, 1); bert.pooler.dense (20.60, 1) | yes: 73 layers, statuses identical=True, max |d theta|=0.0e+00 deg | True |
| mnli_rte_hubish | A | 0 / 73 | 66.31 | none | yes: 73 layers, statuses identical=True, max |d theta|=0.0e+00 deg | True |
| mnli_rte_hubish | B | 0 / 73 | 66.31 | none | no original table for this subspace | True |

## 3. Old vs fixed accuracies

| pair | method | task | head | retain OLD eval (no seg ids) | norm OLD | retain FIXED (BERT seg ids) | norm FIXED |
|---|---|---|---|---:|---:|---:|---:|
| single | mnli_s7_hubish | mnli | mnli_s7_hubish | 0.4525 | 1.000 | **0.8210** | 1.000 |
| single | mnli_s42_hubish | mnli | mnli_s42_hubish | 0.4903 | 1.000 | **0.8239** | 1.000 |
| single | rte_s42_hubish | rte | rte_s42_hubish | 0.4332 | 1.000 | **0.6751** | 1.000 |
| seed_hubish | arith_sum | mnli | mnli_s7_hubish | 0.4290 | 0.948 | **0.6931** | 0.844 |
| seed_hubish | cert_A | mnli | mnli_s7_hubish | 0.4273 | 0.944 | **0.6911** | 0.842 |
| seed_hubish | cert_B | mnli | mnli_s7_hubish | 0.4273 | 0.944 | **0.6911** | 0.842 |
| seed_hubish | ties_lite (new: not run for seed pair originally) | mnli | mnli_s7_hubish | 0.4512 | 0.997 | **0.7074** | 0.862 |
| seed_hubish | cert_then_trim (new: not run for seed pair originally) | mnli | mnli_s7_hubish | 0.4455 | 0.985 | **0.7048** | 0.859 |
| seed_hubish | arith_lam0.3 (exploratory lam) | mnli | mnli_s7_hubish | 0.4214 | 0.931 | **0.7323** | 0.892 |
| seed_hubish | arith_lam0.5 (exploratory lam) | mnli | mnli_s7_hubish | 0.4932 | 1.090 | **0.8251** | 1.005 |
| seed_hubish | arith_lam0.7 (exploratory lam) | mnli | mnli_s7_hubish | 0.4813 | 1.064 | **0.7761** | 0.945 |
| seed_hubish | arith_sum (exploratory head) | mnli | mnli_s42_hubish | 0.4281 | 0.873 | **0.6978** | 0.847 |
| seed_hubish | cert_A (exploratory head) | mnli | mnli_s42_hubish | 0.4249 | 0.867 | **0.6956** | 0.844 |
| seed_hubish | cert_B (exploratory head) | mnli | mnli_s42_hubish | 0.4249 | 0.867 | **0.6956** | 0.844 |
| seed_hubish | ties_lite (exploratory head) (new: not run for seed pair originally) | mnli | mnli_s42_hubish | 0.4424 | 0.902 | **0.7046** | 0.855 |
| seed_hubish | cert_then_trim (exploratory head) (new: not run for seed pair originally) | mnli | mnli_s42_hubish | 0.4389 | 0.895 | **0.7055** | 0.856 |
| seed_hubish | arith_lam0.3 (exploratory head) (exploratory lam) | mnli | mnli_s42_hubish | 0.4362 | 0.890 | **0.7213** | 0.875 |
| seed_hubish | arith_lam0.5 (exploratory head) (exploratory lam) | mnli | mnli_s42_hubish | 0.4940 | 1.008 | **0.8291** | 1.006 |
| seed_hubish | arith_lam0.7 (exploratory head) (exploratory lam) | mnli | mnli_s42_hubish | 0.4796 | 0.978 | **0.7794** | 0.946 |
| mnli_rte_hubish | arith_sum (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4597 | 1.016 | **0.7695** | 0.937 |
| mnli_rte_hubish | cert_A (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4597 | 1.016 | **0.7695** | 0.937 |
| mnli_rte_hubish | cert_B (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4597 | 1.016 | **0.7695** | 0.937 |
| mnli_rte_hubish | ties_lite (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4627 | 1.023 | **0.7956** | 0.969 |
| mnli_rte_hubish | cert_then_trim (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4576 | 1.011 | **0.7822** | 0.953 |
| mnli_rte_hubish | arith_lam0.3 (exploratory lam) (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.3832 | 0.847 | **0.5159** | 0.628 |
| mnli_rte_hubish | arith_lam0.5 (exploratory lam) (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4173 | 0.922 | **0.7606** | 0.926 |
| mnli_rte_hubish | arith_lam0.7 (exploratory lam) (new: full val; original = MNLI n=512 only) | mnli | mnli_s7_hubish | 0.4512 | 0.997 | **0.7974** | 0.971 |
| mnli_rte_hubish | arith_sum (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4729 | 1.092 | **0.6534** | 0.968 |
| mnli_rte_hubish | cert_A (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4729 | 1.092 | **0.6534** | 0.968 |
| mnli_rte_hubish | cert_B (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4729 | 1.092 | **0.6534** | 0.968 |
| mnli_rte_hubish | ties_lite (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4765 | 1.100 | **0.6931** | 1.027 |
| mnli_rte_hubish | cert_then_trim (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4729 | 1.092 | **0.6931** | 1.027 |
| mnli_rte_hubish | arith_lam0.3 (exploratory lam) (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4982 | 1.150 | **0.6318** | 0.936 |
| mnli_rte_hubish | arith_lam0.5 (exploratory lam) (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4838 | 1.117 | **0.7365** | 1.091 |
| mnli_rte_hubish | arith_lam0.7 (exploratory lam) (new: full val; original = MNLI n=512 only) | rte | rte_s42_hubish | 0.4729 | 1.092 | **0.6823** | 1.011 |

## 4. Paired bootstrap (1000 reps over validation examples, seed 0): method minus arithmetic sum, accuracy in pp

| pair | task | head | method - arith_sum | OLD eval: diff [95% CI] | FIXED eval: diff [95% CI] | FIXED verdict |
|---|---|---|---|---|---|---|
| seed_hubish | mnli | mnli_s7_hubish | cert_A | -0.17 [-0.43, +0.09] | -0.20 [-0.34, -0.07] | hurts |
| seed_hubish | mnli | mnli_s7_hubish | cert_B | -0.17 [-0.43, +0.09] | -0.20 [-0.34, -0.07] | hurts |
| seed_hubish | mnli | mnli_s7_hubish | ties_lite | +2.22 [+1.44, +2.93] | +1.43 [+0.89, +1.99] | helps |
| seed_hubish | mnli | mnli_s7_hubish | cert_then_trim | +1.65 [+1.03, +2.28] | +1.17 [+0.78, +1.55] | helps |
| seed_hubish | mnli | mnli_s42_hubish | cert_A | -0.33 [-0.76, +0.09] | -0.22 [-0.46, -0.01] | hurts |
| seed_hubish | mnli | mnli_s42_hubish | cert_B | -0.33 [-0.76, +0.09] | -0.22 [-0.46, -0.01] | hurts |
| seed_hubish | mnli | mnli_s42_hubish | ties_lite | +1.43 [+0.66, +2.20] | +0.68 [+0.14, +1.22] | helps |
| seed_hubish | mnli | mnli_s42_hubish | cert_then_trim | +1.08 [+0.44, +1.73] | +0.76 [+0.37, +1.16] | helps |
| mnli_rte_hubish | mnli | mnli_s7_hubish | cert_A | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | identical (no FAIL layers -> cert == sum) |
| mnli_rte_hubish | mnli | mnli_s7_hubish | cert_B | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | identical (no FAIL layers -> cert == sum) |
| mnli_rte_hubish | mnli | mnli_s7_hubish | ties_lite | +0.30 [-0.42, +0.99] | +2.61 [+2.14, +3.07] | helps |
| mnli_rte_hubish | mnli | mnli_s7_hubish | cert_then_trim | -0.21 [-0.79, +0.26] | +1.26 [+0.91, +1.58] | helps |
| mnli_rte_hubish | rte | rte_s42_hubish | cert_A | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | identical (no FAIL layers -> cert == sum) |
| mnli_rte_hubish | rte | rte_s42_hubish | cert_B | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | identical (no FAIL layers -> cert == sum) |
| mnli_rte_hubish | rte | rte_s42_hubish | ties_lite | +0.36 [+0.00, +1.08] | +3.97 [-2.17, +9.39] | no significant difference |
| mnli_rte_hubish | rte | rte_s42_hubish | cert_then_trim | +0.00 [+0.00, +0.00] | +3.97 [+0.72, +7.22] | helps |

The primary row is seed_hubish / mnli / head mnli_s7_hubish, cert_A and cert_B vs arith_sum. The original table evaluated only that row.

## Notes

- The "OLD eval" column is a re-run with the old convention (no segment ids). For the rows marked "original artifact" in section 1, the re-run matches the original artifacts bit for bit. The mnli_rte_hubish full-val rows, the RTE rows, and the TIES-lite / cert_then_trim / lambda-grid rows for the seed pair were never in the original tables.
- Fixed singles: mnli_s7 0.8210, mnli_s42 0.8239, rte_s42 0.6751. For comparison, the Hub MNLI adapter scores 0.8071 under its own (zero-segment) convention. The hubish adapters are fine; the old 0.45-0.49 was an evaluation bug.
- Seed pair, primary row: the certificate correction removes one shared direction in 2 layers (layer 11 output.dense and the pooler). It lowers MNLI by 0.20pp (95% CI [-0.34, -0.07]) relative to the plain sum. The difference is statistically detectable but tiny. TIES-lite (+1.43pp) and cert_then_trim (+1.17pp) beat the sum.
- Most of the seed-pair drop comes from lambda=1 overshoot, not interference: summing two same-task adapters gives 0.693, while lambda=0.5 gives 0.825 (s7 head), which is at or above both singles. The lambda grid is exploratory and was selected on validation here, so treat it as descriptive.
- mnli x rte: no FAIL layers (theta_min 66.3 deg), so cert == sum exactly (0.7695 MNLI, 0.6534 RTE).
