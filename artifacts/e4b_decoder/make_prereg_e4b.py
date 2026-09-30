"""Generates prereg_e4b.json (run once, before any E4b training; the output is hashed and never edited)."""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
E1BP = json.loads(Path(sys.argv[1]).read_text())        # E1b prereg.json (budgets: n_train_total, n_eval_total)
TASKS = ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "rte", "wic", "snli", "scitail", "ag_news", "imdb", "trec", "yelp_polarity"]
PROBE = {"cola": {"train_mean": 11.11, "hold_tokens": 9397, "eval_tokens": 10055}, "sst2": {"train_mean": 12.1, "hold_tokens": 11843, "eval_tokens": 20716}, "mrpc": {"train_mean": 53.02, "hold_tokens": 53448, "eval_tokens": 21523}, "stsb": {"train_mean": 19.23, "hold_tokens": 25772, "eval_tokens": 43543}, "mnli": {"train_mean": 37.53, "hold_tokens": 37608, "eval_tokens": 180573}, "qnli": {"train_mean": 49.51, "hold_tokens": 50232, "eval_tokens": 251215}, "rte": {"train_mean": 70.0, "hold_tokens": 69734, "eval_tokens": 18854}, "wic": {"train_mean": 19.64, "hold_tokens": 20021, "eval_tokens": 13303}, "snli": {"train_mean": 23.61, "hold_tokens": 23272, "eval_tokens": 122734}, "scitail": {"train_mean": 39.91, "hold_tokens": 36592, "eval_tokens": 48689}, "ag_news": {"train_mean": 55.71, "hold_tokens": 52390, "eval_tokens": 260890}, "imdb": {"train_mean": 202.61, "hold_tokens": 204258, "eval_tokens": 1008268}, "trec": {"train_mean": 11.51, "hold_tokens": 11782, "eval_tokens": 4134}, "yelp_polarity": {"train_mean": 140.05, "hold_tokens": 136913, "eval_tokens": 701707}}


def budget(pool_n, n_used):
    spe = math.ceil(n_used / 32); ep = 10 if pool_n <= 10000 else (2 if pool_n < 30000 else 1); ms = spe * ep
    return {"n_pool": int(pool_n), "n_train_used": int(n_used), "epochs": ep, "steps_per_epoch": spe, "max_steps": ms, "warmup_steps": int(min(500, round(0.1 * ms)))}


per = {}
for t in TASKS:
    b = E1BP["budgets"][t]; pool = b["n_train_total"] - 1000; used = min(pool, 50000)
    e = budget(pool, used); ne = b["n_eval_total"]
    e.update({"n_eval_total": ne, "n_eval_used": min(ne, 5000), "eval_split": b["eval_split"], "metric": b["metric"], "num_labels": b["num_labels"],
              "E1b_max_steps": b["max_steps"]})
    per[t] = e
tot = sum(v["max_steps"] for v in per.values())
P = {
 "study": "E4b: Qwen2.5-0.5B (decoder LLM) second-backbone replication of E1b/E1c (LoRA merge-overlap paper)",
 "written_kst": "2026-09-26, before any E4b training (pilot included); see prereg_e4b.sha256 file time",
 "written_after_results_known": {
  "E1b": "BERT-base seed 0: H1 FAIL rho=-0.194; H2 INCONCLUSIVE rho=-0.044; gate 0/91",
  "E1c": "BERT-base mixed seed P: H1c INCONCLUSIVE rho=-0.025; H2c INCONCLUSIVE rho=0.127; min theta 39.8 deg; gate 0/196",
  "E4a": "RoBERTa-base: pre-registered; pilot chose lr 5e-4 (rte diverged at 1e-3); main training/merges running when this was written; NO E4a eval-set, integrity or merge result was known"
 },
 "base_studies": {
  "E1b_prereg_json_sha256": "f9c43e5140452ea3f1b25e053be16d61b20d8bf47858f545b3e1ddae1646a341",
  "E1b_e1b_py_sha256": "4610d045fec73b9356481e9c6bb14a8c15a99259b43b0518bd2afe87fff1c9d6",
  "E1b_tasks_py_sha256": "34dd6bd222880b35a9a8d4bd2be7b907bfa4995ce46e8699539a0978a2e60b53",
  "E3_e3_py_sha256": "a25e393e60d0e076d0c54a2e41301ebfb7a303b38a94bb9c671cf69e8e1d020c",
  "E3_merges_py_sha256": "7ec5774b491102bfadeeec68f6e9be429b3f6894b40e1ed6c128b2de1bf68300",
  "E4a_prereg_md_sha256": "b833c774d3d776d073440e11ecb39b9be7f598414e4542eca79785c38919e542",
  "E4a_prereg_json_sha256": "c803d8eb7c8522c7197551876c9aae2b6243169f5bd35a6c7b6e8a61c88f3f06",
  "probe_qwen_json_sha256": "4a3d91d0350a5156aab4df1e2ac1f6b0d012a057c9aa8753f5e2b1d2399b67d4"
 },
 "tasks": TASKS,
 "seeds_max": [0, 1],
 "seeds_rule": "seeds [0,1] (primary P) iff the pilot-projected core cost <= pilot.cost_rule.two_seed_if_core_h_le hours, else seed [0] only (primary R0); decided automatically in pilot/pilot_decision.json",
 "recipe": {
  "base_model": "Qwen/Qwen2.5-0.5B", "base_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
  "model_class": "AutoModelForSequenceClassification -> Qwen2ForSequenceClassification (pooling: last non-pad token, i.e. the rightmost token != pad_token_id)",
  "pad_token": "<|endoftext|>", "pad_token_id": 151643, "padding_side": "right",
  "special_tokens": "none added (Qwen2 tokenizer adds no BOS/EOS); the pooled token is the last text token",
  "pair_encoding": "tokenizer(a + '\\n', b): concatenation of the two token sequences; truncation longest_first at 256",
  "max_seq_len": 256,
  "r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "lora_layers_expected": 168,
  "modules_to_save": ["score"],
  "head": "score: Linear(896, num_labels, bias=False), randomly initialized (seeded), fully trained, saved with the adapter",
  "task_type": "SEQ_CLS",
  "lr": None, "lr_note": "chosen by the pre-registered pilot (pilot.lr_candidates, held-out only)",
  "batch_size": 32, "micro_batch": {"default": 32, "imdb": 16, "yelp_polarity": 16},
  "micro_batch_note": "micro-batch x gradient accumulation = 32; OOM fallback: halve the micro-batch (double accumulation), automatic retry",
  "weight_decay": 0.01, "scheduler": "linear with warmup; warmup_steps = min(500, round(0.1*max_steps))", "optimizer": "AdamW (HF Trainer default)",
  "precision_training": "fp32 master weights (model loaded in float32) + bf16 autocast (Trainer bf16=True)",
  "logging_steps": 25,
  "seed_applies_to": ["transformers.set_seed before model construction (LoRA-A init, head init)", "TrainingArguments.seed (dropout)", "TrainingArguments.data_seed (data order)"]
 },
 "budgets": {
  "rule": "pool (= n_train_total - 1000 held-out) <= 10,000: full pool, 10 epochs (identical to E1b); otherwise min(pool, 50,000) examples = the first ones of the E1b seed-0 permutation after the held-out 1,000 (nested in E1b's 60k subset), 1 epoch, or 2 epochs if pool < 30,000; steps = ceil(n/32) x epochs",
  "per_task": per, "total_steps_per_seed": tot, "total_steps_both_seeds": 2 * tot,
  "eval_subset": "all if n_eval <= 5,000 else sorted(default_rng(1).choice(n_eval, 5000, replace=False))",
  "held_out": "E1b tasks.split_indices held-out 1,000 TRAIN examples (asserted equal to the E1b adapter hold_idx_sha256)"
 },
 "pilot": {
  "tasks": ["sst2", "rte"], "seed": 0, "lr_candidates": [0.0001, 0.0003, 0.001],
  "budget": "the E4b budgets of the task (sst2 1,563 steps; rte 470 steps), recipe identical to main; all three lrs are trained",
  "evaluation": "held-out 1,000 TRAIN examples only (bf16 manual merge, as in the study); NO eval-set scoring",
  "divergence_rule": "diverged if ANY: (i) a logged loss NaN/inf; (ii) mean logged loss over the last 20% of logged steps >= 0.9 x entropy of the training-label prior; (iii) held-out accuracy < held-out majority baseline + 10pp",
  "decision": "lr_main = the non-diverged candidate with the highest mean held-out accuracy over sst2 and rte (ties -> smaller lr); if all diverge, the candidate with the highest mean held-out accuracy (documented; the integrity rule then applies)",
  "cost_rule": {
   "formula": "sec/step(L) = alpha + beta*L, line through the (mean train tokens, sec/step) points of the sst2 and rte pilot runs at lr_main (floored at the smaller measured value); train_h_per_seed = sum_t max_steps_t * sec/step(L_t) / 3600 (sequential, no concurrency credit); tau = bf16 forward tokens/s measured on the pilot held-out evaluations; pair_s(i,j) = [4*(hold_i+hold_j+eval_i+eval_j) + 4*(hold_i+hold_j) + 2*(eval_i+eval_j)] / tau + pair_overhead_s; merges_core_h = [sum_P pair_s + (1 + e3_pair_cost_factor) * sum_S2 pair_s] / 3600 / merge_parallel_speedup; core_h = pilot_h + 2*train_h_per_seed + stage01_h + merges_core_h",
   "mean_train_tokens": {t: PROBE[t]["train_mean"] for t in TASKS},
   "hold_tokens": {t: PROBE[t]["hold_tokens"] for t in TASKS},
   "eval_tokens": {t: PROBE[t]["eval_tokens"] for t in TASKS},
   "token_source": "probe_qwen.json (CPU-only tokenization probe; no model forward, no training)",
   "pair_overhead_s": 15, "e3_pair_cost_factor": 3, "merge_parallel_speedup": 1.5, "stage01_h": 0.6,
   "two_seed_if_core_h_le": 11.0
  },
  "pilot_adapters_used_in_study": False
 },
 "integrity": {
  "a": "metric >= majority-class baseline + 0.10 (E1b definition, on the eval subset); stsb Spearman >= 0.70",
  "b": "NOT APPLICABLE: no citable published Qwen2.5-0.5B dev-set references exist for these classification tasks with this setup",
  "c": "max |logit(manual merge W0+dW + own head, fp32) - logit(PeftModel fp32)| <= c_tol_rel * max(1, max|PeftModel logit|) on the first 64 eval examples, each unpadded, TF32 disabled",
  "c_tol_rel": 1e-3,
  "c2": "held-out metric of the single adapter with bf16 merged weights (study precision) vs fp32 merged weights: |diff| <= c2_tol",
  "c2_tol": 0.01,
  "structural": "168 LoRA layers; score head (num_labels, 896), no bias; r/alpha; train/held-out index sha256 = train_meta; main-run adapter of the right seed",
  "exclusion": "a task failing any criterion in ANY used seed is removed with all its pairs from EVERY population",
  "stop_if_valid_tasks_below": 10
 },
 "predictors": {
  "primary": ["O_A", "tv_cosine"],
  "O_A": "mean over the 168 LoRA layers of mean_k cos^2 theta_k, principal angles between orth(B1) and orth(B2), float64",
  "tv_cosine": "cosine of flattened concatenated dW = 2 B A (168 layers)",
  "secondary": ["mean_theta_min_A_deg", "min_theta_min_A_deg", "n_layers_theta_min_lt30_A", "sign_conflict_top20", "norm_ratio", "null_z_O_A", "sign_conflict_all"],
  "dropped": {"O_B": "identical to O_A (E1c N3.1)", "mean_theta_min_B_deg": "identical"},
  "null": "1000 random rank-8 subspace pairs per layer (seed 0)",
  "frozen": "predictors_e4b.sha256 written before any merge"
 },
 "merge": {
  "method": "task arithmetic W0 + lam*(dW1+dW2) on all 168 LoRA layers; each task with its own score head",
  "lambda_grid": [0.3, 0.5, 0.7, 1.0],
  "lambda_selection": "per pair, maximize mean normalized score on the two 1,000-example held-out TRAIN sets; ties -> smaller lam",
  "normalized_score": "merged / single on the same set (accuracy; stsb Spearman)",
  "D": "1 - mean of the two normalized eval-set scores at the selected lam",
  "exploratory": "TIES-lite and theta*=30 deg hard gate exactly as E1b",
  "precision": "weights = bf16(W0 + delta computed in fp32); W0 = bf16 checkpoint (exact); head in fp32; singles computed identically in stage0",
  "eval_batch": 128
 },
 "populations": {
  "two_seeds": {"P_primary": "mixed-seed cross-task pairs, alphabetically-first task seed 0 = t1, other seed 1 = t2 (E1c rule)", "R0": "seed-0 cross-task pairs (E1b order)", "S1": "seed-1 cross-task pairs", "S2": "same-task seed pairs t@s0 x t@s1"},
  "one_seed": {"R0_primary": "seed-0 cross-task pairs (E1b confirmatory design)"},
  "n_expected_if_14_valid": {"P": 91, "R0": 91, "S1": 91, "S2": 14}
 },
 "analysis": {
  "primary_population": "P if two seeds, else R0",
  "hypotheses": {"H1": "Spearman rho(O_A, D) > 0", "H2": "Spearman rho(tv_cosine, D) > 0"},
  "multiplicity": "Holm over {H1, H2} on the one-sided permutation p (primary population only)",
  "permutation": "10,000 task relabelings, default_rng(0); p = (1 + #{rho_perm >= rho_obs}) / 10001",
  "bootstrap": "task-block, 2,000 reps, default_rng(0), skip resamples with < 4 distinct pairs, percentile 95% CI",
  "loto": "rho with each valid task removed",
  "rule": "PASS iff rho >= 0.4 AND Holm p < 0.05 AND >= 75% LOTO rho > 0.2; FAIL iff not PASS and bootstrap upper < 0.3; else INCONCLUSIVE",
  "code": "E4a e4a_analysis.py (statistics verbatim from E1c pop_stats) with TAG e4b",
  "secondary": ["R0/S1 same statistics (rule descriptive)", "pooled 3-population summary (descriptive)", "theta_min distribution, gate firings, O_A vs random null, shared-vs-mixed O_A",
                "D/O_A/tv_cosine/lam* reliability across seed configurations", "S2: theta_min, gate, D, E1b-stage2 gate vs TA; E3 run() TA/PICO_TA/GATE/FORCEGATE with A2 grids",
                "fixed-lambda rho, secondary predictors, TIES-lite/gate vs TA", "cross-backbone: per-task-pair D/O_A/tv_cosine Spearman vs E1b R0, E1c P (BERT) and E4a R0/P (RoBERTa)"]
 },
 "compute_budget": {
  "cap_gpu_hours": 14,
  "definition": "wall-clock from the start of the first E4b GPU job (pilot) to the end of the last E4b GPU job",
  "guard": "E4B_DEADLINE_EPOCH = pilot start + 13.6 h: no new merge pair / E3 run starts after it; stage3 (CPU-light) then runs",
  "max_concurrent_gpu_processes": 2,
  "priority_if_cap_at_risk": ["pilot", "training", "integrity", "predictors (frozen)", "P merges", "S2 merges + E3 subset", "R0 merges", "S1 merges"],
  "unfinished_work": "reported as not run; never used as partial evidence for a verdict"
 },
 "files": "artifacts/e4b_decoder/ on ubuntu-4070 (mirror box /workspace/lora-paper/e4b/); nothing written to other artifacts/ folders; no git operations"
}
(HERE / "prereg_e4b.json").write_text(json.dumps(P, indent=1) + "\n")
print("total steps/seed", tot); print({t: (v["n_train_used"], v["epochs"], v["max_steps"], v["warmup_steps"]) for t, v in per.items()})
