"""Builds prereg.json (machine-readable pre-registration). Run once, before any training."""
import json, math, sys, time
sys.path.insert(0, "."); from tasks import TASKS, ALL_SPECS, N_HOLD, TRAIN_CAP, EVAL_CAP, MAXLEN, REVISION, TASK_TYPE
probe = json.load(open("data_probe.json"))
N_TRAIN = {t: probe[t]["n_train"] for t in TASKS if probe[t]["ok"]}; N_TRAIN["trec"] = 5452
N_EVAL = {t: probe[t]["n_eval"] for t in TASKS if probe[t]["ok"]}; N_EVAL["trec"] = 500
BS = 32; STEP_CAP = math.ceil(TRAIN_CAP / BS) * 3   # 5625 = large-task budget
budgets = {}
for t in TASKS:
    pool = N_TRAIN[t] - N_HOLD
    used = min(pool, TRAIN_CAP)
    ep = 10 if pool <= 10000 else 3
    steps = min(math.ceil(used / BS) * ep, STEP_CAP)
    budgets[t] = {"n_train_total": N_TRAIN[t], "n_holdout": N_HOLD, "n_pool": pool, "n_train_used": used,
                  "epochs": ep, "steps_per_epoch": math.ceil(used / BS), "max_steps": steps,
                  "warmup_steps": min(500, round(0.1 * steps)), "n_eval_total": N_EVAL[t],
                  "n_eval_used": min(N_EVAL[t], EVAL_CAP), "eval_split": ALL_SPECS[t][3], "num_labels": ALL_SPECS[t][7],
                  "metric": ALL_SPECS[t][8], "hf_path": ALL_SPECS[t][0], "hf_config": ALL_SPECS[t][1],
                  "hf_revision": REVISION.get(t), "task_type_group": TASK_TYPE[t]}
P = {
 "study": "E1b confirmatory: do weights-only adapter-overlap predictors predict task-arithmetic merge loss?",
 "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
 "tasks": list(TASKS), "substitutions": {"trec": "none: CogComp/trec main branch has an unsupported loading script; the same repo's Hub-generated parquet branch refs/convert/parquet (commit 65752bf53af25bc935a0dce92fb5b6c930728450) is used. Same dataset, same splits; not a substitute."},
 "substitute_order_if_load_fails": ["rotten_tomatoes", "emotion", "paws"],
 "dataset_revisions_at_prereg": {"nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "aps/super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
     "stanfordnlp/snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b", "allenai/scitail": "0cc4353235b289165dfde1c7c5d1be983f99ce44",
     "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4", "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
     "CogComp/trec@refs/convert/parquet": "65752bf53af25bc935a0dce92fb5b6c930728450", "fancyzhx/yelp_polarity": "bbf1c97a1f0cf005e5aded43839fd814654a1557"},
 "recipe": {"base_model": "bert-base-uncased", "base_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["query", "key", "value", "dense"],
            "modules_to_save": ["classifier"], "task_type": "SEQ_CLS", "lr": 1e-3, "batch_size": BS, "weight_decay": 0.01,
            "scheduler": "linear with warmup; warmup_steps = min(500, round(0.1*max_steps))", "optimizer": "AdamW (HF Trainer default)",
            "fp16_amp": True, "seed": 0, "max_seq_len": MAXLEN,
            "segment_ids": "standard BERT token_type_ids (tokenizer output) in BOTH training and evaluation",
            "lora_layers_expected": 73, "step_cap": STEP_CAP},
 "budgets": budgets,
 "splits": {"holdout": "rng=np.random.default_rng(0); perm=rng.permutation(n_train); hold=perm[:1000] (never trained on; lambda selection only); pool=perm[1000:]; train=pool[:60000] if len(pool)>60000 else pool",
            "eval": "if n_eval>10000: np.random.default_rng(1).choice(n_eval,10000,replace=False) else all; sorted",
            "snli": "label -1 removed before indexing", "code": "tasks.py:split_indices"},
 "integrity": {"a_majority": "metric >= majority-class baseline + 0.10 (majority label = most frequent label in the task's training set used; baseline = its accuracy on the eval set used; ties -> the tied label with the higher eval accuracy); stsb: Spearman >= 0.70",
               "b_glue_reference": {"source": "HuggingFace transformers examples/pytorch/text-classification/README.md, run_glue.py dev-set results for bert-base-cased (FP32 column), seq len 128, bs 32, lr 2e-5, 3 epochs (MRPC 5)",
                                    "reference": {"cola": {"metric": "mcc", "value": 0.5653}, "sst2": {"metric": "accuracy", "value": 0.9232},
                                                  "mrpc": {"metric": "accuracy", "value": 0.8407}, "stsb": {"metric": "spearman", "value": 0.8848},
                                                  "qqp": {"metric": "accuracy", "value": 0.9071}, "mnli": {"metric": "accuracy", "value": 0.8391},
                                                  "qnli": {"metric": "accuracy", "value": 0.9066}, "rte": {"metric": "accuracy", "value": 0.6570}},
                                    "rule": "ours >= reference - 0.05 on the listed metric (CoLA: MCC, because no accuracy is published)"},
               "c_logit_equivalence": "max |logit(manual merge W0+dW + own head) - logit(PeftModel)| <= 1e-4 on the first 64 eval examples, fp32, TF32 disabled",
               "exclusion": "a task failing any applicable criterion is excluded with all its pairs BEFORE predictors/merges; if < 12 valid tasks remain: STOP and report"},
 "predictors": {"layers": "all 73 LoRA layers (12 x {query,key,value,attention.output.dense,intermediate.dense,output.dense} + pooler.dense)",
                "O_A": "mean over layers of mean_k cos^2(theta_k), theta = principal angles between orth(B1) and orth(B2) (QR, float64)",
                "tv_cosine": "cosine of flattened concatenated dW = (alpha/r) B@A over all LoRA layers",
                "secondary": ["O_B (top-r left singular vectors of B@A)", "mean theta_min", "min theta_min", "n layers theta_min<30deg",
                              "TIES sign conflict (top-20% |dW| per task, global; fraction of jointly kept coords with opposite signs)",
                              "norm ratio max/min ||dW||", "null z-score of O_A vs 1000 random rank-8 subspace pairs (seed 0)"],
                "freeze": "predictors.csv + predictors.sha256 written before any merge; merges refuse to run if the hash changes"},
 "merge": {"method": "task arithmetic W0 + lam*(dW1+dW2) on all LoRA layers; each task evaluated with its own classifier head",
           "lambda_grid": [0.3, 0.5, 0.7, 1.0], "lambda_selection": "per pair, maximize mean normalized score on the two 1000-example held-out TRAIN sets; ties -> smaller lam",
           "normalized_score": "merged metric / single-adapter metric on the same eval set (accuracy; stsb Spearman)",
           "D": "1 - mean of the two normalized eval-set scores at the selected lam",
           "pair_order": "t1 precedes t2 in the task list order"},
 "analysis": {"hypotheses": {"H1": "Spearman rho(O_A, D) > 0", "H2": "Spearman rho(tv_cosine, D) > 0"},
              "permutation": "10000 random relabelings of tasks applied to the D matrix (predictor fixed), np.random.default_rng(0); one-sided p=(1+#{rho_perm>=rho_obs})/(1+10000)",
              "multiplicity": "Holm over {H1,H2} on the one-sided permutation p",
              "bootstrap": "task-block: 2000 reps, resample K tasks with replacement (np.random.default_rng(0), separate generator), use all pairs of distinct resampled indices, percentile 95% CI",
              "loto": "leave-one-task-out rho for each valid task",
              "PASS": "rho >= 0.4 AND Holm-adjusted one-sided permutation p < 0.05 AND >= 75% of LOTO rho > 0.2",
              "FAIL": "(if not PASS) task-block bootstrap 95% upper bound < 0.3 (meaningful effect ruled out)",
              "INCONCLUSIVE": "otherwise"},
 "exploratory": ["secondary predictors (rho, bootstrap CI, permutation p; not used for verdicts)",
                 "rho(predictor, D_lam) for each fixed lam in the grid (eval set evaluated at all 4 lam)",
                 "method comparison per pair: TA at selected lam; TIES-lite (trim top-20% per task globally, sign election by sum, disjoint mean, merged = W0 + lam*tau) at the TA-selected lam and at its own held-out-selected lam; theta*=30deg hard gate (per layer, shared directions S = principal vectors of orth(B1) with theta<30deg vs orth(B2); on such FAIL layers dW2 <- (I - S S^T) dW2; merged = W0 + lam*(dW1+dW2')) at the TA-selected lam and at its own held-out-selected lam. Summary: mean normalized score, win/tie/loss vs TA, paired Wilcoxon (exploratory)."],
 "known_properties": ["All adapters use seed 0: LoRA A matrices start from identical initializations across tasks (B starts at 0); classifier init differs only by shape. Any shared-init overlap is part of the measured predictor.",
                      "O_B spans the same space as O_A when A has full row rank (expected O_B == O_A up to numerics).",
                      "The null z-score is a linear transform of O_A (identical layer shapes for all pairs) -> same rho."],
 "compute_budget": {"training_gpu_hours_max": 12},
}
json.dump(P, open("prereg.json", "w"), indent=1)
tot = sum(b["max_steps"] for b in budgets.values())
print("total steps", tot)
for t, b in budgets.items(): print(t, b["n_train_used"], b["epochs"], b["max_steps"], b["warmup_steps"], b["n_eval_used"])
