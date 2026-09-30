"""Generates prereg_e6.json from the E4b prereg_e4b.json (read-only): every E4b field is kept unless listed in E6_CHANGES below.
Run once before any E6 study GPU job (pilot); the output is hashed (prereg_e6.sha256) and never edited."""
import copy, json, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
E4B = HERE.parent / "e4b_decoder"
src = E4B / "prereg_e4b.json"
P4 = json.loads(src.read_text())
P = copy.deepcopy(P4)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
P["study"] = "E6: Qwen2.5-1.5B second-decoder replication of the E4b (Qwen2.5-0.5B) decoder analysis (LoRA merge-overlap paper)"
P["written_kst"] = "2026-09-27, before any E6 study GPU job (pilot included) and before any E6 merge; see prereg_e6.sha256 / PREREG_RECORD_E6.txt"
P["written_after_results_known"] = {
  "E1b": P4["written_after_results_known"]["E1b"], "E1c": P4["written_after_results_known"]["E1c"],
  "E4a": "RoBERTa-base P (K=13, rte excluded): H1 INCONCLUSIVE rho(O_A,D)=0.161; H2 INCONCLUSIVE rho(cos,D)=0.112",
  "E4b": "Qwen2.5-0.5B P (primary): rho(O_A,D)=0.345 CI[-0.096,0.609] Holm p=0.030 LOTO 1.00 -> INCONCLUSIVE; rho(cos,D)=0.368 CI[-0.079,0.652] Holm p=0.024 -> INCONCLUSIVE; R0 0.367/0.395, S1 0.211/0.105; gate fires in 66/91 P pairs (sub-30 deg overlap mostly layers.0.self_attn.k_proj); gate30 vs TA ~0",
  "E5": "on the E4b adapters: no method beats tuned TA by >=0.5pp; lambda grid up to 2.0 leaves rho unchanged (P 0.3475/0.3699); evaluation-sampling reliability of D high (split-half SB ~0.90 on P)",
  "consequence": "E6 is a replication with ALL E4b analysis choices frozen; nothing is tuned to E6 data. The disclosure is repeated in PREREG_E6.md sec. 0."}
P["base_studies"] = dict(P4["base_studies"], E4b_prereg_json_sha256=sha(src), E4b_prereg_md_sha256=sha(E4B / "PREREG_E4B.md"),
                         E4b_e4b_py_sha256=sha(E4B / "e4b.py"), E4b_e4b_analysis_py_sha256=sha(E4B / "e4b_analysis.py"),
                         E4b_train_e4b_py_sha256=sha(E4B / "train_e4b.py"), E4b_tasks_e4b_py_sha256=sha(E4B / "tasks_e4b.py"))
P["seeds_max"] = [0, 1]
P["seeds_rule"] = "E6: seeds [0, 1] FIXED (primary P, as realized in E4b); the E4b projection-dependent seed rule is not used (the pilot still records the projection)"
R = P["recipe"]
R.update({"base_model": "Qwen/Qwen2.5-1.5B", "base_revision": "8faed761d45a263340a0528343f099c05c9a4323",
          "lora_layers_expected": 196, "hidden_size": 1536, "n_decoder_layers": 28,
          "head": "score: Linear(1536, num_labels, bias=False), randomly initialized (seeded), fully trained, saved with the adapter",
          "gradient_checkpointing": True,
          "gradient_checkpointing_note": "E6: HF Trainer gradient_checkpointing=True, use_reentrant=False (activations recomputed; dropout RNG state preserved). Needed for memory: feasibility smoke on throwaway data showed sst2 micro-batch 32 without GC at 14.9 GB, rte micro 16 and imdb micro 8 without GC out of memory; with GC <= 11.6 GB for all probed tasks at the E4b micro-batch table",
          "d_out": {"q_proj": 1536, "k_proj": 256, "v_proj": 256, "o_proj": 1536, "gate_proj": 8960, "up_proj": 8960, "down_proj": 1536}})
P["pilot"]["budget"] = "the E4b budgets of the task (sst2 1,563 steps; rte 470 steps), recipe identical to main (E6 recipe); all three lrs are trained"
P["pilot"]["cost_rule"]["two_seed_if_core_h_le"] = None
P["pilot"]["cost_rule"]["note_e6"] = "E6: the projection is computed and recorded exactly as in E4b but does NOT decide the seeds (fixed [0, 1]); token counts reuse probe_qwen.json because Qwen2.5-1.5B has the identical Qwen2 tokenizer (asserted in the E6 smoke/equivalence checks)"
P["integrity"]["b"] = "NOT APPLICABLE: no citable published Qwen2.5-1.5B dev-set references exist for these classification tasks with this setup"
P["integrity"]["structural"] = "196 LoRA layers; score head (num_labels, 1536), no bias; r/alpha; train/held-out index sha256 = train_meta; main-run adapter of the right seed"
P["predictors"]["O_A"] = "mean over the 196 LoRA layers of mean_k cos^2 theta_k, principal angles between orth(B1) and orth(B2), float64"
P["predictors"]["tv_cosine"] = "cosine of flattened concatenated dW = 2 B A (196 layers)"
P["predictors"]["frozen"] = "predictors_e6.sha256 written before any merge"
P["merge"]["method"] = "task arithmetic W0 + lam*(dW1+dW2) on all 196 LoRA layers; each task with its own score head"
P["merge"]["implementation_e6"] = "memory-only restructuring (deltas recomputed per layer on demand with the same op; merged weights written layer by layer; fp32 evaluator W0 on CPU; TIES threshold by exact CPU order statistic): identical arithmetic; verified bit-for-bit against E4b stage-0/1/2 outputs on the E4b Qwen2.5-0.5B adapters before hashing"
P["merge"]["e3_subset"] = "NOT RUN in E6: unmodified e3.run() keeps both all-layer fp32 deltas on the GPU (~10.5 GB for Qwen2.5-1.5B) next to the model; the E4b E3 subset was secondary and E5a covered those methods on Qwen2.5-0.5B"
P["merge"]["lambda_grid_extended_secondary"] = [1.3, 1.5, 2.0]
P["merge"]["lambda_grid_extended_note"] = "E6: after the E4b-grid selection (lam_selected / D unchanged, E4b grid), TA is also evaluated at lam 1.3/1.5/2.0 on held-out and evaluation sets with eval predictions saved (E5b field names); lam_selected_G7 = held-out argmax over the 7-point grid; secondary only (G7 analyses of the lambda-rule hypotheses)"
P["merge"]["predictions_saved"] = "per-example eval predictions (argmax class / regression value) for both single adapters (preds/single_*.npz) and for the merged model at every lambda of both grids (preds/pair_*.npz keys TA_<adapter>_lam<lam>), as in E4b/E5b"
FZ = HERE / "e6_lambda_code_frozen"
fz = {l.split()[1]: l.split()[0] for l in (FZ / "SHA256SUMS").read_text().splitlines() if l.strip()}
P["lambda_rule_hypotheses"] = {
  "status": "CONFIRMATORY SECONDARY (do not affect the H1/H2 verdict). Rules, thresholds, splits and statistics frozen from the post hoc exploratory analysis /workspace/lora-paper/e6_lambda (780 cross-task pairs: BERT/RoBERTa/Qwen2.5-0.5B, R0/S1/P), whose results were known when this was written; nothing is re-tuned on Qwen2.5-1.5B data",
  "frozen_code_copy": "e6_lambda_code_frozen/ (verbatim copy of e6_lambda/code/*.py and RULES.md)", "frozen_code_sha256": fz,
  "implementation": "lam_confirm_e6.py (CPU, after stage2): functions copied verbatim from the frozen code; self-test reproduces the e6_lambda Qwen2.5-0.5B numbers from the E4b/E5b files (U1 removed vs lam=1 0.984 [0.881, 1.054]; M3 AUROC 0.929 [0.840, 0.986]); exact ties in agreement rates -> smaller lambda (made explicit by rounding scores to 12 decimals, which reproduces e6_lambda's selections on all 273 pairs)",
  "data": "E6 cross-task pairs R0 + S1 + P (S2 same-task pairs excluded), populations used only if complete",
  "U1": "lambda in grid maximizing the mean over the pair's two tasks of the agreement rate between merged-model and that task's single-adapter argmax predictions (regression: Spearman) on the unlabeled calibration inputs; calibration = min(200, floor(n_eval/2)) evaluation examples per task, sampled once per task with numpy default_rng(20260927 + index of the task in the 14-task list); labels never used; ties -> smaller lambda",
  "scoring": "all rules scored on the evaluation examples NOT in the calibration set: D_test(lam) = 1 - mean_t(score_merged,test / score_single,test); reference lam_sel = held-out selected lambda (the paper's tuned TA; E4b grid); regret_R = 100 (D_test(lam_R) - D_test(lam_sel)) pp",
  "inference": "task-block bootstrap, 2,000 replicates, weights c_a c_b, default_rng(20260927 + sum(ord(scope key))), percentile 95% CI (frozen Boot class)",
  "primary_scope": "qwen15/all = R0 + S1 + P pooled, grid G4 = {0.3, 0.5, 0.7, 1.0}, tau = 0.05",
  "H-U1": "SUPPORTED iff the 95% CI upper bounds of mean[regret(U1) - regret(lam=0.7)] AND mean[regret(U1) - regret(lam=1.0)] are both < 0",
  "H-U1b": "SUPPORTED iff 1 - mean regret(U1) / mean regret(lam=1.0) >= 0.80 (point estimate; CI reported); NOT APPLICABLE if mean regret(lam=1.0) <= 0 (ratio undefined / no avoidable loss; guard added before any E6 merge after the smoke test showed the ratio is meaningless with a non-positive denominator)",
  "H-M3": "M3 score Dhat_U = 1 - max_{lam in G4} agreement(lam); y = 1[D_test(lam_sel) > 0.05]; SUPPORTED iff AUROC >= 0.75 AND task-block CI lower bound > 0.5",
  "H-M3acc": "decision 'harmful merge' iff Dhat_U >= 0.1174999999999999 (threshold FROZEN: accuracy-maximizing cut on the 780 e6_lambda pairs, in-sample accuracy 0.879, prevalence 0.247); SUPPORTED iff the task-block CI lower bound of accuracy(M3 decision) - accuracy(always merge) > 0; sensitivity thresholds tau 0.02 -> 0.0504428860721516, tau 0.10 -> 0.2375",
  "secondary": "per population (P, R0, S1); grid G7 (lam_sel over G7 as reference); L-cal (labeled calibration) reference; evaluation-optimal floor; U1 vs L-cal; unfitted M3 decision Dhat_U > tau; recall/precision",
  "multiplicity": "four confirmatory secondary hypotheses reported separately, no correction; they do not enter the H1/H2 verdict"}
P["populations"]["one_seed"] = "not used in E6"
P["analysis"]["primary_population"] = "P (mixed seed), fixed"
P["analysis"]["code"] = "e6_analysis.py = E4b e4b_analysis.py with TAG e6 (statistics verbatim), plus cross-backbone vs E4b"
P["analysis"]["secondary"] = P4["analysis"]["secondary"][:4] + ["S2: theta_min, gate, D, E1b-stage2 gate vs TA (no E3 subset)"] + P4["analysis"]["secondary"][5:6] + [
    "cross-backbone: per-task-pair D/O_A/tv_cosine Spearman vs E1b R0, E1c P (BERT), E4a R0/P (RoBERTa) and E4b P/R0/S1 (Qwen2.5-0.5B)"]
P["analysis"]["extra_e6"] = {
  "file": "e6_extra.py -> e6_extra.json (secondary/exploratory, no decision rule)",
  "location": "sub-30 deg (theta_min < 30) pair-layers by module type, layer index, relative depth quartile; share in layers.0.self_attn.k_proj; E4b recomputed for comparison",
  "gate": "gate firing (pairs with >= 1 layer theta_min < 30) and GATE - TA (at lam*_TA and own lam) on gate-active pairs: mean/median, task-block bootstrap CI, Wilcoxon two-sided, W/T/L",
  "pooled_decoder": "EXPLORATORY: per population and predictor, rho_E4b, rho_E6, pooled decoder rho = their mean, difference rho_E6 - rho_E4b; joint task-block bootstrap (same resampled tasks for both backbones, 2,000, default_rng(0)) CIs; joint task-relabeling permutation p (10,000, default_rng(0)) for the pooled mean; common valid tasks"}
P["analysis"]["reading_fixed_in_advance"] = {
  "verdict": "H1/H2 outcome = the E4b rule, verbatim (PASS iff rho >= 0.4 AND Holm p < 0.05 AND >= 75% LOTO rho > 0.2; FAIL iff not PASS and bootstrap upper < 0.3; else INCONCLUSIVE)",
  "replication_reading": "descriptive, not an extra test: the E4b decoder pattern (positive, permutation-significant after Holm, below the 0.4 bar) is 'reproduced' on Qwen2.5-1.5B iff E6 P gives rho > 0 AND Holm p < 0.05 for that predictor; PASS = the association clears the pre-registered bar on the larger decoder; FAIL or rho <= 0 = the E4b association does not replicate; INCONCLUSIVE with Holm p >= 0.05 = not reproduced at this power"}
P["compute_budget"] = {"cap_gpu_hours": 40, "definition": "wall-clock from the start of the E6 pilot to the end of the last E6 GPU job",
  "guard": "E6_DEADLINE_EPOCH = pilot start + 39 h: no new merge pair starts after it; stage3 then runs",
  "max_concurrent_gpu_processes": 1,
  "priority_if_cap_at_risk": ["pilot", "training", "integrity", "predictors (frozen)", "P merges", "S2 merges", "R0 merges", "S1 merges"],
  "reduction_rule_if_projection_exceeds_cap": "not needed: feasibility projection ~29 GPU-h <= 40 including the extended lambda grid (pilot ~1 h, training ~8.5 h, stage0+1 ~0.7 h, stage2 ~18.5 h; PREREG_E6.md sec. 7); if the cap is still hit, lower-priority populations are reported as not run",
  "unfinished_work": "reported as not run; never used as partial evidence for a verdict"}
P["files"] = "artifacts/e6_decoder2/ on ubuntu-4070 (mirror box /workspace/lora-paper/e6_decoder2/); nothing written to other artifacts/ folders; prereg committed locally (branch e6-prereg, git plumbing, no push)"
(HERE / "prereg_e6.json").write_text(json.dumps(P, indent=1) + "\n")
print("wrote prereg_e6.json; tasks", P["tasks"], "steps/seed", P["budgets"]["total_steps_per_seed"])
