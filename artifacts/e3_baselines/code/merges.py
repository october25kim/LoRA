"""E3 merge library (LoRA merge baselines).

Common per-layer interface
    f(dWs, hp, fac=None, ctx=None) -> merged delta (tensor, same shape as dWs[0]); the result ALREADY includes the
    method's global scale (lambda / c), i.e. the merged model is W0 + f(...).
      dWs : list of T dense per-layer updates dW_t = (alpha/r) B_t A_t   (d_out x d_in, float32)
      hp  : dict of hyper-parameters (always contains 'lam' or 'c')
      fac : list of T (B_t_scaled, A_t) with dW_t = B_t_scaled @ A_t (LoRA scaling absorbed into B); needed by PICO/GATE
      ctx : per-layer extras computed at MODEL level (global magnitude thresholds for TIES-style trimming, DARE rng, ...)
Model-level entry point: merge_model(method, hp, names, dW, fac) -> {layer: merged delta}.
Every method is lam * base(hp without lam); merge_model(..., hp with lam=1.0) returns the base, which the runner scales.

Methods: TA, TIES, DARE_TA, DARE_TIES, TSVM, KNOTS, PICO_TA, PICO_TIES, GATE (hard: beta=1; soft: beta<1).
Reference code consulted (read-only): prateeky2806/ties-merging (src/ties_minimal.ipynb), yule-BUAA/MergeLM
(model_merging_methods/mask_weights_utils.py, merging_methods.py), AntoAndGar/task_singular_vectors
(src/utils/TSVM_utils.py: compute_and_sum_svd_mem_reduction), gstoica27/KnOTS (task_merger.py SVDMerger, masking_ops.py),
Tang & Yang arXiv:2604.16826 (Pico, Sec. 4, no public code found).
"""
import math
from functools import reduce
import operator
import torch


def _sum(xs):
    return reduce(operator.add, xs)          # d1 + d2 (+ ...) : same op order as E1 `lam * (dW1 + dW2)`


# ----------------------------------------------------------------------------- TIES helpers (official semantics)
def official_kth_threshold(flat_abs, K):
    """ties-merging topk_values_mask: keep entries with |x| >= kthvalue(|x|, d - int(d*K/100)).  K in percent (keep top K%).
    Returns None for K >= 100 (no trimming)."""
    if K >= 100:
        return None
    d = flat_abs.numel()
    k = d - int(d * (K / 100.0))
    return float(torch.kthvalue(flat_abs.float().cpu(), k).values)


def global_thresholds(dW_task, names, K):
    """per-task global threshold over the concatenation of all LoRA-covered matrices (official TIES flattens the whole task
    vector; for LoRA the task vector is zero outside the LoRA-covered matrices, so the trim domain = those matrices)."""
    if K >= 100:
        return None
    flat = torch.cat([dW_task[n].detach().float().flatten().cpu() for n in names]).abs()
    return official_kth_threshold(flat, K)


def ties_disjoint(xs, thr=None, merge="mean"):
    """TIES core on already-trimmed-or-not tensors. xs: list of T tensors. thr: list of T thresholds (None = no trim).
    Steps (official): trim (|x| >= thr_t), elect sign = sign(sum of trimmed values) (zeros -> majority sign), disjoint mean
    over entries whose sign agrees with the elected sign."""
    X = torch.stack([x if (thr is None or thr[i] is None) else x * (x.abs() >= thr[i]) for i, x in enumerate(xs)])
    sign = torch.sign(X.sum(0))
    maj = torch.sign(sign.sum())                     # official: majority over the whole flattened vector; here per layer.
    sign[sign == 0] = maj                            # (only affects exact-cancellation / all-zero coordinates, see notes)
    keep = torch.where(sign.unsqueeze(0) > 0, X > 0, X < 0)
    sel = X * keep
    if merge == "mean":
        cnt = (sel != 0).sum(0).float()
        return sel.sum(0) / torch.clamp(cnt, min=1)
    if merge == "sum":
        return sel.sum(0)
    raise ValueError(merge)


# ----------------------------------------------------------------------------- per-layer merge functions
def ta(dWs, hp, fac=None, ctx=None):
    """Task arithmetic: lam * sum_t dW_t."""
    return hp["lam"] * _sum(dWs)


def ties(dWs, hp, fac=None, ctx=None):
    """TIES-Merging (Yadav et al. 2023): trim to top-k% magnitude per task (threshold global over the task's LoRA
    matrices, passed via ctx['thr']; per-layer threshold if ctx is None), sign election by mass, disjoint mean, * lam."""
    thr = (ctx or {}).get("thr")
    if thr is None and hp.get("k", 100) < 100:
        thr = [official_kth_threshold(x.abs().flatten(), hp["k"]) for x in dWs]
    return hp["lam"] * ties_disjoint(dWs, thr, "mean")


def dare_drop(x, p, gen):
    """DARE (Yu et al. 2024; MergeLM mask_input_with_mask_rate): drop each entry with prob p, rescale survivors by 1/(1-p)."""
    if p <= 0:
        return x
    mask = torch.bernoulli(torch.full_like(x, p), generator=gen)
    return x * (1 - mask) / (1 - p)


def dare_ta(dWs, hp, fac=None, ctx=None):
    gen = ctx["gen"]
    return hp["lam"] * _sum([dare_drop(x, hp["p"], gen) for x in dWs])


def dare_ties(dWs, hp, fac=None, ctx=None):
    """DARE then TIES (MergeLM mask_apply_method='ties_merging': DARE on each delta, then full TIES incl. its own top-k trim,
    default param_value_mask_rate=0.8 -> k=20). Thresholds for the trim are computed on the DARE'd deltas (ctx['thr'])."""
    xs = ctx["dared"] if "dared" in ctx else [dare_drop(x, hp["p"], ctx["gen"]) for x in dWs]
    thr = ctx.get("thr")
    if thr is None and hp.get("k", 100) < 100:
        thr = [official_kth_threshold(x.abs().flatten(), hp["k"]) for x in xs]
    return hp["lam"] * ties_disjoint(xs, thr, "mean")


def lowrank_svd(B, A):
    """SVD of B @ A (d_out x r)(r x d_in) through QR(B), QR(A^T) and an r x r core SVD (float64). Returns (U_r, s_r, Vh_r)."""
    Qb, Rb = torch.linalg.qr(B.double()); Qa, Ra = torch.linalg.qr(A.double().T)
    u, s, vh = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ u, s, vh @ Qa.T


def tsvm(dWs, hp, fac=None, ctx=None):
    """TSV-M (Gargiulo et al. CVPR 2025; official compute_and_sum_svd_mem_reduction):
    per task SVD dW_t = U_t S_t V_t^T, keep k components per task, concatenate U's / S's / V's, orthogonalize the
    concatenated U and V by their polar factor (U = P Q^T from SVD P Sigma Q^T), merged = U_perp diag(S) V_perp^T, * lam.
      hp['kmode'] = 'official': k = int(min(d_out,d_in) / T)  (exact official rule; for a rank-r LoRA dW this includes
                                 min_dim/T - r numerically-null singular directions whose basis is SVD-implementation-defined)
                  = 'lora'    : k = min(r, int(min_dim/T)) (keeps exactly the r non-null task singular vectors; concatenation
                                 has T*k columns; LoRA-faithful adaptation, see notes). With LoRA factors the per-task SVD is
                                 computed exactly from (B, A) (lowrank_svd); unit-tested equal to the dense path.)"""
    T = len(dWs)
    dt = dWs[0].dtype
    X = [x.float() for x in dWs]
    mind = min(X[0].shape)
    if hp.get("kmode", "lora") == "official":
        k = int(mind * (1.0 / T))
        ncol = mind
    else:
        k = min(int(hp["r"]), int(mind * (1.0 / T)))
        ncol = T * k
    d_out, d_in = X[0].shape
    su = torch.zeros(d_out, ncol, device=X[0].device, dtype=X[0].dtype)
    ss = torch.zeros(ncol, device=X[0].device, dtype=X[0].dtype)
    sv = torch.zeros(ncol, d_in, device=X[0].device, dtype=X[0].dtype)
    for i, x in enumerate(X):
        if hp.get("kmode", "lora") != "official" and fac is not None and hp.get("fast", True):
            u, s, v = lowrank_svd(fac[i][0], fac[i][1])             # exact SVD of B A via thin QRs (fp64), top-r only
            u, s, v = u.to(X[0].dtype), s.to(X[0].dtype), v.to(X[0].dtype)
        else:
            u, s, v = torch.linalg.svd(x, full_matrices=False)
        su[:, i * k:(i + 1) * k] = u[:, :k]
        ss[i * k:(i + 1) * k] = s[:k]
        sv[i * k:(i + 1) * k, :] = v[:k, :]
    u_u, _, v_u = torch.linalg.svd(su, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sv, full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(ss), u_v, v_v))
    return (hp["lam"] * merged).to(dt)


def knots_align(dWs, fac=None, fast=True, tol=1e-5):
    """KnOTS alignment (Stoica et al. ICLR 2025; official SVDMerger.apply_svd, concat_across_output=True):
    SVD of the column-concatenation [dW_1 | ... | dW_T] (d_out x T*d_in) in float64, keep s > 1e-5, cast to float32;
    task representation rep_t = diag(s) V_t (V_t = task-t block of V^T). Returns (U, [rep_t]).
    fast=True with LoRA factors computes the identical SVD through the rank-(T*r) factorisation
    [B_1..B_T] blockdiag(A_1..A_T) (exact up to fp64 round-off; unit-tested against the dense path)."""
    T = len(dWs)
    d_out, d_in = dWs[0].shape
    if fast and fac is not None:
        Bc = torch.cat([b.double() for b, _ in fac], 1)                       # d_out x T r
        Qb, Rb = torch.linalg.qr(Bc)
        rs = [a.shape[0] for _, a in fac]
        Ac = torch.zeros(sum(rs), T * d_in, dtype=torch.float64, device=Bc.device)
        o = 0
        for i, (_, a) in enumerate(fac):
            Ac[o:o + rs[i], i * d_in:(i + 1) * d_in] = a.double(); o += rs[i]
        Qa, Ra = torch.linalg.qr(Ac.T)                                        # T d_in x T r
        u, s, vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
        U = Qb @ u; Vh = vh @ Qa.T
    else:
        M = torch.cat([x.double() for x in dWs], 1)
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    keep = s > tol
    U = U[:, keep].float(); Vh = Vh[keep].float(); s = s[keep].float()
    reps = [torch.diag(s) @ Vh[:, i * d_in:(i + 1) * d_in] for i in range(T)]
    return U, reps


def knots(dWs, hp, fac=None, ctx=None):
    """KnOTS-TIES: align (knots_align), then TIES on the task representations (trim top-K% per task with the threshold
    global over all layers' representations -> ctx['thr']; sign by mass; disjoint mean), scale by lam, map back with U.
    Official masked_merge multiplies the vectors by scaling_coeffs[0] BEFORE the disjoint mean (== lam * disjoint mean)."""
    if ctx is not None and "U" in ctx:
        U, reps = ctx["U"], ctx["reps"]
    else:
        U, reps = knots_align(dWs, fac, fast=hp.get("fast", True))
    thr = (ctx or {}).get("thr")
    if thr is None and hp.get("topK", 100) < 100:
        thr = [official_kth_threshold(r_.abs().flatten(), hp["topK"]) for r_ in reps]
    if hp.get("mask", "ties") == "ties":
        merged_rep = ties_disjoint(reps, thr, hp.get("merging_type", "mean"))
    else:                                            # official tv_masking: magnitude prune only, no sign election
        X = torch.stack([r_ if (thr is None or thr[i] is None) else r_ * (r_.abs() >= thr[i]) for i, r_ in enumerate(reps)])
        if hp.get("merging_type", "mean") == "mean":
            cnt = (X != 0).sum(0).float(); merged_rep = X.sum(0) / torch.clamp(cnt, min=1)
        else:
            merged_rep = X.sum(0)
    return (U @ (hp["lam"] * merged_rep)).to(dWs[0].dtype)


def pico_calibrate(fac):
    """Pico shared-direction calibration (Tang & Yang 2026, Sec. 4.1): B_all = [B_1..B_T] = U Sigma V^T (thin SVD, m = min(d_out,
    T r) columns), s_j = sigma_j^2 / sum_k sigma_k^2, alpha_j = 1 / (1 + (T-1) s_j), S = I + U diag(alpha - 1) U^T,
    B~_t = S B_t, dW~_t = B~_t A_t.  (LoRA scaling absorbed into B as in the paper; s_j is scale-invariant anyway.)"""
    T = len(fac)
    Bs = [b.double() for b, _ in fac]
    U, sig, _ = torch.linalg.svd(torch.cat(Bs, 1), full_matrices=False)
    s = sig ** 2 / (sig ** 2).sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    out = []
    for (b, a), B in zip(fac, Bs):
        Bt = B + U @ ((alpha - 1.0).unsqueeze(1) * (U.T @ B))              # S B without forming the d_out x d_out S
        out.append((Bt.float() @ a.float()).to(a.dtype))
    return out, {"alpha": alpha.float().cpu(), "s": s.float().cpu()}


def pico(dWs, hp, fac=None, ctx=None):
    """Pico = calibration + base merger + magnitude rescaling (Sec. 4.2):
    dW_calib = M(dW~_1..dW~_T);  gamma = (1/T) sum_t ||dW_t||_F / ||dW_calib||_F  (dW_t = ORIGINAL updates);
    return c * gamma * dW_calib.  Base merger hp['base'] in {'TA','TIES'}; because of the rescaling, the base merger's own lam
    cancels exactly, so the only global scale is the outer multiplier c (c = 1 is the paper's method; c != 1 is our tuning knob)."""
    cal = ctx["cal"] if (ctx is not None and "cal" in ctx) else pico_calibrate(fac)[0]
    if hp.get("base", "TA") == "TA":
        m = _sum(cal)
    else:
        thr = (ctx or {}).get("thr")
        if thr is None and hp.get("k", 100) < 100:
            thr = [official_kth_threshold(x.abs().flatten(), hp["k"]) for x in cal]
        m = ties_disjoint(cal, thr, "mean")
    target = sum(float(x.double().norm()) for x in dWs) / len(dWs)
    nm = float(m.double().norm())
    gamma = target / nm if nm > 0 else 0.0
    return hp["c"] * gamma * m


def gate_shared_basis(B1, B2, theta_deg):
    """principal vectors of orth(B1) whose principal angle to orth(B2) is < theta (E1/E1b definition, float64).
    Returns (Sd float32 or None, theta_min_deg)."""
    U1 = torch.linalg.qr(B1.double())[0]; U2 = torch.linalg.qr(B2.double())[0]
    Pm, S, _ = torch.linalg.svd(U1.T @ U2)
    th_min = math.degrees(math.acos(min(1.0, float(S.max()))))
    cs = math.cos(math.radians(theta_deg))
    if not bool((S > cs).any()):
        return None, th_min
    return (U1 @ Pm[:, S > cs]).float(), th_min


def gate_forced_basis(B1, B2, k=1):
    """threshold-free ('forced') gate basis: the k most-aligned principal vectors of orth(B1) w.r.t. orth(B2) (largest cosines),
    regardless of their angle. Returns (Sd float32, theta_min_deg)."""
    U1 = torch.linalg.qr(B1.double())[0]; U2 = torch.linalg.qr(B2.double())[0]
    Pm, S, _ = torch.linalg.svd(U1.T @ U2)          # S sorted descending
    th_min = math.degrees(math.acos(min(1.0, float(S.max()))))
    return (U1 @ Pm[:, :k]).float(), th_min


def gate(dWs, hp, fac=None, ctx=None):
    """theta*-gate (project-off correction). Pairs only (T=2; 'second' = t2 in task-list order).
    FAIL layer iff theta_min(orth(B1), orth(B2)) < theta*; on FAIL layers dW2 <- dW2 - beta * S S^T dW2, S = shared principal
    vectors (angle < theta*) of orth(B1); beta = 1 is the hard gate (E1b definition), beta < 1 the soft projection strength.
    Non-FAIL layers: exactly lam * (dW1 + dW2) (bitwise TA).
    hp['force_k'] = k (> 0): threshold-free 'forced' gate, every layer projects dW2 off the k most-aligned principal vectors
    of orth(B1) regardless of angle (PREREG_E3 amendment A2)."""
    assert len(dWs) == 2, "gate is defined for pairs"
    d1, d2 = dWs
    if ctx is not None and "Sd" in ctx:
        Sd = ctx["Sd"]
    elif hp.get("force_k"):
        Sd, _ = gate_forced_basis(fac[0][0], fac[1][0], int(hp["force_k"]))
    else:
        Sd, _ = gate_shared_basis(fac[0][0], fac[1][0], hp.get("theta", 30.0))
    if Sd is None:
        return hp["lam"] * (d1 + d2)
    d2g = d2 - hp.get("beta", 1.0) * (Sd @ (Sd.T @ d2))
    return hp["lam"] * (d1 + d2g)


LAYER_FN = {"TA": ta, "TIES": ties, "DARE_TA": dare_ta, "DARE_TIES": dare_ties, "TSVM": tsvm, "KNOTS": knots,
            "PICO_TA": pico, "PICO_TIES": pico, "GATE": gate}


# ----------------------------------------------------------------------------- model level
def merge_model(method, hp, names, dW, fac=None, info=None, thr_cache=None, task_keys=None):
    """dW: list over tasks of {layer: dense dW}; fac: list over tasks of {layer: (B_scaled, A)}.
    Handles the global (cross-layer) parts: TIES/KnOTS/Pico-TIES thresholds, DARE seeding (torch.Generator seeded 0 at the
    start of every call, consumed in task order within layer order), and returns {layer: merged delta}.
    info (dict) receives diagnostics (e.g. number of gate FAIL layers, Pico alpha stats).
    thr_cache: optional dict memoising per-task global TIES thresholds on the ORIGINAL dWs (key (task_keys[t], K))."""
    T = len(dW)
    f = LAYER_FN[method]
    facl = (lambda n: [fac[t][n] for t in range(T)]) if fac is not None else (lambda n: None)
    out = {}
    info = info if info is not None else {}
    if method == "TA":
        for n in names:
            out[n] = f([dW[t][n] for t in range(T)], hp)
    elif method == "TIES":
        thr = []
        for t in range(T):
            key = ((task_keys[t] if task_keys else id(dW[t])), hp["k"])
            if thr_cache is not None and key in thr_cache:
                thr.append(thr_cache[key])
            else:
                v = global_thresholds(dW[t], names, hp["k"]); thr.append(v)
                if thr_cache is not None: thr_cache[key] = v
        for n in names:
            out[n] = f([dW[t][n] for t in range(T)], hp, ctx={"thr": thr})
    elif method in ("DARE_TA", "DARE_TIES"):
        dev = dW[0][names[0]].device
        gen = torch.Generator(device=dev); gen.manual_seed(int(hp.get("seed", 0)))
        dared = {n: [dare_drop(dW[t][n], hp["p"], gen) for t in range(T)] for n in names}   # layer-major, task-minor order
        if method == "DARE_TA":
            for n in names:
                out[n] = hp["lam"] * _sum(dared[n])
        else:
            k = hp.get("k", 20)
            thr = [global_thresholds({n: dared[n][t] for n in names}, names, k) for t in range(T)]
            for n in names:
                out[n] = f(None, hp, ctx={"dared": dared[n], "thr": thr})
        del dared
    elif method == "TSVM":
        for n in names:
            out[n] = f([dW[t][n] for t in range(T)], hp, facl(n))
    elif method == "KNOTS":
        al = {n: knots_align([dW[t][n] for t in range(T)], facl(n), fast=hp.get("fast", True)) for n in names}
        thr = None
        if hp.get("topK", 100) < 100:
            thr = [official_kth_threshold(torch.cat([al[n][1][t].flatten().cpu() for n in names]).abs(), hp["topK"])
                   for t in range(T)]
        for n in names:
            out[n] = f([dW[t][n] for t in range(T)], hp, ctx={"U": al[n][0], "reps": al[n][1], "thr": thr})
        info["knots_rank_kept_mean"] = sum(al[n][0].shape[1] for n in names) / len(names)
        del al
    elif method in ("PICO_TA", "PICO_TIES"):
        cal, amin = {}, []
        for n in names:
            c_, st = pico_calibrate(facl(n)); cal[n] = c_; amin.append(float(st["alpha"].min()))
        info["pico_alpha_min_mean"] = sum(amin) / len(amin); info["pico_alpha_min_min"] = min(amin)
        thr = None
        if method == "PICO_TIES" and hp.get("k", 100) < 100:
            thr = [global_thresholds({n: cal[n][t] for n in names}, names, hp["k"]) for t in range(T)]
        hp2 = dict(hp); hp2["base"] = "TA" if method == "PICO_TA" else "TIES"
        for n in names:
            out[n] = f([dW[t][n] for t in range(T)], hp2, ctx={"cal": cal[n], "thr": thr})
        del cal
    elif method == "GATE":
        nfail = 0; thmins = []
        for n in names:
            if hp.get("force_k"):
                Sd, thm = gate_forced_basis(fac[0][n][0], fac[1][n][0], int(hp["force_k"]))
            else:
                Sd, thm = gate_shared_basis(fac[0][n][0], fac[1][n][0], hp.get("theta", 30.0))
            thmins.append(thm); nfail += Sd is not None
            out[n] = f([dW[0][n], dW[1][n]], hp, ctx={"Sd": Sd})
        info["gate_n_fail_layers"] = nfail; info["gate_min_theta_min_deg"] = min(thmins)
    else:
        raise ValueError(method)
    return out
