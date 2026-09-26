"""Unit tests for merges.py (run: python -m pytest -q test_merges.py). CPU, float32/float64, seconds."""
import math
import torch
import pytest
import merges as M

torch.manual_seed(0)
SHAPES = {"a.q": (24, 16), "b.inter": (40, 16), "c.out": (16, 40)}
R = 3


def make_tasks(T=2, seed=0, shapes=SHAPES, r=R, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    fac, dW = [], []
    for t in range(T):
        f, d = {}, {}
        for n, (o, i) in shapes.items():
            B = torch.randn(o, r, generator=g) * 0.3; A = torch.randn(r, i, generator=g) * 0.3
            f[n] = (scale * B, A); d[n] = scale * (B @ A)
        fac.append(f); dW.append(d)
    return fac, dW


NAMES = sorted(SHAPES)
ALL_CFGS = [("TA", {"lam": 0.7}), ("TIES", {"k": 20, "lam": 1.5}), ("DARE_TA", {"p": 0.5, "lam": 1.0}),
            ("DARE_TIES", {"p": 0.5, "k": 20, "lam": 1.0}), ("TSVM", {"kmode": "lora", "r": R, "lam": 1.0}),
            ("TSVM", {"kmode": "official", "lam": 1.0}), ("KNOTS", {"topK": 20, "lam": 1.4}), ("KNOTS", {"topK": 100, "lam": 1.4}),
            ("PICO_TA", {"c": 1.0}), ("PICO_TIES", {"k": 20, "c": 1.0}), ("GATE", {"theta": 30.0, "lam": 0.7}),
            ("GATE", {"theta": 30.0, "beta": 0.5, "lam": 0.7}), ("GATE", {"force_k": 1, "lam": 0.7}),
            ("GATE", {"force_k": 1, "beta": 0.5, "lam": 0.7})]


# ---------------------------------------------------------------- vendored official references (minimal copies)
def off_topk_values_mask(M_, K=0.7):                       # ties-merging/src/ties_minimal.ipynb
    if K > 1: K /= 100
    if M_.dim() == 1: M_ = M_.unsqueeze(0)
    n, d = M_.shape; k = int(d * K); k = d - k
    kth, _ = M_.abs().kthvalue(k, dim=1, keepdim=True)
    mask = M_.abs() >= kth
    return M_ * mask


def off_ties(flat, K, merge_func="mean"):
    upd = off_topk_values_mask(flat.clone(), K)
    sgn = torch.sign(upd.sum(0)); maj = torch.sign(sgn.sum()); sgn[sgn == 0] = maj
    keep = torch.where(sgn.unsqueeze(0) > 0, upd > 0, upd < 0); sel = upd * keep
    return sel.sum(0) / torch.clamp((sel != 0).sum(0).float(), min=1)


def off_tsvm(task_mats):                                   # task_singular_vectors compute_and_sum_svd_mem_reduction (2-D keys)
    T = len(task_mats); red = 1 / T; out = {}
    for key in task_mats[0]:
        for i, tm in enumerate(task_mats):
            u, s, v = torch.linalg.svd(tm[key], full_matrices=False)
            if i == 0:
                su = torch.zeros_like(u); ss = torch.zeros_like(s); sv = torch.zeros_like(v)
            k = int(s.shape[0] * red)
            su[:, i * k:(i + 1) * k] = u[:, :k]; ss[i * k:(i + 1) * k] = s[:k]; sv[i * k:(i + 1) * k, :] = v[:k, :]
        uu, _, vu = torch.linalg.svd(su, full_matrices=False); uv, _, vv = torch.linalg.svd(sv, full_matrices=False)
        out[key] = torch.linalg.multi_dot((uu, vu, torch.diag(ss), uv, vv))
    return out


def off_knots(task_mats, topK, lam, merging_type="mean"):   # KnOTS SVDMerger.apply_svd + ties_masking + masked_merge
    keys = sorted(task_mats[0]); T = len(task_mats)
    U, reps = {}, [dict() for _ in range(T)]
    for key in keys:
        val = torch.cat([tm[key] for tm in task_mats], dim=1)
        u, s, V = torch.linalg.svd(val.to(torch.float64), full_matrices=False)
        u = u[:, s > 1e-5].float(); V = V[s > 1e-5].float(); s = s[s > 1e-5].float()
        U[key] = u; hid = V.shape[1] // T
        for idx, Vt in enumerate(torch.split(V, hid, dim=1)):
            reps[idx][key] = torch.diag(s) @ Vt
    flat = torch.vstack([torch.nn.utils.parameters_to_vector([r[k].reshape(-1) for k in keys]) for r in reps])
    if topK < 100:
        n, d = flat.shape; k = d - int(d * topK / 100)
        kth, _ = flat.abs().kthvalue(k, dim=1, keepdim=True); pm = flat.abs() >= kth
    else:
        pm = torch.ones_like(flat, dtype=torch.bool)
    pr = flat * pm
    sg = torch.sign(pr.sum(0)); maj = torch.sign(sg.sum()); sg[sg == 0] = maj
    mask = torch.where(sg.unsqueeze(0) > 0, pr > 0, pr < 0) * pm
    v = flat * mask * lam
    if merging_type == "mean":
        tot = v.sum(0); cnt = (v != 0).sum(0).float(); merged = tot / torch.clamp(cnt, min=1); merged[cnt == 0] = 0
    else:
        merged = v.sum(0)
    out, o = {}, 0
    for key in keys:
        sz = reps[0][key].numel(); out[key] = U[key] @ merged[o:o + sz].reshape(reps[0][key].shape); o += sz
    return out


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("method,hp", ALL_CFGS)
def test_shapes_and_finite(method, hp):
    fac, dW = make_tasks()
    out = M.merge_model(method, dict(hp), NAMES, dW, fac)
    for n in NAMES:
        assert out[n].shape == SHAPES[n] and out[n].dtype == torch.float32 and torch.isfinite(out[n]).all()


def test_ta_special_cases():
    fac, dW = make_tasks()
    for n in NAMES:
        assert torch.equal(M.ta([dW[0][n]], {"lam": 1.0}), dW[0][n])
        assert torch.equal(M.ta([dW[0][n], dW[1][n]], {"lam": 0.7}), 0.7 * (dW[0][n] + dW[1][n]))   # bitwise E1 formula
        assert torch.equal(M.ta([dW[0][n], dW[1][n]], {"lam": 0.0}), torch.zeros(SHAPES[n]))
    base = M.merge_model("TA", {"lam": 1.0}, NAMES, dW)
    for n in NAMES:
        assert torch.equal(0.5 * base[n], 0.5 * (dW[0][n] + dW[1][n]))       # lam * base == E1 lam*(d1+d2) bitwise


def test_ties_toy_by_hand():
    t1 = torch.tensor([3.0, -1.0, 0.5, -4.0, 0.1]); t2 = torch.tensor([-2.0, -2.0, 1.0, 1.0, -0.2])
    # keep top 40%: d=5, k = 5 - int(2) = 3 -> kthvalue(3) of |t1| = {0.1,0.5,1,3,4} -> 1.0 -> keep |x|>=1 : [3,-1,0,-4,0]
    # |t2| = {0.2,1,1,2,2} -> kth(3)=1.0 -> keep [-2,-2,1,1,0]
    # sum = [1,-3,1,-3,0] -> sign [1,-1,1,-1,0->majority sign(sum signs)=sign(1-1+1-1+0)=0 -> 0]
    # disjoint mean: c0: only +3 agrees -> 3 ; c1: -1,-2 -> -1.5 ; c2: +1 -> 1 ; c3: -4 -> -4 ; c4: 0
    thr = [M.official_kth_threshold(t1.abs(), 40), M.official_kth_threshold(t2.abs(), 40)]
    out = M.ties([t1, t2], {"k": 40, "lam": 1.0}, ctx={"thr": thr})
    assert torch.allclose(out, torch.tensor([3.0, -1.5, 1.0, -4.0, 0.0]))
    assert torch.allclose(off_ties(torch.stack([t1, t2]), 40), out)


def test_ties_matches_official_global():
    fac, dW = make_tasks(T=3, seed=1)
    for K in (10, 20, 30):
        out = M.merge_model("TIES", {"k": K, "lam": 1.0}, NAMES, dW)
        flat = torch.vstack([torch.cat([d[n].flatten() for n in NAMES]) for d in dW])
        ref = off_ties(flat, K)
        mine = torch.cat([out[n].flatten() for n in NAMES])
        assert torch.allclose(mine, ref, atol=1e-7), K


def test_ties_k100_disjoint_mean_properties():
    a = torch.tensor([1.0, 2.0, -1.0]); b = torch.tensor([3.0, -1.0, -3.0])
    out = M.ties([a, b], {"k": 100, "lam": 1.0})
    assert torch.allclose(out, torch.tensor([2.0, 2.0, -2.0]))


def test_dare_properties():
    fac, dW = make_tasks()
    g = torch.Generator().manual_seed(0)
    x = dW[0]["b.inter"]
    assert torch.equal(M.dare_drop(x, 0.0, g), x)
    for p in (0.5, 0.9):
        big = torch.randn(200000)
        y = M.dare_drop(big, p, torch.Generator().manual_seed(0))
        frac_kept = float((y != 0).float().mean())
        assert abs(frac_kept - (1 - p)) < 0.01
        assert torch.allclose(y[y != 0], big[y != 0] / (1 - p))
        ys = torch.stack([M.dare_drop(torch.ones(1000), p, torch.Generator().manual_seed(s)) for s in range(400)])
        assert abs(float(ys.mean()) - 1.0) < 0.03                               # unbiased
    o1 = M.merge_model("DARE_TA", {"p": 0.5, "lam": 1.0}, NAMES, dW)
    o2 = M.merge_model("DARE_TA", {"p": 0.5, "lam": 1.0}, NAMES, dW)
    assert all(torch.equal(o1[n], o2[n]) for n in NAMES)                         # seed-0 deterministic
    o0 = M.merge_model("DARE_TA", {"p": 0.0, "lam": 0.7}, NAMES, dW); ota = M.merge_model("TA", {"lam": 0.7}, NAMES, dW)
    assert all(torch.equal(o0[n], ota[n]) for n in NAMES)                        # p=0 -> TA
    t0 = M.merge_model("DARE_TIES", {"p": 0.0, "k": 20, "lam": 1.0}, NAMES, dW); tt = M.merge_model("TIES", {"k": 20, "lam": 1.0}, NAMES, dW)
    assert all(torch.allclose(t0[n], tt[n]) for n in NAMES)                      # p=0 -> TIES


def test_tsvm_official_matches_reference():
    g = torch.Generator().manual_seed(3)
    mats = [{n: torch.randn(*SHAPES[n], generator=g) for n in NAMES} for _ in range(2)]
    ref = off_tsvm(mats)
    for n in NAMES:
        mine = M.tsvm([mats[0][n], mats[1][n]], {"kmode": "official", "lam": 1.0})
        assert torch.allclose(mine, ref[n], atol=1e-5)


def test_tsvm_lora_special_cases():
    fac, dW = make_tasks()
    for n in NAMES:                                                               # T=1: reconstructs dW
        assert torch.allclose(M.tsvm([dW[0][n]], {"kmode": "lora", "r": R, "lam": 1.0}), dW[0][n], atol=1e-5)
    # disjoint (orthogonal) supports in both output and input spaces -> TSV-M == TA
    d1 = torch.zeros(20, 20); d2 = torch.zeros(20, 20)
    d1[:5, :5] = torch.randn(5, 5, generator=torch.Generator().manual_seed(0)) @ torch.diag(torch.tensor([1., 1, 1, 0, 0])) @ torch.randn(5, 5)
    d2[10:15, 10:15] = torch.randn(5, 3) @ torch.randn(3, 5)
    out = M.tsvm([d1, d2], {"kmode": "lora", "r": 3, "lam": 1.0})
    assert torch.allclose(out, d1 + d2, atol=1e-4)


def test_tsvm_lora_fast_equals_dense():
    fac, dW = make_tasks(seed=9)
    for n in NAMES:
        a = M.tsvm([dW[0][n], dW[1][n]], {"kmode": "lora", "r": R, "lam": 1.0}, [fac[0][n], fac[1][n]])
        b = M.tsvm([dW[0][n], dW[1][n]], {"kmode": "lora", "r": R, "lam": 1.0}, None)
        assert torch.allclose(a, b, atol=1e-5)


def test_knots_fast_equals_dense_and_reference():
    fac, dW = make_tasks(T=2, seed=5)
    for n in NAMES:
        U1, r1 = M.knots_align([dW[0][n], dW[1][n]], [fac[0][n], fac[1][n]], fast=True)
        U2, r2 = M.knots_align([dW[0][n], dW[1][n]], None, fast=False)
        assert U1.shape == U2.shape
        rec1 = [U1 @ r for r in r1]; rec2 = [U2 @ r for r in r2]
        for t in range(2):
            assert torch.allclose(rec1[t], dW[t][n], atol=1e-5) and torch.allclose(rec2[t], dW[t][n], atol=1e-5)
    for topK in (20, 100):
        mine = M.merge_model("KNOTS", {"topK": topK, "lam": 1.3, "fast": False}, NAMES, dW, fac)
        ref = off_knots(dW, topK, 1.3)
        for n in NAMES:
            assert torch.allclose(mine[n], ref[n], atol=1e-5), (topK, n)
        fastm = M.merge_model("KNOTS", {"topK": topK, "lam": 1.3, "fast": True}, NAMES, dW, fac)
        # fast path: singular vectors identical up to sign; masks/signs are sign-equivariant -> same merged update
        for n in NAMES:
            assert torch.allclose(fastm[n], ref[n], atol=1e-4), (topK, n)


def test_knots_tv_mask_topK100_sum_is_TA():
    """KnOTS-TA (tv mask, no trim, sum) reduces exactly to task arithmetic: U s (V1 + V2) = dW1 + dW2."""
    fac, dW = make_tasks()
    ta = M.merge_model("TA", {"lam": 0.7}, NAMES, dW)
    for n in NAMES:
        out = M.knots([dW[0][n], dW[1][n]], {"topK": 100, "lam": 0.7, "merging_type": "sum", "mask": "tv"}, [fac[0][n], fac[1][n]])
        assert torch.allclose(out, ta[n], atol=1e-5)


def test_pico_properties():
    fac, dW = make_tasks()
    for n in NAMES:                                      # T=1: s_j sums to 1 but (T-1)=0 -> alpha=1 -> identity; rescale -> dW
        out = M.pico([dW[0][n]], {"c": 1.0, "base": "TA"}, [fac[0][n]])
        assert torch.allclose(out, dW[0][n], atol=1e-5)
    out = M.merge_model("PICO_TA", {"c": 1.3}, NAMES, dW, fac)
    for n in NAMES:
        target = (dW[0][n].norm() + dW[1][n].norm()) / 2
        assert abs(float(out[n].norm()) - 1.3 * float(target)) < 1e-4 * float(target)
        _, st = M.pico_calibrate([fac[0][n], fac[1][n]])
        assert float(st["alpha"].min()) >= 0.5 - 1e-6 and float(st["alpha"].max()) <= 1 + 1e-6
    # hand example: B1 = [3 e1], B2 = [1 e1 + 0 ...], A = I -> B_all = [3e1, e1] -> one component sigma^2 = 10 (s=1), alpha=1/2
    e = torch.eye(4)
    B1 = 3 * e[:, :1]; B2 = 1 * e[:, :1]; A1 = e[:1, :]; A2 = e[1:2, :]
    cal, st = M.pico_calibrate([(B1, A1), (B2, A2)])
    assert torch.allclose(cal[0], 0.5 * B1 @ A1) and torch.allclose(cal[1], 0.5 * B2 @ A2)
    # two orthogonal equal-energy directions: s = [.5,.5] -> alpha = 2/3 both -> uniform shrink -> after rescale == TA direction
    B1 = e[:, :1]; B2 = e[:, 1:2]
    cal, st = M.pico_calibrate([(B1, A1), (B2, A2)])
    assert torch.allclose(st["alpha"], torch.tensor([2 / 3, 2 / 3]))
    out = M.pico([B1 @ A1, B2 @ A2], {"c": 1.0, "base": "TA"}, [(B1, A1), (B2, A2)])
    ta = B1 @ A1 + B2 @ A2
    assert torch.allclose(out, ta / ta.norm() * 1.0, atol=1e-6)


def test_gate():
    fac, dW = make_tasks(seed=7)
    # random 3-dim subspaces in 16..40 dims: theta_min >> 30deg -> no FAIL layer -> bitwise TA
    info = {}
    g = M.merge_model("GATE", {"theta": 30.0, "lam": 0.7}, NAMES, dW, fac, info=info)
    ta = M.merge_model("TA", {"lam": 0.7}, NAMES, dW)
    assert info["gate_n_fail_layers"] == 0 and all(torch.equal(g[n], ta[n]) for n in NAMES)
    # construct a shared direction: B2's first column = B1's first column
    n = "b.inter"
    B1, A1 = fac[0][n]; B2, A2 = fac[1][n]
    B2s = B2.clone(); B2s[:, 0] = B1[:, 0]
    d2s = B2s @ A2
    Sd, th = M.gate_shared_basis(B1, B2s, 30.0)
    assert Sd is not None and th < 1e-3 and Sd.shape[1] >= 1
    hard = M.gate([dW[0][n], d2s], {"theta": 30.0, "lam": 1.0}, [(B1, A1), (B2s, A2)])
    corr = hard - dW[0][n]
    assert float((Sd.T @ corr).abs().max()) < 1e-4                     # projected dW2 has no component on shared dirs
    zero = M.gate([dW[0][n], d2s], {"theta": 30.0, "lam": 1.0, "beta": 0.0}, [(B1, A1), (B2s, A2)])
    assert torch.allclose(zero, dW[0][n] + d2s)
    half = M.gate([dW[0][n], d2s], {"theta": 30.0, "lam": 1.0, "beta": 0.5}, [(B1, A1), (B2s, A2)])
    assert torch.allclose(half, 0.5 * (hard + zero), atol=1e-5)


def test_forced_gate():
    fac, dW = make_tasks(seed=7)
    info = {}
    g = M.merge_model("GATE", {"force_k": 1, "lam": 1.0}, NAMES, dW, fac, info=info)
    for n in NAMES:
        Sd, th = M.gate_forced_basis(fac[0][n][0], fac[1][n][0], 1)
        assert Sd.shape[1] == 1 and th > 30                      # fires although no layer is below 30 deg
        corr = g[n] - dW[0][n]
        assert float((Sd.T @ corr).abs().max()) < 1e-5            # dW2 has no component along the top aligned direction
        # the projected direction is the principal vector of orth(B1) closest to span(B2)
        U2 = torch.linalg.qr(fac[1][n][0].double())[0].float()
        cos_top = float((U2.T @ Sd).norm())
        U1 = torch.linalg.qr(fac[0][n][0].double())[0].float()
        assert cos_top >= float(torch.linalg.svdvals(U1.T @ U2).max()) - 1e-5
    assert info["gate_n_fail_layers"] == len(NAMES)


def test_lambda_linearity_all_methods():
    """runner relies on merged(lam) == lam * merged(lam=1) (c for Pico)."""
    fac, dW = make_tasks(seed=11)
    for method, hp in ALL_CFGS:
        key = "c" if method.startswith("PICO") else "lam"
        h1 = dict(hp); h1[key] = 1.0; h2 = dict(hp); h2[key] = 1.7
        a = M.merge_model(method, h1, NAMES, dW, fac); b = M.merge_model(method, h2, NAMES, dW, fac)
        for n in NAMES:
            assert torch.allclose(1.7 * a[n], b[n], atol=1e-5, rtol=1e-5), (method, n)
