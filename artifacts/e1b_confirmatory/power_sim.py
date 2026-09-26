"""E1b pre-registration power simulation (run BEFORE any training).
Synthetic pair-level predictor X and pair loss D over K tasks with task-block (crossed random effect) structure:
  X_ij = sqrt(w)*(u_i+u_j)/sqrt(2) + sqrt(1-w)*e_ij,   D_ij = sqrt(w)*(v_i+v_j)/sqrt(2) + sqrt(1-w)*n_ij
  corr(u,v) = corr(e,n) = r_lat, so Pearson(X,D) = r_lat; r_lat = 2 sin(pi*rho/6) gives population Spearman rho.
Decision rule = the pre-registered one, same resampling (2000 task-block bootstrap, 10000 task-label permutations),
evaluated per hypothesis at one-sided alpha 0.025 (Holm, worst case: this hypothesis has the smaller p) and 0.05
(Holm, best case: the other hypothesis already rejected)."""
import itertools, json, sys, time
import numpy as np
from scipy.stats import rankdata
from multiprocessing import Pool

K = 16; NBOOT = 2000; NPERM = 10000
PAIRS = list(itertools.combinations(range(K), 2)); I = np.array([p[0] for p in PAIRS]); J = np.array([p[1] for p in PAIRS])
PI = np.full((K, K), -1); PI[I, J] = np.arange(len(PAIRS)); PI[J, I] = np.arange(len(PAIRS))


def spear(x, y):
    rx = rankdata(x); ry = rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def one(args):
    rho, w, seed = args
    rng = np.random.default_rng(seed)
    r = 2 * np.sin(np.pi * rho / 6)
    C = np.array([[1, r], [r, 1]])
    uv = rng.multivariate_normal([0, 0], C, size=K)
    en = rng.multivariate_normal([0, 0], C, size=len(PAIRS))
    X = np.sqrt(w) * (uv[I, 0] + uv[J, 0]) / np.sqrt(2) + np.sqrt(1 - w) * en[:, 0]
    D = np.sqrt(w) * (uv[I, 1] + uv[J, 1]) / np.sqrt(2) + np.sqrt(1 - w) * en[:, 1]
    obs = spear(X, D)
    # permutation (task labels of D permuted)
    rX = rankdata(X); rX = (rX - rX.mean()) / np.sqrt(((rX - rX.mean()) ** 2).sum())
    rD = rankdata(D); rD = (rD - rD.mean()) / np.sqrt(((rD - rD.mean()) ** 2).sum())
    prng = np.random.default_rng(seed + 10 ** 6)
    perms = np.argsort(prng.random((NPERM, K)), axis=1)
    maps = PI[perms[:, I], perms[:, J]]
    rp = (rD[maps] * rX[None, :]).sum(1)
    p1 = (1 + np.sum(rp >= obs - 1e-12)) / (1 + NPERM)
    # task-block bootstrap
    brng = np.random.default_rng(seed + 2 * 10 ** 6)
    Dm = np.zeros((K, K)); Xm = np.zeros((K, K)); Dm[I, J] = D; Dm[J, I] = D; Xm[I, J] = X; Xm[J, I] = X
    iu = np.triu_indices(K, 1)
    bs = []
    for _ in range(NBOOT):
        s = brng.integers(0, K, K)
        a = s[iu[0]]; b = s[iu[1]]; m = a != b
        bs.append(spear(Xm[a[m], b[m]], Dm[a[m], b[m]]))
    hi = float(np.nanpercentile(bs, 97.5)); lo = float(np.nanpercentile(bs, 2.5))
    loto = []
    for t in range(K):
        m = (I != t) & (J != t)
        loto.append(spear(X[m], D[m]))
    loto_ok = np.mean(np.array(loto) > 0.2) >= 0.75
    out = {}
    for alpha in (0.025, 0.05):
        if obs >= 0.4 and p1 < alpha and loto_ok:
            v = "PASS"
        elif hi < 0.3:
            v = "FAIL"
        else:
            v = "INCONCLUSIVE"
        out[alpha] = v
    return dict(rho=rho, w=w, obs=obs, p1=p1, lo=lo, hi=hi, loto_ok=bool(loto_ok), v025=out[0.025], v05=out[0.05])


if __name__ == "__main__":
    nsim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()
    jobs = [(rho, w, 1000 * k + int(rho * 10) * 7 + int(w * 100)) for rho in (0.0, 0.3, 0.5) for w in (0.25, 0.5, 0.75) for k in range(nsim)]
    with Pool(8) as p:
        res = p.map(one, jobs, chunksize=4)
    summ = {}
    for rho in (0.0, 0.3, 0.5):
        for w in (0.25, 0.5, 0.75):
            rr = [r for r in res if r["rho"] == rho and r["w"] == w]
            e = {"n_sim": len(rr), "mean_obs_rho": float(np.mean([r["obs"] for r in rr])),
                 "sd_obs_rho": float(np.std([r["obs"] for r in rr])),
                 "mean_boot_upper": float(np.mean([r["hi"] for r in rr])), "mean_boot_width": float(np.mean([r["hi"] - r["lo"] for r in rr])),
                 "P(perm p1<0.05)": float(np.mean([r["p1"] < 0.05 for r in rr]))}
            for a in ("v025", "v05"):
                for v in ("PASS", "FAIL", "INCONCLUSIVE"):
                    e[f"P({v})@alpha{'0.025' if a == 'v025' else '0.05'}"] = float(np.mean([r[a] == v for r in rr]))
            summ[f"rho={rho},w={w}"] = e
            print(f"rho={rho} w={w}: " + json.dumps({k: round(v, 3) for k, v in e.items()}), flush=True)
    json.dump({"K": K, "n_boot": NBOOT, "n_perm": NPERM, "summary": summ, "wall_s": time.time() - t0,
               "model": __doc__}, open(sys.argv[2] if len(sys.argv) > 2 else "power_sim.json", "w"), indent=1)
    print("wall", time.time() - t0)
