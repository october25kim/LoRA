#!/usr/bin/env python3
"""E7 (EXPLORATORY / POST HOC) stage a1 for the Qwen2.5-0.5B E4b pairs (+ E5b extended-lambda predictions): same computation as e7.py st_a1
(singles + merged TA at G7 on the <=200 E6-split cal inputs -> entropy, agreement, timings), using the UNMODIFIED e4b.py Enc/get_data/load_ad.
Writes a1_entropy_q05.jsonl into artifacts/e7_baselines/. Added after PLAN_E7.md (see DEVIATIONS_E7.md)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; E4D = HERE.parent / "e4b_decoder"; E5B = HERE.parent / "e5_accept" / "e5b"
sys.path.insert(0, str(E4D)); sys.path.insert(0, str(HERE.parent / "e6_decoder2"))
import e4b                                  # noqa
import lam_confirm_e6 as LC                 # noqa
import e7                                   # noqa  (helpers: entropy, Timer, LazyD, status, jl_append, done_keys, log)
bm = e4b.bm; G7 = e7.G7; BS = e4b.BS


def main():
    torch = bm.setup_torch(); dev = torch.device("cuda")
    PO = json.loads((E4D / "populations_e4b.json").read_text())
    pairs = [(p, tuple(x.split("__"))) for p in ("P", "R0", "S1") for x in PO["populations"][p]]
    names = PO["lora_layers"]; out = HERE / "a1_entropy_q05.jsonl"; done = e7.done_keys(out)
    enc = e4b.Enc(dev, names); ADS = {}; D, C, S = {}, {}, {}
    def ad(k):
        if k not in ADS: ADS[k] = e4b.load_ad(k)
        return ADS[k]
    def data(t):
        if t not in D: D[t] = e4b.get_data(t)
        return D[t]
    def cal(t):
        if t not in C:
            d = data(t)["eval"]; m = LC.split("qwen", t, len(d["labels"])); idx = np.where(m)[0]
            C[t] = (m, idx, {"ids": [d["ids"][i] for i in idx], "tt": [d["tt"][i] for i in idx]})
        return C[t]
    def single(k):
        if k not in S: S[k] = dict(np.load(E4D / "preds" / f"single_{k}.npz"))
        return S[k]
    for t in sorted({e4b.task_of(k) for _, pr in pairs for k in pr}): enc.batches(cal(t)[2], BS)
    for i, (pop, (x, y)) in enumerate(pairs):
        key = e4b.pid(x, y)
        if key in done: continue
        e7.status("a1_q05", i, len(pairs), key)
        rec = {"id": key, "pop": pop, "t1": x, "t2": y, "timing": {}, "peak_gb": {}, "n_fwd": {}}
        Z = dict(np.load(E4D / "preds" / f"pair_{key}.npz")); Z.update(dict(np.load(E5B / "preds" / f"pair_{key}.npz")))
        ks = [(x, e4b.task_of(x)), (y, e4b.task_of(y))]
        for k, _ in ks: ad(k)
        sl = {}
        with e7.Timer(torch) as T:
            for k, t in ks:
                enc.set_delta({n: bm.delta(ad(k), n, dev) for n in names}); sl[k] = enc.predict(cal(t)[2], ad(k)["head"], BS)
        rec["timing"]["singles_cal_s"] = T.s; rec["peak_gb"]["singles_cal"] = T.peak
        rec["n_fwd"]["singles_cal_examples"] = int(sum(len(cal(t)[1]) for _, t in ks))
        for lam in G7:
            with e7.Timer(torch) as T:
                enc.set_delta({n: lam * (bm.delta(ad(x), n, dev) + bm.delta(ad(y), n, dev)) for n in names})
                ml = {k: enc.predict(cal(t)[2], ad(k)["head"], BS) for k, t in ks}
            rec["timing"][f"merged_cal_lam{lam}_s"] = T.s; rec["peak_gb"][f"merged_cal_lam{lam}"] = T.peak
            for j, (k, t) in enumerate(ks, 1):
                m, idx, _ = cal(t); reg = e4b.metric_name(t) == "spearman"
                rec[f"ent{j}_{lam}"], rec[f"entn{j}_{lam}"] = (None, None) if reg else e7.entropy(ml[k])
                rec[f"agree{j}_{lam}"] = LC.score(bm.compact(t, ml[k]), single(k)["eval"][idx])
                cm = Z[f"TA_{k}_lam{lam}"][idx]; rm = bm.compact(t, ml[k])
                rec[f"match{j}_{lam}"] = float(np.mean(rm == cm)) if not reg else float(np.max(np.abs(rm - cm)))
        rec["n_fwd"]["merged_cal_examples_per_lam"] = int(sum(len(cal(t)[1]) for _, t in ks))
        e7.jl_append(out, rec)
        e7.log(f"a1_q05 {i+1}/{len(pairs)} {key}: t_lam1={rec['timing']['merged_cal_lam1.0_s']:.2f}s match={[rec[f'match1_{l}'] for l in G7]}")
        if len(ADS) > 8: ADS.clear()
    e7.status("a1_q05", len(pairs), len(pairs), "DONE")


if __name__ == "__main__":
    e7.log(f"=== e7_q05.py {sys.argv}"); main()
