import json, sys, time, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from tasks import *
out = {}
for t in list(TASKS) + list(SUBSTITUTES):
    spec = ALL_SPECS[t]
    try:
        tr = load_raw(t, spec[2]); ev = load_raw(t, spec[3])
        a, b, y = columns(t, tr); ae, be, ye = columns(t, ev)
        rec = {"ok": True, "n_train": len(tr), "n_eval": len(ev), "cols": tr.column_names,
               "ex_a": a[0][:120], "ex_b": (b[0][:120] if b else None)}
        if spec[7] > 1:
            ct = collections.Counter(y.tolist()); ce = collections.Counter(ye.tolist())
            maj = ct.most_common(1)[0][0]
            rec["train_label_counts"] = dict(ct); rec["eval_label_counts"] = dict(ce)
            rec["majority_label_train"] = maj; rec["majority_baseline_eval_full"] = ce.get(maj, 0) / len(ye)
            rec["labels_in_range"] = bool(set(ct) <= set(range(spec[7])) and set(ce) <= set(range(spec[7])))
        else:
            rec["label_range"] = [float(y.min()), float(y.max())]
    except Exception as e:
        rec = {"ok": False, "error": repr(e)[:500]}
    out[t] = rec
    print(t, json.dumps(rec)[:600], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
