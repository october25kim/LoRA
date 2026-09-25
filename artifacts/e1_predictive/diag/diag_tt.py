# Stage-0 diagnostic: effect of token_type_ids on single-adapter scores + card-slice reproduction
import sys, json, numpy as np, torch
sys.path.insert(0, ".")
import e1
from pathlib import Path
e1.OUT = Path(".").resolve()
from transformers import AutoTokenizer
s0 = json.load(open("stage0.json"))
tok = AutoTokenizer.from_pretrained(e1.BASE, revision=s0["base_model"]["revision"])
enc = e1.Enc(torch.device("cuda"), s0["lora_layers"], s0["base_model"]["revision"])
res = {}
for t in e1.TASKS:
    ad = e1.load_adapter(t); enc.set_delta(e1.delta_weights(ad, torch.device("cuda")))
    dv = e1.get_data(t, "val", tok)
    z = dict(dv); z["tt"] = [[0]*len(x) for x in dv["tt"]]
    full = e1.load_split(t, "train"); sp = full.train_test_split(test_size=100, seed=0)
    dc = e1.tokenize(t, sp["test"], tok); zc = dict(dc); zc["tt"] = [[0]*len(x) for x in dc["tt"]]
    r = {}
    for name, d in [("val_tt", dv), ("val_tt0", z), ("card_slice_seed0_tt", dc), ("card_slice_seed0_tt0", zc)]:
        m, _ = e1.metrics(enc.predict(d, ad["head"]), d["labels"]); r[name] = {k: round(v, 4) for k, v in m.items()}
    res[t] = r; print(t, "card", e1.CARD[t], json.dumps(r), flush=True)
json.dump(res, open("diag/diag_tt.json", "w"), indent=1)
