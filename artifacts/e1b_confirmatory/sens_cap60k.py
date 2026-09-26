"""EXPLORATORY, post-verdict sensitivity check (not pre-registered, not used for any verdict):
score the kept 60k-capped MNLI/QQP adapters (adapters/*_cap60k) on the same eval sets with the same integrity criteria (a)/(b)."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import e1b
e1b.OUT = HERE; e1b.ADIR = HERE / "adapters"; e1b.LOGF = HERE / "run.log"
torch = e1b.setup_torch()
from transformers import AutoTokenizer
assert (HERE / "analysis.json").exists(), "run only after the verdict"
tok = AutoTokenizer.from_pretrained(e1b.BASE, revision=e1b.BREV)
s0 = json.loads((HERE / "stage0.json").read_text())
res = {"note": "EXPLORATORY post-verdict sensitivity check; capped adapters are not part of the confirmatory analysis", "tasks": {}}
enc = None
for t in ("mnli", "qqp"):
    ad = e1b.load_adapter(f"{t}_cap60k"); d = e1b.get_data(t, tok)
    names = sorted(ad["layers"])
    if enc is None: enc = e1b.Enc(torch.device("cuda"), names)
    enc.set_delta({n: e1b.delta(ad, n, "cuda") for n in names})
    se = e1b.score(t, enc.predict(d["eval"], ad["head"]), d["eval"]["labels"])
    ref = e1b.REF[t]; pt = s0["per_task"][t]
    res["tasks"][t] = {"cap60k_eval": se, "a_threshold": pt["a_threshold"], "a_ok": se["main"] >= pt["a_threshold"],
                       "b_threshold": ref["value"] - 0.05, "b_ok": se["main"] >= ref["value"] - 0.05,
                       "full_data_D1_eval": pt["eval"]["main"], "train_loss_cap60k": ad["meta"]["train_loss"]}
    e1b.log(f"EXPLORATORY cap60k {t}: eval={se['main']:.4f} a_ok={res['tasks'][t]['a_ok']} b_ok={res['tasks'][t]['b_ok']} (full-data D1 adapter: {pt['eval']['main']:.4f})")
(HERE / "sensitivity_cap60k.json").write_text(json.dumps(res, indent=1))
