# Stage-0 item 4 follow-up (report only, no training): MNLI/RTE accuracy of local hubish adapters
# with BERT segment ids vs all-zero segment ids. train_lora_glue.py trains WITH token_type_ids
# (DataCollatorWithPadding keeps them) while lora_merge_cert/eval.py evaluates WITHOUT them.
import sys, json, torch, numpy as np
sys.path.insert(0, "artifacts/e1_predictive")
import e1
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
from peft import PeftModel
e1.OUT = Path("artifacts/e1_predictive/diag").resolve()
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(e1.BASE)
res = {}
for a, t in [("mnli_s7_hubish", "mnli"), ("mnli_s42_hubish", "mnli"), ("rte_s42_hubish", "rte")]:
    ds = e1.load_split(t, e1.VAL_SPLIT[t]); k1, k2 = e1.KEYS[t]
    enc = tok([str(x) for x in ds[k1]], [str(x) for x in ds[k2]], truncation=True, max_length=128)
    labels = np.array(list(ds["label"]))
    base = BertForSequenceClassification.from_pretrained(e1.BASE, num_labels=e1.NUM_LABELS[t])
    m = PeftModel.from_pretrained(base, f"adapters/{a}").to(dev).eval()
    r = {}
    for mode in ("bert_segment_ids", "zero_segment_ids"):
        preds = []
        with torch.inference_mode():
            for i in range(0, len(labels), 128):
                ids = enc["input_ids"][i:i+128]; L = max(map(len, ids))
                ii = torch.zeros(len(ids), L, dtype=torch.long); tt = torch.zeros_like(ii); am = torch.zeros_like(ii)
                for j, x in enumerate(ids):
                    ii[j, :len(x)] = torch.tensor(x); am[j, :len(x)] = 1
                    if mode == "bert_segment_ids": tt[j, :len(x)] = torch.tensor(enc["token_type_ids"][i+j])
                preds.append(m(input_ids=ii.to(dev), token_type_ids=tt.to(dev), attention_mask=am.to(dev)).logits.argmax(-1).cpu().numpy())
        r[mode] = float((np.concatenate(preds) == labels).mean())
    res[a] = r; print(a, r, flush=True)
    del m, base; torch.cuda.empty_cache()
json.dump(res, open("artifacts/e1_predictive/diag/hubish_segment_ids.json", "w"), indent=1)
