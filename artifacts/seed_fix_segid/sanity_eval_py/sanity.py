# Sanity check of the fixed lora_merge_cert/eval.py (fix/segment-ids): per-adapter auto-resolved mode.
import json, sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from lora_merge_cert.eval import evaluate_glue, resolve_segment_ids
tok = AutoTokenizer.from_pretrained("bert-base-uncased"); dev = torch.device("cuda")
out = []
for path, nl, task, modes in [("adapters/mnli_s7_hubish", 3, "mnli", ["auto", "none"]),
                              ("prateeky2806/bert-base-uncased-mnli-lora-epochs-2-lr-0.001", 3, "mnli", ["auto", "bert"]),
                              ("adapters/rte_s42_hubish", 2, "rte", ["auto"])]:
    m = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nl), path).to(dev)
    for mode in modes:
        seg = resolve_segment_ids(path, mode)
        for n in (512, None):
            r = evaluate_glue(m, tok, task, device=dev, num_samples=n, segment_ids=seg)
            r.update(adapter=path, requested=mode); out.append(r); print(json.dumps(r), flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open("artifacts/seed_fix_segid/sanity_eval_py/sanity.json", "w"), indent=1)
