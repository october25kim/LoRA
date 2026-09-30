"""E6 (post hoc): per-layer task-vector norms and inner products from LoRA weights. CPU only.
For each backbone, each adapter t and LoRA module l: n2[t,l] = ||s B A||_F^2; for each adapter pair (t,u): ip[t,u,l] = <s B_t A_t, s B_u A_u>_F.
Uses rank-8 factor identities (no dense d_out x d_in products beyond 8x8). Classification heads excluded."""
import os, sys, json, itertools, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from safetensors.torch import load_file
torch.set_num_threads(4)
ROOT = os.path.expanduser("~/Desktop/Workspace/LoRA/artifacts")
BACK = {
 "bert":    {"s0": "e1b_confirmatory/adapters", "s1": "e1c_seed_diverse/adapters_s1"},
 "roberta": {"s0": "e4a_roberta/adapters_s0",   "s1": "e4a_roberta/adapters_s1"},
 "qwen":    {"s0": "e4b_decoder/adapters_s0",   "s1": "e4b_decoder/adapters_s1"},
}
TASKS = ['cola','sst2','mrpc','stsb','mnli','qnli','rte','wic','snli','scitail','ag_news','imdb','trec','yelp_polarity']
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for bb, dirs in BACK.items():
    ads = {}
    for seed, d in dirs.items():
        for t in TASKS:
            p = os.path.join(ROOT, d, t)
            f = os.path.join(p, "adapter_model.safetensors")
            if not os.path.exists(f): continue
            cfg = json.load(open(os.path.join(p, "adapter_config.json")))
            s = cfg["lora_alpha"] / cfg["r"]
            sd = load_file(f)
            mods = {}
            for k, v in sd.items():
                if ".lora_A." in k:
                    m = k.split(".lora_A.")[0].replace("base_model.model.", "")
                    B = sd[k.replace(".lora_A.", ".lora_B.")].double()
                    mods[m] = (v.double(), B * s)
            ads[f"{t}@{seed}"] = mods
    names = sorted(ads)
    layers = sorted(next(iter(ads.values())).keys())
    print(bb, len(names), "adapters", len(layers), "modules", flush=True)
    # precompute gram pieces
    with open(os.path.join(out, f"layer_norms_{bb}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["adapter", "layer", "n2"])
        for a in names:
            for l in layers:
                A, B = ads[a][l]
                n2 = torch.trace((B.T @ B) @ (A @ A.T)).item()
                w.writerow([a, l, repr(n2)])
    with open(os.path.join(out, f"layer_ip_{bb}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["a1", "a2", "layer", "ip"])
        for a1, a2 in itertools.combinations(names, 2):
            if a1.split("@")[0] == a2.split("@")[0] and False: pass
            for l in layers:
                A1, B1 = ads[a1][l]; A2, B2 = ads[a2][l]
                ip = torch.trace((B1.T @ B2) @ (A2 @ A1.T)).item()
                w.writerow([a1, a2, l, repr(ip)])
    print(bb, "done", flush=True)
