#!/usr/bin/env python3
"""Forest plot (derived, exploratory pooled rows) of Spearman rho with task-block 95% CIs per population and pooled, per backbone.
Inputs: analysis/pooled_backbones.json only. Output: analysis/fig_forest_backbones.png"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
o = json.loads((ROOT / "analysis" / "pooled_backbones.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
labels = {"R0": "R0 seed 0", "P": "P mixed seed (primary*)", "S1": "S1 seed 1"}
for ax, c, title in zip(axes, ["O_A", "tv_cosine"], ["O_A (output-subspace overlap)", "task-vector cosine"]):
    y = 0; ticks = []; names = []
    for b, col in zip(["BERT", "RoBERTa", "Qwen"], ["tab:blue", "tab:green", "tab:red"]):
        r = o["per_backbone"][b]["results"][c]
        for p in ["R0", "P", "S1"]:
            lo, hi = r["per_population_boot_ci95"][p]; v = r["rho_per_population"][p]
            ax.plot([lo, hi], [y, y], color=col, lw=1.5); ax.plot(v, y, "o", color=col)
            ticks.append(y); names.append(f"{b}: {labels[p]}"); y -= 1
        lo, hi = r["rho_bar_boot_ci95"]; v = r["rho_bar"]
        ax.plot([lo, hi], [y, y], color=col, lw=3); ax.plot(v, y, "D", color=col, ms=8)
        ticks.append(y); names.append(f"{b}: pooled ρ̄ (exploratory)"); y -= 1.6
    a = o["across_backbone"]["results"][c]["contrasts"]["across_mean"]["joint"]
    ax.plot(a["ci95"], [y, y], color="k", lw=3); ax.plot(a["obs"], y, "D", color="k", ms=8); ticks.append(y); names.append("across-backbone mean (exploratory)")
    for xv, ls in ((0, "-"), (0.3, ":"), (0.4, "--")): ax.axvline(xv, color="grey", ls=ls, lw=0.8)
    ax.set_title(title); ax.set_xlabel("Spearman ρ with D (task-block 95% CI)"); ax.set_xlim(-0.8, 0.9)
axes[0].set_yticks(ticks); axes[0].set_yticklabels(names, fontsize=8)
fig.text(0.01, 0.01, "*P is the pre-registered primary population for E1c (BERT), E4a and E4b; for E1b the confirmatory population is R0. Dotted: FAIL bound 0.3; dashed: PASS threshold 0.4.", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(ROOT / "analysis" / "fig_forest_backbones.png", dpi=150)
print("ok")
