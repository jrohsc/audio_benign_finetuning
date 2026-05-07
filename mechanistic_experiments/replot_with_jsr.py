#!/usr/bin/env python3
"""Regenerate refusal feature figures with correct JSR numbers from paper."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = Path("mechanistic_experiments/results/refusal_features_af3_sbert_audio_semantic")
fig_dir = results_dir / "figures"

# Correct JSR from paper (Table: AF3, Semantic, AdvBench)
JSR = {
    "pretrained": 7.69,
    "sbert_semantic_25pct": 20.19,
    "sbert_semantic_50pct": 14.23,
    "sbert_semantic_75pct": 32.12,
}

DISPLAY_NAMES = {
    "pretrained": "Pretrained",
    "sbert_semantic_25pct": "Semantic 25%",
    "sbert_semantic_50pct": "Semantic 50%",
    "sbert_semantic_75pct": "Semantic 75%",
}

# Load analysis
with open(results_dir / "refusal_analysis.json") as f:
    analysis = json.load(f)

# Get layers
layers = sorted(int(k) for k in analysis["pretrained"]["projections"].keys())

# ── Figure 1: Refusal direction projection across layers ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"pretrained": "#1f77b4", "sbert_semantic_25pct": "#ff7f0e",
          "sbert_semantic_50pct": "#2ca02c", "sbert_semantic_75pct": "#d62728"}

for model_name in ["pretrained", "sbert_semantic_25pct", "sbert_semantic_50pct", "sbert_semantic_75pct"]:
    proj = analysis[model_name]["projections"]
    means = [proj[str(l)]["mean"] for l in layers]
    jsr = JSR[model_name]
    label = f"{DISPLAY_NAMES[model_name]} (JSR={jsr:.1f}%)"
    linestyle = '--' if model_name == "pretrained" else '-'
    linewidth = 2.5 if model_name == "pretrained" else 1.8
    ax.plot(layers, means, marker='o', markersize=5, label=label,
            color=colors[model_name], linestyle=linestyle, linewidth=linewidth)

ax.set_xlabel("Layer", fontsize=14)
ax.set_ylabel("Projection onto Refusal Direction", fontsize=14)
ax.set_title("Refusal Feature Strength Across Layers (AF3, SBERT Semantic, AdvBench)", fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
fig.savefig(fig_dir / "refusal_projection_by_layer.pdf", dpi=300, bbox_inches='tight')
fig.savefig(fig_dir / "refusal_projection_by_layer.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig_dir / 'refusal_projection_by_layer.pdf'}")

# ── Figure 2: Shift bar chart at key layers ──
ft_models = ["sbert_semantic_25pct", "sbert_semantic_50pct", "sbert_semantic_75pct"]

key_layers = [0, 6, 14, 22, 27]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(key_layers))
width = 0.8 / len(ft_models)

for i, model_name in enumerate(ft_models):
    proj = analysis[model_name]["projections"]
    shifts = [proj[str(l)].get("shift_from_pretrained", 0) for l in key_layers]
    jsr = JSR[model_name]
    offset = (i - len(ft_models)/2 + 0.5) * width
    ax.bar(x + offset, shifts, width,
           label=f"{DISPLAY_NAMES[model_name]} (JSR={jsr:.1f}%)",
           color=colors[model_name], alpha=0.85)

ax.set_xlabel("Layer", fontsize=14)
ax.set_ylabel("Shift from Pretrained", fontsize=14)
ax.set_title("Refusal Direction Suppression by Layer\n(negative = safety eroded)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in key_layers])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
fig.savefig(fig_dir / "refusal_shift_bar.pdf", dpi=300, bbox_inches='tight')
fig.savefig(fig_dir / "refusal_shift_bar.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig_dir / 'refusal_shift_bar.pdf'}")

# ── Figure 3: Heatmap ──
shift_matrix = []
ylabels = []
for model_name in ft_models:
    proj = analysis[model_name]["projections"]
    row = [proj[str(l)].get("shift_from_pretrained", 0) for l in layers]
    shift_matrix.append(row)
    ylabels.append(f"{DISPLAY_NAMES[model_name]} (JSR={JSR[model_name]:.1f}%)")

shift_matrix = np.array(shift_matrix)

fig, ax = plt.subplots(figsize=(12, 3.5))
vmax = max(abs(shift_matrix.min()), abs(shift_matrix.max()))
im = ax.imshow(shift_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
               vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
ax.set_yticks(range(len(ft_models)))
ax.set_yticklabels(ylabels, fontsize=11)
ax.set_xlabel("Layer", fontsize=14)
ax.set_title("Refusal Direction Shift Heatmap\n(blue = refusal suppressed, red = refusal enhanced)", fontsize=13)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Shift from pretrained", fontsize=11)
plt.tight_layout()
fig.savefig(fig_dir / "refusal_shift_heatmap.pdf", dpi=300, bbox_inches='tight')
fig.savefig(fig_dir / "refusal_shift_heatmap.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig_dir / 'refusal_shift_heatmap.pdf'}")
