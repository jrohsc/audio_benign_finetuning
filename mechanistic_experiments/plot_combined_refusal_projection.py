#!/usr/bin/env python3
"""Combined 2x2 figure of refusal projection by layer for all 4 model/modality combos."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Panel configs: (results_dir, title, model_order, jsr_map, display_names) ──
panels = [
    {
        "dir": Path("mechanistic_experiments/results/refusal_features_qwen_sbert_text_semantic"),
        "title": "Qwen2.5-Omni, Text Semantic",
        "order": ["pretrained", "text_semantic_25pct", "text_semantic_50pct", "text_semantic_75pct"],
        "jsr": {"pretrained": 0.19, "text_semantic_25pct": 16.35, "text_semantic_50pct": 6.35, "text_semantic_75pct": 6.35},
        "names": {"pretrained": "Pretrained", "text_semantic_25pct": "Text Semantic 25%",
                  "text_semantic_50pct": "Text Semantic 50%", "text_semantic_75pct": "Text Semantic 75%"},
    },
    {
        "dir": Path("mechanistic_experiments/results/refusal_features_qwen_sbert_audio_semantic"),
        "title": "Qwen2.5-Omni, Audio Semantic",
        "order": ["pretrained", "audio_semantic_25pct", "audio_semantic_50pct", "audio_semantic_75pct"],
        "jsr": {"pretrained": 0.19, "audio_semantic_25pct": 9.42, "audio_semantic_50pct": 12.50, "audio_semantic_75pct": 2.69},
        "names": {"pretrained": "Pretrained", "audio_semantic_25pct": "Audio Semantic 25%",
                  "audio_semantic_50pct": "Audio Semantic 50%", "audio_semantic_75pct": "Audio Semantic 75%"},
    },
    {
        "dir": Path("mechanistic_experiments/results/refusal_features_af3_text_semantic"),
        "title": "Audio Flamingo, Text Semantic",
        "order": ["pretrained", "text_semantic_25pct", "text_semantic_50pct", "text_semantic_75pct"],
        "jsr": {"pretrained": 7.69, "text_semantic_25pct": 2.12, "text_semantic_50pct": 3.85, "text_semantic_75pct": 6.73},
        "names": {"pretrained": "Pretrained", "text_semantic_25pct": "Text Semantic 25%",
                  "text_semantic_50pct": "Text Semantic 50%", "text_semantic_75pct": "Text Semantic 75%"},
    },
    {
        "dir": Path("mechanistic_experiments/results/refusal_features_af3_sbert_audio_semantic"),
        "title": "Audio Flamingo, Audio Semantic",
        "order": ["pretrained", "sbert_semantic_25pct", "sbert_semantic_50pct", "sbert_semantic_75pct"],
        "jsr": {"pretrained": 7.69, "sbert_semantic_25pct": 20.19, "sbert_semantic_50pct": 14.23, "sbert_semantic_75pct": 32.12},
        "names": {"pretrained": "Pretrained", "sbert_semantic_25pct": "Semantic 25%",
                  "sbert_semantic_50pct": "Semantic 50%", "sbert_semantic_75pct": "Semantic 75%"},
    },
]

# Check AF3 audio semantic model keys
af3_audio_path = panels[3]["dir"] / "refusal_analysis.json"
with open(af3_audio_path) as f:
    af3_audio_keys = list(json.load(f).keys())
print(f"AF3 audio semantic keys: {af3_audio_keys}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

tab_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for idx, (panel, ax) in enumerate(zip(panels, axes)):
    with open(panel["dir"] / "refusal_analysis.json") as f:
        analysis = json.load(f)

    layers = sorted(int(k) for k in analysis["pretrained"]["projections"].keys())
    available = [m for m in panel["order"] if m in analysis]

    colors = {"pretrained": "#1f77b4"}
    for i, m in enumerate(panel["order"]):
        if m != "pretrained":
            colors[m] = tab_colors[i % len(tab_colors)]

    for model_name in available:
        proj = analysis[model_name]["projections"]
        means = [proj[str(l)]["mean"] for l in layers]
        jsr = panel["jsr"].get(model_name, 0)
        label = f"{panel['names'][model_name]} (JSR={jsr:.1f}%)"
        linestyle = '--' if model_name == "pretrained" else '-'
        linewidth = 2.5 if model_name == "pretrained" else 1.8
        ax.plot(layers, means, marker='o', markersize=4, label=label,
                color=colors[model_name], linestyle=linestyle, linewidth=linewidth)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Projection onto Refusal Direction", fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Place subtitle inside the plot area (two lines: model name + modality)
    model_name_str, modality_str = panel["title"].split(", ")
    subtitle = f"{model_name_str}\n({modality_str})"
    ax.text(0.72, 0.95, subtitle, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='center', linespacing=1.3,
)

plt.tight_layout()

out_dir = Path("mechanistic_experiments/results/combined_figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "refusal_projection_combined.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out_dir / "refusal_projection_combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_dir / 'refusal_projection_combined.png'}")
