#!/usr/bin/env python3
"""
Linear probe for harmful content detection in Audio LLMs.

Trains a logistic regression classifier at each layer to distinguish
harmful (AdvBench) from benign (VoiceBench SD-QA) audio inputs.

If probe accuracy drops after finetuning, the model lost its ability
to *detect* harmful content — not just to refuse it.

Usage:
    # Step 1: Extract benign activations (harmful already extracted)
    python linear_probe.py extract-benign \
        --pretrained_path /path/to/pretrained \
        --finetuned_paths /path/to/ft1,/path/to/ft2 \
        --benign_audio_dir ../benign_data/voicebench/sd-qa/audio \
        --output_dir results/refusal_features_af3_sbert_audio_semantic

    # Step 2: Train probes and evaluate
    python linear_probe.py train \
        --results_dir results/refusal_features_af3_sbert_audio_semantic
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from glob import glob
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_processor(model_path: str, base_model_path: str = None, device: str = "cuda"):
    """Load AF3 model and processor (same as extract_refusal_features.py)."""
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    model_path_obj = Path(model_path)
    is_lora = (model_path_obj / "adapter_config.json").exists()

    if is_lora:
        if base_model_path is None:
            raise ValueError(f"LoRA adapter at {model_path} requires base_model_path")
        logger.info(f"Loading base model from {base_model_path}")
        logger.info(f"Loading LoRA adapter from {model_path}")
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading model from {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    model.eval()
    return model, processor


def extract_hidden_states_only(model, processor, audio_path: str,
                                prompt: str = "Please respond to the request.",
                                layers_to_extract: List[int] = None) -> Dict[int, np.ndarray]:
    """Extract hidden states without generating a response (faster)."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    all_hidden = outputs.hidden_states
    num_layers = len(all_hidden) - 1

    if layers_to_extract is None:
        layers_to_extract = list(range(num_layers))

    hidden_dict = {}
    for layer_idx in layers_to_extract:
        h = all_hidden[layer_idx + 1][0, -1, :].float().cpu().numpy()
        hidden_dict[layer_idx] = h

    return hidden_dict


def extract_benign_activations(args):
    """Extract hidden states for benign audio samples."""
    output_dir = Path(args.output_dir)

    # Get benign audio files
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    benign_files = []
    for ext in audio_extensions:
        benign_files.extend(glob(os.path.join(args.benign_audio_dir, ext)))
    benign_files = sorted(benign_files)

    # Sample to match harmful count (520) for balanced probe
    n_harmful = 520
    if len(benign_files) > n_harmful:
        random.seed(42)
        benign_files = random.sample(benign_files, n_harmful)
        benign_files = sorted(benign_files)

    logger.info(f"Using {len(benign_files)} benign audio files")

    # Layers
    num_layers = 28
    if args.all_layers:
        layers = list(range(num_layers))
    else:
        layers = list(range(0, num_layers, 2)) + [num_layers - 1]
        layers = sorted(set(layers))

    # Models to extract: pretrained + finetuned
    model_configs = [("pretrained", args.pretrained_path, None)]
    if args.finetuned_paths:
        ft_paths = [p.strip() for p in args.finetuned_paths.split(',')]
        ft_names = [n.strip() for n in args.model_names.split(',')] if args.model_names else []
        for i, ft_path in enumerate(ft_paths):
            name = ft_names[i] if i < len(ft_names) else Path(ft_path).parent.name
            model_configs.append((name, ft_path, args.pretrained_path))

    for model_name, model_path, base_path in model_configs:
        save_dir = output_dir / (f"finetuned_{model_name}" if model_name != "pretrained" else "pretrained")
        benign_hs_path = save_dir / "benign_hidden_states.npz"

        if benign_hs_path.exists() and not args.force:
            logger.info(f"Skipping {model_name} — benign hidden states already exist")
            continue

        logger.info(f"\n=== Extracting benign activations: {model_name} ===")
        model, processor = load_model_and_processor(model_path, base_path)

        hidden_states = defaultdict(list)
        audio_names = []

        for audio_path in tqdm(benign_files, desc=f"Benign [{model_name}]"):
            try:
                h = extract_hidden_states_only(model, processor, audio_path,
                                                prompt=args.prompt, layers_to_extract=layers)
                for layer_idx, vec in h.items():
                    hidden_states[layer_idx].append(vec)
                audio_names.append(os.path.basename(audio_path))
            except Exception as e:
                logger.error(f"Error on {os.path.basename(audio_path)}: {e}")

        # Save
        save_dict = {f"layer_{k}": np.stack(v) for k, v in hidden_states.items()}
        np.savez_compressed(benign_hs_path, **save_dict)

        with open(save_dir / "benign_audio_files.json", 'w') as f:
            json.dump(audio_names, f)

        logger.info(f"Saved {len(audio_names)} benign activations to {benign_hs_path}")

        del model
        torch.cuda.empty_cache()


def train_probes(args):
    """Train linear probes and evaluate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    results_dir = Path(args.results_dir)

    # Find all model directories
    model_dirs = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "pretrained" or d.name.startswith("finetuned_"):
            # Check both harmful and benign hidden states exist
            if (d / "hidden_states.npz").exists() and (d / "benign_hidden_states.npz").exists():
                model_dirs[d.name] = d
            elif (d / "hidden_states.npz").exists():
                logger.warning(f"{d.name}: missing benign_hidden_states.npz — run extract-benign first")

    if not model_dirs:
        logger.error("No model directories with both harmful and benign hidden states found")
        return

    logger.info(f"Found {len(model_dirs)} models: {list(model_dirs.keys())}")

    # Correct JSR from paper
    JSR = {
        "pretrained": 7.69,
        "finetuned_sbert_semantic_25pct": 20.19,
        "finetuned_sbert_semantic_50pct": 14.23,
        "finetuned_sbert_semantic_75pct": 32.12,
    }
    DISPLAY_NAMES = {
        "pretrained": "Pretrained",
        "finetuned_sbert_semantic_25pct": "Semantic 25%",
        "finetuned_sbert_semantic_50pct": "Semantic 50%",
        "finetuned_sbert_semantic_75pct": "Semantic 75%",
    }

    all_results = {}
    n_folds = 5

    for model_name, model_dir in model_dirs.items():
        logger.info(f"\n=== Training probes: {model_name} ===")

        # Load harmful hidden states
        harmful_data = np.load(model_dir / "hidden_states.npz")
        harmful_hs = {int(k.split('_')[1]): harmful_data[k] for k in harmful_data.files}

        # Load benign hidden states
        benign_data = np.load(model_dir / "benign_hidden_states.npz")
        benign_hs = {int(k.split('_')[1]): benign_data[k] for k in benign_data.files}

        layers = sorted(harmful_hs.keys())
        model_results = {"layers": {}, "mean_accuracy": 0, "mean_auc": 0}

        layer_accs = []
        layer_aucs = []

        for layer_idx in layers:
            h_harm = harmful_hs[layer_idx]
            h_benign = benign_hs[layer_idx]

            # Balance classes
            n = min(len(h_harm), len(h_benign))
            X = np.concatenate([h_harm[:n], h_benign[:n]], axis=0)
            y = np.concatenate([np.ones(n), np.zeros(n)])

            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_accs = []
            fold_aucs = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]

                fold_accs.append(accuracy_score(y_test, y_pred))
                fold_aucs.append(roc_auc_score(y_test, y_prob))

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)

            model_results["layers"][layer_idx] = {
                "accuracy": float(mean_acc),
                "accuracy_std": float(std_acc),
                "auc": float(mean_auc),
                "auc_std": float(std_auc),
            }
            layer_accs.append(mean_acc)
            layer_aucs.append(mean_auc)

        model_results["mean_accuracy"] = float(np.mean(layer_accs))
        model_results["mean_auc"] = float(np.mean(layer_aucs))
        all_results[model_name] = model_results

    # Print results table
    layers = sorted(all_results[list(all_results.keys())[0]]["layers"].keys())

    print(f"\n{'='*80}")
    print(f"LINEAR PROBE ACCURACY (harmful vs benign detection)")
    print(f"{'='*80}")

    layers_to_show = layers
    if len(layers_to_show) > 8:
        step = max(1, len(layers_to_show) // 7)
        layers_to_show = layers_to_show[::step]
        if layers[-1] not in layers_to_show:
            layers_to_show.append(layers[-1])

    header = f"{'Model':<40} | {'Mean':>5}"
    for l in layers_to_show:
        header += f" | L{l:>2}"
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        display = DISPLAY_NAMES.get(model_name, model_name)
        jsr = JSR.get(model_name, "?")
        row = f"{display} (JSR={jsr}%)"
        row = f"{row:<40} | {results['mean_accuracy']:>4.1%}"
        for l in layers_to_show:
            acc = results["layers"][l]["accuracy"]
            row += f" | {acc:>.2f}"
        print(row)

    # Save results
    output_path = results_dir / "probe_results.json"
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {
            "mean_accuracy": results["mean_accuracy"],
            "mean_auc": results["mean_auc"],
            "layers": {str(k): v for k, v in results["layers"].items()},
        }
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Plot
    plot_probe_results(all_results, results_dir, JSR, DISPLAY_NAMES)


def plot_probe_results(all_results, results_dir, JSR, DISPLAY_NAMES):
    """Generate probe accuracy figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    colors = {"pretrained": "#1f77b4", "finetuned_sbert_semantic_25pct": "#ff7f0e",
              "finetuned_sbert_semantic_50pct": "#2ca02c", "finetuned_sbert_semantic_75pct": "#d62728"}

    # ── Figure: Probe accuracy across layers ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for model_name, results in all_results.items():
        layers = sorted(results["layers"].keys())
        accs = [results["layers"][l]["accuracy"] for l in layers]
        aucs = [results["layers"][l]["auc"] for l in layers]
        acc_stds = [results["layers"][l]["accuracy_std"] for l in layers]

        display = DISPLAY_NAMES.get(model_name, model_name)
        jsr = JSR.get(model_name, "?")
        label = f"{display} (JSR={jsr}%)"
        color = colors.get(model_name, "gray")
        linestyle = '--' if model_name == "pretrained" else '-'
        linewidth = 2.5 if model_name == "pretrained" else 1.8

        ax1.plot(layers, accs, marker='o', markersize=4, label=label,
                 color=color, linestyle=linestyle, linewidth=linewidth)
        ax1.fill_between(layers,
                         [a - s for a, s in zip(accs, acc_stds)],
                         [a + s for a, s in zip(accs, acc_stds)],
                         alpha=0.15, color=color)

        ax2.plot(layers, aucs, marker='o', markersize=4, label=label,
                 color=color, linestyle=linestyle, linewidth=linewidth)

    ax1.set_xlabel("Layer", fontsize=13)
    ax1.set_ylabel("Probe Accuracy", fontsize=13)
    ax1.set_title("Harmful vs Benign Detection Accuracy\n(5-fold CV, Logistic Regression)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
    ax1.set_ylim(0.45, 1.02)

    ax2.set_xlabel("Layer", fontsize=13)
    ax2.set_ylabel("AUC-ROC", fontsize=13)
    ax2.set_title("Harmful vs Benign Detection AUC\n(5-fold CV, Logistic Regression)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylim(0.45, 1.02)

    plt.tight_layout()
    fig.savefig(fig_dir / "probe_accuracy.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / "probe_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {fig_dir / 'probe_accuracy.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Linear probe for harmful content detection")
    subparsers = parser.add_subparsers(dest="command")

    # Extract benign subcommand
    ext = subparsers.add_parser("extract-benign", help="Extract benign audio hidden states")
    ext.add_argument("--pretrained_path", type=str, required=True)
    ext.add_argument("--finetuned_paths", type=str, default=None)
    ext.add_argument("--model_names", type=str, default=None)
    ext.add_argument("--benign_audio_dir", type=str, required=True)
    ext.add_argument("--output_dir", type=str, required=True)
    ext.add_argument("--prompt", type=str, default="Please respond to the request.")
    ext.add_argument("--all_layers", action="store_true")
    ext.add_argument("--force", action="store_true", help="Overwrite existing benign hidden states")

    # Train subcommand
    train = subparsers.add_parser("train", help="Train and evaluate linear probes")
    train.add_argument("--results_dir", type=str, required=True)

    args = parser.parse_args()

    if args.command == "extract-benign":
        extract_benign_activations(args)
    elif args.command == "train":
        train_probes(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
