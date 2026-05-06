#!/usr/bin/env python3
"""
Random k% sampling of benign dataset (baseline for comparison with distance-based filtering).

This script randomly selects k% of samples WITHOUT considering any distance/similarity
to harmful audio. This serves as a baseline to compare against the acoustic/semantic
filtering approaches.

Supports two modes:
  1. Load VoiceBench from HuggingFace (--benign_dataset voicebench)
  2. Load pre-existing JSONL file (--input_jsonl path/to/file.jsonl)

Usage:
    # VoiceBench from HuggingFace
    python 0_filter_random.py \
        --benign_dataset voicebench \
        --percentage 25 \
        --output data_random/voicebench_filtered_random_percentage_25.jsonl \
        --seed 42

    # Pre-existing JSONL
    python 0_filter_random.py \
        --input_jsonl data/full_dataset.jsonl \
        --percentage 25 \
        --output data_random/dataset_filtered_random_percentage_25.jsonl \
        --seed 42
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

# VoiceBench audio directory (shared across all models)
VOICEBENCH_AUDIO_DIR = "/work/anon/audio_benign_finetuning/shared_audio/voicebench"


def load_voicebench_data():
    """Load VoiceBench dataset from HuggingFace (same as in the acoustic/semantic filters)."""
    from datasets import load_dataset

    print("Loading VoiceBench dataset...")

    voicebench_data = []

    subsets = [
        ("sd-qa", ["aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl",
                   "phl", "usa", "zaf"]),
    ]

    for subset_name, splits in subsets:
        try:
            ds = load_dataset("hlt-lab/voicebench", subset_name)
            for split in splits:
                if split in ds:
                    split_ds = ds[split].remove_columns(["audio"])
                    for idx, item in enumerate(split_ds):
                        voicebench_data.append({
                            "subset": subset_name,
                            "split": split,
                            "question": item.get("prompt", ""),
                            "answer": item.get("reference", ""),
                            "audio_path": os.path.join(VOICEBENCH_AUDIO_DIR, f"{split}_{idx}.wav")
                        })
        except Exception as e:
            print(f"Error loading {subset_name}: {e}")

    # Note: Unlike acoustic/semantic filters, random filtering does NOT need to
    # access audio files (no embedding extraction), so we skip the existence check.
    # The audio paths are still recorded for downstream semantic code extraction.
    print(f"Loaded {len(voicebench_data)} VoiceBench samples")

    return voicebench_data


def voicebench_to_kimi_format(data, seed):
    """Convert VoiceBench data to Kimi-Audio JSONL conversation format."""
    conversations = []
    for item in data:
        conv = {
            "task_type": "understanding",
            "conversation": [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": "Please answer the following question based on the audio."
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": item["audio_path"]
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": item["answer"]
                }
            ],
            "_metadata": {
                "filter_type": "random",
                "seed": seed,
                "subset": item.get("subset", ""),
                "split": item.get("split", "")
            }
        }
        conversations.append(conv)
    return conversations


def filter_random_samples(
    input_jsonl=None,
    benign_dataset=None,
    output_path=None,
    percentage=None,
    num_samples=None,
    seed=42,
):
    """
    Randomly filter samples from a benign dataset.

    Args:
        input_jsonl: Path to pre-existing JSONL file (if provided, loads from file)
        benign_dataset: Name of benign dataset to load from source (e.g., "voicebench")
        output_path: Path to save filtered JSONL
        percentage: Percentage of samples to keep (e.g., 25 for 25%)
        num_samples: Exact number of samples to keep (overrides percentage)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load data
    if input_jsonl is not None:
        # Load from pre-existing JSONL file
        print(f"Loading data from {input_jsonl}")
        data = []
        with open(input_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        is_raw = False
    elif benign_dataset == "voicebench":
        # Load VoiceBench from HuggingFace
        raw_data = load_voicebench_data()
        is_raw = True
    else:
        raise ValueError(f"Unsupported benign_dataset '{benign_dataset}' without --input_jsonl. "
                         f"Currently supports: voicebench (from HuggingFace), "
                         f"or provide --input_jsonl for any pre-existing JSONL.")

    if is_raw:
        total_samples = len(raw_data)
    else:
        total_samples = len(data)

    print(f"Total samples: {total_samples}")

    # Determine number of samples to keep
    if num_samples is not None:
        keep_count = min(num_samples, total_samples)
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        keep_count = int(total_samples * percentage / 100)
        keep_count = max(1, keep_count)  # Ensure at least 1 sample
        print(f"\n=== Converting {percentage}% to {keep_count} samples ===")
    else:
        raise ValueError("Either percentage or num_samples must be specified")

    # Randomly select indices
    all_indices = np.arange(total_samples)
    selected_indices = np.random.choice(all_indices, size=keep_count, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort to maintain original order

    print(f"Randomly selected {keep_count} samples (seed={seed})")

    # Create filtered dataset
    if is_raw:
        selected_raw = [raw_data[i] for i in selected_indices]
        filtered_data = voicebench_to_kimi_format(selected_raw, seed)
    else:
        filtered_data = [data[i] for i in selected_indices]
        # Add random selection metadata
        for i, idx in enumerate(selected_indices):
            if isinstance(filtered_data[i], dict):
                if "_metadata" not in filtered_data[i]:
                    filtered_data[i]["_metadata"] = {}
                filtered_data[i]["_metadata"]["filter_type"] = "random"
                filtered_data[i]["_metadata"]["seed"] = seed
                filtered_data[i]["_metadata"]["original_index"] = int(idx)

    # Save filtered data as JSONL
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_path}")
    print(f"  Samples: {len(filtered_data)} / {total_samples} ({100*len(filtered_data)/total_samples:.1f}%)")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Randomly filter samples from a benign dataset (baseline)"
    )
    parser.add_argument("--input_jsonl", type=str, default=None,
                       help="Path to pre-existing JSONL file to randomly subsample")
    parser.add_argument("--benign_dataset", type=str, default=None,
                       help="Benign dataset to load from source (e.g., 'voicebench')")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save filtered JSONL")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Percentage of samples to keep (e.g., 25 for 25%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Exact number of samples to keep (overrides percentage)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.input_jsonl is None and args.benign_dataset is None:
        parser.error("Either --input_jsonl or --benign_dataset must be specified")

    if args.percentage is None and args.num_samples is None:
        parser.error("Either --percentage or --num_samples must be specified")

    filter_random_samples(
        input_jsonl=args.input_jsonl,
        benign_dataset=args.benign_dataset,
        output_path=args.output,
        percentage=args.percentage,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
