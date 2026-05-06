#!/usr/bin/env python3
"""
Random k% sampling of any dataset (baseline for comparison with distance-based filtering).

This script randomly selects k% of samples WITHOUT considering
any distance/similarity to harmful audio. This serves as a baseline to compare
against the acoustic/semantic filtering approaches.

Supports both JSON (Audio Flamingo format) and JSONL (Kimi Audio format).

Usage:
    python 0_filter_random.py \
        --input_json data/voicebench/sd-qa/sd_qa_full.json \
        --percentage 25 \
        --output_json data_random/voicebench/sd-qa/sd_qa_random_25.json \
        --seed 42
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def filter_random_samples(
    input_json: str,
    output_json: str,
    percentage: float = None,
    num_samples: int = None,
    seed: int = 42,
):
    """
    Randomly filter samples from any JSON/JSONL dataset.

    Args:
        input_json: Path to input JSON or JSONL file
        output_json: Path to save filtered samples
        percentage: Percentage of samples to keep (e.g., 25 for 25%)
        num_samples: Exact number of samples to keep (overrides percentage)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load data (support both JSON and JSONL)
    print(f"Loading data from {input_json}")
    input_path = Path(input_json)

    if input_path.suffix == '.jsonl':
        # JSONL format (one JSON object per line)
        data = []
        with open(input_json, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        is_jsonl = True
    else:
        # JSON format (single JSON array)
        with open(input_json, 'r') as f:
            data = json.load(f)
        is_jsonl = False

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
    selected_indices = np.sort(selected_indices)  # Sort to maintain order

    print(f"Randomly selected {keep_count} samples (seed={seed})")

    # Create filtered dataset
    filtered_data = [data[i] for i in selected_indices]

    # Add metadata about random selection
    for i, idx in enumerate(selected_indices):
        if isinstance(filtered_data[i], dict):
            filtered_data[i]["_random_selection"] = {
                "original_index": int(idx),
                "seed": seed,
                "filter_type": "random"
            }

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_jsonl or output_path.suffix == '.jsonl':
        # Save as JSONL
        with open(output_json, 'w') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        # Save as JSON
        with open(output_json, 'w') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)} / {total_samples} ({100*len(filtered_data)/total_samples:.1f}%)")

    return str(output_json)


def main():
    parser = argparse.ArgumentParser(description="Randomly filter samples from any dataset (baseline)")
    parser.add_argument("--input_json", type=str, required=True,
                       help="Path to input JSON/JSONL file")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Path to save filtered samples")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Percentage of samples to keep (e.g., 25 for 25%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Exact number of samples to keep (overrides percentage)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    # Backwards compatibility aliases
    parser.add_argument("--voicebench_json", type=str, default=None,
                       help="Alias for --input_json (backwards compatibility)")

    args = parser.parse_args()

    # Handle backwards compatibility
    input_json = args.input_json or args.voicebench_json
    if input_json is None:
        parser.error("--input_json is required")

    if args.percentage is None and args.num_samples is None:
        parser.error("Either --percentage or --num_samples must be specified")

    filter_random_samples(
        input_json=input_json,
        output_json=args.output_json,
        percentage=args.percentage,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
