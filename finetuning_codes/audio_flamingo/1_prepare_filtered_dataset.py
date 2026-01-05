#!/usr/bin/env python3
# Copyright (c) 2025
# Script to prepare filtered VoiceBench dataset for finetuning Audio Flamingo 3

"""
Prepare a filtered VoiceBench dataset based on embedding distance threshold.

Usage:
    python 1_prepare_filtered_dataset.py \
        --analysis_file data/voicebench/sd-qa/sd_qa_closest_to_advbench_analysis.npz \
        --voicebench_json data/voicebench/sd-qa/sd_qa_full.json \
        --threshold 0.025 \
        --output_dir data/filtered_voicebench

This will:
1. Load the pre-computed distance analysis
2. Filter samples based on the threshold
3. Save a training-ready JSON file with paths to audio files
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np


def prepare_filtered_dataset(
    analysis_file: str,
    voicebench_json: str,
    threshold: float,
    output_dir: str,
    top_k: int = None,
    percentage: float = None,
    num_samples: int = None,
    filter_closest: bool = True,
):
    """
    Prepare filtered dataset for finetuning.

    Args:
        analysis_file: Path to the analysis .npz file from filter_closest_to_advbench.py
        voicebench_json: Path to original VoiceBench JSON
        threshold: Distance threshold (keep samples with distance <= threshold)
        output_dir: Directory to save filtered dataset
        top_k: If set, keep top-k samples with smallest distance (overrides threshold)
        percentage: If set, keep top percentage of samples (e.g., 50 for 50%)
        num_samples: If set, keep exact number of samples (overrides percentage and top_k)
    """
    # Load analysis data
    print(f"Loading analysis from {analysis_file}")
    analysis = np.load(analysis_file, allow_pickle=True)
    min_distances = analysis['min_distances']
    voicebench_files = analysis['voicebench_files']
    advbench_files = analysis['advbench_files']
    all_distances = analysis['all_distances']

    # Load original VoiceBench data
    print(f"Loading VoiceBench data from {voicebench_json}")
    with open(voicebench_json) as f:
        voicebench_data = json.load(f)

    print(f"Total samples: {len(voicebench_data)}")

    # Print distance statistics
    print(f"\nDistance statistics:")
    print(f"  Min:    {min_distances.min():.6f}")
    print(f"  Max:    {min_distances.max():.6f}")
    print(f"  Mean:   {min_distances.mean():.6f}")
    print(f"  Median: {np.median(min_distances):.6f}")

    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(min_distances, p)
        count = (min_distances <= val).sum()
        print(f"  {p}th: {val:.6f} ({count} samples)")

    # Apply filtering
    # Priority: num_samples > percentage > top_k > threshold
    original_percentage = percentage  # Keep track for filename
    original_num_samples = num_samples  # Keep track for filename

    if num_samples is not None:
        top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        top_k = int(len(voicebench_data) * percentage / 100)
        top_k = max(1, top_k)  # Ensure at least 1 sample
        print(f"\n=== Converting {percentage}% to top_k={top_k} samples ===")

    if top_k is not None:
        if filter_closest:
            # Keep top-k closest samples (smallest distances)
            sorted_indices = np.argsort(min_distances)[:top_k]
            print(f"\n=== Keeping top {top_k} CLOSEST samples ===")
        else:
            # Keep top-k farthest samples (largest distances)
            sorted_indices = np.argsort(min_distances)[-top_k:][::-1]
            print(f"\n=== Keeping top {top_k} FARTHEST samples (most benign) ===")
        filtered_indices = sorted_indices
        effective_threshold = min_distances[sorted_indices[-1]]
        print(f"  Effective threshold: {effective_threshold:.6f}")
    else:
        if filter_closest:
            # Keep samples with distance <= threshold (closest to harmful)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} (closest) ===")
        else:
            # Keep samples with distance >= threshold (farthest from harmful)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (most benign) ===")
        effective_threshold = threshold

    print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = []
    for idx in filtered_indices:
        sample = voicebench_data[idx].copy()

        # Add distance metadata
        sample['min_distance_to_advbench'] = float(min_distances[idx])
        closest_advbench_idx = np.argmin(all_distances[idx])
        sample['closest_advbench_file'] = str(advbench_files[closest_advbench_idx])

        # Ensure audio path is absolute
        audio_path = sample['audio']
        if not os.path.isabs(audio_path):
            # Try to resolve relative path
            json_dir = Path(voicebench_json).parent
            full_path = json_dir.parent.parent.parent / audio_path
            if full_path.exists():
                audio_path = str(full_path.resolve())
            else:
                # Try from current directory
                cwd_path = Path(audio_path)
                if cwd_path.exists():
                    audio_path = str(cwd_path.resolve())
        sample['audio'] = audio_path

        filtered_data.append(sample)

    # Sort by distance
    filtered_data.sort(key=lambda x: x['min_distance_to_advbench'])

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered dataset JSON
    if original_num_samples is not None:
        threshold_str = f"n{original_num_samples}"
    elif original_percentage is not None:
        threshold_str = f"percentage_{int(original_percentage) if original_percentage == int(original_percentage) else original_percentage}"
    elif top_k is not None:
        threshold_str = f"top{top_k}"
    else:
        threshold_str = f"{threshold}"
    filter_type = "closest" if filter_closest else "benign"
    output_json = output_dir / f"voicebench_filtered_{filter_type}_{threshold_str}.json"

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Also save in HuggingFace conversation format for training
    hf_format_data = []
    for sample in filtered_data:
        conversations = sample.get('conversations', [])
        if len(conversations) >= 2:
            # Extract question and answer
            user_msg = conversations[0].get('value', '')
            assistant_msg = conversations[1].get('value', '')

            # Remove <audio> tag from user message to get the text question
            question_text = user_msg.replace('<audio>', '').replace('\n', ' ').strip()

            hf_sample = {
                "id": sample.get('id', ''),
                "audio_path": sample['audio'],
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": sample['audio']},
                            {"type": "text", "text": question_text},
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_msg}
                        ]
                    }
                ],
                "min_distance_to_advbench": sample['min_distance_to_advbench'],
            }
            hf_format_data.append(hf_sample)

    hf_output_json = output_dir / f"voicebench_filtered_{filter_type}_{threshold_str}_hf.json"
    with open(hf_output_json, 'w') as f:
        json.dump(hf_format_data, f, indent=2)

    print(f"  HuggingFace format: {hf_output_json}")

    # Print some examples
    print(f"\n=== Sample entries ===")
    for i in range(min(3, len(filtered_data))):
        sample = filtered_data[i]
        print(f"\n{i+1}. Distance: {sample['min_distance_to_advbench']:.6f}")
        print(f"   Audio: {sample['audio']}")
        if 'conversations' in sample:
            q = sample['conversations'][0].get('value', '')[:80]
            a = sample['conversations'][1].get('value', '')[:80]
            print(f"   Q: {q}...")
            print(f"   A: {a}...")

    return str(output_json), str(hf_output_json)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare filtered VoiceBench dataset for finetuning"
    )
    parser.add_argument(
        "--analysis_file",
        type=str,
        default="data/voicebench/sd-qa/sd_qa_closest_to_advbench_analysis.npz",
        help="Path to analysis .npz file from filter_closest_to_advbench.py"
    )
    parser.add_argument(
        "--voicebench_json",
        type=str,
        default="data/voicebench/sd-qa/sd_qa_full.json",
        help="Path to original VoiceBench JSON"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.025,
        help="Distance threshold (keep samples <= threshold)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep top-k closest samples (overrides threshold)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Keep top percentage of samples (e.g., 50 for 50%%)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Keep exact number of samples (overrides percentage and top_k)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/filtered_voicebench",
        help="Output directory for filtered dataset"
    )
    parser.add_argument(
        "--filter_closest",
        action="store_true",
        default=True,
        help="Filter for closest samples (default: True). Use --filter_benign for farthest."
    )
    parser.add_argument(
        "--filter_benign",
        action="store_true",
        help="Filter for farthest/most benign samples instead of closest"
    )

    args = parser.parse_args()

    # Determine filter mode
    filter_closest = not args.filter_benign

    prepare_filtered_dataset(
        analysis_file=args.analysis_file,
        voicebench_json=args.voicebench_json,
        threshold=args.threshold,
        top_k=args.top_k,
        percentage=args.percentage,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        filter_closest=filter_closest,
    )


if __name__ == "__main__":
    main()
