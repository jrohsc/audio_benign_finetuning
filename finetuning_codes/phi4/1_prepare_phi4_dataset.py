#!/usr/bin/env python3
"""
Prepare filtered VoiceBench dataset for Phi-4 finetuning.

This script takes the filtered VoiceBench samples and prepares them in a format
suitable for Phi-4 multimodal finetuning.

Usage:
    python 1_prepare_phi4_dataset.py \
        --analysis_file data/voicebench/sd-qa/sd_qa_filtered_phi4_analysis.npz \
        --voicebench_json data/voicebench/sd-qa/sd_qa_full.json \
        --threshold 0.5 \
        --output_dir data/phi4_filtered_voicebench
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np


def prepare_filtered_dataset(
    analysis_file: str,
    voicebench_json: str,
    output_dir: str,
    num_samples: int,
    filter_benign: bool = True,
):
    """
    Prepare filtered dataset for Phi-4 finetuning.

    Args:
        analysis_file: Path to the analysis .npz file from 0_filter_phi4.py
        voicebench_json: Path to original VoiceBench JSON
        output_dir: Directory to save filtered dataset
        num_samples: Number of samples to keep
        filter_benign: If True, keep samples far from AdvBench; if False, keep close ones
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
        count_below = (min_distances <= val).sum()
        count_above = (min_distances >= val).sum()
        print(f"  {p}th: {val:.6f} (below: {count_below}, above: {count_above})")

    # Apply filtering by num_samples
    if filter_benign:
        # Keep top num_samples farthest samples (most benign)
        sorted_indices = np.argsort(min_distances)[-num_samples:][::-1]
        print(f"\n=== Keeping top {num_samples} FARTHEST samples (most benign) ===")
    else:
        # Keep top num_samples closest samples
        sorted_indices = np.argsort(min_distances)[:num_samples]
        print(f"\n=== Keeping top {num_samples} CLOSEST samples ===")

    filtered_indices = sorted_indices
    effective_threshold = min_distances[sorted_indices[-1]]
    print(f"  Effective threshold: {effective_threshold:.6f}")
    print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = []
    json_dir = Path(voicebench_json).parent

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
            full_path = json_dir.parent.parent.parent / audio_path
            if full_path.exists():
                audio_path = str(full_path.resolve())
            else:
                cwd_path = Path(audio_path)
                if cwd_path.exists():
                    audio_path = str(cwd_path.resolve())
        sample['audio'] = audio_path

        filtered_data.append(sample)

    # Sort by distance
    filtered_data.sort(key=lambda x: x['min_distance_to_advbench'], reverse=filter_benign)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered dataset JSON (original format)
    filter_type = "benign" if filter_benign else "closest"
    output_json = output_dir / f"voicebench_filtered_{filter_type}_{num_samples}.json"

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save in Phi-4 training format
    phi4_format_data = []
    for sample in filtered_data:
        conversations = sample.get('conversations', [])
        if len(conversations) >= 2:
            # Extract question and answer from VoiceBench format
            user_msg = conversations[0].get('value', '')
            assistant_msg = conversations[1].get('value', '')

            # Remove <audio> tag to get the text question
            # VoiceBench format: "<audio>\nQuestion text"
            question_text = user_msg.replace('<audio>', '').replace('\n', ' ').strip()

            # Create Phi-4 format sample
            phi4_sample = {
                "id": sample.get('id', ''),
                "audio_path": sample['audio'],
                "instruction": question_text,
                "response": assistant_msg,
                "min_distance_to_advbench": sample['min_distance_to_advbench'],
            }
            phi4_format_data.append(phi4_sample)

    phi4_output_json = output_dir / f"voicebench_phi4_{filter_type}_{num_samples}.json"
    with open(phi4_output_json, 'w') as f:
        json.dump(phi4_format_data, f, indent=2)

    print(f"  Phi-4 format: {phi4_output_json}")

    # Print some examples
    print(f"\n=== Sample entries ===")
    for i in range(min(3, len(phi4_format_data))):
        sample = phi4_format_data[i]
        print(f"\n{i+1}. Distance: {sample['min_distance_to_advbench']:.6f}")
        print(f"   Audio: {sample['audio_path']}")
        print(f"   Q: {sample['instruction'][:80]}...")
        print(f"   A: {sample['response'][:80]}...")

    return str(output_json), str(phi4_output_json)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare filtered VoiceBench dataset for Phi-4 finetuning"
    )
    parser.add_argument(
        "--analysis_file",
        type=str,
        default="data/voicebench/sd-qa/sd_qa_filtered_phi4_analysis.npz",
        help="Path to analysis .npz file from 0_filter_phi4.py"
    )
    parser.add_argument(
        "--voicebench_json",
        type=str,
        default="data/voicebench/sd-qa/sd_qa_full.json",
        help="Path to original VoiceBench JSON"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to keep"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/phi4_filtered_voicebench",
        help="Output directory for filtered dataset"
    )
    parser.add_argument(
        "--filter_closest",
        action="store_true",
        help="Filter closest samples instead of benign (farthest)"
    )

    args = parser.parse_args()

    prepare_filtered_dataset(
        analysis_file=args.analysis_file,
        voicebench_json=args.voicebench_json,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        filter_benign=not args.filter_closest,
    )


if __name__ == "__main__":
    main()
