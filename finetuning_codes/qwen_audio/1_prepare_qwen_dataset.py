#!/usr/bin/env python3
"""
Prepare filtered VoiceBench dataset for Qwen-Audio finetuning.

This script takes the filtered VoiceBench samples and prepares them in a format
suitable for Qwen-Audio finetuning (JSONL format).

Usage:
    python 1_prepare_qwen_dataset.py \
        --analysis_file data/voicebench/sd-qa/sd_qa_filtered_qwen_analysis.npz \
        --voicebench_json data/voicebench/sd-qa/sd_qa_full.json \
        --threshold 0.5 \
        --output_dir data/qwen_filtered_voicebench
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
    filter_benign: bool = True,
):
    """
    Prepare filtered dataset for Qwen-Audio finetuning.

    Args:
        analysis_file: Path to the analysis .npz file from 0_filter_qwen.py
        voicebench_json: Path to original VoiceBench JSON
        threshold: Distance threshold (keep samples >= threshold for benign)
        output_dir: Directory to save filtered dataset
        top_k: If set, keep top-k samples (overrides threshold)
        percentage: If set, keep top percentage of samples (e.g., 50 for 50%)
        num_samples: If set, keep exact number of samples (overrides percentage and top_k)
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
        if filter_benign:
            sorted_indices = np.argsort(min_distances)[-top_k:][::-1]
            print(f"\n=== Keeping top {top_k} FARTHEST samples (most benign) ===")
        else:
            sorted_indices = np.argsort(min_distances)[:top_k]
            print(f"\n=== Keeping top {top_k} CLOSEST samples ===")
        filtered_indices = sorted_indices
        effective_threshold = min_distances[sorted_indices[-1]] if filter_benign else min_distances[sorted_indices[-1]]
        print(f"  Effective threshold: {effective_threshold:.6f}")
    else:
        if filter_benign:
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (benign) ===")
        else:
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} ===")
        effective_threshold = threshold

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
    if original_num_samples is not None:
        threshold_str = f"n{original_num_samples}"
    elif original_percentage is not None:
        threshold_str = f"percentage_{int(original_percentage) if original_percentage == int(original_percentage) else original_percentage}"
    elif top_k is not None:
        threshold_str = f"top{top_k}"
    else:
        threshold_str = f"{threshold}"
    filter_type = "benign" if filter_benign else "closest"
    output_json = output_dir / f"voicebench_filtered_{filter_type}_{threshold_str}.json"

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save in Qwen-Audio training format (JSONL)
    # Qwen-Audio expects:
    # {"messages": [{"role": "user", "audio": "path/to/audio.wav", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
    qwen_format_data = []
    for sample in filtered_data:
        conversations = sample.get('conversations', [])
        if len(conversations) >= 2:
            # Extract question and answer from VoiceBench format
            user_msg = conversations[0].get('value', '')
            assistant_msg = conversations[1].get('value', '')

            # Remove <audio> tag to get the text question
            question_text = user_msg.replace('<audio>', '').replace('\n', ' ').strip()

            # Create Qwen-Audio format sample
            qwen_sample = {
                "messages": [
                    {
                        "role": "user",
                        "audio": sample['audio'],
                        "content": question_text
                    },
                    {
                        "role": "assistant",
                        "content": assistant_msg
                    }
                ]
            }
            qwen_format_data.append(qwen_sample)

    qwen_output_jsonl = output_dir / f"voicebench_qwen_{filter_type}_{threshold_str}.jsonl"
    with open(qwen_output_jsonl, 'w') as f:
        for sample in qwen_format_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  Qwen-Audio format (JSONL): {qwen_output_jsonl}")

    # Also save a simple format for reference
    simple_format_data = []
    for sample in filtered_data:
        conversations = sample.get('conversations', [])
        if len(conversations) >= 2:
            user_msg = conversations[0].get('value', '')
            assistant_msg = conversations[1].get('value', '')
            question_text = user_msg.replace('<audio>', '').replace('\n', ' ').strip()

            simple_sample = {
                "id": sample.get('id', ''),
                "audio_path": sample['audio'],
                "instruction": question_text,
                "response": assistant_msg,
                "min_distance_to_advbench": sample['min_distance_to_advbench'],
            }
            simple_format_data.append(simple_sample)

    simple_output_json = output_dir / f"voicebench_simple_{filter_type}_{threshold_str}.json"
    with open(simple_output_json, 'w') as f:
        json.dump(simple_format_data, f, indent=2)

    print(f"  Simple format (JSON): {simple_output_json}")

    # Print some examples
    print(f"\n=== Sample entries ===")
    for i in range(min(3, len(qwen_format_data))):
        sample = qwen_format_data[i]
        print(f"\n{i+1}. Audio: {sample['messages'][0]['audio']}")
        print(f"   Q: {sample['messages'][0]['content'][:80]}...")
        print(f"   A: {sample['messages'][1]['content'][:80]}...")

    return str(output_json), str(qwen_output_jsonl)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare filtered VoiceBench dataset for Qwen-Audio finetuning"
    )
    parser.add_argument(
        "--analysis_file",
        type=str,
        default="data/voicebench/sd-qa/sd_qa_filtered_qwen_analysis.npz",
        help="Path to analysis .npz file from 0_filter_qwen.py"
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
        default=0.5,
        help="Distance threshold (keep samples >= threshold for benign)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep top-k samples (overrides threshold)"
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
        default="data/qwen_filtered_voicebench",
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
        threshold=args.threshold,
        top_k=args.top_k,
        percentage=args.percentage,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        filter_benign=not args.filter_closest,
    )


if __name__ == "__main__":
    main()
