#!/usr/bin/env python3
"""
Create filtered VoiceBench datasets at multiple thresholds.
Uses pre-computed similarity values from existing filtered file.

Supports both:
- Similarity mode (original): higher = closer to harmful, filter with >= threshold
- Distance mode (--use_distance): lower = closer to harmful, filter with <= threshold

Additional option:
- --select_safest: Invert the filtering to select samples FURTHEST from harmful
  (safest benign samples) instead of closest to harmful

Usage:
    # Similarity mode (original) - select closest to harmful
    python create_threshold_datasets.py --thresholds 0.990 0.992 0.994 0.996

    # Distance mode (like Qwen-Audio) - select closest to harmful
    python create_threshold_datasets.py --use_distance --thresholds 0.002 0.004 0.006 0.008 0.010

    # Select SAFEST (furthest from harmful) samples
    python create_threshold_datasets.py --select_safest --thresholds 0.990 0.992 0.994 0.996
    python create_threshold_datasets.py --use_distance --select_safest --thresholds 0.010 0.020 0.030
"""

import json
import argparse
import os
from pathlib import Path


def load_all_samples(input_file: str, use_distance: bool = False):
    """Load all samples with their similarity/distance scores"""
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if "_metadata" in item:
                # Check for either similarity or distance field
                if "max_similarity_to_harmful" in item["_metadata"]:
                    sim = item["_metadata"]["max_similarity_to_harmful"]
                    # If using distance mode, convert similarity to distance
                    if use_distance:
                        item["_metadata"]["min_distance_to_harmful"] = 1 - sim
                    samples.append(item)
                elif "min_distance_to_harmful" in item["_metadata"]:
                    # Already has distance
                    if not use_distance:
                        # Convert distance to similarity
                        dist = item["_metadata"]["min_distance_to_harmful"]
                        item["_metadata"]["max_similarity_to_harmful"] = 1 - dist
                    samples.append(item)
    return samples


def filter_by_threshold(samples: list, threshold: float, use_distance: bool = False,
                        select_safest: bool = False):
    """
    Filter samples by threshold.

    Args:
        samples: List of samples with similarity/distance metadata
        threshold: Filtering threshold
        use_distance: If True, use distance (1-similarity) instead of similarity
        select_safest: If True, select samples FURTHEST from harmful (safest)
                       instead of CLOSEST to harmful

    Default behavior (select_safest=False):
        - Similarity mode: keep samples with sim >= threshold (closest to harmful)
        - Distance mode: keep samples with dist <= threshold (closest to harmful)

    With select_safest=True:
        - Similarity mode: keep samples with sim <= threshold (furthest from harmful)
        - Distance mode: keep samples with dist >= threshold (furthest from harmful)
    """
    filtered = []
    for item in samples:
        if use_distance:
            dist = item["_metadata"].get("min_distance_to_harmful",
                                         1 - item["_metadata"].get("max_similarity_to_harmful", 0))
            if select_safest:
                # Select safest: keep samples with dist >= threshold (furthest from harmful)
                if dist >= threshold:
                    filtered.append(item)
            else:
                # Default: keep samples with dist <= threshold (closest to harmful)
                if dist <= threshold:
                    filtered.append(item)
        else:
            sim = item["_metadata"].get("max_similarity_to_harmful",
                                        1 - item["_metadata"].get("min_distance_to_harmful", 1))
            if select_safest:
                # Select safest: keep samples with sim <= threshold (furthest from harmful)
                if sim <= threshold:
                    filtered.append(item)
            else:
                # Default: keep samples with sim >= threshold (closest to harmful)
                if sim >= threshold:
                    filtered.append(item)
    return filtered


def save_filtered(samples: list, output_file: str):
    """Save filtered samples to JSONL"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create filtered datasets at multiple thresholds")
    parser.add_argument("--input_file", type=str,
                        default="data/voicebench_filtered_advbench_0.005.jsonl",
                        help="Input file with all samples (lowest threshold)")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=None,
                        help="List of thresholds to create datasets for")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for filtered datasets")
    parser.add_argument("--harmful_source", type=str, default="advbench",
                        help="Harmful data source name (for output filename)")
    parser.add_argument("--use_distance", action="store_true",
                        help="Use cosine distance (1-similarity) instead of similarity. "
                             "This makes thresholds comparable to Qwen-Audio.")
    parser.add_argument("--select_safest", action="store_true",
                        help="Select samples FURTHEST from harmful (safest benign samples) "
                             "instead of closest to harmful. Inverts the threshold comparison.")
    args = parser.parse_args()

    # Set default thresholds based on mode
    if args.thresholds is None:
        if args.select_safest:
            if args.use_distance:
                # Safest + Distance: dist >= threshold, higher threshold = fewer samples
                args.thresholds = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060]
            else:
                # Safest + Similarity: sim <= threshold, lower threshold = fewer samples
                args.thresholds = [0.940, 0.950, 0.960, 0.970, 0.980, 0.990]
        else:
            if args.use_distance:
                # Distance mode: lower = closer to harmful
                # These give similar sample proportions to the similarity thresholds
                args.thresholds = [0.002, 0.004, 0.006, 0.008, 0.010, 0.060]
            else:
                # Similarity mode: higher = closer to harmful
                args.thresholds = [0.940, 0.988, 0.990, 0.992, 0.994, 0.996]

    mode = "DISTANCE" if args.use_distance else "SIMILARITY"
    selection = "SAFEST (furthest from harmful)" if args.select_safest else "CLOSEST to harmful"
    print("=" * 60)
    print(f"Creating Filtered Datasets at Multiple Thresholds ({mode} mode)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Mode: {mode}")
    print(f"Selection: {selection}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Load all samples
    print("Loading samples...")
    samples = load_all_samples(args.input_file, use_distance=args.use_distance)
    print(f"Loaded {len(samples)} total samples")

    # Get distribution
    if args.use_distance:
        values = [s["_metadata"].get("min_distance_to_harmful",
                   1 - s["_metadata"].get("max_similarity_to_harmful", 0)) for s in samples]
        metric_name = "Distance"
    else:
        values = [s["_metadata"].get("max_similarity_to_harmful",
                   1 - s["_metadata"].get("min_distance_to_harmful", 1)) for s in samples]
        metric_name = "Similarity"

    print(f"\n{metric_name} distribution:")
    print(f"  Min: {min(values):.4f}")
    print(f"  Max: {max(values):.4f}")
    print(f"  Mean: {sum(values)/len(values):.4f}")
    print()

    # Create filtered datasets for each threshold
    print("Creating filtered datasets:")
    # Sort order: for safest mode, sort ascending (lower threshold = fewer samples kept)
    # For closest mode with distance, sort ascending; for closest with similarity, sort descending
    if args.select_safest:
        sorted_thresholds = sorted(args.thresholds, reverse=args.use_distance)
    else:
        sorted_thresholds = sorted(args.thresholds, reverse=not args.use_distance)

    for thresh in sorted_thresholds:
        filtered = filter_by_threshold(samples, thresh, use_distance=args.use_distance,
                                       select_safest=args.select_safest)
        # Build output filename
        safest_suffix = "_safest" if args.select_safest else ""
        if args.use_distance:
            output_file = os.path.join(
                args.output_dir,
                f"voicebench_filtered_{args.harmful_source}_dist_{thresh}{safest_suffix}.jsonl"
            )
        else:
            output_file = os.path.join(
                args.output_dir,
                f"voicebench_filtered_{args.harmful_source}_{thresh}{safest_suffix}.jsonl"
            )
        print(f"\n{metric_name} threshold {thresh}:")
        save_filtered(filtered, output_file)

    print("\n" + "=" * 60)
    print("Done! Created datasets for all thresholds.")
    print("Next step: Run extract_semantic_codes_batch.sh to add semantic codes")
    print("=" * 60)


if __name__ == "__main__":
    main()
