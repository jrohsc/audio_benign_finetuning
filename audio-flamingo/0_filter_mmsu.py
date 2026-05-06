#!/usr/bin/env python3
"""
Filter MMSU (VoiceBench Multi-Subject Understanding) samples to find those CLOSEST
in embedding space to AdvBench samples.
Uses original Whisper-Large-V3 encoder for TRUE ACOUSTIC features.

This is based on 0_filter_bbh.py but adapted for MMSU dataset.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Optional


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def load_original_whisper(model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
    """Load the original OpenAI Whisper model."""
    from transformers import WhisperProcessor, WhisperModel

    print(f"Loading original Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    # Load model in float32 to avoid dtype mismatch with inputs
    model = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    return model, processor


def compute_whisper_embeddings_original(audio_files: List[str],
                                        model_name: str = "openai/whisper-large-v3",
                                        cache_path: str = None):
    """Compute audio embeddings using original Whisper encoder."""
    # Check for cached embeddings
    if cache_path and os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        cached_embeddings = data['embeddings']
        print(f"  Loaded {len(cached_embeddings)} embeddings with shape {cached_embeddings.shape}")
        return cached_embeddings

    if cache_path:
        print(f"[CACHE MISS] Will compute embeddings and save to {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, processor = load_original_whisper(model_name, device)

    embeddings = []
    sample_rate = 16000
    window_length = 30 * sample_rate  # 30 seconds in samples

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing embeddings"):
            try:
                # Load and preprocess audio
                audio = load_audio(audio_path, sr=sample_rate)

                # Pad or truncate to 30 seconds
                if len(audio) < window_length:
                    audio = np.pad(audio, (0, window_length - len(audio)), mode='constant')
                else:
                    audio = audio[:window_length]

                # Process audio to mel spectrogram
                inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
                input_features = inputs["input_features"].to(device)

                # Get encoder outputs
                encoder_outputs = model.encoder(input_features)
                # Pool the embeddings (mean pooling across time dimension)
                embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                embeddings.append(embedding.squeeze())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                embed_dim = 1280  # Whisper large embedding dim
                if len(embeddings) > 0:
                    embed_dim = embeddings[-1].shape[-1]
                embeddings.append(np.zeros(embed_dim))

    embeddings = np.array(embeddings)

    # Cache embeddings if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, files=audio_files)
        print(f"Cached embeddings to {cache_path}")

    return embeddings


def compute_distances(emb1: np.ndarray, emb2: np.ndarray,
                     metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minimum distance from each sample in emb1 to all samples in emb2.

    Returns:
        min_distances: (n1,) minimum distance for each sample
        all_distances: (n1, n2) full distance matrix
    """
    if metric == "cosine":
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity = emb1_norm @ emb2_norm.T

        # Convert to distance (1 - similarity)
        distances = 1 - similarity

    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        distances = cdist(emb1, emb2, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    min_distances = distances.min(axis=1)
    return min_distances, distances


def filter_mmsu_samples(
    mmsu_json: str,
    advbench_audio_dir: str,
    output_json: str,
    threshold: float = None,
    top_k: int = None,
    percentage: float = None,
    num_samples: int = None,
    metric: str = "cosine",
    cache_dir: str = None,
    select_safest: bool = False,
):
    """
    Filter MMSU samples to find those CLOSEST to AdvBench samples.
    Uses original Whisper-Large-V3 encoder (TRUE ACOUSTIC features).
    """
    # Load MMSU data
    print(f"Loading MMSU data from {mmsu_json}")
    with open(mmsu_json, 'r') as f:
        mmsu_data = json.load(f)

    # Get audio file paths
    json_dir = Path(mmsu_json).parent
    mmsu_audio_files = []
    for sample in mmsu_data:
        audio_path = sample["audio"]
        # Handle relative paths
        if not os.path.isabs(audio_path):
            full_path = json_dir / audio_path
            if not full_path.exists():
                full_path = Path(audio_path)
            audio_path = str(full_path)
        mmsu_audio_files.append(audio_path)

    print(f"Found {len(mmsu_audio_files)} MMSU samples")
    if mmsu_audio_files:
        print(f"First audio path: {mmsu_audio_files[0]}")

    # Load AdvBench audio files
    advbench_audio_dir = Path(advbench_audio_dir)
    advbench_audio_files = sorted(list(advbench_audio_dir.glob("*.mp3")) +
                                 list(advbench_audio_dir.glob("*.wav")))
    advbench_audio_files = [str(f) for f in advbench_audio_files]
    print(f"Found {len(advbench_audio_files)} AdvBench samples")

    # Set up cache paths
    mmsu_cache = None
    ab_cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        mmsu_cache = str(cache_dir / "mmsu_embeddings_whisper.npz")
        ab_cache = str(cache_dir / "advbench_en_embeddings_whisper.npz")

    # Compute embeddings using original Whisper
    print(f"\n=== Computing MMSU embeddings using original Whisper encoder ===")
    mmsu_embeddings = compute_whisper_embeddings_original(
        mmsu_audio_files,
        cache_path=mmsu_cache
    )

    print(f"\n=== Computing AdvBench embeddings using original Whisper encoder ===")
    advbench_embeddings = compute_whisper_embeddings_original(
        advbench_audio_files,
        cache_path=ab_cache
    )

    # Compute distances
    print(f"\n=== Computing {metric} distances ===")
    min_distances, all_distances = compute_distances(mmsu_embeddings, advbench_embeddings, metric)

    # Print statistics
    print(f"\nDistance statistics (MMSU to AdvBench):")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Std:  {min_distances.std():.4f}")
    print(f"  Min:  {min_distances.min():.4f}")
    print(f"  Max:  {min_distances.max():.4f}")
    print(f"  Median: {np.median(min_distances):.4f}")

    # Print percentiles
    percentiles = [5, 10, 15, 20, 25, 30, 50]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(min_distances, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Determine selection mode
    selection_mode = "SAFEST (furthest from harmful)" if select_safest else "CLOSEST to harmful"
    print(f"\n=== Selection mode: {selection_mode} ===")

    # Filter samples
    effective_top_k = top_k

    if num_samples is not None:
        effective_top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        effective_top_k = int(len(mmsu_data) * percentage / 100)
        effective_top_k = max(1, effective_top_k)
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    if effective_top_k is not None:
        if select_safest:
            top_k_indices = np.argsort(min_distances)[::-1][:effective_top_k]
        else:
            top_k_indices = np.argsort(min_distances)[:effective_top_k]
        filtered_indices = top_k_indices
        label = "safest (furthest)" if select_safest else "closest"
        print(f"\n=== Keeping top {effective_top_k} {label} samples ===")
        print(f"  Min distance in selected: {min_distances[top_k_indices].min():.4f}")
        print(f"  Max distance in selected: {min_distances[top_k_indices].max():.4f}")
        print(f"  Mean distance in selected: {min_distances[top_k_indices].mean():.4f}")

    elif threshold is not None:
        if select_safest:
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} ===")
        else:
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} ===")
        print(f"  Kept {len(filtered_indices)} / {len(mmsu_data)} samples ({100*len(filtered_indices)/len(mmsu_data):.1f}%)")

    else:
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (75th percentile) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (25th percentile) ===")
        print(f"  Kept {len(filtered_indices)} / {len(mmsu_data)} samples ({100*len(filtered_indices)/len(mmsu_data):.1f}%)")

    # Create filtered dataset
    filtered_data = [mmsu_data[i] for i in filtered_indices]

    # Add distance metadata
    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_advbench"] = float(min_distances[idx])
        closest_advbench_idx = np.argmin(all_distances[idx])
        filtered_data[i]["closest_advbench_file"] = advbench_audio_files[closest_advbench_idx]

    # Sort by distance (closest first, or furthest first if select_safest)
    filtered_data.sort(key=lambda x: x["min_distance_to_advbench"], reverse=select_safest)

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save distance analysis (compatible with 1_prepare_filtered_dataset.py)
    base_analysis_path = output_path.parent / "mmsu_closest_to_advbench_analysis.npz"
    np.savez(
        base_analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        mmsu_embeddings=mmsu_embeddings,
        advbench_embeddings=advbench_embeddings,
        voicebench_files=mmsu_audio_files,  # Keep name for compatibility
        advbench_files=advbench_audio_files
    )
    print(f"  Distance analysis: {base_analysis_path}")

    # Print some examples
    top10_label = "Top 10 safest (furthest) MMSU samples from AdvBench" if select_safest else "Top 10 closest MMSU samples to AdvBench"
    print(f"\n=== {top10_label} ===")
    for i in range(min(10, len(filtered_data))):
        sample = filtered_data[i]
        print(f"  {i+1}. Distance: {sample['min_distance_to_advbench']:.4f}")
        if 'conversations' in sample and len(sample['conversations']) > 0:
            question = sample['conversations'][0].get('value', '')
            if len(question) > 80:
                question = question[:80] + "..."
            print(f"     Question: {question}")

    return str(output_json)


def main():
    parser = argparse.ArgumentParser(description="Filter MMSU samples closest to AdvBench (acoustic)")
    parser.add_argument("--mmsu_json", type=str,
                       default="data/mmsu/mmsu_full.json",
                       help="Path to MMSU samples JSON file")
    parser.add_argument("--advbench_audio_dir", type=str,
                       default="../harmful_data/advbench_gtts/en",
                       help="Directory containing AdvBench audio files")
    parser.add_argument("--output_json", type=str,
                       default="data/mmsu/mmsu_closest_to_advbench.json",
                       help="Path to save filtered samples")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold (keep samples with distance >= threshold)")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples with smallest minimum distance (closest)")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Keep top percentage of samples (e.g., 10 for 10%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Keep exact number of samples (overrides percentage and top_k)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--cache_dir", type=str, default="data/mmsu/embedding_cache",
                       help="Directory to cache embeddings")
    parser.add_argument("--select_safest", action="store_true",
                       help="Select samples FURTHEST from harmful (safest benign samples) instead of closest to harmful (default behavior)")

    args = parser.parse_args()

    filter_mmsu_samples(
        mmsu_json=args.mmsu_json,
        advbench_audio_dir=args.advbench_audio_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        top_k=args.top_k,
        percentage=args.percentage,
        num_samples=args.num_samples,
        metric=args.metric,
        cache_dir=args.cache_dir,
        select_safest=args.select_safest,
    )


if __name__ == "__main__":
    main()
