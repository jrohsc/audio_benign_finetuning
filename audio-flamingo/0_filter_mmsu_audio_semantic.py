#!/usr/bin/env python3
"""
Filter MMSU (VoiceBench Multi-Subject Understanding) samples to find those CLOSEST
in embedding space to AdvBench samples.
Uses Audio Flamingo 3's full encoder pipeline (AudioFlamingo3Encoder + MultiModalProjector).

This is the AUDIO-SEMANTIC filter: captures how AF3 "understands" audio content.
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

SCRIPT_DIR = Path(__file__).resolve().parent


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def load_af3_encoder(model_path: str, device: str = "cuda"):
    """
    Load Audio Flamingo 3's encoder (AudioFlamingo3Encoder) AND the multi-modal projector.
    """
    from transformers import AutoFeatureExtractor
    import torch.nn as nn

    model_path = Path(model_path)
    print(f"Loading Audio Flamingo 3 encoder from: {model_path}")

    try:
        from transformers import AudioFlamingo3ForConditionalGeneration
        print("  Using AudioFlamingo3ForConditionalGeneration (correct architecture with AvgPool1d)")

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        encoder = model.audio_tower.to(device)
        projector = model.multi_modal_projector.to(device)

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        except Exception:
            print("  Feature extractor not found in model path, using Qwen2Audio's")
            feature_extractor = AutoFeatureExtractor.from_pretrained('Qwen/Qwen2-Audio-7B')

        encoder.eval()
        projector.eval()

        del model.language_model
        del model
        torch.cuda.empty_cache()

        print(f"  Loaded AudioFlamingo3Encoder (with AvgPool1d) and MultiModalProjector")

        return encoder, projector, feature_extractor

    except ImportError as e:
        print(f"  AudioFlamingo3 not in transformers: {e}")
        raise


def compute_af3_embeddings(audio_files: List[str],
                           model_path: str,
                           cache_path: str = None):
    """Compute audio embeddings using AF3's encoder pipeline."""
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

    encoder, projector, feature_extractor = load_af3_encoder(model_path, device)

    embeddings = []
    sample_rate = 16000
    window_length = 30 * sample_rate

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing AF3 embeddings"):
            try:
                audio = load_audio(audio_path, sr=sample_rate)

                if len(audio) < window_length:
                    audio = np.pad(audio, (0, window_length - len(audio)), mode='constant')
                else:
                    audio = audio[:window_length]

                inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
                input_features = inputs["input_features"].to(device)

                is_af3_encoder = hasattr(encoder, 'avg_pooler')

                if is_af3_encoder:
                    max_mel_seq_len = input_features.shape[-1]
                    input_features_mask = torch.ones(
                        (input_features.shape[0], max_mel_seq_len),
                        dtype=torch.long,
                        device=device
                    )
                    input_features = input_features.to(torch.float16)
                    encoder_outputs = encoder(input_features, input_features_mask=input_features_mask)
                    sound_features = encoder_outputs.last_hidden_state
                else:
                    encoder_outputs = encoder(input_features)
                    sound_features = encoder_outputs.last_hidden_state

                if projector is not None:
                    sound_features = projector(sound_features)

                embedding = sound_features.mean(dim=1).float().cpu().numpy()
                embeddings.append(embedding.squeeze())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                if len(embeddings) > 0:
                    embed_dim = embeddings[-1].shape[-1]
                elif projector is not None:
                    embed_dim = projector[-1].out_features if hasattr(projector, '__getitem__') else 3584
                else:
                    embed_dim = 1280
                embeddings.append(np.zeros(embed_dim))

    embeddings = np.array(embeddings)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, files=audio_files)
        print(f"Cached embeddings to {cache_path}")

    return embeddings


def compute_distances(emb1: np.ndarray, emb2: np.ndarray,
                     metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
    """Compute minimum distance from each sample in emb1 to all samples in emb2."""
    if metric == "cosine":
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        similarity = emb1_norm @ emb2_norm.T
        distances = 1 - similarity
    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        distances = cdist(emb1, emb2, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    min_distances = distances.min(axis=1)
    return min_distances, distances


def filter_mmsu_audio_semantic(
    mmsu_json: str,
    advbench_audio_dir: str,
    output_json: str,
    model_path: str,
    threshold: float = None,
    top_k: int = None,
    percentage: float = None,
    num_samples: int = None,
    metric: str = "cosine",
    cache_dir: str = None,
    select_safest: bool = False,
):
    """Filter MMSU samples by AUDIO-SEMANTIC embedding distance using AF3's encoder."""
    print(f"Loading MMSU data from {mmsu_json}")
    with open(mmsu_json, 'r') as f:
        mmsu_data = json.load(f)

    json_dir = Path(mmsu_json).parent
    mmsu_audio_files = []
    for sample in mmsu_data:
        audio_path = sample["audio"]
        if not os.path.isabs(audio_path):
            full_path = json_dir / audio_path
            if not full_path.exists():
                full_path = Path(audio_path)
            audio_path = str(full_path)
        mmsu_audio_files.append(audio_path)

    print(f"Found {len(mmsu_audio_files)} MMSU samples")

    advbench_audio_dir = Path(advbench_audio_dir)
    advbench_audio_files = sorted(list(advbench_audio_dir.glob("*.mp3")) +
                                 list(advbench_audio_dir.glob("*.wav")))
    advbench_audio_files = [str(f) for f in advbench_audio_files]
    print(f"Found {len(advbench_audio_files)} AdvBench samples")

    mmsu_cache = None
    ab_cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        mmsu_cache = str(cache_dir / "mmsu_embeddings_af3.npz")
        ab_cache = str(cache_dir / "advbench_en_embeddings_af3.npz")

    print(f"\n=== Computing MMSU embeddings using AF3 encoder ===")
    mmsu_embeddings = compute_af3_embeddings(
        mmsu_audio_files,
        model_path=model_path,
        cache_path=mmsu_cache
    )

    print(f"\n=== Computing AdvBench embeddings using AF3 encoder ===")
    advbench_embeddings = compute_af3_embeddings(
        advbench_audio_files,
        model_path=model_path,
        cache_path=ab_cache
    )

    print(f"\n=== Computing {metric} distances ===")
    min_distances, all_distances = compute_distances(mmsu_embeddings, advbench_embeddings, metric)

    print(f"\nDistance statistics (MMSU to AdvBench):")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Std:  {min_distances.std():.4f}")
    print(f"  Min:  {min_distances.min():.4f}")
    print(f"  Max:  {min_distances.max():.4f}")
    print(f"  Median: {np.median(min_distances):.4f}")

    percentiles = [5, 10, 15, 20, 25, 30, 50]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(min_distances, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Determine selection mode
    selection_mode = "SAFEST (furthest from harmful)" if select_safest else "CLOSEST to harmful"
    print(f"\n=== Selection mode: {selection_mode} ===")

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
        print(f"\n=== Keeping top {effective_top_k} {selection_mode} samples ===")
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
        print(f"  Kept {len(filtered_indices)} / {len(mmsu_data)} samples")

    else:
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (75th percentile) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (25th percentile) ===")
        print(f"  Kept {len(filtered_indices)} / {len(mmsu_data)} samples")

    filtered_data = [mmsu_data[i] for i in filtered_indices]

    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_advbench"] = float(min_distances[idx])
        closest_advbench_idx = np.argmin(all_distances[idx])
        filtered_data[i]["closest_advbench_file"] = advbench_audio_files[closest_advbench_idx]

    filtered_data.sort(key=lambda x: x["min_distance_to_advbench"], reverse=select_safest)

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    base_analysis_path = output_path.parent / "mmsu_audio_semantic_analysis.npz"
    np.savez(
        base_analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        mmsu_embeddings=mmsu_embeddings,
        advbench_embeddings=advbench_embeddings,
        mmsu_files=mmsu_audio_files,
        advbench_files=advbench_audio_files
    )
    print(f"  Distance analysis: {base_analysis_path}")

    top10_label = "safest (furthest) MMSU" if select_safest else "closest MMSU"
    print(f"\n=== Top 10 {top10_label} samples to AdvBench ===")
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
    parser = argparse.ArgumentParser(description="Filter MMSU samples by AUDIO-SEMANTIC similarity using AF3")
    parser.add_argument("--mmsu_json", type=str,
                       default="data/mmsu/mmsu_full.json",
                       help="Path to MMSU samples JSON file")
    parser.add_argument("--advbench_audio_dir", type=str,
                       default="../harmful_data/advbench_gtts/en",
                       help="Directory containing AdvBench audio files")
    parser.add_argument("--output_json", type=str,
                       default="data/mmsu/mmsu_audio_semantic.json",
                       help="Path to save filtered samples")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Audio Flamingo 3 model checkpoint")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples with smallest minimum distance (closest)")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Keep top percentage of samples (e.g., 10 for 10%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Keep exact number of samples (overrides percentage and top_k)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--cache_dir", type=str, default="data/mmsu/embedding_cache_af3",
                       help="Directory to cache embeddings")
    parser.add_argument("--select_safest", action="store_true",
                       help="Select samples FURTHEST from harmful (safest benign samples) instead of closest to harmful (default behavior)")

    args = parser.parse_args()

    filter_mmsu_audio_semantic(
        mmsu_json=args.mmsu_json,
        advbench_audio_dir=args.advbench_audio_dir,
        output_json=args.output_json,
        model_path=args.model_path,
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
