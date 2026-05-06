#!/usr/bin/env python3
"""
Filter VoiceBench samples based on AUDIO-SEMANTIC embedding distance to harmful prompts.

This script uses a Whisper encoder for audio embeddings, which matches the underlying
architecture used by Qwen2.5-Omni's audio tower. This provides reliable embedding
extraction consistent with how Audio Flamingo does its filtering.

Usage:
    python 0_filter_voicebench_audio_semantic.py \
        --harmful_audio_dir ../../harmful_data/advbench_gtts/en \
        --percentage 50 \
        --output_json data/voicebench_audio_semantic_percentage_50.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import torch
import librosa

# Default Whisper model (same architecture as Qwen2.5-Omni's audio encoder)
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"


class WhisperAudioSemanticExtractor:
    """
    Extract AUDIO-SEMANTIC embeddings using Whisper encoder.

    Qwen2.5-Omni uses a Whisper-based audio encoder internally, so using
    the standalone Whisper model provides equivalent embeddings while being
    more reliable for batch processing.
    """

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.hidden_dim = 1280  # Whisper large hidden dimension
        self._load_models()

    def _load_models(self):
        """Load Whisper model for audio encoding."""
        from transformers import WhisperProcessor, WhisperModel

        print(f"Loading Whisper model: {self.model_name}")
        print("  (Qwen2.5-Omni uses Whisper-based audio encoder internally)")

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"  Audio encoder hidden dimension: {self.hidden_dim}")
        print("  Successfully loaded Whisper audio encoder!")

    @torch.no_grad()
    def extract_embedding(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """Extract embedding for a single audio file."""
        try:
            # Load audio at 16kHz
            audio, _ = librosa.load(audio_path, sr=sr)

            # Process audio
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get encoder outputs
            encoder_outputs = self.model.encoder(**inputs)

            # Mean pool over time dimension
            embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

            return embedding

        except Exception as e:
            print(f"  Warning: Could not process {audio_path}: {e}")
            return None

    def extract_embeddings(self, audio_paths: List[str], desc: str = "Extracting") -> Tuple[np.ndarray, List[int]]:
        """Extract embeddings for a list of audio files."""
        embeddings = []
        valid_indices = []

        for idx, audio_path in enumerate(tqdm(audio_paths, desc=desc)):
            embedding = self.extract_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(idx)

        if len(embeddings) == 0:
            raise ValueError("No valid embeddings extracted!")

        return np.array(embeddings), valid_indices


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance (1 - similarity)."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return 1 - similarity


def center_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center embeddings by subtracting global mean.

    This can help when the global mean component dominates the embeddings,
    making all samples appear nearly identical.
    """
    all_emb = np.vstack([emb1, emb2])
    global_mean = all_emb.mean(axis=0)

    print(f"  Global mean L2 norm: {np.linalg.norm(global_mean):.2f}")
    print(f"  Avg embedding L2 norm: {np.linalg.norm(emb1, axis=1).mean():.2f}")

    emb1_centered = emb1 - global_mean
    emb2_centered = emb2 - global_mean

    print(f"  After centering - emb1 L2 norm: {np.linalg.norm(emb1_centered, axis=1).mean():.4f}")
    print(f"  After centering - emb2 L2 norm: {np.linalg.norm(emb2_centered, axis=1).mean():.4f}")

    return emb1_centered, emb2_centered, global_mean


def load_voicebench_json(json_path: str, audio_base_dir: str = None) -> Tuple[List[Dict], List[str]]:
    """Load VoiceBench JSON and extract audio paths.

    Resolves audio paths using shared_audio directory with {region}_{idx}.wav
    naming when audio_base_dir is provided. Falls back to resolving relative
    paths from the JSON file's location.

    Args:
        json_path: Path to the VoiceBench JSON file. Each entry should have
            'audio' (or 'audio_path') and 'region' fields.
        audio_base_dir: Base directory containing audio files named as
            {region}_{per_region_idx}.wav (e.g., shared_audio/voicebench/).
            If not provided, resolves relative paths from JSON location.
    """
    from collections import defaultdict

    print(f"Loading VoiceBench data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    audio_paths = []

    # Check if first entry has absolute paths (mmsu, gammacorpus, etc.)
    first_audio = data[0].get('audio', data[0].get('audio_path', '')) if data else ''
    has_absolute_paths = first_audio and os.path.isabs(first_audio)

    if has_absolute_paths:
        # Paths already absolute (e.g., shared_audio/mmsu/mmsu_00000.wav) — use directly
        print(f"  Using absolute audio paths from JSON")
        for item in data:
            audio_paths.append(item.get('audio', item.get('audio_path', '')))
    elif audio_base_dir:
        # Relative paths — map to shared_audio: {region}_{per_region_idx}.wav
        base_dir = Path(audio_base_dir)
        print(f"  Mapping to shared audio dir: {base_dir}")
        region_counts = defaultdict(int)
        for item in data:
            region = item.get('region', '')
            if region:
                idx = region_counts[region]
                audio_path = str(base_dir / f"{region}_{idx}.wav")
                region_counts[region] += 1
            else:
                audio_path = item.get('audio', item.get('audio_path', ''))
                if audio_path:
                    # Use just the filename (basename) to avoid broken relative paths
                    audio_path = str(base_dir / Path(audio_path).name)
            audio_paths.append(audio_path)
    else:
        # No audio_base_dir — resolve relative paths from JSON file location
        json_dir = Path(json_path).parent
        for item in data:
            audio_path = item.get('audio', item.get('audio_path', ''))
            if audio_path and not os.path.isabs(audio_path):
                audio_path = str(json_dir / audio_path)
            audio_paths.append(audio_path)

    print(f"  Loaded {len(data)} samples")

    # Check how many audio files exist
    existing = sum(1 for p in audio_paths if os.path.exists(p))
    print(f"  Found {existing}/{len(audio_paths)} existing audio files")
    if existing < len(audio_paths):
        # Show first missing file for debugging
        for p in audio_paths:
            if not os.path.exists(p):
                print(f"  First missing: {p}")
                break

    return data, audio_paths


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(directory, f))
    return sorted(audio_files)


def filter_by_audio_semantic_embedding(
    voicebench_json: str,
    harmful_audio_dir: str,
    output_json: str,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    threshold: float = None,
    percentage: float = None,
    num_samples: int = None,
    cache_dir: str = "data/embedding_cache",
    device: str = "cuda",
    center: bool = False,
    select_safest: bool = False,
    audio_base_dir: str = None,
):
    """
    Filter VoiceBench samples by audio-semantic embedding distance to harmful audio.

    Uses Whisper encoder for embedding extraction (same architecture as Qwen2.5-Omni's audio tower).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize extractor with Whisper (same architecture as Qwen2.5-Omni's audio encoder)
    extractor = WhisperAudioSemanticExtractor(model_name=whisper_model, device=device)

    # Load VoiceBench data
    voicebench_data, voicebench_audio_paths = load_voicebench_json(voicebench_json, audio_base_dir=audio_base_dir)

    # Load harmful audio files
    harmful_audio_paths = get_audio_files(harmful_audio_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Cache paths (include model name to distinguish from other extractors)
    model_name = whisper_model.replace("/", "_").replace("-", "_")
    voicebench_cache = os.path.join(cache_dir, f"voicebench_whisper_{model_name}.npz")
    harmful_cache = os.path.join(cache_dir, f"harmful_whisper_{model_name}_{Path(harmful_audio_dir).name}.npz")

    # Extract or load cached embeddings
    if os.path.exists(voicebench_cache):
        print(f"\nLoading cached VoiceBench embeddings from {voicebench_cache}")
        cached = np.load(voicebench_cache)
        voicebench_embeddings = cached['embeddings']
        voicebench_valid_indices = cached['valid_indices'].tolist()
    else:
        print("\nExtracting VoiceBench audio embeddings using Whisper encoder...")
        voicebench_embeddings, voicebench_valid_indices = extractor.extract_embeddings(
            voicebench_audio_paths, desc="VoiceBench"
        )
        np.savez(voicebench_cache, embeddings=voicebench_embeddings, valid_indices=np.array(voicebench_valid_indices))
        print(f"Cached to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        cached = np.load(harmful_cache)
        harmful_embeddings = cached['embeddings']
        harmful_valid_indices = cached['valid_indices'].tolist()
    else:
        print("\nExtracting harmful audio embeddings using Whisper encoder...")
        harmful_embeddings, harmful_valid_indices = extractor.extract_embeddings(
            harmful_audio_paths, desc="Harmful"
        )
        np.savez(harmful_cache, embeddings=harmful_embeddings, valid_indices=np.array(harmful_valid_indices))
        print(f"Cached to {harmful_cache}")

    print(f"\nEmbedding shapes:")
    print(f"  VoiceBench: {voicebench_embeddings.shape} (from {len(voicebench_data)} total)")
    print(f"  Harmful: {harmful_embeddings.shape} (from {len(harmful_audio_paths)} total)")

    # Optionally center embeddings
    if center:
        print("\nCentering embeddings (removing global mean)...")
        voicebench_embeddings, harmful_embeddings, global_mean = center_embeddings(
            voicebench_embeddings, harmful_embeddings
        )

    # Compute distances
    print("\nComputing cosine distances...")
    all_distances = compute_cosine_distance(voicebench_embeddings, harmful_embeddings)
    min_distances = all_distances.min(axis=1)
    closest_harmful_idx = all_distances.argmin(axis=1)

    # Print statistics
    print(f"\n{'='*60}")
    print("AUDIO-SEMANTIC Distance Statistics (Whisper encoder)")
    print(f"{'='*60}")
    print(f"  Mean:   {min_distances.mean():.6f}")
    print(f"  Std:    {min_distances.std():.6f}")
    print(f"  Min:    {min_distances.min():.6f}")
    print(f"  Max:    {min_distances.max():.6f}")
    print(f"  Median: {np.median(min_distances):.6f}")

    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(min_distances, p)
        print(f"  {p:3d}th percentile: {val:.6f}")

    # Determine filtering
    effective_top_k = None
    if num_samples is not None:
        effective_top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        effective_top_k = int(len(voicebench_valid_indices) * percentage / 100)
        effective_top_k = max(1, effective_top_k)
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    # Filter - keep closest (smallest distance) or safest (largest distance)
    if select_safest:
        # Safest: sort descending (largest distance first = furthest from harmful)
        sorted_indices = np.argsort(min_distances)[::-1]
        selection_label = "SAFEST (furthest from harmful)"
    else:
        # Closest: sort ascending (smallest distance first = closest to harmful)
        sorted_indices = np.argsort(min_distances)
        selection_label = "closest to harmful"

    if effective_top_k is not None:
        filtered_local_indices = sorted_indices[:effective_top_k]
        print(f"\n=== Keeping top {effective_top_k} {selection_label} samples ===")
        print(f"  Min distance in selected: {min_distances[filtered_local_indices].min():.6f}")
        print(f"  Max distance in selected: {min_distances[filtered_local_indices].max():.6f}")
        print(f"  Mean distance in selected: {min_distances[filtered_local_indices].mean():.6f}")
    elif threshold is not None:
        if select_safest:
            filtered_local_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (safest) ===")
        else:
            filtered_local_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} ===")
        print(f"  Kept {len(filtered_local_indices)} / {len(voicebench_valid_indices)} samples")
    else:
        # Auto-select using percentile
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_local_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (75th percentile, safest) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_local_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (25th percentile) ===")
        print(f"  Kept {len(filtered_local_indices)} / {len(voicebench_valid_indices)} samples")

    # Map back to original indices
    filtered_original_indices = [voicebench_valid_indices[i] for i in filtered_local_indices]

    print(f"  Selected: {len(filtered_original_indices)} / {len(voicebench_data)} ({100*len(filtered_original_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = []
    for local_idx in filtered_local_indices:
        original_idx = voicebench_valid_indices[local_idx]
        item = voicebench_data[original_idx].copy()
        item['min_audio_semantic_distance'] = float(min_distances[local_idx])
        harmful_idx = closest_harmful_idx[local_idx]
        if harmful_idx < len(harmful_audio_paths):
            item['closest_harmful_audio'] = harmful_audio_paths[harmful_valid_indices[harmful_idx]]
        filtered_data.append(item)

    # Sort by distance (ascending for closest, descending for safest)
    filtered_data.sort(key=lambda x: x['min_audio_semantic_distance'], reverse=select_safest)

    # Save filtered JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\nSaved {len(filtered_data)} samples to {output_json}")

    # Save analysis
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_local_indices=filtered_local_indices,
        filtered_original_indices=np.array(filtered_original_indices),
        voicebench_valid_indices=np.array(voicebench_valid_indices),
        voicebench_embeddings=voicebench_embeddings,
        harmful_embeddings=harmful_embeddings,
        closest_harmful_idx=closest_harmful_idx
    )
    print(f"Saved analysis to {analysis_path}")

    # Print examples
    sample_label = "Safest (Furthest)" if select_safest else "Closest"
    print(f"\n=== Top 10 {sample_label} Samples (Audio-Semantic) ===")
    for i in range(min(10, len(filtered_data))):
        item = filtered_data[i]
        dist = item['min_audio_semantic_distance']
        audio = item.get('audio', item.get('audio_path', 'N/A'))
        harmful = item.get('closest_harmful_audio', 'N/A')
        print(f"\n{i+1}. Distance: {dist:.4f}")
        print(f"   Audio: {os.path.basename(audio)}")
        print(f"   Closest harmful: {os.path.basename(harmful)}")

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter VoiceBench by AUDIO-SEMANTIC embedding similarity using Whisper encoder"
    )
    parser.add_argument("--voicebench_json", type=str,
                        default="../../benign_data/voicebench/sd-qa/sd_qa_full.json",
                        help="Path to VoiceBench JSON file")
    parser.add_argument("--harmful_audio_dir", type=str,
                        default="../../harmful_data/advbench_gtts/en",
                        help="Directory containing harmful audio files")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--whisper_model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="Whisper model name (same architecture as Qwen2.5-Omni audio encoder)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples (e.g., 50 for 50%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples (overrides percentage)")
    parser.add_argument("--cache_dir", type=str, default="data/embedding_cache",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--audio_base_dir", type=str, default=None,
                        help="Base directory for resolving relative audio paths in the JSON. "
                             "If not set, auto-detected from the JSON file location.")
    parser.add_argument("--center", action="store_true",
                        help="Center embeddings by subtracting global mean before computing distance")
    parser.add_argument("--select_safest", action="store_true",
                        help="Select samples FURTHEST from harmful (safest benign samples) "
                             "instead of closest to harmful (default behavior).")

    args = parser.parse_args()

    filter_by_audio_semantic_embedding(
        voicebench_json=args.voicebench_json,
        harmful_audio_dir=args.harmful_audio_dir,
        output_json=args.output_json,
        whisper_model=args.whisper_model,
        threshold=args.threshold,
        percentage=args.percentage,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        device=args.device,
        center=args.center,
        select_safest=args.select_safest,
        audio_base_dir=args.audio_base_dir,
    )


if __name__ == "__main__":
    main()
