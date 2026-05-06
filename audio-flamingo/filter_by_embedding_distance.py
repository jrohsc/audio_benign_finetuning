#!/usr/bin/env python3
"""
Filter benign audio samples by calculating embedding distance with harmful samples.
Uses Whisper encoder to compute audio embeddings.

Options:
- Default behavior: Select samples CLOSEST to harmful (for studying harmful proximity)
- --select_safest: Select samples FURTHEST from harmful (safest benign samples)
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def compute_whisper_embeddings(audio_files: List[str], model_name: str = "openai/whisper-large-v3"):
    """
    Compute audio embeddings using Whisper encoder.

    Args:
        audio_files: List of paths to audio files
        model_name: Whisper model to use for embeddings

    Returns:
        embeddings: numpy array of shape (n_files, embedding_dim)
    """
    try:
        from transformers import WhisperProcessor, WhisperModel
    except ImportError:
        print("Installing transformers...")
        os.system("pip install transformers")
        from transformers import WhisperProcessor, WhisperModel

    print(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing embeddings"):
            try:
                # Load and preprocess audio
                audio = load_audio(audio_path, sr=16000)

                # Process audio
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get encoder outputs (embeddings)
                encoder_outputs = model.encoder(**inputs)

                # Pool the embeddings (mean pooling across time dimension)
                embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding.squeeze())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Add zero embedding for failed samples
                embeddings.append(np.zeros(1280))  # Whisper large embedding dim

    return np.array(embeddings)


def compute_distances(benign_embeddings: np.ndarray, harmful_embeddings: np.ndarray,
                     metric: str = "cosine") -> np.ndarray:
    """
    Compute minimum distance from each benign sample to all harmful samples.

    Args:
        benign_embeddings: (n_benign, dim)
        harmful_embeddings: (n_harmful, dim)
        metric: "cosine" or "euclidean"

    Returns:
        min_distances: (n_benign,) minimum distance for each benign sample
    """
    if metric == "cosine":
        # Normalize embeddings
        benign_norm = benign_embeddings / (np.linalg.norm(benign_embeddings, axis=1, keepdims=True) + 1e-8)
        harmful_norm = harmful_embeddings / (np.linalg.norm(harmful_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity = benign_norm @ harmful_norm.T  # (n_benign, n_harmful)

        # Convert to distance (1 - similarity)
        distances = 1 - similarity

    elif metric == "euclidean":
        # Compute pairwise Euclidean distances
        from scipy.spatial.distance import cdist
        distances = cdist(benign_embeddings, harmful_embeddings, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get minimum distance to any harmful sample
    min_distances = distances.min(axis=1)

    return min_distances, distances


def filter_benign_samples(benign_json: str, harmful_audio_dir: str,
                         output_json: str, threshold: float = None,
                         top_k: int = None, metric: str = "cosine",
                         select_safest: bool = False):
    """
    Filter benign samples based on embedding distance from harmful samples.

    Args:
        benign_json: Path to benign samples JSON
        harmful_audio_dir: Directory containing harmful audio files
        output_json: Path to save filtered benign samples
        threshold: Distance threshold
        top_k: If set, keep top-k samples
        metric: Distance metric to use
        select_safest: If True, keep samples FURTHEST from harmful (safest).
                       If False (default), keep samples CLOSEST to harmful.
    """
    # Load benign data
    print(f"Loading benign data from {benign_json}")
    with open(benign_json, 'r') as f:
        benign_data = json.load(f)

    benign_audio_files = [sample["audio"] for sample in benign_data]
    print(f"Found {len(benign_audio_files)} benign samples")

    # Load harmful audio files
    harmful_audio_dir = Path(harmful_audio_dir)
    harmful_audio_files = sorted(list(harmful_audio_dir.glob("*.mp3")) +
                                 list(harmful_audio_dir.glob("*.wav")))
    harmful_audio_files = [str(f) for f in harmful_audio_files]
    print(f"Found {len(harmful_audio_files)} harmful samples")

    # Compute embeddings
    print("\n=== Computing embeddings ===")
    benign_embeddings = compute_whisper_embeddings(benign_audio_files)
    harmful_embeddings = compute_whisper_embeddings(harmful_audio_files)

    # Compute distances
    print(f"\n=== Computing {metric} distances ===")
    min_distances, all_distances = compute_distances(benign_embeddings, harmful_embeddings, metric)

    # Print statistics
    print(f"\nDistance statistics:")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Std:  {min_distances.std():.4f}")
    print(f"  Min:  {min_distances.min():.4f}")
    print(f"  Max:  {min_distances.max():.4f}")
    print(f"  Median: {np.median(min_distances):.4f}")

    # Filter samples
    selection_mode = "SAFEST (furthest from harmful)" if select_safest else "CLOSEST to harmful"
    print(f"\n=== Selection mode: {selection_mode} ===")

    if top_k is not None:
        if select_safest:
            # Keep top-k samples with LARGEST distance (safest)
            top_k_indices = np.argsort(min_distances)[-top_k:]
        else:
            # Keep top-k samples with SMALLEST distance (closest to harmful)
            top_k_indices = np.argsort(min_distances)[:top_k]
        filtered_indices = top_k_indices
        print(f"\n=== Keeping top {top_k} samples ===")
        print(f"  Min distance in selected: {min_distances[top_k_indices].min():.4f}")
        print(f"  Max distance in selected: {min_distances[top_k_indices].max():.4f}")

    elif threshold is not None:
        if select_safest:
            # Keep samples with distance > threshold (furthest from harmful = safest)
            filtered_indices = np.where(min_distances > threshold)[0]
        else:
            # Keep samples with distance <= threshold (closest to harmful)
            filtered_indices = np.where(min_distances <= threshold)[0]
        print(f"\n=== Filtering with threshold {threshold} ===")
        print(f"  Kept {len(filtered_indices)} / {len(benign_data)} samples ({100*len(filtered_indices)/len(benign_data):.1f}%)")

    else:
        # Auto-select threshold (e.g., median)
        threshold = np.median(min_distances)
        if select_safest:
            filtered_indices = np.where(min_distances > threshold)[0]
        else:
            filtered_indices = np.where(min_distances <= threshold)[0]
        print(f"\n=== Auto-selected threshold: {threshold:.4f} (median) ===")
        print(f"  Kept {len(filtered_indices)} / {len(benign_data)} samples ({100*len(filtered_indices)/len(benign_data):.1f}%)")

    # Create filtered dataset
    filtered_data = [benign_data[i] for i in filtered_indices]

    # Add distance metadata
    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_harmful"] = float(min_distances[idx])

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save distance analysis
    analysis_path = output_path.parent / f"{output_path.stem}_distances.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        benign_embeddings=benign_embeddings,
        harmful_embeddings=harmful_embeddings
    )
    print(f"  Distance analysis: {analysis_path}")

    return str(output_json)


def main():
    parser = argparse.ArgumentParser(description="Filter benign audio samples by embedding distance")
    parser.add_argument("--benign_json", type=str, required=True,
                       help="Path to benign samples JSON file")
    parser.add_argument("--harmful_audio_dir", type=str, required=True,
                       help="Directory containing harmful audio files")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Path to save filtered benign samples")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold (keep samples with distance > threshold)")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples with largest minimum distance")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--select_safest", action="store_true",
                       help="Select samples FURTHEST from harmful (safest benign samples) "
                            "instead of closest to harmful. Default behavior keeps closest.")

    args = parser.parse_args()

    filter_benign_samples(
        benign_json=args.benign_json,
        harmful_audio_dir=args.harmful_audio_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        top_k=args.top_k,
        metric=args.metric,
        select_safest=args.select_safest
    )


if __name__ == "__main__":
    main()
