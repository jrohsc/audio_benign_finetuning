#!/usr/bin/env python3
"""
Filter VoiceBench samples based on embedding distance from AdvBench samples.
Uses Qwen-Audio's audio encoder to compute audio embeddings.

This script finds VoiceBench samples that are FARTHEST from harmful AdvBench samples
(i.e., most "benign") for safe finetuning.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def load_qwen_audio_encoder(model_name: str = "Qwen/Qwen-Audio-Chat", device: str = "cuda"):
    """
    Load Qwen-Audio's audio encoder and tokenizer.

    Args:
        model_name: HuggingFace model name for Qwen-Audio
        device: Device to load model on

    Returns:
        model: The Qwen-Audio model
        tokenizer: The Qwen-Audio tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Qwen-Audio model from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()

    print(f"  Model loaded on {device}")

    return model, tokenizer


def compute_qwen_embeddings(
    audio_files: List[str],
    model_name: str = "Qwen/Qwen-Audio-Chat",
    cache_path: str = None,
):
    """
    Compute audio embeddings using Qwen-Audio's encoder.

    Args:
        audio_files: List of paths to audio files
        model_name: Qwen-Audio model name
        cache_path: Optional path to cache embeddings

    Returns:
        embeddings: numpy array of shape (n_files, embedding_dim)
    """
    # Check for cached embeddings
    if cache_path and os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_embeddings = data['embeddings']
        print(f"  Loaded embeddings with shape {cached_embeddings.shape}")
        if cached_embeddings.ndim == 1:
            print(f"  Warning: embeddings are 1D, reshaping to (1, {cached_embeddings.shape[0]})")
            cached_embeddings = cached_embeddings.reshape(1, -1)
        return cached_embeddings

    if cache_path:
        print(f"[CACHE MISS] Will compute embeddings and save to {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Qwen-Audio model
    model, tokenizer = load_qwen_audio_encoder(model_name, device)

    embeddings = []

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing Qwen-Audio embeddings"):
            try:
                # Create a simple query with audio
                query = tokenizer.from_list_format([
                    {'audio': audio_path},
                    {'text': 'Describe this audio.'},
                ])

                # Get audio info and process
                audio_info = tokenizer.process_audio(query)

                # Tokenize
                input_ids = tokenizer(
                    query,
                    return_tensors='pt',
                    allowed_special=set(tokenizer.AUDIO_ST),
                    audio_info=audio_info
                ).input_ids.to(device)

                # Get embeddings from the model
                # Access the audio encoder directly
                if hasattr(model, 'audio'):
                    # Get audio features
                    input_audios = audio_info['input_audios'].to(device, dtype=torch.bfloat16)
                    audio_lengths = audio_info['input_audio_lengths'].to(device)

                    # Get audio encoder output
                    audio_features = model.audio.audio_encoder(
                        input_audios,
                        audio_lengths
                    )

                    # Mean pooling over time dimension
                    embedding = audio_features.mean(dim=1).squeeze(0).float().cpu().numpy()
                else:
                    # Fallback: use hidden states from transformer
                    outputs = model.transformer(
                        input_ids=input_ids,
                        audio_info=audio_info,
                        output_hidden_states=True,
                    )
                    # Use last hidden state, mean pooled
                    hidden_states = outputs.last_hidden_state
                    embedding = hidden_states.mean(dim=1).squeeze(0).float().cpu().numpy()

                embeddings.append(embedding)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                # Add zero embedding for failed samples
                if len(embeddings) > 0:
                    embed_dim = embeddings[-1].shape[-1]
                else:
                    embed_dim = 1024  # Qwen-Audio's likely dimension
                embeddings.append(np.zeros(embed_dim))

    embeddings = np.array(embeddings)

    # Cache embeddings if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, files=np.array(audio_files, dtype=object))
        print(f"Cached embeddings to {cache_path}")

    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

    return embeddings


def compute_distances(voicebench_embeddings: np.ndarray, advbench_embeddings: np.ndarray,
                     metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minimum distance from each VoiceBench sample to all AdvBench samples.

    Args:
        voicebench_embeddings: (n_voicebench, dim)
        advbench_embeddings: (n_advbench, dim)
        metric: "cosine" or "euclidean"

    Returns:
        min_distances: (n_voicebench,) minimum distance for each VoiceBench sample
        all_distances: (n_voicebench, n_advbench) full distance matrix
    """
    if voicebench_embeddings.ndim == 1:
        voicebench_embeddings = voicebench_embeddings.reshape(1, -1)
    if advbench_embeddings.ndim == 1:
        advbench_embeddings = advbench_embeddings.reshape(1, -1)

    print(f"  VoiceBench embeddings shape: {voicebench_embeddings.shape}")
    print(f"  AdvBench embeddings shape: {advbench_embeddings.shape}")

    if metric == "cosine":
        # Normalize embeddings
        vb_norm = voicebench_embeddings / (np.linalg.norm(voicebench_embeddings, axis=1, keepdims=True) + 1e-8)
        ab_norm = advbench_embeddings / (np.linalg.norm(advbench_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity = vb_norm @ ab_norm.T

        # Convert to distance (1 - similarity)
        distances = 1 - similarity

    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        distances = cdist(voicebench_embeddings, advbench_embeddings, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get minimum distance to any AdvBench sample
    min_distances = distances.min(axis=1)

    return min_distances, distances


def filter_samples(
    voicebench_json: str,
    advbench_audio_dir: str,
    output_json: str,
    threshold: float = None,
    top_k: int = None,
    percentage: float = None,
    metric: str = "cosine",
    cache_dir: str = None,
    model_name: str = "Qwen/Qwen-Audio-Chat",
    filter_closest: bool = False,
):
    """
    Filter VoiceBench samples based on distance to AdvBench samples.

    By default, keeps samples FARTHEST from AdvBench (most benign).
    Use filter_closest=True to keep samples closest to AdvBench.
    """
    # Load VoiceBench data
    print(f"Loading VoiceBench data from {voicebench_json}")
    with open(voicebench_json, 'r') as f:
        voicebench_data = json.load(f)

    json_dir = Path(voicebench_json).parent

    voicebench_audio_files = []
    for sample in voicebench_data:
        audio_path = sample["audio"]
        if not os.path.isabs(audio_path):
            full_path = json_dir.parent.parent.parent / audio_path
            if not full_path.exists():
                full_path = Path(audio_path)
            audio_path = str(full_path)
        voicebench_audio_files.append(audio_path)

    print(f"Found {len(voicebench_audio_files)} VoiceBench samples")
    if len(voicebench_audio_files) > 0:
        print(f"First audio path: {voicebench_audio_files[0]}")

    # Load AdvBench audio files
    advbench_audio_dir = Path(advbench_audio_dir)
    advbench_audio_files = sorted(list(advbench_audio_dir.glob("*.mp3")) +
                                 list(advbench_audio_dir.glob("*.wav")))
    advbench_audio_files = [str(f) for f in advbench_audio_files]
    print(f"Found {len(advbench_audio_files)} AdvBench samples")

    # Set up cache paths
    vb_cache = None
    ab_cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vb_cache = str(cache_dir / "voicebench_embeddings_qwen.npz")
        ab_cache = str(cache_dir / "advbench_embeddings_qwen.npz")

    # Compute embeddings using Qwen-Audio
    print(f"\n=== Computing VoiceBench embeddings using Qwen-Audio encoder ===")
    voicebench_embeddings = compute_qwen_embeddings(
        voicebench_audio_files,
        model_name=model_name,
        cache_path=vb_cache
    )

    print(f"\n=== Computing AdvBench embeddings using Qwen-Audio encoder ===")
    advbench_embeddings = compute_qwen_embeddings(
        advbench_audio_files,
        model_name=model_name,
        cache_path=ab_cache
    )

    # Compute distances
    print(f"\n=== Computing {metric} distances ===")
    min_distances, all_distances = compute_distances(voicebench_embeddings, advbench_embeddings, metric)

    # Print statistics
    print(f"\nDistance statistics (VoiceBench to AdvBench):")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Std:  {min_distances.std():.4f}")
    print(f"  Min:  {min_distances.min():.4f}")
    print(f"  Max:  {min_distances.max():.4f}")
    print(f"  Median: {np.median(min_distances):.4f}")

    # Print percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(min_distances, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Filter samples
    # Convert percentage to top_k if provided
    if percentage is not None:
        top_k = int(len(voicebench_data) * percentage / 100)
        top_k = max(1, top_k)  # Ensure at least 1 sample
        print(f"\n=== Converting {percentage}% to top_k={top_k} samples ===")

    if top_k is not None:
        if filter_closest:
            sorted_indices = np.argsort(min_distances)[:top_k]
            print(f"\n=== Keeping top {top_k} CLOSEST samples ===")
        else:
            sorted_indices = np.argsort(min_distances)[-top_k:][::-1]
            print(f"\n=== Keeping top {top_k} FARTHEST samples (most benign) ===")

        filtered_indices = sorted_indices
        print(f"  Min distance in selected: {min_distances[sorted_indices].min():.4f}")
        print(f"  Max distance in selected: {min_distances[sorted_indices].max():.4f}")
        print(f"  Mean distance in selected: {min_distances[sorted_indices].mean():.4f}")

    elif threshold is not None:
        if filter_closest:
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} (closest) ===")
        else:
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (most benign) ===")

        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    else:
        threshold = np.percentile(min_distances, 75)
        filtered_indices = np.where(min_distances >= threshold)[0]
        print(f"\n=== Auto-selected threshold: {threshold:.4f} (75th percentile, most benign) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = [voicebench_data[i] for i in filtered_indices]

    # Add distance metadata
    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_advbench"] = float(min_distances[idx])
        closest_advbench_idx = np.argmin(all_distances[idx])
        filtered_data[i]["closest_advbench_file"] = advbench_audio_files[closest_advbench_idx]

    # Sort by distance
    filtered_data.sort(key=lambda x: x["min_distance_to_advbench"], reverse=not filter_closest)

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save distance analysis
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        voicebench_embeddings=voicebench_embeddings,
        advbench_embeddings=advbench_embeddings,
        voicebench_files=np.array(voicebench_audio_files, dtype=object),
        advbench_files=np.array(advbench_audio_files, dtype=object)
    )
    print(f"  Distance analysis: {analysis_path}")

    # Print some examples
    filter_type = "closest" if filter_closest else "farthest (most benign)"
    print(f"\n=== Top 10 {filter_type} VoiceBench samples ===")
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
    parser = argparse.ArgumentParser(description="Filter VoiceBench samples using Qwen-Audio embeddings")
    parser.add_argument("--voicebench_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_full.json",
                       help="Path to VoiceBench samples JSON file")
    parser.add_argument("--advbench_audio_dir", type=str,
                       default="data/advbench/en",
                       help="Directory containing AdvBench audio files")
    parser.add_argument("--output_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_filtered_qwen.json",
                       help="Path to save filtered VoiceBench samples")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold for filtering")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Keep top percentage of samples (e.g., 50 for 50%%)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--cache_dir", type=str, default="data/embedding_cache_qwen",
                       help="Directory to cache embeddings")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Audio-Chat",
                       help="Qwen-Audio model name")
    parser.add_argument("--filter_closest", action="store_true",
                       help="Keep samples closest to AdvBench (default: keep farthest/most benign)")

    args = parser.parse_args()

    filter_samples(
        voicebench_json=args.voicebench_json,
        advbench_audio_dir=args.advbench_audio_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        top_k=args.top_k,
        percentage=args.percentage,
        metric=args.metric,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        filter_closest=args.filter_closest,
    )


if __name__ == "__main__":
    main()
