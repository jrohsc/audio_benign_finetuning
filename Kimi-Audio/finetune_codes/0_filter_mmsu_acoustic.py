#!/usr/bin/env python3
"""
Filter MMSU samples by TRUE ACOUSTIC embedding distance to harmful audio.
Uses Whisper-Large-V3 encoder for acoustic features.

Output: data_acoustic_mmsu/ folder
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KimiAcousticEmbeddingExtractor:
    """Extract TRUE ACOUSTIC embeddings using Whisper-Large-V3 encoder."""

    def __init__(self, model_path: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.device = device
        self.whisper_model = None
        self.feature_extractor = None
        self.hidden_dim = 1280
        self._load_models(model_path)

    def _load_models(self, model_path: str):
        from transformers import WhisperModel, WhisperFeatureExtractor

        print(f"Loading Whisper-Large-V3 for TRUE ACOUSTIC embeddings...")
        print(f"  Model: {model_path}")

        self.whisper_model = WhisperModel.from_pretrained(model_path)
        self.whisper_model = self.whisper_model.encoder
        self.whisper_model = self.whisper_model.to(self.device).to(torch.bfloat16)
        self.whisper_model.eval()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

        print(f"  Hidden dimension: {self.hidden_dim}")
        print("  Model loaded!")

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract ACOUSTIC embedding for a single audio file."""
        audio, sr = librosa.load(audio_path, sr=16000)

        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = features.input_features.to(self.device).to(torch.bfloat16)

        outputs = self.whisper_model(
            input_features=input_features,
            return_dict=True
        )

        last_hidden = outputs.last_hidden_state
        embedding = last_hidden.mean(dim=1)

        return embedding.cpu().float().numpy().squeeze()

    def extract_embeddings_batch(self, audio_paths: List[str], desc: str = "Extracting") -> np.ndarray:
        """Extract ACOUSTIC embeddings for a list of audio files."""
        embeddings = []
        for path in tqdm(audio_paths, desc=desc):
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                embeddings.append(np.zeros(self.hidden_dim))
        return np.stack(embeddings)


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return 1 - similarity


def load_mmsu_data(jsonl_path: str) -> List[Dict]:
    """Load MMSU dataset from JSONL file."""
    print(f"Loading MMSU dataset from {jsonl_path}...")

    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            # Extract audio path from conversation
            audio_path = None
            for msg in entry.get("conversation", []):
                if msg.get("message_type") == "audio":
                    audio_path = msg.get("content")
                    break

            if audio_path and os.path.exists(audio_path):
                data.append({
                    "entry": entry,
                    "audio_path": audio_path
                })

    print(f"Loaded {len(data)} samples with existing audio files")
    return data


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(directory, f))
    return sorted(audio_files)


def filter_and_save(
    data: List[Dict],
    embeddings: np.ndarray,
    harmful_embeddings: np.ndarray,
    threshold: float,
    output_path: str,
    percentage: float = None,
    num_samples: int = None,
):
    """Filter samples and save to JSONL."""

    print("Computing pairwise cosine distances...")
    distance = compute_cosine_distance(embeddings, harmful_embeddings)
    min_distance = distance.min(axis=1)

    # Priority: num_samples > percentage > threshold
    if num_samples is not None:
        top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        top_k = int(len(data) * percentage / 100)
        top_k = max(1, top_k)
        print(f"\n=== Converting {percentage}% to top_k={top_k} samples ===")
    else:
        top_k = None

    if top_k is not None:
        sorted_indices = np.argsort(min_distance)
        filtered_indices = sorted_indices[:top_k]
        mask = np.zeros(len(data), dtype=bool)
        mask[filtered_indices] = True
        effective_threshold = min_distance[filtered_indices[-1]] if len(filtered_indices) > 0 else threshold
        print(f"  Effective threshold: {effective_threshold:.6f}")
    else:
        mask = min_distance <= threshold

    # Print stats
    print(f"\n{'='*60}")
    print(f"ACOUSTIC Distance Distribution:")
    print(f"  Mean:   {min_distance.mean():.4f}")
    print(f"  Std:    {min_distance.std():.4f}")
    print(f"  Min:    {min_distance.min():.4f}")
    print(f"  Max:    {min_distance.max():.4f}")
    print(f"  Median: {np.median(min_distance):.4f}")
    print(f"{'='*60}")

    print(f"\n  Total samples: {len(data)}")
    print(f"  Samples kept: {mask.sum()}")
    print(f"  Samples removed: {(~mask).sum()}")

    # Save filtered data
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for i, (item, dist_val) in enumerate(zip(data, min_distance)):
            if mask[i]:
                entry = item["entry"].copy()
                if "_metadata" not in entry:
                    entry["_metadata"] = {}
                entry["_metadata"]["min_acoustic_distance_to_harmful"] = float(dist_val)
                # Fix text instruction to use generic format (like VoiceBench)
                # This ensures model learns to extract question FROM audio
                for msg in entry.get("conversation", []):
                    if msg.get("message_type") == "text" and msg.get("role") == "user":
                        msg["content"] = "Please answer the following question based on the audio."
                        break
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"\nSaved {count} samples to {output_path}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Filter MMSU by TRUE ACOUSTIC embedding distance"
    )
    parser.add_argument("--mmsu_jsonl", type=str, default="data_mmsu/mmsu_full.jsonl",
                        help="Path to MMSU JSONL file")
    parser.add_argument("--harmful_dir", type=str, required=True,
                        help="Directory containing harmful audio files")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Distance threshold")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--output", type=str, default="data_acoustic_mmsu/mmsu_filtered.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_acoustic_mmsu",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3",
                        help="Whisper model for acoustic embeddings")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize extractor
    extractor = KimiAcousticEmbeddingExtractor(model_path=args.model, device=args.device)

    # Load data
    data = load_mmsu_data(args.mmsu_jsonl)
    audio_paths = [d["audio_path"] for d in data]

    harmful_audio_paths = get_audio_files(args.harmful_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Extract or load cached embeddings
    mmsu_cache = os.path.join(args.cache_dir, "mmsu_acoustic_embeddings.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_acoustic_embeddings_{Path(args.harmful_dir).name}.npy")

    if os.path.exists(mmsu_cache):
        print(f"Loading cached MMSU embeddings from {mmsu_cache}")
        embeddings = np.load(mmsu_cache)
    else:
        print("Extracting MMSU ACOUSTIC embeddings...")
        embeddings = extractor.extract_embeddings_batch(audio_paths, desc="MMSU")
        np.save(mmsu_cache, embeddings)
        print(f"Cached embeddings to {mmsu_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful audio embeddings...")
        harmful_embeddings = extractor.extract_embeddings_batch(harmful_audio_paths, desc="Harmful")
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached harmful embeddings to {harmful_cache}")

    # Filter and save
    filter_and_save(
        data,
        embeddings,
        harmful_embeddings,
        args.threshold,
        args.output,
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
