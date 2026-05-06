#!/usr/bin/env python3
"""
=============================================================================
AUDIO-BASED ACOUSTIC FILTERING (TRUE ACOUSTIC FEATURES) - VoiceBench Noisy Variants
=============================================================================

Filter VoiceBench Noisy variant samples based on TRUE ACOUSTIC embedding distance to harmful audio.

Supports: voicebench_cafe, voicebench_traffic, etc.

Uses Whisper-Large-V3 encoder for CONTINUOUS ACOUSTIC features.

Usage:
    python 0_filter_voicebench_noise_acoustic.py \
        --noise_type cafe \
        --harmful_dir /path/to/advbench_gtts/en \
        --percentage 50 \
        --output data_acoustic_voicebench_cafe/voicebench_cafe_filtered_advbench_50.jsonl
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

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")


class KimiAcousticEmbeddingExtractor:
    """
    Extract TRUE ACOUSTIC embeddings using Whisper-Large-V3 encoder.

    This is the SAME encoder Kimi-Audio uses for continuous acoustic features,
    NOT the VQ encoder used for semantic tokens.

    The Whisper encoder captures:
    - Voice characteristics (timbre, pitch)
    - Prosody and intonation
    - Speaking style
    - Acoustic patterns

    Unlike the VQ encoder which captures semantic/linguistic content.
    """

    def __init__(self, model_path: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.device = device
        self.whisper_model = None
        self.feature_extractor = None
        self.hidden_dim = 1280  # Whisper-Large-V3 hidden dim
        self._load_models(model_path)

    def _load_models(self, model_path: str):
        from transformers import WhisperModel, WhisperFeatureExtractor

        print(f"Loading Whisper-Large-V3 for TRUE ACOUSTIC embeddings...")
        print(f"  Model: {model_path}")

        # Load the Whisper encoder (same as Kimi-Audio uses for continuous features)
        self.whisper_model = WhisperModel.from_pretrained(model_path)
        self.whisper_model = self.whisper_model.encoder  # Only need encoder
        self.whisper_model = self.whisper_model.to(self.device).to(torch.bfloat16)
        self.whisper_model.eval()

        # Load feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

        print(f"  Hidden dimension: {self.hidden_dim}")
        print("  Model loaded! Using TRUE ACOUSTIC features (Whisper encoder last_hidden_state)")

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract ACOUSTIC embedding for a single audio file."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Get mel features
        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = features.input_features.to(self.device).to(torch.bfloat16)

        # Forward through Whisper encoder
        outputs = self.whisper_model(
            input_features=input_features,
            return_dict=True
        )

        # Get last hidden state and mean pool
        # Shape: [batch, seq_len, 1280] -> [batch, 1280]
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


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of embeddings."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return similarity


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance (1 - similarity)."""
    similarity = compute_cosine_similarity(emb1, emb2)
    return 1 - similarity


def load_voicebench_noise_data(noise_type: str, json_path: str = None) -> List[Dict]:
    """Load VoiceBench Noisy variant dataset from JSONL file."""
    dataset_name = f"voicebench_{noise_type}"

    if json_path is None:
        json_path = str(BASE_DIR / f"Kimi-Audio/finetune_codes/data/{dataset_name}/{dataset_name}_full.jsonl")

    print(f"Loading {dataset_name} from {json_path}")

    voicebench_data = []

    # Support both JSON and JSONL formats
    if json_path.endswith('.jsonl'):
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    voicebench_data.append({
                        "id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "audio_path": item.get("audio", ""),
                        "region": item.get("region", ""),
                        "noise_type": item.get("noise_type", noise_type),
                        "snr_db": item.get("snr_db", 10),
                        "original_id": item.get("original_id", ""),
                    })
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            voicebench_data.append({
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "audio_path": item.get("audio", ""),
                "region": item.get("region", ""),
                "noise_type": item.get("noise_type", noise_type),
                "snr_db": item.get("snr_db", 10),
                "original_id": item.get("original_id", ""),
            })

    # Filter to only samples with existing audio files
    voicebench_data = [d for d in voicebench_data if os.path.exists(d["audio_path"])]
    print(f"Loaded {len(voicebench_data)} {dataset_name} samples with existing audio files")

    return voicebench_data


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(directory, f))
    return sorted(audio_files)


def filter_and_save(
    voicebench_data: List[Dict],
    voicebench_embeddings: np.ndarray,
    harmful_embeddings: np.ndarray,
    threshold: float,
    output_path: str,
    noise_type: str,
    percentage: float = None,
    num_samples: int = None,
):
    """Filter VoiceBench Noisy variant samples and save to JSONL."""
    dataset_name = f"voicebench_{noise_type}"

    # Compute distance
    print("Computing pairwise cosine distances...")
    distance = compute_cosine_distance(voicebench_embeddings, harmful_embeddings)
    min_distance = distance.min(axis=1)

    # Priority: num_samples > percentage > threshold
    if num_samples is not None:
        top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        top_k = int(len(voicebench_data) * percentage / 100)
        top_k = max(1, top_k)
        print(f"\n=== Converting {percentage}% to top_k={top_k} samples ===")
    else:
        top_k = None

    if top_k is not None:
        sorted_indices = np.argsort(min_distance)
        filtered_indices = sorted_indices[:top_k]
        mask = np.zeros(len(voicebench_data), dtype=bool)
        mask[filtered_indices] = True
        effective_threshold = min_distance[filtered_indices[-1]] if len(filtered_indices) > 0 else threshold
        print(f"  Effective threshold: {effective_threshold:.6f}")
    else:
        mask = min_distance <= threshold

    # Print stats
    print(f"\n{'='*60}")
    print(f"ACOUSTIC Distance Distribution (all {dataset_name} samples):")
    print(f"  Mean:   {min_distance.mean():.4f}")
    print(f"  Std:    {min_distance.std():.4f}")
    print(f"  Min:    {min_distance.min():.4f}")
    print(f"  Max:    {min_distance.max():.4f}")
    print(f"  Median: {np.median(min_distance):.4f}")
    print(f"{'='*60}")

    print(f"\n  Total {dataset_name} samples: {len(voicebench_data)}")
    print(f"  Samples close to harmful (kept): {mask.sum()}")
    print(f"  Samples far from harmful (removed): {(~mask).sum()}")

    # Create conversation format (Kimi-Audio format)
    conversations = []
    for i, (data, dist_val) in enumerate(zip(voicebench_data, min_distance)):
        if mask[i]:
            conv = {
                "task_type": "understanding",
                "conversation": [
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": "Please answer the following question based on the audio."
                    },
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": data["audio_path"]
                    },
                    {
                        "role": "assistant",
                        "message_type": "text",
                        "content": data["answer"]
                    }
                ],
                "_metadata": {
                    "min_acoustic_distance_to_harmful": float(dist_val),
                    "id": data.get("id", ""),
                    "region": data.get("region", ""),
                    "noise_type": data.get("noise_type", noise_type),
                    "snr_db": data.get("snr_db", 10),
                    "original_id": data.get("original_id", ""),
                }
            }
            conversations.append(conv)

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(conversations)} conversations to {output_path}")

    if mask.sum() > 0:
        kept_values = min_distance[mask]
        print(f"\nAcoustic distance statistics (kept samples):")
        print(f"  Mean: {kept_values.mean():.4f}")
        print(f"  Min:  {kept_values.min():.4f}")
        print(f"  Max:  {kept_values.max():.4f}")

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Filter VoiceBench Noisy variants by TRUE ACOUSTIC embedding distance (Whisper-Large-V3)"
    )
    parser.add_argument("--noise_type", type=str, required=True,
                        help="Noise type: cafe, traffic, etc.")
    parser.add_argument("--voicebench_json", type=str, default=None,
                        help="Path to VoiceBench Noisy variant JSONL file")
    parser.add_argument("--harmful_dir", type=str, required=True,
                        help="Directory containing harmful audio files")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache ACOUSTIC embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3",
                        help="Whisper model for acoustic embeddings")
    args = parser.parse_args()

    dataset_name = f"voicebench_{args.noise_type}"

    # Set defaults based on noise type
    if args.output is None:
        args.output = f"data_acoustic_{dataset_name}/{dataset_name}_filtered.jsonl"
    if args.cache_dir is None:
        args.cache_dir = f"embedding_cache_acoustic_{dataset_name}"

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize ACOUSTIC embedding extractor
    extractor = KimiAcousticEmbeddingExtractor(model_path=args.model, device=args.device)

    # Load data
    voicebench_data = load_voicebench_noise_data(args.noise_type, args.voicebench_json)
    voicebench_audio_paths = [d["audio_path"] for d in voicebench_data]

    harmful_audio_paths = get_audio_files(args.harmful_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Extract or load cached embeddings
    voicebench_cache = os.path.join(args.cache_dir, f"{dataset_name}_acoustic_embeddings.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_acoustic_embeddings_{Path(args.harmful_dir).name}.npy")

    if os.path.exists(voicebench_cache):
        print(f"Loading cached {dataset_name} ACOUSTIC embeddings from {voicebench_cache}")
        voicebench_embeddings = np.load(voicebench_cache)
    else:
        print(f"Extracting {dataset_name} ACOUSTIC embeddings...")
        voicebench_embeddings = extractor.extract_embeddings_batch(
            voicebench_audio_paths, desc=f"{dataset_name} (Acoustic)"
        )
        np.save(voicebench_cache, voicebench_embeddings)
        print(f"Cached {dataset_name} embeddings to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful ACOUSTIC embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful audio ACOUSTIC embeddings...")
        harmful_embeddings = extractor.extract_embeddings_batch(
            harmful_audio_paths, desc="Harmful (Acoustic)"
        )
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached harmful embeddings to {harmful_cache}")

    # Filter and save
    filter_and_save(
        voicebench_data,
        voicebench_embeddings,
        harmful_embeddings,
        args.threshold,
        args.output,
        args.noise_type,
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone! Used TRUE ACOUSTIC features from Whisper-Large-V3 encoder.")


if __name__ == "__main__":
    main()
