#!/usr/bin/env python3
"""
=============================================================================
TEXT-BASED SEMANTIC FILTERING
=============================================================================

Filter VoiceBench samples based on SEMANTIC embedding distance to harmful prompts.

This script uses sentence-transformers to extract text embeddings that capture
the MEANING of the spoken content.

IMPORTANT DISTINCTION - Three filtering approaches:
1. TEXT-BASED SEMANTIC (this script):
   - Input: Text transcriptions/questions
   - Model: sentence-transformers (all-MiniLM-L6-v2)
   - Captures: Text meaning similarity

2. AUDIO-BASED SEMANTIC (0_filter_voicebench_audio_semantic.py):
   - Input: Audio files
   - Model: GLM-4 Voice Tokenizer (WhisperVQEncoder)
   - Captures: Semantic content from audio (what is said)

3. AUDIO-BASED ACOUSTIC (0_filter_voicebench_acoustic.py):
   - Input: Audio files
   - Model: Whisper-Large-V3 encoder
   - Captures: Acoustic features (how it sounds)

Usage:
    python 0_filter_voicebench_text_semantic.py \
        --harmful_texts_csv ../../harmful_data/advbench.csv \
        --percentage 10 \
        --output data_semantic/voicebench_filtered_advbench_10.jsonl
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
from datasets import load_dataset

# Paths
VOICEBENCH_AUDIO_DIR = "/work/anon/audio_benign_finetuning/shared_audio/voicebench"


class SemanticEmbeddingExtractor:
    """Extract SEMANTIC embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("ERROR: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)

        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.hidden_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.hidden_dim}")

    def extract_embeddings(self, texts: List[str], batch_size: int = 32, desc: str = "Extracting") -> np.ndarray:
        """Extract embeddings for a list of texts."""
        print(f"{desc}: {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance (1 - similarity)."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return 1 - similarity


def load_voicebench_data() -> List[Dict]:
    """Load VoiceBench dataset from HuggingFace."""
    print("Loading VoiceBench dataset...")

    voicebench_data = []
    subsets = [
        ("sd-qa", ["aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl", "phl", "usa", "zaf"]),
    ]

    for subset_name, splits in subsets:
        try:
            ds = load_dataset("hlt-lab/voicebench", subset_name)
            for split in splits:
                if split in ds:
                    split_ds = ds[split].remove_columns(["audio"])
                    for idx, item in enumerate(split_ds):
                        voicebench_data.append({
                            "subset": subset_name,
                            "split": split,
                            "question": item.get("prompt", ""),
                            "answer": item.get("reference", ""),
                            "audio_path": os.path.join(VOICEBENCH_AUDIO_DIR, f"{split}_{idx}.wav")
                        })
        except Exception as e:
            print(f"Error loading {subset_name}: {e}")

    # Filter to only include files that exist
    voicebench_data = [d for d in voicebench_data if os.path.exists(d["audio_path"])]
    print(f"Loaded {len(voicebench_data)} VoiceBench samples with existing audio files")

    return voicebench_data


def load_harmful_texts(csv_path: str, text_column: str = "goal") -> List[str]:
    """Load harmful texts from CSV file."""
    print(f"Loading harmful texts from {csv_path}")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{text_column}' not found in CSV")

    texts = df[text_column].dropna().tolist()
    print(f"  Found {len(texts)} harmful texts")
    return texts


def filter_by_semantic_embedding(
    harmful_texts_csv: str,
    output_path: str,
    threshold: float = None,
    percentage: float = None,
    num_samples: int = None,
    cache_dir: str = "embedding_cache_semantic_text",
    device: str = "cuda",
    text_column: str = "goal",
    model_name: str = "all-MiniLM-L6-v2",
):
    """
    Filter VoiceBench samples by semantic (text) embedding distance to harmful prompts.

    Args:
        harmful_texts_csv: CSV file with harmful texts
        output_path: Output JSONL file
        threshold: Distance threshold (keep samples with distance <= threshold)
        percentage: Keep top percentage of closest samples
        num_samples: Keep exact number of samples
        cache_dir: Directory to cache embeddings
        device: cuda or cpu
        text_column: Column name in CSV for harmful texts
        model_name: Sentence transformer model name
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize extractor
    extractor = SemanticEmbeddingExtractor(model_name=model_name, device=device)

    # Load VoiceBench data
    voicebench_data = load_voicebench_data()
    voicebench_texts = [d["question"] for d in voicebench_data]

    # Load harmful texts
    harmful_texts = load_harmful_texts(harmful_texts_csv, text_column)

    # Cache paths
    model_suffix = model_name.replace("/", "_")
    voicebench_cache = os.path.join(cache_dir, f"voicebench_semantic_{model_suffix}.npy")
    harmful_cache = os.path.join(cache_dir, f"harmful_semantic_{model_suffix}_{Path(harmful_texts_csv).stem}.npy")

    # Extract or load cached embeddings
    if os.path.exists(voicebench_cache):
        print(f"\nLoading cached VoiceBench embeddings from {voicebench_cache}")
        voicebench_embeddings = np.load(voicebench_cache)
    else:
        print("\nExtracting VoiceBench text embeddings...")
        voicebench_embeddings = extractor.extract_embeddings(voicebench_texts, desc="VoiceBench")
        np.save(voicebench_cache, voicebench_embeddings)
        print(f"Cached to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("\nExtracting harmful text embeddings...")
        harmful_embeddings = extractor.extract_embeddings(harmful_texts, desc="Harmful")
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached to {harmful_cache}")

    print(f"\nEmbedding shapes:")
    print(f"  VoiceBench: {voicebench_embeddings.shape}")
    print(f"  Harmful: {harmful_embeddings.shape}")

    # Compute distances
    print("\nComputing cosine distances...")
    all_distances = compute_cosine_distance(voicebench_embeddings, harmful_embeddings)
    min_distances = all_distances.min(axis=1)
    closest_harmful_idx = all_distances.argmin(axis=1)

    # Print statistics
    print(f"\n{'='*60}")
    print("SEMANTIC Distance Statistics")
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
        effective_top_k = int(len(voicebench_data) * percentage / 100)
        effective_top_k = max(1, effective_top_k)
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    # Filter - keep closest (smallest distance)
    sorted_indices = np.argsort(min_distances)

    if effective_top_k is not None:
        filtered_indices = sorted_indices[:effective_top_k]
        print(f"\n=== Keeping top {effective_top_k} semantically closest samples ===")
        print(f"  Min distance in selected: {min_distances[filtered_indices].min():.6f}")
        print(f"  Max distance in selected: {min_distances[filtered_indices].max():.6f}")
        print(f"  Mean distance in selected: {min_distances[filtered_indices].mean():.6f}")
    elif threshold is not None:
        filtered_indices = np.where(min_distances <= threshold)[0]
        print(f"\n=== Filtering with threshold <= {threshold} ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples")
    else:
        # Auto-select using 25th percentile
        threshold = np.percentile(min_distances, 25)
        filtered_indices = np.where(min_distances <= threshold)[0]
        print(f"\n=== Auto-selected threshold: {threshold:.6f} (25th percentile) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples")

    print(f"  Selected: {len(filtered_indices)} / {len(voicebench_data)} ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create output in conversation format
    conversations = []
    for idx in filtered_indices:
        data = voicebench_data[idx]
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
                "min_semantic_distance_to_harmful": float(min_distances[idx]),
                "closest_harmful_idx": int(closest_harmful_idx[idx]),
                "closest_harmful_text": harmful_texts[closest_harmful_idx[idx]][:100],
                "question": data["question"],
                "subset": data.get("subset", ""),
                "split": data.get("split", "")
            }
        }
        conversations.append(conv)

    # Sort by distance (closest first)
    conversations.sort(key=lambda x: x["_metadata"]["min_semantic_distance_to_harmful"])

    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(conversations)} conversations to {output_path}")

    # Save analysis
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        voicebench_embeddings=voicebench_embeddings,
        harmful_embeddings=harmful_embeddings,
        closest_harmful_idx=closest_harmful_idx
    )
    print(f"Saved analysis to {analysis_path}")

    # Print examples
    print(f"\n=== Top 10 Semantically Closest Samples ===")
    for i in range(min(10, len(conversations))):
        meta = conversations[i]["_metadata"]
        print(f"\n{i+1}. Distance: {meta['min_semantic_distance_to_harmful']:.4f}")
        print(f"   Benign Q: {meta['question'][:60]}...")
        print(f"   Closest harmful: {meta['closest_harmful_text'][:60]}...")

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Filter VoiceBench by SEMANTIC text embedding similarity")
    parser.add_argument("--harmful_texts_csv", type=str, required=True,
                        help="CSV file with harmful texts (e.g., advbench.csv)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples (e.g., 10 for 10%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples (overrides percentage)")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_semantic_text",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--text_column", type=str, default="goal",
                        help="Column name in CSV for harmful texts")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")

    args = parser.parse_args()

    filter_by_semantic_embedding(
        harmful_texts_csv=args.harmful_texts_csv,
        output_path=args.output,
        threshold=args.threshold,
        percentage=args.percentage,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        device=args.device,
        text_column=args.text_column,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
