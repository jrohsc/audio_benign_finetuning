#!/usr/bin/env python3
"""
Filter VoiceBench samples based on SEMANTIC (text) embedding distance to harmful prompts.

This script uses sentence-transformers to extract text embeddings that capture
the MEANING of the spoken content, independent of audio characteristics.

DIFFERENCE FROM ACOUSTIC FILTERING (0_filter_closest_to_advbench.py):
- ACOUSTIC filtering: selects samples with similar audio characteristics (voice, tone)
- SEMANTIC filtering (this script): selects samples with similar text meaning

Usage:
    python 0_filter_voicebench_semantic.py \
        --harmful_texts_csv ../harmful_data/advbench.csv \
        --percentage 10 \
        --output_json data_semantic/voicebench/sd-qa/sd_qa_semantic_percentage_10.json
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


def load_voicebench_json(json_path: str) -> Tuple[List[Dict], List[str]]:
    """Load VoiceBench JSON and extract question texts."""
    print(f"Loading VoiceBench data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    texts = []
    for item in data:
        # Extract question text from conversations
        text = ""
        if 'conversations' in item:
            for conv in item['conversations']:
                if conv.get('from') == 'human':
                    # Remove <audio> tag
                    text = conv.get('value', '').replace('<audio>\n', '').replace('<audio>', '').strip()
                    break
        texts.append(text)

    print(f"  Loaded {len(data)} samples")
    return data, texts


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
    voicebench_json: str,
    harmful_texts_csv: str,
    output_json: str,
    threshold: float = None,
    percentage: float = None,
    num_samples: int = None,
    cache_dir: str = "data_semantic/embedding_cache",
    device: str = "cuda",
    text_column: str = "goal",
    model_name: str = "all-MiniLM-L6-v2",
    select_safest: bool = False,
):
    """
    Filter VoiceBench samples by semantic (text) embedding distance to harmful prompts.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize extractor
    extractor = SemanticEmbeddingExtractor(model_name=model_name, device=device)

    # Load VoiceBench data
    voicebench_data, voicebench_texts = load_voicebench_json(voicebench_json)

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

    # Selection mode
    selection_mode = "SAFEST (furthest from harmful)" if select_safest else "CLOSEST to harmful"
    print(f"\n=== Selection mode: {selection_mode} ===")

    # Determine filtering
    effective_top_k = None
    if num_samples is not None:
        effective_top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        effective_top_k = int(len(voicebench_data) * percentage / 100)
        effective_top_k = max(1, effective_top_k)
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    # Filter - sort by distance
    sorted_indices = np.argsort(min_distances)[::-1] if select_safest else np.argsort(min_distances)

    if effective_top_k is not None:
        filtered_indices = sorted_indices[:effective_top_k]
        print(f"\n=== Keeping top {effective_top_k} semantically {'safest' if select_safest else 'closest'} samples ===")
        print(f"  Min distance in selected: {min_distances[filtered_indices].min():.6f}")
        print(f"  Max distance in selected: {min_distances[filtered_indices].max():.6f}")
        print(f"  Mean distance in selected: {min_distances[filtered_indices].mean():.6f}")
    elif threshold is not None:
        filtered_indices = np.where(min_distances >= threshold)[0] if select_safest else np.where(min_distances <= threshold)[0]
        print(f"\n=== Filtering with threshold {'>=' if select_safest else '<='} {threshold} ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples")
    else:
        # Auto-select using percentile
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (75th percentile) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (25th percentile) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples")

    print(f"  Selected: {len(filtered_indices)} / {len(voicebench_data)} ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset (same format as original)
    filtered_data = []
    for idx in filtered_indices:
        item = voicebench_data[idx].copy()
        item['min_semantic_distance_to_harmful'] = float(min_distances[idx])
        item['closest_harmful_idx'] = int(closest_harmful_idx[idx])
        item['closest_harmful_text'] = harmful_texts[closest_harmful_idx[idx]][:200]
        filtered_data.append(item)

    # Sort by distance
    filtered_data.sort(key=lambda x: x['min_semantic_distance_to_harmful'], reverse=select_safest)

    # Save filtered JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\nSaved {len(filtered_data)} samples to {output_json}")

    # Extract audio paths for compatibility with 1_prepare_filtered_dataset.py
    voicebench_files = np.array([item.get('audio', '') for item in voicebench_data])
    # Use harmful texts as "advbench_files" for compatibility
    advbench_files = np.array(harmful_texts)

    # Save analysis (compatible with 1_prepare_filtered_dataset.py)
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        voicebench_files=voicebench_files,  # Required by 1_prepare_filtered_dataset.py
        advbench_files=advbench_files,       # Required by 1_prepare_filtered_dataset.py
        voicebench_embeddings=voicebench_embeddings,
        harmful_embeddings=harmful_embeddings,
        closest_harmful_idx=closest_harmful_idx
    )
    print(f"Saved analysis to {analysis_path}")

    # Print examples
    print(f"\n=== Top 10 Semantically {'Safest (Furthest)' if select_safest else 'Closest'} Samples ===")
    for i in range(min(10, len(filtered_data))):
        item = filtered_data[i]
        dist = item['min_semantic_distance_to_harmful']
        question = voicebench_texts[filtered_indices[i]][:60]
        harmful = item['closest_harmful_text'][:60]
        print(f"\n{i+1}. Distance: {dist:.4f}")
        print(f"   Benign Q: {question}...")
        print(f"   Closest harmful: {harmful}...")

    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Filter VoiceBench by SEMANTIC text embedding similarity")
    parser.add_argument("--voicebench_json", type=str,
                        default="data/voicebench/sd-qa/sd_qa_full.json",
                        help="Path to VoiceBench JSON file")
    parser.add_argument("--harmful_texts_csv", type=str,
                        default="../harmful_data/advbench.csv",
                        help="CSV file with harmful texts (e.g., advbench.csv)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples (e.g., 10 for 10%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples (overrides percentage)")
    parser.add_argument("--cache_dir", type=str, default="data_semantic/embedding_cache",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--text_column", type=str, default="goal",
                        help="Column name in CSV for harmful texts")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--select_safest", action="store_true",
                        help="Select samples FURTHEST from harmful (safest benign samples) instead of closest to harmful (default behavior)")

    args = parser.parse_args()

    filter_by_semantic_embedding(
        voicebench_json=args.voicebench_json,
        harmful_texts_csv=args.harmful_texts_csv,
        output_json=args.output_json,
        threshold=args.threshold,
        percentage=args.percentage,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        device=args.device,
        text_column=args.text_column,
        model_name=args.model,
        select_safest=args.select_safest,
    )


if __name__ == "__main__":
    main()
