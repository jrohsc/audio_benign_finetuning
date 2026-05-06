#!/usr/bin/env python3
"""
Filter Spoken-SQuAD samples by TEXT SEMANTIC embedding distance to harmful texts.
Uses sentence-transformers for text embeddings.

Output: data_semantic_spoken_squad/ folder
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict


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
    """Compute pairwise cosine distance."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return 1 - similarity


def load_spoken_squad_data(jsonl_path: str) -> List[Dict]:
    """Load Spoken-SQuAD dataset from JSONL file."""
    print(f"Loading Spoken-SQuAD dataset from {jsonl_path}...")

    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            # Extract question/text and audio path
            metadata = entry.get("_metadata", {})
            question = metadata.get("question", "")

            # Get audio path from conversation
            audio_path = None
            for msg in entry.get("conversation", []):
                if msg.get("message_type") == "audio":
                    audio_path = msg.get("content")
                    break

            if audio_path and os.path.exists(audio_path):
                data.append({
                    "entry": entry,
                    "audio_path": audio_path,
                    "text": question  # The question text for semantic comparison
                })

    print(f"Loaded {len(data)} samples with existing audio files")
    return data


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


def filter_and_save(
    data: List[Dict],
    embeddings: np.ndarray,
    harmful_embeddings: np.ndarray,
    harmful_texts: List[str],
    output_path: str,
    threshold: float = None,
    percentage: float = None,
    num_samples: int = None,
):
    """Filter samples and save to JSONL."""

    print("Computing pairwise cosine distances...")
    distance = compute_cosine_distance(embeddings, harmful_embeddings)
    min_distance = distance.min(axis=1)
    closest_harmful_idx = distance.argmin(axis=1)

    # Print stats
    print(f"\n{'='*60}")
    print("SEMANTIC Distance Statistics")
    print(f"{'='*60}")
    print(f"  Mean:   {min_distance.mean():.6f}")
    print(f"  Std:    {min_distance.std():.6f}")
    print(f"  Min:    {min_distance.min():.6f}")
    print(f"  Max:    {min_distance.max():.6f}")
    print(f"  Median: {np.median(min_distance):.6f}")

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
        print(f"  Min distance in selected: {min_distance[filtered_indices].min():.6f}")
        print(f"  Max distance in selected: {min_distance[filtered_indices].max():.6f}")
    elif threshold is not None:
        mask = min_distance <= threshold
        print(f"\n=== Filtering with threshold <= {threshold} ===")
    else:
        # Auto-select using 25th percentile
        threshold = np.percentile(min_distance, 25)
        mask = min_distance <= threshold
        print(f"\n=== Auto-selected threshold: {threshold:.6f} (25th percentile) ===")

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
                entry["_metadata"]["min_semantic_distance_to_harmful"] = float(dist_val)
                entry["_metadata"]["closest_harmful_idx"] = int(closest_harmful_idx[i])
                entry["_metadata"]["closest_harmful_text"] = harmful_texts[closest_harmful_idx[i]][:100]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"\nSaved {count} samples to {output_path}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Filter Spoken-SQuAD by TEXT SEMANTIC embedding distance"
    )
    parser.add_argument("--spoken_squad_jsonl", type=str, default="data_spoken_squad/spoken_squad_full.jsonl",
                        help="Path to Spoken-SQuAD JSONL file")
    parser.add_argument("--harmful_texts_csv", type=str, required=True,
                        help="CSV file with harmful texts")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--output", type=str, default="data_semantic_spoken_squad/spoken_squad_filtered.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_semantic_spoken_squad",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--text_column", type=str, default="goal",
                        help="Column name in CSV for harmful texts")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize extractor
    extractor = SemanticEmbeddingExtractor(model_name=args.model, device=args.device)

    # Load data
    data = load_spoken_squad_data(args.spoken_squad_jsonl)
    texts = [d["text"] for d in data]

    # Load harmful texts
    harmful_texts = load_harmful_texts(args.harmful_texts_csv, args.text_column)

    # Cache paths
    model_suffix = args.model.replace("/", "_")
    data_cache = os.path.join(args.cache_dir, f"spoken_squad_semantic_{model_suffix}.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_semantic_{model_suffix}_{Path(args.harmful_texts_csv).stem}.npy")

    # Extract or load cached embeddings
    if os.path.exists(data_cache):
        print(f"Loading cached embeddings from {data_cache}")
        embeddings = np.load(data_cache)
    else:
        print("Extracting Spoken-SQuAD text embeddings...")
        embeddings = extractor.extract_embeddings(texts, desc="Spoken-SQuAD")
        np.save(data_cache, embeddings)
        print(f"Cached to {data_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful text embeddings...")
        harmful_embeddings = extractor.extract_embeddings(harmful_texts, desc="Harmful")
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached to {harmful_cache}")

    # Filter and save
    filter_and_save(
        data,
        embeddings,
        harmful_embeddings,
        harmful_texts,
        args.output,
        threshold=args.threshold,
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
