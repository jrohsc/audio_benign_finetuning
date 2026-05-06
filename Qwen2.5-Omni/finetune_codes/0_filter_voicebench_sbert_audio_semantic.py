#!/usr/bin/env python3
"""
Filter VoiceBench samples based on TEXT-SEMANTIC embedding distance to harmful prompts.

This script uses sentence-transformers (all-MiniLM-L6-v2) to compute semantic
similarity between benign text questions and harmful text prompts.

Unlike audio_acoustic (WavLM) and audio_semantic (Whisper) filtering which operate
on audio signals, this filters based on the TEXT MEANING of the questions.

Usage:
    python 0_filter_voicebench_text_semantic.py \
        --voicebench_json ../../audio-flamingo/data/voicebench/sd-qa/sd_qa_full.json \
        --harmful_texts_csv ../../harmful_data/advbench.csv \
        --percentage 50 \
        --output_json data_semantic/filtered_voicebench/voicebench_filtered_closest_percentage_50.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


class TextSemanticExtractor:
    """Extract text semantic embeddings using sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL, device: str = "cuda"):
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

    def extract_embeddings(self, texts: List[str], batch_size: int = 64, desc: str = "Extracting") -> np.ndarray:
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


def load_voicebench_json(json_path: str, audio_base_dir: str = None) -> Tuple[List[Dict], List[str], List[str]]:
    """Load VoiceBench JSON and extract audio paths + text questions.

    Returns:
        data: Original JSON data
        audio_paths: Resolved audio paths
        questions: Text questions for embedding
    """
    from collections import defaultdict

    print(f"Loading VoiceBench data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    audio_paths = []
    questions = []

    first_audio = data[0].get('audio', data[0].get('audio_path', '')) if data else ''
    has_absolute_paths = first_audio and os.path.isabs(first_audio)

    if has_absolute_paths:
        print(f"  Using absolute audio paths from JSON")
        for item in data:
            audio_paths.append(item.get('audio', item.get('audio_path', '')))
    elif audio_base_dir:
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
        json_dir = Path(json_path).parent
        for item in data:
            audio_path = item.get('audio', item.get('audio_path', ''))
            if audio_path and not os.path.isabs(audio_path):
                audio_path = str(json_dir / audio_path)
            audio_paths.append(audio_path)

    # Extract text questions from conversations
    for item in data:
        question = ""
        if "conversations" in item:
            for conv in item["conversations"]:
                role = conv.get("from", conv.get("role", ""))
                if role in ("human", "user"):
                    # Extract text from conversation content
                    value = conv.get("value", "")
                    content = conv.get("content", "")
                    if value:
                        # ShareGPT format: "from"/"value" with <audio> tag prefix
                        question = value.replace("<audio>\n", "").replace("<audio>", "").strip()
                    elif isinstance(content, list):
                        # Multi-modal content format
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                question = part.get("text", "")
                                break
                    elif isinstance(content, str) and content:
                        question = content.replace("<audio>\n", "").replace("<audio>", "").strip()
                    break
        elif "question" in item:
            question = item["question"]
        elif "prompt" in item:
            question = item["prompt"]
        questions.append(question)

    print(f"  Loaded {len(data)} samples")
    existing = sum(1 for p in audio_paths if os.path.exists(p))
    print(f"  Found {existing}/{len(audio_paths)} existing audio files")
    non_empty_q = sum(1 for q in questions if q.strip())
    print(f"  Extracted {non_empty_q}/{len(questions)} non-empty text questions")

    if existing < len(audio_paths):
        for p in audio_paths:
            if not os.path.exists(p):
                print(f"  First missing: {p}")
                break

    return data, audio_paths, questions


def load_harmful_texts(csv_path: str, text_column: str = "goal") -> List[str]:
    """Load harmful texts from CSV file."""
    import pandas as pd

    print(f"Loading harmful texts from {csv_path}")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{text_column}' not found in CSV")

    texts = df[text_column].dropna().tolist()
    print(f"  Found {len(texts)} harmful texts")
    return texts


def filter_by_text_semantic_embedding(
    voicebench_json: str,
    harmful_texts_csv: str,
    output_json: str,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    threshold: float = None,
    percentage: float = None,
    num_samples: int = None,
    cache_dir: str = "data_semantic/embedding_cache",
    device: str = "cuda",
    select_safest: bool = False,
    audio_base_dir: str = None,
    text_column: str = "goal",
):
    """
    Filter VoiceBench samples by text-semantic embedding distance to harmful text prompts.
    """
    os.makedirs(cache_dir, exist_ok=True)

    extractor = TextSemanticExtractor(model_name=sbert_model, device=device)

    voicebench_data, voicebench_audio_paths, voicebench_questions = load_voicebench_json(
        voicebench_json, audio_base_dir=audio_base_dir
    )

    harmful_texts = load_harmful_texts(harmful_texts_csv, text_column)

    # Filter out samples with empty questions
    valid_indices = [i for i, q in enumerate(voicebench_questions) if q.strip()]
    valid_questions = [voicebench_questions[i] for i in valid_indices]
    print(f"\nUsing {len(valid_questions)}/{len(voicebench_questions)} samples with non-empty questions")

    # Cache paths
    model_tag = sbert_model.replace("/", "_").replace("-", "_")
    harmful_tag = Path(harmful_texts_csv).stem
    voicebench_cache = os.path.join(cache_dir, f"voicebench_text_semantic_{model_tag}.npz")
    harmful_cache = os.path.join(cache_dir, f"harmful_text_semantic_{model_tag}_{harmful_tag}.npz")

    # Extract or load cached embeddings
    if os.path.exists(voicebench_cache):
        print(f"\nLoading cached VoiceBench text embeddings from {voicebench_cache}")
        cached = np.load(voicebench_cache)
        voicebench_embeddings = cached['embeddings']
        cached_valid_indices = cached['valid_indices'].tolist()
        # Verify cache matches current data
        if len(cached_valid_indices) != len(valid_indices):
            print(f"  WARNING: Cache size mismatch ({len(cached_valid_indices)} vs {len(valid_indices)}), recomputing...")
            voicebench_embeddings = extractor.extract_embeddings(valid_questions, desc="VoiceBench texts")
            np.savez(voicebench_cache, embeddings=voicebench_embeddings, valid_indices=np.array(valid_indices))
    else:
        print("\nExtracting VoiceBench text embeddings...")
        voicebench_embeddings = extractor.extract_embeddings(valid_questions, desc="VoiceBench texts")
        np.savez(voicebench_cache, embeddings=voicebench_embeddings, valid_indices=np.array(valid_indices))
        print(f"Cached to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful text embeddings from {harmful_cache}")
        cached = np.load(harmful_cache)
        harmful_embeddings = cached['embeddings']
    else:
        print("\nExtracting harmful text embeddings...")
        harmful_embeddings = extractor.extract_embeddings(harmful_texts, desc="Harmful texts")
        np.savez(harmful_cache, embeddings=harmful_embeddings)
        print(f"Cached to {harmful_cache}")

    print(f"\nEmbedding shapes:")
    print(f"  VoiceBench: {voicebench_embeddings.shape} (from {len(voicebench_data)} total)")
    print(f"  Harmful: {harmful_embeddings.shape}")

    # Compute distances
    print("\nComputing cosine distances...")
    all_distances = compute_cosine_distance(voicebench_embeddings, harmful_embeddings)
    min_distances = all_distances.min(axis=1)
    closest_harmful_idx = all_distances.argmin(axis=1)

    print(f"\n{'='*60}")
    print("TEXT-SEMANTIC Distance Statistics (sentence-transformers)")
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
        effective_top_k = int(len(valid_indices) * percentage / 100)
        effective_top_k = max(1, effective_top_k)
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    # Filter
    if select_safest:
        sorted_indices = np.argsort(min_distances)[::-1]
        selection_label = "SAFEST (furthest from harmful)"
    else:
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
        print(f"  Kept {len(filtered_local_indices)} / {len(valid_indices)} samples")
    else:
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_local_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (75th percentile, safest) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_local_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.6f} (25th percentile) ===")
        print(f"  Kept {len(filtered_local_indices)} / {len(valid_indices)} samples")

    # Map back to original indices
    filtered_original_indices = [valid_indices[i] for i in filtered_local_indices]

    print(f"  Selected: {len(filtered_original_indices)} / {len(voicebench_data)} ({100*len(filtered_original_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset (same format as audio_acoustic filter output)
    filtered_data = []
    for local_idx in filtered_local_indices:
        original_idx = valid_indices[local_idx]
        item = voicebench_data[original_idx].copy()
        item['min_text_semantic_distance'] = float(min_distances[local_idx])
        h_idx = closest_harmful_idx[local_idx]
        if h_idx < len(harmful_texts):
            item['closest_harmful_text'] = harmful_texts[h_idx]
        filtered_data.append(item)

    filtered_data.sort(key=lambda x: x['min_text_semantic_distance'], reverse=select_safest)

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
        valid_indices=np.array(valid_indices),
        voicebench_embeddings=voicebench_embeddings,
        harmful_embeddings=harmful_embeddings,
        closest_harmful_idx=closest_harmful_idx
    )
    print(f"Saved analysis to {analysis_path}")

    # Print examples
    sample_label = "Safest (Furthest)" if select_safest else "Closest"
    print(f"\n=== Top 10 {sample_label} Samples (Text-Semantic / sentence-BERT) ===")
    for i in range(min(10, len(filtered_data))):
        item = filtered_data[i]
        dist = item['min_text_semantic_distance']
        question = voicebench_questions[filtered_original_indices[i]] if i < len(filtered_original_indices) else "N/A"
        harmful = item.get('closest_harmful_text', 'N/A')
        print(f"\n{i+1}. Distance: {dist:.4f}")
        print(f"   Question: {question[:80]}")
        print(f"   Closest harmful: {harmful[:80]}")

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter VoiceBench by TEXT-SEMANTIC embedding similarity using sentence-transformers"
    )
    parser.add_argument("--voicebench_json", type=str,
                        default="../../audio-flamingo/data/voicebench/sd-qa/sd_qa_full.json",
                        help="Path to VoiceBench JSON file")
    parser.add_argument("--harmful_texts_csv", type=str,
                        default="../../harmful_data/advbench.csv",
                        help="CSV file with harmful texts")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--sbert_model", type=str, default=DEFAULT_SBERT_MODEL,
                        help="Sentence-transformer model name")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Distance threshold")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples (e.g., 50 for 50%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--cache_dir", type=str, default="data_semantic/embedding_cache",
                        help="Directory to cache embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--audio_base_dir", type=str, default=None,
                        help="Base directory for resolving relative audio paths")
    parser.add_argument("--select_safest", action="store_true",
                        help="Select samples FURTHEST from harmful instead of closest")
    parser.add_argument("--text_column", type=str, default="goal",
                        help="Column name in CSV for harmful texts")

    args = parser.parse_args()

    filter_by_text_semantic_embedding(
        voicebench_json=args.voicebench_json,
        harmful_texts_csv=args.harmful_texts_csv,
        output_json=args.output_json,
        sbert_model=args.sbert_model,
        threshold=args.threshold,
        percentage=args.percentage,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        device=args.device,
        select_safest=args.select_safest,
        audio_base_dir=args.audio_base_dir,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
