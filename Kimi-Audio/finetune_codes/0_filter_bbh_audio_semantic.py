#!/usr/bin/env python3
"""
=============================================================================
AUDIO-BASED SEMANTIC FILTERING - BBH (BIG-Bench Hard)
=============================================================================

Filter BBH samples based on SEMANTIC embedding distance to harmful audio.

Uses GLM-4 Voice Tokenizer (WhisperVQEncoder) - captures WHAT is being said.
This extracts embeddings that capture the MEANING of speech, not just acoustic features.

Output: data_bbh/ folder (for audio_semantic filter type)

IMPORTANT: Use --center flag (embeddings have 99.96% norm in global mean)
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
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GLM4_TOKENIZER_PATH = "THUDM/glm-4-voice-tokenizer"


class KimiSemanticEmbeddingExtractor:
    """
    Extract SEMANTIC embeddings using GLM-4 Voice Tokenizer (WhisperVQEncoder).

    This extracts embeddings that capture the MEANING of speech, not just acoustic features.
    Uses the hidden states from the VQ encoder, which are trained to represent speech semantics.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vq_encoder = None
        self.feature_extractor = None
        self.hidden_dim = None
        self._load_models()

    def _load_models(self):
        from transformers import WhisperFeatureExtractor
        from kimia_infer.models.tokenizer.glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder

        print("Loading GLM-4 Voice Tokenizer (WhisperVQEncoder) for SEMANTIC embeddings...")

        # Load the VQ encoder - this is trained to extract semantic representations
        self.vq_encoder = WhisperVQEncoder.from_pretrained(GLM4_TOKENIZER_PATH)
        self.vq_encoder = self.vq_encoder.to(self.device).to(torch.bfloat16)
        self.vq_encoder.eval()

        # Get the hidden dimension from config
        self.hidden_dim = self.vq_encoder.config.d_model

        # Get quantization position - we want hidden states BEFORE this layer
        self.quantize_position = getattr(self.vq_encoder.config, 'quantize_position', None)
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Quantize position: {self.quantize_position}")

        # Load feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(GLM4_TOKENIZER_PATH)

        print("Models loaded! Using SEMANTIC embeddings (PRE-quantization hidden states)")

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract SEMANTIC embedding for a single audio file (PRE-quantization)"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Get mel features
        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = features.input_features.to(self.device).to(torch.bfloat16)

        # Create attention mask based on input features shape
        seq_len = input_features.shape[-1]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

        # Forward through VQ encoder with output_hidden_states=True
        outputs = self.vq_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the hidden states BEFORE quantization
        if outputs.hidden_states is not None and self.quantize_position is not None:
            pre_quant_hidden = outputs.hidden_states[self.quantize_position]
            embedding = pre_quant_hidden.mean(dim=1)
        else:
            print("Warning: Could not get pre-quantization hidden states, using last_hidden_state")
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1)

        return embedding.cpu().float().numpy().squeeze()

    def extract_embeddings_batch(self, audio_paths: List[str], desc: str = "Extracting") -> np.ndarray:
        """Extract SEMANTIC embeddings for a list of audio files"""
        embeddings = []
        for path in tqdm(audio_paths, desc=desc):
            try:
                emb = self.extract_embedding(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                embeddings.append(np.zeros(self.hidden_dim))
        return np.stack(embeddings)


def center_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center embeddings by subtracting global mean.

    This is critical for Kimi-Audio embeddings because the global mean component
    dominates ~99.96% of the L2 norm, making all samples appear nearly identical.
    Centering removes this common baseline and reveals the actual semantic variation.
    """
    all_emb = np.vstack([emb1, emb2])
    global_mean = all_emb.mean(axis=0)

    print(f"  Global mean L2 norm: {np.linalg.norm(global_mean):.2f}")
    print(f"  Avg embedding L2 norm: {np.linalg.norm(emb1, axis=1).mean():.2f}")
    print(f"  Mean dominance ratio: {100 * np.linalg.norm(global_mean) / np.linalg.norm(emb1, axis=1).mean():.1f}%")

    emb1_centered = emb1 - global_mean
    emb2_centered = emb2 - global_mean

    print(f"  After centering - emb1 L2 norm: {np.linalg.norm(emb1_centered, axis=1).mean():.4f}")
    print(f"  After centering - emb2 L2 norm: {np.linalg.norm(emb2_centered, axis=1).mean():.4f}")

    return emb1_centered, emb2_centered, global_mean


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of embeddings"""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return similarity


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance (1 - similarity)"""
    similarity = compute_cosine_similarity(emb1, emb2)
    return 1 - similarity


def load_bbh_data(jsonl_path: str) -> List[Dict]:
    """Load BBH dataset from JSONL file."""
    print(f"Loading BBH dataset from {jsonl_path}...")

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
    """Get all audio files from a directory"""
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
    use_distance: bool = False,
    percentage: float = None,
    num_samples: int = None,
):
    """Filter samples and save to JSONL"""

    if use_distance:
        print("Computing pairwise cosine distances...")
        distance = compute_cosine_distance(embeddings, harmful_embeddings)
        min_distance = distance.min(axis=1)
        metric_values = min_distance
        metric_name = "Distance"
        sort_ascending = True
    else:
        print("Computing pairwise cosine similarities...")
        similarity = compute_cosine_similarity(embeddings, harmful_embeddings)
        max_similarity = similarity.max(axis=1)
        metric_values = max_similarity
        metric_name = "Similarity"
        sort_ascending = False

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
        sorted_indices = np.argsort(metric_values)
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]
        filtered_indices = sorted_indices[:top_k]
        mask = np.zeros(len(data), dtype=bool)
        mask[filtered_indices] = True
        effective_threshold = metric_values[filtered_indices[-1]] if len(filtered_indices) > 0 else threshold
        print(f"  Effective threshold: {effective_threshold:.6f}")
        filter_desc = f"top-{top_k}"
    else:
        if use_distance:
            mask = metric_values <= threshold
            filter_desc = f"<= {threshold}"
        else:
            mask = metric_values >= threshold
            filter_desc = f">= {threshold}"

    # Print stats
    print(f"\n{'='*60}")
    print(f"SEMANTIC {metric_name} Distribution:")
    print(f"  Mean:   {metric_values.mean():.4f}")
    print(f"  Std:    {metric_values.std():.4f}")
    print(f"  Min:    {metric_values.min():.4f}")
    print(f"  Max:    {metric_values.max():.4f}")
    print(f"  Median: {np.median(metric_values):.4f}")
    print(f"{'='*60}")

    print(f"\nFiltering with {metric_name.lower()} {filter_desc}")
    print(f"  Total samples: {len(data)}")
    print(f"  Samples kept: {mask.sum()}")
    print(f"  Samples removed: {(~mask).sum()}")

    metric_field = "min_distance_to_harmful" if use_distance else "max_similarity_to_harmful"

    # Save filtered data
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for i, (item, metric_val) in enumerate(zip(data, metric_values)):
            if mask[i]:
                entry = item["entry"].copy()
                if "_metadata" not in entry:
                    entry["_metadata"] = {}
                entry["_metadata"][metric_field] = float(metric_val)
                # Fix text instruction to use generic format (like VoiceBench)
                # This ensures model learns to extract question FROM audio
                for msg in entry.get("conversation", []):
                    if msg.get("message_type") == "text" and msg.get("role") == "user":
                        msg["content"] = "Please answer the following question based on the audio."
                        break
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"\nSaved {count} samples to {output_path}")

    if mask.sum() > 0:
        kept_values = metric_values[mask]
        print(f"\n{metric_name} statistics (kept samples):")
        print(f"  Mean: {kept_values.mean():.4f}")
        print(f"  Min:  {kept_values.min():.4f}")
        print(f"  Max:  {kept_values.max():.4f}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Filter BBH by SEMANTIC embedding similarity to harmful audio"
    )
    parser.add_argument("--bbh_jsonl", type=str, default="data_bbh/bbh_full.jsonl",
                        help="Path to BBH JSONL file")
    parser.add_argument("--harmful_dir", type=str, required=True,
                        help="Directory containing harmful audio files")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Threshold for filtering")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--output", type=str, default="data_bbh/bbh_filtered_semantic.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_semantic_bbh",
                        help="Directory to cache SEMANTIC embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--use_distance", action="store_true",
                        help="Use cosine distance (1-similarity) instead of similarity")
    parser.add_argument("--center", action="store_true",
                        help="Center embeddings by subtracting global mean (RECOMMENDED)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize SEMANTIC embedding extractor
    extractor = KimiSemanticEmbeddingExtractor(device=args.device)

    # Load data
    data = load_bbh_data(args.bbh_jsonl)
    audio_paths = [d["audio_path"] for d in data]

    harmful_audio_paths = get_audio_files(args.harmful_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Extract or load cached embeddings
    bbh_cache = os.path.join(args.cache_dir, "bbh_semantic_embeddings.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_semantic_embeddings_{Path(args.harmful_dir).name}.npy")

    if os.path.exists(bbh_cache):
        print(f"Loading cached BBH embeddings from {bbh_cache}")
        embeddings = np.load(bbh_cache)
    else:
        print("Extracting BBH SEMANTIC embeddings...")
        embeddings = extractor.extract_embeddings_batch(audio_paths, desc="BBH")
        np.save(bbh_cache, embeddings)
        print(f"Cached embeddings to {bbh_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful audio SEMANTIC embeddings...")
        harmful_embeddings = extractor.extract_embeddings_batch(harmful_audio_paths, desc="Harmful")
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached harmful embeddings to {harmful_cache}")

    # Optionally center embeddings (RECOMMENDED for Kimi-Audio)
    if args.center:
        print("\nCentering embeddings (removing global mean)...")
        embeddings, harmful_embeddings, global_mean = center_embeddings(embeddings, harmful_embeddings)
        np.save(os.path.join(args.cache_dir, "bbh_semantic_embeddings_centered.npy"), embeddings)
        np.save(os.path.join(args.cache_dir, f"harmful_semantic_embeddings_{Path(args.harmful_dir).name}_centered.npy"), harmful_embeddings)
        np.save(os.path.join(args.cache_dir, "global_mean.npy"), global_mean)
        print("Saved centered embeddings to cache.")

    # Filter and save
    filter_and_save(
        data,
        embeddings,
        harmful_embeddings,
        args.threshold,
        args.output,
        use_distance=args.use_distance,
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
