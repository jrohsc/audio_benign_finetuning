"""
Filter VoiceBench samples based on SEMANTIC embedding distance to harmful audio samples.

This script uses the GLM-4 Voice Tokenizer (WhisperVQEncoder) to extract SEMANTIC
embeddings that capture the meaning of speech, not just acoustic features.

This script:
1. Extracts SEMANTIC embeddings from VoiceBench audio files
2. Extracts SEMANTIC embeddings from harmful audio files (AdvBench/SafetyBench GTTS)
3. Computes pairwise cosine similarity
4. Filters VoiceBench samples that are semantically close to harmful samples
5. Saves filtered data in conversation format for finetuning

Usage:
    python 0_filter_voicebench_by_embedding.py \
        --harmful_dir /path/to/advbench_gtts/en \
        --threshold 0.90 \
        --output voicebench_filtered_0.90.jsonl
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
from datasets import load_dataset
import torch.nn as nn

# Add the parent directory to path for kimia_infer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
MODEL_PATH = "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
VOICEBENCH_AUDIO_DIR = "/work/pi_ahoumansadr_umass_edu/jroh/voicebench_temp_audio"
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

    def _get_attention_mask(self, audio_length: int, stride: int) -> torch.Tensor:
        """Compute attention mask based on audio length"""
        # Compute the sequence length after conv layers
        seq_length = audio_length // (self.feature_extractor.hop_length * stride)
        return torch.ones(1, seq_length, dtype=torch.long, device=self.device)

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
        # Shape: [batch, feature_dim, seq_len] -> mask shape: [batch, seq_len]
        seq_len = input_features.shape[-1]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)

        # Forward through VQ encoder with output_hidden_states=True
        # This returns hidden states from ALL layers
        outputs = self.vq_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the hidden states BEFORE quantization
        # outputs.hidden_states is a tuple of (embedding, layer1, layer2, ..., layerN)
        # Quantization happens at layer `quantize_position`, so we want the layer just before
        if outputs.hidden_states is not None and self.quantize_position is not None:
            # Get hidden state just BEFORE quantization (index = quantize_position)
            # hidden_states[0] = input embeddings
            # hidden_states[i] = output of layer i-1
            # So hidden_states[quantize_position] = output of layer (quantize_position-1)
            #    which is the INPUT to the quantization at layer quantize_position
            pre_quant_hidden = outputs.hidden_states[self.quantize_position]
            embedding = pre_quant_hidden.mean(dim=1)
        else:
            # Fallback: use last hidden state (not ideal, but better than nothing)
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
                embeddings.append(np.zeros(self.hidden_dim))  # placeholder with correct dim
        return np.stack(embeddings)


def center_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center embeddings by subtracting global mean.

    This is critical for Kimi-Audio embeddings because the global mean component
    dominates ~99.96% of the L2 norm, making all samples appear nearly identical.
    Centering removes this common baseline and reveals the actual semantic variation.

    Returns:
        emb1_centered, emb2_centered, global_mean
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
    # Normalize
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    # Pairwise similarity
    similarity = np.dot(emb1_norm, emb2_norm.T)
    return similarity


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance (1 - similarity) between two sets of embeddings"""
    similarity = compute_cosine_similarity(emb1, emb2)
    return 1 - similarity


def load_voicebench_data() -> List[Dict]:
    """Load VoiceBench dataset from HuggingFace"""
    print("Loading VoiceBench dataset...")

    voicebench_data = []

    # Load different subsets
    subsets = [
        ("sd-qa", ["aus", "gbr", "ind_n", "ind_s", "irl", "kenya", "nga", "nzl",
                   "phl", "usa", "zaf"]),
    ]

    for subset_name, splits in subsets:
        try:
            # Load dataset without audio column to avoid torchcodec issues
            ds = load_dataset("hlt-lab/voicebench", subset_name)
            for split in splits:
                if split in ds:
                    # Remove audio column to avoid decoding issues
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


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory"""
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
    use_distance: bool = False,
    top_k: int = None,
    percentage: float = None,
    num_samples: int = None,
):
    """Filter VoiceBench samples and save to JSONL

    Args:
        use_distance: If True, use cosine distance (1-similarity) like Qwen-Audio.
                      Lower distance = closer to harmful, filter with <= threshold.
                      If False, use cosine similarity (original behavior).
                      Higher similarity = closer to harmful, filter with >= threshold.
        top_k: If set, keep top-k samples (overrides threshold)
        percentage: If set, keep top percentage of samples (e.g., 50 for 50%)
        num_samples: If set, keep exact number of samples (overrides percentage and top_k)
    """

    if use_distance:
        # Compute distance (like Qwen-Audio)
        print("Computing pairwise cosine distances...")
        distance = compute_cosine_distance(voicebench_embeddings, harmful_embeddings)
        # Get min distance for each VoiceBench sample (closest harmful sample)
        min_distance = distance.min(axis=1)
        metric_values = min_distance
        metric_name = "Distance"
        # For distance: lower = closer to harmful
        sort_ascending = True  # Sort ascending so smallest distances come first
    else:
        # Compute similarity (original behavior)
        print("Computing pairwise cosine similarities...")
        similarity = compute_cosine_similarity(voicebench_embeddings, harmful_embeddings)
        # Get max similarity for each VoiceBench sample
        max_similarity = similarity.max(axis=1)
        metric_values = max_similarity
        metric_name = "Similarity"
        # For similarity: higher = closer to harmful
        sort_ascending = False  # Sort descending so highest similarities come first

    # Priority: num_samples > percentage > top_k > threshold
    original_percentage = percentage
    original_num_samples = num_samples

    if num_samples is not None:
        top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        top_k = int(len(voicebench_data) * percentage / 100)
        top_k = max(1, top_k)  # Ensure at least 1 sample
        print(f"\n=== Converting {percentage}% to top_k={top_k} samples ===")

    # Apply filtering based on top_k or threshold
    if top_k is not None:
        # Sort by metric and take top-k closest to harmful
        sorted_indices = np.argsort(metric_values)
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]  # Reverse for similarity
        filtered_indices = sorted_indices[:top_k]
        mask = np.zeros(len(voicebench_data), dtype=bool)
        mask[filtered_indices] = True
        effective_threshold = metric_values[filtered_indices[-1]] if len(filtered_indices) > 0 else threshold
        print(f"\n=== Keeping top {top_k} samples closest to harmful ===")
        print(f"  Effective threshold: {effective_threshold:.6f}")
        filter_desc = f"top-{top_k}"
    else:
        # Use threshold-based filtering
        if use_distance:
            # Lower distance = closer to harmful, so keep samples with distance <= threshold
            mask = metric_values <= threshold
            filter_desc = f"<= {threshold}"
        else:
            # Higher similarity = closer to harmful, so keep samples with similarity >= threshold
            mask = metric_values >= threshold
            filter_desc = f">= {threshold}"

    # Print overall distribution
    print(f"\n{'='*60}")
    print(f"OVERALL {metric_name} Distribution (all VoiceBench samples):")
    print(f"  Mean:   {metric_values.mean():.4f}")
    print(f"  Std:    {metric_values.std():.4f}")
    print(f"  Min:    {metric_values.min():.4f}")
    print(f"  Max:    {metric_values.max():.4f}")
    print(f"  Median: {np.median(metric_values):.4f}")
    print(f"{'='*60}")

    print(f"\nFiltering with {metric_name.lower()} {filter_desc}")
    print(f"  Total VoiceBench samples: {len(voicebench_data)}")
    print(f"  Samples close to harmful (kept): {mask.sum()}")
    print(f"  Samples far from harmful (removed): {(~mask).sum()}")

    # Create conversation format
    # Metadata field name depends on metric type
    metric_field = "min_distance_to_harmful" if use_distance else "max_similarity_to_harmful"

    conversations = []
    for i, (data, metric_val) in enumerate(zip(voicebench_data, metric_values)):
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
                    metric_field: float(metric_val),
                    "subset": data.get("subset", ""),
                    "split": data.get("split", "")
                }
            }
            conversations.append(conv)

    # Save to JSONL
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(conversations)} conversations to {output_path}")

    # Print statistics
    if mask.sum() > 0:
        kept_values = metric_values[mask]
        print(f"\n{metric_name} statistics (kept samples):")
        print(f"  Mean: {kept_values.mean():.4f}")
        print(f"  Min:  {kept_values.min():.4f}")
        print(f"  Max:  {kept_values.max():.4f}")

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Filter VoiceBench by SEMANTIC embedding similarity to harmful audio")
    parser.add_argument("--harmful_dir", type=str, required=True,
                        help="Directory containing harmful audio files (e.g., advbench_gtts/en)")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Threshold for filtering. With --use_distance: keep samples with distance <= threshold. "
                             "Without: keep samples with similarity >= threshold.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Keep top-k samples closest to harmful (overrides threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of samples closest to harmful (e.g., 50 for 50%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples (overrides percentage and top_k)")
    parser.add_argument("--output", type=str, default="voicebench_filtered.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_semantic",
                        help="Directory to cache SEMANTIC embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--use_distance", action="store_true",
                        help="Use cosine distance (1-similarity) instead of similarity. "
                             "This makes thresholds comparable to Qwen-Audio (lower = closer to harmful).")
    parser.add_argument("--center", action="store_true",
                        help="Center embeddings by subtracting global mean before computing similarity. "
                             "RECOMMENDED for Kimi-Audio: the raw embeddings have ~99.96%% of their norm "
                             "in a common baseline, making all samples appear nearly identical. "
                             "Centering reveals the actual semantic variation.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize SEMANTIC embedding extractor (using GLM-4 Voice Tokenizer)
    extractor = KimiSemanticEmbeddingExtractor(device=args.device)

    # Load VoiceBench data
    voicebench_data = load_voicebench_data()
    voicebench_audio_paths = [d["audio_path"] for d in voicebench_data]

    # Load harmful audio files
    harmful_audio_paths = get_audio_files(args.harmful_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Extract or load cached SEMANTIC embeddings
    # Note: Using different cache names to distinguish from acoustic embeddings
    voicebench_cache = os.path.join(args.cache_dir, "voicebench_semantic_embeddings.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_semantic_embeddings_{Path(args.harmful_dir).name}.npy")

    if os.path.exists(voicebench_cache):
        print(f"Loading cached VoiceBench embeddings from {voicebench_cache}")
        voicebench_embeddings = np.load(voicebench_cache)
    else:
        print("Extracting VoiceBench embeddings...")
        voicebench_embeddings = extractor.extract_embeddings_batch(
            voicebench_audio_paths, desc="VoiceBench"
        )
        np.save(voicebench_cache, voicebench_embeddings)
        print(f"Cached VoiceBench embeddings to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful audio embeddings...")
        harmful_embeddings = extractor.extract_embeddings_batch(
            harmful_audio_paths, desc="Harmful"
        )
        np.save(harmful_cache, harmful_embeddings)
        print(f"Cached harmful embeddings to {harmful_cache}")

    # Optionally center embeddings (RECOMMENDED for Kimi-Audio)
    if args.center:
        print("\nCentering embeddings (removing global mean)...")
        voicebench_embeddings, harmful_embeddings, global_mean = center_embeddings(
            voicebench_embeddings, harmful_embeddings
        )
        # Save centered embeddings and global mean for reproducibility
        np.save(os.path.join(args.cache_dir, "voicebench_semantic_embeddings_centered.npy"), voicebench_embeddings)
        np.save(os.path.join(args.cache_dir, f"harmful_semantic_embeddings_{Path(args.harmful_dir).name}_centered.npy"), harmful_embeddings)
        np.save(os.path.join(args.cache_dir, "global_mean.npy"), global_mean)
        print("Saved centered embeddings to cache.")

    # Filter and save
    filter_and_save(
        voicebench_data,
        voicebench_embeddings,
        harmful_embeddings,
        args.threshold,
        args.output,
        use_distance=args.use_distance,
        top_k=args.top_k,
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
