"""
=============================================================================
AUDIO-BASED SEMANTIC FILTERING - VoiceBench Noisy
=============================================================================

Filter VoiceBench Noisy samples based on SEMANTIC embedding distance to harmful audio.

VoiceBench Noisy contains the original VoiceBench dataset augmented with realistic
background noise (cafe, car, traffic, office, etc.) at configurable SNR levels.

IMPORTANT DISTINCTION - Kimi-Audio has TWO encoders:
1. GLM-4 Voice Tokenizer (WhisperVQEncoder) -> Discrete SEMANTIC tokens
   - Captures WHAT is being said (linguistic content)
   - This script uses pre-quantization hidden states from this encoder

2. Whisper-Large-V3 Encoder -> Continuous ACOUSTIC features
   - Captures HOW it sounds (voice characteristics, prosody)
   - See: 0_filter_voicebench_noisy_acoustic.py for true acoustic filtering

Usage:
    python 0_filter_voicebench_noisy_audio_semantic.py \
        --harmful_dir /path/to/advbench_gtts/en \
        --percentage 50 \
        --output data_voicebench_noisy/voicebench_noisy_filtered.jsonl
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
import torch.nn as nn

# Add the parent directory to path for kimia_infer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")
MODEL_PATH = "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
GLM4_TOKENIZER_PATH = "THUDM/glm-4-voice-tokenizer"
VOICEBENCH_NOISY_JSON = BASE_DIR / "Kimi-Audio/finetune_codes/data/voicebench_noisy/voicebench_noisy_full.jsonl"


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
        """Extract SEMANTIC embeddings for a list of audio files."""
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


def load_voicebench_noisy_data(json_path: str = None) -> List[Dict]:
    """Load VoiceBench Noisy dataset from JSONL file."""
    if json_path is None:
        json_path = str(VOICEBENCH_NOISY_JSON)

    print(f"Loading VoiceBench Noisy from {json_path}")

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
                        "noise_type": item.get("noise_type", ""),
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
                "noise_type": item.get("noise_type", ""),
                "snr_db": item.get("snr_db", 10),
                "original_id": item.get("original_id", ""),
            })

    # Filter to only samples with existing audio files
    voicebench_data = [d for d in voicebench_data if os.path.exists(d["audio_path"])]
    print(f"Loaded {len(voicebench_data)} VoiceBench Noisy samples with existing audio files")

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
    percentage: float = None,
    num_samples: int = None,
):
    """Filter VoiceBench Noisy samples and save to JSONL."""

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
    print(f"SEMANTIC Distance Distribution (all VoiceBench Noisy samples):")
    print(f"  Mean:   {min_distance.mean():.4f}")
    print(f"  Std:    {min_distance.std():.4f}")
    print(f"  Min:    {min_distance.min():.4f}")
    print(f"  Max:    {min_distance.max():.4f}")
    print(f"  Median: {np.median(min_distance):.4f}")
    print(f"{'='*60}")

    print(f"\n  Total VoiceBench Noisy samples: {len(voicebench_data)}")
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
                    "min_semantic_distance_to_harmful": float(dist_val),
                    "id": data.get("id", ""),
                    "region": data.get("region", ""),
                    "noise_type": data.get("noise_type", ""),
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
        print(f"\nSemantic distance statistics (kept samples):")
        print(f"  Mean: {kept_values.mean():.4f}")
        print(f"  Min:  {kept_values.min():.4f}")
        print(f"  Max:  {kept_values.max():.4f}")

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Filter VoiceBench Noisy by AUDIO SEMANTIC embedding distance (WhisperVQEncoder)"
    )
    parser.add_argument("--voicebench_json", type=str, default=None,
                        help="Path to VoiceBench Noisy JSONL file")
    parser.add_argument("--harmful_dir", type=str, required=True,
                        help="Directory containing harmful audio files")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Distance threshold (keep samples with distance <= threshold)")
    parser.add_argument("--percentage", type=float, default=None,
                        help="Keep top percentage of closest samples")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Keep exact number of samples")
    parser.add_argument("--output", type=str, default="data_voicebench_noisy/voicebench_noisy_filtered.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--cache_dir", type=str, default="embedding_cache_voicebench_noisy",
                        help="Directory to cache SEMANTIC embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Initialize SEMANTIC embedding extractor
    extractor = KimiSemanticEmbeddingExtractor(device=args.device)

    # Load data
    voicebench_data = load_voicebench_noisy_data(args.voicebench_json)
    voicebench_audio_paths = [d["audio_path"] for d in voicebench_data]

    harmful_audio_paths = get_audio_files(args.harmful_dir)
    print(f"Found {len(harmful_audio_paths)} harmful audio files")

    # Extract or load cached embeddings
    voicebench_cache = os.path.join(args.cache_dir, "voicebench_noisy_semantic_embeddings.npy")
    harmful_cache = os.path.join(args.cache_dir, f"harmful_semantic_embeddings_{Path(args.harmful_dir).name}.npy")

    if os.path.exists(voicebench_cache):
        print(f"Loading cached VoiceBench Noisy SEMANTIC embeddings from {voicebench_cache}")
        voicebench_embeddings = np.load(voicebench_cache)
    else:
        print("Extracting VoiceBench Noisy SEMANTIC embeddings...")
        voicebench_embeddings = extractor.extract_embeddings_batch(
            voicebench_audio_paths, desc="VoiceBench Noisy (Semantic)"
        )
        np.save(voicebench_cache, voicebench_embeddings)
        print(f"Cached VoiceBench Noisy embeddings to {voicebench_cache}")

    if os.path.exists(harmful_cache):
        print(f"Loading cached harmful SEMANTIC embeddings from {harmful_cache}")
        harmful_embeddings = np.load(harmful_cache)
    else:
        print("Extracting harmful audio SEMANTIC embeddings...")
        harmful_embeddings = extractor.extract_embeddings_batch(
            harmful_audio_paths, desc="Harmful (Semantic)"
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
        percentage=args.percentage,
        num_samples=args.num_samples,
    )

    print("\nDone! Used SEMANTIC features from WhisperVQEncoder.")


if __name__ == "__main__":
    main()
