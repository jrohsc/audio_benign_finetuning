#!/usr/bin/env python3
"""
Filter VoiceBench samples to find those CLOSEST in embedding space to AdvBench samples.
Uses Audio Flamingo 3's pretrained Whisper encoder to compute audio embeddings and finds samples with minimum distance.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def load_af3_encoder(model_path: str, device: str = "cuda"):
    """
    Load Audio Flamingo 3's FINETUNED Whisper encoder (AudioFlamingo3Encoder) AND the multi-modal projector.

    IMPORTANT: This uses the actual AudioFlamingo3Encoder architecture which differs from Qwen2AudioEncoder:
    - Has AvgPool1d(2, stride=2) after transformer layers
    - Different attention mask handling (uses input_features_mask)
    - Layer norm applied AFTER avg pooling

    The full audio embedding pipeline in AF3 is:
        audio -> audio_tower (AudioFlamingo3Encoder) -> multi_modal_projector (adapter) -> embedding

    Args:
        model_path: Path to the Audio Flamingo 3 model checkpoint
        device: Device to load model on

    Returns:
        encoder: The AudioFlamingo3Encoder model (finetuned whisper with avg pooling)
        projector: The multi-modal projector (adapter)
        feature_extractor: The audio feature extractor
    """
    from transformers import AutoFeatureExtractor
    import torch.nn as nn

    model_path = Path(model_path)
    print(f"Loading Audio Flamingo 3 encoder from: {model_path}")

    # Use transformers' native AudioFlamingo3 support (correct architecture with AvgPool1d)
    try:
        from transformers import AudioFlamingo3ForConditionalGeneration
        print("  Using AudioFlamingo3ForConditionalGeneration (correct architecture with AvgPool1d)")

        # Load the full model
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Extract the audio tower (AudioFlamingo3Encoder) and projector
        encoder = model.audio_tower.to(device)
        projector = model.multi_modal_projector.to(device)

        # Get feature extractor
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        except Exception:
            print("  Feature extractor not found in model path, using Qwen2Audio's")
            feature_extractor = AutoFeatureExtractor.from_pretrained('Qwen/Qwen2-Audio-7B')

        encoder.eval()
        projector.eval()

        # Free the language model to save memory
        del model.language_model
        del model
        torch.cuda.empty_cache()

        print(f"  Loaded AudioFlamingo3Encoder (with AvgPool1d) and MultiModalProjector")
        print(f"  Architecture: conv1 -> conv2 -> transformer layers -> AvgPool1d(2) -> LayerNorm")

        return encoder, projector, feature_extractor

    except ImportError as e:
        print(f"  AudioFlamingo3 not in transformers: {e}")
        print("  Falling back to manual loading (Qwen2AudioEncoder - different architecture!)")

    # Fallback to old method
    from transformers import AutoConfig
    from safetensors.torch import load_file

    # Load the feature extractor (same as Qwen2Audio)
    feature_extractor = AutoFeatureExtractor.from_pretrained('Qwen/Qwen2-Audio-7B')

    # Check if this is HuggingFace format (has config.json with audioflamingo3 model_type)
    config_path = model_path / "config.json"
    is_hf_format = False
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        is_hf_format = config.get("model_type") == "audioflamingo3"

    if is_hf_format:
        print("  Detected HuggingFace format checkpoint")

        # Load the full model's state dict from safetensors
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            import json
            with open(index_path) as f:
                index = json.load(f)

            # Find which shard files contain audio_tower and multi_modal_projector
            audio_tower_weights = {}
            projector_weights = {}

            # Group weights by shard file
            shards_to_load = set()
            for weight_name, shard_file in index["weight_map"].items():
                if weight_name.startswith("audio_tower."):
                    shards_to_load.add(shard_file)
                elif weight_name.startswith("multi_modal_projector."):
                    shards_to_load.add(shard_file)

            # Load only the necessary shards
            print(f"  Loading weights from shards: {shards_to_load}")
            for shard_file in shards_to_load:
                shard_path = model_path / shard_file
                shard_weights = load_file(shard_path)
                for k, v in shard_weights.items():
                    if k.startswith("audio_tower."):
                        # Remove 'audio_tower.' prefix
                        new_key = k[len("audio_tower."):]
                        audio_tower_weights[new_key] = v
                    elif k.startswith("multi_modal_projector."):
                        # Remove 'multi_modal_projector.' prefix
                        new_key = k[len("multi_modal_projector."):]
                        projector_weights[new_key] = v

            # Build the encoder using transformers' native Qwen2AudioEncoder
            from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
            from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig

            audio_config = config["audio_config"]
            encoder_config = Qwen2AudioEncoderConfig(
                d_model=audio_config["hidden_size"],
                encoder_layers=audio_config["num_hidden_layers"],
                encoder_attention_heads=audio_config["num_attention_heads"],
                encoder_ffn_dim=audio_config["intermediate_size"],
                dropout=audio_config.get("dropout", 0.0),
                attention_dropout=audio_config.get("attention_dropout", 0.0),
                activation_dropout=audio_config.get("activation_dropout", 0.0),
                activation_function=audio_config.get("activation_function", "gelu"),
                num_mel_bins=audio_config.get("num_mel_bins", 128),
                max_source_positions=audio_config.get("max_source_positions", 1500),
                scale_embedding=audio_config.get("scale_embedding", False),
            )

            encoder = Qwen2AudioEncoder(encoder_config)
            encoder.load_state_dict(audio_tower_weights)
            encoder = encoder.to(device)
            encoder.eval()
            print(f"  Loaded audio_tower with {len(audio_tower_weights)} weight tensors")

            # Build the projector
            # Config: hidden_size=1280 (audio) -> hidden_size=3584 (text)
            text_hidden_size = config["text_config"]["hidden_size"]
            audio_hidden_size = audio_config["hidden_size"]

            projector = nn.Sequential(
                nn.Linear(audio_hidden_size, text_hidden_size, bias=config.get("projector_bias", True)),
                nn.GELU(),
                nn.Linear(text_hidden_size, text_hidden_size, bias=config.get("projector_bias", True)),
            )

            # Map weights: linear_1 -> 0, linear_2 -> 2
            projector_state = {
                "0.weight": projector_weights["linear_1.weight"],
                "0.bias": projector_weights["linear_1.bias"],
                "2.weight": projector_weights["linear_2.weight"],
                "2.bias": projector_weights["linear_2.bias"],
            }
            projector.load_state_dict(projector_state)
            projector = projector.to(device)
            projector.eval()
            print(f"  Loaded multi_modal_projector: Linear({audio_hidden_size}, {text_hidden_size}) -> GELU -> Linear({text_hidden_size}, {text_hidden_size})")

        else:
            raise ValueError(f"HuggingFace format detected but model.safetensors.index.json not found at {model_path}")

    else:
        # Custom format with sound_tower/ and sound_mm_projector/ subdirectories
        print("  Detected custom format checkpoint")
        from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

        # Load AF3's pretrained encoder (sound_tower)
        sound_tower_path = model_path / "sound_tower"
        if sound_tower_path.exists():
            print(f"  Loading sound_tower from: {sound_tower_path}")
            encoder = Qwen2AudioEncoder.from_pretrained(sound_tower_path)
        else:
            print(f"  Loading sound_tower from: {model_path}")
            encoder = Qwen2AudioEncoder.from_pretrained(model_path)

        encoder = encoder.to(device)
        encoder.eval()

        # Load AF3's pretrained projector (sound_mm_projector / adapter)
        projector = None
        sound_mm_projector_path = model_path / "sound_mm_projector"
        if sound_mm_projector_path.exists():
            print(f"  Loading sound_mm_projector (adapter) from: {sound_mm_projector_path}")
            # Load projector weights and build nn.Sequential
            projector_weights = load_file(sound_mm_projector_path / "model.safetensors")
            # Determine dimensions from weights
            in_features = projector_weights["layers.0.weight"].shape[1]
            out_features = projector_weights["layers.2.weight"].shape[0]
            projector = nn.Sequential(
                nn.Linear(in_features, out_features, bias=True),
                nn.GELU(),
                nn.Linear(out_features, out_features, bias=True),
            )
            projector_state = {
                "0.weight": projector_weights["layers.0.weight"],
                "0.bias": projector_weights["layers.0.bias"],
                "2.weight": projector_weights["layers.2.weight"],
                "2.bias": projector_weights["layers.2.bias"],
            }
            projector.load_state_dict(projector_state)
            projector = projector.to(device)
            projector.eval()
            print(f"  Projector loaded: Linear({in_features}, {out_features}) -> GELU -> Linear({out_features}, {out_features})")
        else:
            print(f"  WARNING: sound_mm_projector not found at {sound_mm_projector_path}")
            print(f"  Will use encoder-only embeddings (without adapter projection)")

    return encoder, projector, feature_extractor


def load_original_whisper(model_name: str = "openai/whisper-large-v3", device: str = "cuda"):
    """
    Load the original OpenAI Whisper model (fallback option).

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        model: The WhisperModel
        processor: The WhisperProcessor
    """
    from transformers import WhisperProcessor, WhisperModel

    print(f"Loading original Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, processor


def compute_whisper_embeddings(audio_files: List[str],
                                model_path: Optional[str] = None,
                                model_name: str = "openai/whisper-large-v3",
                                use_af3_encoder: bool = True,
                                batch_size: int = 8,
                                cache_path: str = None):
    """
    Compute audio embeddings using Whisper encoder.

    Args:
        audio_files: List of paths to audio files
        model_path: Path to Audio Flamingo 3 checkpoint (required if use_af3_encoder=True)
        model_name: Whisper model to use for embeddings (fallback if use_af3_encoder=False)
        use_af3_encoder: Whether to use AF3's pretrained encoder or original Whisper
        batch_size: Number of files to process at once
        cache_path: Optional path to cache embeddings

    Returns:
        embeddings: numpy array of shape (n_files, embedding_dim)
    """
    # Check for cached embeddings
    if cache_path and os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path)
        cached_embeddings = data['embeddings']
        print(f"  Loaded {len(cached_embeddings)} embeddings with shape {cached_embeddings.shape}")
        return cached_embeddings

    if cache_path:
        print(f"[CACHE MISS] Will compute embeddings and save to {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if use_af3_encoder:
        if model_path is None:
            raise ValueError("model_path is required when use_af3_encoder=True")
        encoder, projector, feature_extractor = load_af3_encoder(model_path, device)
        use_af3 = True
    else:
        model, processor = load_original_whisper(model_name, device)
        feature_extractor = processor
        projector = None
        use_af3 = False

    embeddings = []
    sample_rate = 16000
    window_length = 30 * sample_rate  # 30 seconds in samples

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing embeddings"):
            try:
                # Load and preprocess audio
                audio = load_audio(audio_path, sr=sample_rate)

                # Pad or truncate to 30 seconds (required input length for Whisper)
                if len(audio) < window_length:
                    audio = np.pad(audio, (0, window_length - len(audio)), mode='constant')
                else:
                    audio = audio[:window_length]

                # Process audio to mel spectrogram
                inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
                input_features = inputs["input_features"].to(device)

                if use_af3:
                    # Check if this is AudioFlamingo3Encoder (has avg_pooler) or Qwen2AudioEncoder (fallback)
                    is_af3_encoder = hasattr(encoder, 'avg_pooler')

                    if is_af3_encoder:
                        # AudioFlamingo3Encoder uses input_features_mask (1=valid, 0=padding)
                        # The mask should match the mel spectrogram sequence length
                        max_mel_seq_len = input_features.shape[-1]  # 3000 for 30s audio

                        # Create input_features_mask: all 1s for full audio (no padding)
                        input_features_mask = torch.ones(
                            (input_features.shape[0], max_mel_seq_len),
                            dtype=torch.long,
                            device=device
                        )

                        # Cast to float16 to match model dtype
                        input_features = input_features.to(torch.float16)

                        # Get encoder outputs using AudioFlamingo3Encoder interface
                        encoder_outputs = encoder(input_features, input_features_mask=input_features_mask)
                        sound_features = encoder_outputs.last_hidden_state  # (batch, seq_len/4, hidden_dim)
                    else:
                        # Fallback: Qwen2AudioEncoder uses attention_mask with -inf
                        orig_length = min(len(audio), window_length)
                        melspec_frames = int(math.ceil(orig_length / 160))

                        batch_size_curr = input_features.shape[0]
                        max_mel_seq_len = input_features.shape[-1]
                        max_seq_len = (max_mel_seq_len - 2) // 2 + 1

                        audio_feat_lengths = torch.tensor([melspec_frames], device=device)
                        audio_feat_lengths = (audio_feat_lengths - 1) // 2 + 1

                        seq_range = torch.arange(0, max_seq_len, dtype=torch.long, device=device).unsqueeze(0)
                        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size_curr, max_seq_len)
                        padding_mask = seq_range >= lengths_expand

                        audio_attention_mask_ = padding_mask.view(batch_size_curr, 1, 1, max_seq_len).expand(
                            batch_size_curr, 1, max_seq_len, max_seq_len
                        )
                        audio_attention_mask = audio_attention_mask_.to(dtype=input_features.dtype)
                        audio_attention_mask[audio_attention_mask_] = float("-inf")

                        encoder_outputs = encoder(input_features, attention_mask=audio_attention_mask)
                        sound_features = encoder_outputs.last_hidden_state

                    # Apply projector/adapter if available
                    if projector is not None:
                        sound_features = projector(sound_features)  # (batch, seq_len, llm_hidden_dim)

                    # Pool the embeddings (mean pooling across time dimension)
                    embedding = sound_features.mean(dim=1).float().cpu().numpy()
                else:
                    # Original Whisper model
                    encoder_outputs = model.encoder(input_features)
                    # Pool the embeddings (mean pooling across time dimension)
                    embedding = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                embeddings.append(embedding.squeeze())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                # Add zero embedding for failed samples
                # Dimension depends on whether projector is used:
                # - Without projector: 1280 (Whisper encoder hidden dim)
                # - With projector: LLM hidden dim (e.g., 3584 for Qwen2)
                if len(embeddings) > 0:
                    embed_dim = embeddings[-1].shape[-1]
                elif use_af3 and projector is not None:
                    # Get output dim from projector (works for both nn.Sequential and custom modules)
                    if hasattr(projector, 'layers'):
                        embed_dim = projector.layers[-1].out_features
                    else:
                        # nn.Sequential case
                        embed_dim = projector[-1].out_features
                else:
                    embed_dim = 1280  # Whisper large embedding dim
                embeddings.append(np.zeros(embed_dim))

    embeddings = np.array(embeddings)

    # Cache embeddings if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, files=audio_files)
        print(f"Cached embeddings to {cache_path}")

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
    if metric == "cosine":
        # Normalize embeddings
        vb_norm = voicebench_embeddings / (np.linalg.norm(voicebench_embeddings, axis=1, keepdims=True) + 1e-8)
        ab_norm = advbench_embeddings / (np.linalg.norm(advbench_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity = vb_norm @ ab_norm.T  # (n_voicebench, n_advbench)

        # Convert to distance (1 - similarity)
        distances = 1 - similarity

    elif metric == "euclidean":
        # Compute pairwise Euclidean distances
        from scipy.spatial.distance import cdist
        distances = cdist(voicebench_embeddings, advbench_embeddings, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get minimum distance to any AdvBench sample
    min_distances = distances.min(axis=1)

    return min_distances, distances


def filter_closest_samples(voicebench_json: str, advbench_audio_dir: str,
                          output_json: str, threshold: float = None,
                          top_k: int = None, percentage: float = None,
                          num_samples: int = None, metric: str = "cosine",
                          cache_dir: str = None,
                          model_path: str = None,
                          use_af3_encoder: bool = True,
                          select_safest: bool = False):
    """
    Filter VoiceBench samples by embedding distance to AdvBench samples.

    Args:
        voicebench_json: Path to VoiceBench samples JSON
        advbench_audio_dir: Directory containing AdvBench audio files
        output_json: Path to save filtered VoiceBench samples
        threshold: Distance threshold (keep samples with distance <= threshold)
        top_k: If set, keep top-k samples with SMALLEST minimum distance
        percentage: If set, keep top percentage of samples (e.g., 10 for 10%)
        num_samples: If set, keep exact number of samples (overrides percentage and top_k)
        metric: Distance metric to use
        cache_dir: Directory to cache embeddings
        model_path: Path to Audio Flamingo 3 model checkpoint
        use_af3_encoder: Whether to use AF3's encoder (True) or original Whisper (False)
        select_safest: If True, select samples FURTHEST from harmful (safest).
                       If False (default), select samples CLOSEST to harmful.
    """
    # Load VoiceBench data
    print(f"Loading VoiceBench data from {voicebench_json}")
    with open(voicebench_json, 'r') as f:
        voicebench_data = json.load(f)

    # Get the base directory for audio paths
    json_dir = Path(voicebench_json).parent

    voicebench_audio_files = []
    for sample in voicebench_data:
        audio_path = sample["audio"]
        # Handle relative paths
        if not os.path.isabs(audio_path):
            # Try relative to JSON file's parent directory
            full_path = json_dir.parent.parent.parent / audio_path
            if not full_path.exists():
                # Try relative to current working directory
                full_path = Path(audio_path)
            audio_path = str(full_path)
        voicebench_audio_files.append(audio_path)

    print(f"Found {len(voicebench_audio_files)} VoiceBench samples")
    print(f"First audio path: {voicebench_audio_files[0]}")

    # Load AdvBench audio files
    advbench_audio_dir = Path(advbench_audio_dir)
    advbench_audio_files = sorted(list(advbench_audio_dir.glob("*.mp3")) +
                                 list(advbench_audio_dir.glob("*.wav")))
    advbench_audio_files = [str(f) for f in advbench_audio_files]
    print(f"Found {len(advbench_audio_files)} AdvBench samples")

    # Set up cache paths - use different filenames for AF3 vs original Whisper
    vb_cache = None
    ab_cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoder_suffix = "af3" if use_af3_encoder else "whisper"
        vb_cache = str(cache_dir / f"voicebench_embeddings_{encoder_suffix}.npz")
        ab_cache = str(cache_dir / f"advbench_en_embeddings_{encoder_suffix}.npz")

    # Compute embeddings
    encoder_type = "AF3" if use_af3_encoder else "original Whisper"
    print(f"\n=== Computing VoiceBench embeddings using {encoder_type} encoder ===")
    voicebench_embeddings = compute_whisper_embeddings(
        voicebench_audio_files,
        model_path=model_path,
        use_af3_encoder=use_af3_encoder,
        cache_path=vb_cache
    )

    print(f"\n=== Computing AdvBench embeddings using {encoder_type} encoder ===")
    advbench_embeddings = compute_whisper_embeddings(
        advbench_audio_files,
        model_path=model_path,
        use_af3_encoder=use_af3_encoder,
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

    # Print percentiles to help choose threshold
    percentiles = [5, 10, 15, 20, 25, 30, 50]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(min_distances, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Filter samples
    # Priority: num_samples > percentage > top_k > threshold
    selection_mode = "SAFEST (furthest from harmful)" if select_safest else "CLOSEST to harmful"
    print(f"\n=== Selection mode: {selection_mode} ===")

    effective_top_k = top_k

    if num_samples is not None:
        effective_top_k = num_samples
        print(f"\n=== Using exact num_samples={num_samples} ===")
    elif percentage is not None:
        effective_top_k = int(len(voicebench_data) * percentage / 100)
        effective_top_k = max(1, effective_top_k)  # Ensure at least 1 sample
        print(f"\n=== Converting {percentage}% to top_k={effective_top_k} samples ===")

    if effective_top_k is not None:
        if select_safest:
            # Keep top-k samples with LARGEST distance (furthest from harmful = safest)
            top_k_indices = np.argsort(min_distances)[::-1][:effective_top_k]
        else:
            # Keep top-k samples with SMALLEST distance (closest to harmful)
            top_k_indices = np.argsort(min_distances)[:effective_top_k]
        filtered_indices = top_k_indices
        label = "safest" if select_safest else "closest"
        print(f"\n=== Keeping top {effective_top_k} {label} samples ===")
        print(f"  Min distance in selected: {min_distances[top_k_indices].min():.4f}")
        print(f"  Max distance in selected: {min_distances[top_k_indices].max():.4f}")
        print(f"  Mean distance in selected: {min_distances[top_k_indices].mean():.4f}")

    elif threshold is not None:
        if select_safest:
            # Keep samples with distance >= threshold (furthest from harmful)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (safest) ===")
        else:
            # Keep samples with distance <= threshold (closest to harmful)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} (closest) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    else:
        # Auto-select threshold
        if select_safest:
            threshold = np.percentile(min_distances, 75)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (75th percentile, safest) ===")
        else:
            threshold = np.percentile(min_distances, 25)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Auto-selected threshold: {threshold:.4f} (25th percentile, closest) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = [voicebench_data[i] for i in filtered_indices]

    # Add distance metadata
    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_advbench"] = float(min_distances[idx])
        # Also add the index of the closest AdvBench sample
        closest_advbench_idx = np.argmin(all_distances[idx])
        filtered_data[i]["closest_advbench_file"] = advbench_audio_files[closest_advbench_idx]

    # Sort by distance (closest first if closest mode, furthest first if safest mode)
    filtered_data.sort(key=lambda x: x["min_distance_to_advbench"], reverse=select_safest)

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save distance analysis with a consistent base name (reusable for different filtering modes)
    # This file contains ALL embeddings and distances, so it can be reused
    base_analysis_path = output_path.parent / "sd_qa_closest_to_advbench_analysis.npz"
    np.savez(
        base_analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        voicebench_embeddings=voicebench_embeddings,
        advbench_embeddings=advbench_embeddings,
        voicebench_files=voicebench_audio_files,
        advbench_files=advbench_audio_files
    )
    print(f"  Distance analysis: {base_analysis_path}")

    # Print some examples
    label = "safest (furthest)" if select_safest else "closest"
    print(f"\n=== Top 10 {label} VoiceBench samples to AdvBench ===")
    for i in range(min(10, len(filtered_data))):
        sample = filtered_data[i]
        print(f"  {i+1}. Distance: {sample['min_distance_to_advbench']:.4f}")
        if 'conversations' in sample and len(sample['conversations']) > 0:
            question = sample['conversations'][0].get('value', '')
            # Truncate for display
            if len(question) > 80:
                question = question[:80] + "..."
            print(f"     Question: {question}")

    return str(output_json)


def main():
    parser = argparse.ArgumentParser(description="Filter VoiceBench samples closest to AdvBench")
    parser.add_argument("--voicebench_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_full.json",
                       help="Path to VoiceBench samples JSON file")
    parser.add_argument("--advbench_audio_dir", type=str,
                       default="advbench/en",
                       help="Directory containing AdvBench audio files")
    parser.add_argument("--output_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_closest_to_advbench.json",
                       help="Path to save filtered VoiceBench samples")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold (keep samples with distance >= threshold)")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples with smallest minimum distance (closest)")
    parser.add_argument("--percentage", type=float, default=None,
                       help="Keep top percentage of samples (e.g., 10 for 10%%)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Keep exact number of samples (overrides percentage and top_k)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--cache_dir", type=str, default="data/embedding_cache",
                       help="Directory to cache embeddings")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to Audio Flamingo 3 model checkpoint (required for AF3 encoder)")
    parser.add_argument("--use_af3_encoder", action="store_true", default=True,
                       help="Use Audio Flamingo 3's pretrained encoder (default: True)")
    parser.add_argument("--use_original_whisper", action="store_true",
                       help="Use original OpenAI Whisper instead of AF3 encoder")
    parser.add_argument("--select_safest", action="store_true",
                       help="Select samples FURTHEST from harmful (safest benign samples) "
                            "instead of closest to harmful (default behavior)")

    args = parser.parse_args()

    # Determine which encoder to use
    use_af3 = args.use_af3_encoder and not args.use_original_whisper

    if use_af3 and args.model_path is None:
        parser.error("--model_path is required when using AF3 encoder. "
                    "Use --use_original_whisper to fall back to OpenAI Whisper.")

    filter_closest_samples(
        voicebench_json=args.voicebench_json,
        advbench_audio_dir=args.advbench_audio_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        top_k=args.top_k,
        percentage=args.percentage,
        num_samples=args.num_samples,
        metric=args.metric,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        use_af3_encoder=use_af3,
        select_safest=args.select_safest
    )


if __name__ == "__main__":
    main()
