#!/usr/bin/env python3
"""
Filter VoiceBench samples based on embedding distance from AdvBench samples.
Uses Phi-4's audio encoder to compute audio embeddings.

This script finds VoiceBench samples that are FARTHEST from harmful AdvBench samples
(i.e., most "benign") for safe finetuning.
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
import types

# Mock flash_attn BEFORE any transformers imports to avoid import errors
# The model will fall back to SDPA attention
try:
    import flash_attn
    _ = flash_attn.__version__  # Verify it's actually working
except (ImportError, OSError, AttributeError):
    # Create a proper mock module with __spec__ to satisfy importlib checks
    flash_attn_mock = types.ModuleType('flash_attn')
    flash_attn_mock.__version__ = "0.0.0"
    flash_attn_mock.__spec__ = types.SimpleNamespace(name='flash_attn', loader=None, origin=None, submodule_search_locations=[])
    flash_attn_mock.flash_attn_func = lambda *args, **kwargs: None
    flash_attn_mock.flash_attn_varlen_func = lambda *args, **kwargs: None

    flash_attn_interface = types.ModuleType('flash_attn.flash_attn_interface')
    flash_attn_interface.__spec__ = types.SimpleNamespace(name='flash_attn.flash_attn_interface', loader=None, origin=None, submodule_search_locations=[])

    bert_padding = types.ModuleType('flash_attn.bert_padding')
    bert_padding.__spec__ = types.SimpleNamespace(name='flash_attn.bert_padding', loader=None, origin=None, submodule_search_locations=[])
    bert_padding.index_first_axis = lambda *args, **kwargs: None
    bert_padding.pad_input = lambda *args, **kwargs: None
    bert_padding.unpad_input = lambda *args, **kwargs: None

    sys.modules['flash_attn'] = flash_attn_mock
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    sys.modules['flash_attn.bert_padding'] = bert_padding
    print("Note: flash_attn mocked - will use SDPA attention")

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


def load_phi4_encoder(model_name: str = "microsoft/Phi-4-multimodal-instruct", device: str = "cuda"):
    """
    Load Phi-4's audio encoder and processor.

    Phi-4's audio pipeline:
        audio -> processor -> input_audio_embeds -> audio_embed.encoder -> audio_embed.audio_projection -> embedding

    Args:
        model_name: HuggingFace model name for Phi-4
        device: Device to load model on

    Returns:
        model: The full Phi-4 model (we'll extract encoder from it)
        processor: The Phi-4 processor
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import importlib.util

    print(f"Loading Phi-4 model from: {model_name}")

    # Load the custom processor classes directly from the model directory
    model_path = Path(model_name)
    processing_module_path = model_path / "processing_phi4mm.py"

    # Dynamically load the processing module
    spec = importlib.util.spec_from_file_location("processing_phi4mm", processing_module_path)
    processing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processing_module)

    # Get the processor classes
    Phi4MMProcessor = processing_module.Phi4MMProcessor
    Phi4MMImageProcessor = processing_module.Phi4MMImageProcessor
    Phi4MMAudioFeatureExtractor = processing_module.Phi4MMAudioFeatureExtractor

    # Load the tokenizer from Xenova/gpt-4o as specified in auto_map
    # The auto_map in config.json has "AutoTokenizer": "Xenova/gpt-4o" which is a hub path, not module.class
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4o", trust_remote_code=True)

    # Set pad token if not defined (required for Phi-4 processor padding)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set chat template for Phi-4 style prompts
    # This is needed because the Xenova/gpt-4o tokenizer doesn't have a default chat template
    phi4_chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}<|end|>
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}<|end|>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}<|end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""
    tokenizer.chat_template = phi4_chat_template

    # Create the image processor with dynamic_hd parameter
    image_processor = Phi4MMImageProcessor(dynamic_hd=36)

    # Create the audio feature extractor with parameters from config
    # These values come from config.json:
    # - audio_compression_rate: embd_layer.audio_embd_layer.compression_rate (8)
    # - audio_downsample_rate: embd_layer.audio_embd_layer.downsample_rate (1)
    # - audio_feat_stride: audio_processor.config.time_reduction (8)
    audio_processor = Phi4MMAudioFeatureExtractor(
        audio_compression_rate=8,
        audio_downsample_rate=1,
        audio_feat_stride=8
    )

    # Assemble the processor
    processor = Phi4MMProcessor(
        image_processor=image_processor,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )

    # Load model config with audio enabled
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        audio_enabled=True
    )
    # Override flash attention to avoid compatibility issues
    config._attn_implementation = "sdpa"  # Use PyTorch's scaled_dot_product_attention instead of flash_attn

    # Patch the model class to skip LoRA initialization
    # This is needed because we only want the audio encoder, not the full generation model
    # and there are PEFT compatibility issues with the current versions
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    model_class = get_class_from_dynamic_module(
        "modeling_phi4mm.Phi4MMForCausalLM",
        model_name,
        trust_remote_code=True
    )
    original_init = model_class.__init__

    def patched_init(self, config):
        # Store original lora configs temporarily
        orig_vision_lora = getattr(config, 'vision_lora', None)
        orig_speech_lora = getattr(config, 'speech_lora', None)
        # Set to None to skip LoRA in the original init
        config.vision_lora = None
        config.speech_lora = None

        # Call parent class __init__ directly instead of the patched class
        # This initializes the model without LoRA adapters
        from transformers import PreTrainedModel
        # Get the base class (should be Phi4MMPreTrainedModel or similar)
        import torch.nn as nn

        # Initialize as PreTrainedModel
        PreTrainedModel.__init__(self, config)

        # Manually initialize the model components that we need
        # Get the inner model class
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        inner_model_class = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel",
            model_name,
            trust_remote_code=True
        )
        self.model = inner_model_class(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)

        # Restore config
        config.vision_lora = orig_vision_lora
        config.speech_lora = orig_speech_lora

        # Call post_init to load weights properly
        self.post_init()

    model_class.__init__ = patched_init

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa"  # Also pass to from_pretrained
    ).to(device)

    model.eval()

    # Access the audio encoder components
    audio_embed = model.model.embed_tokens_extend.audio_embed
    encoder = audio_embed.encoder
    audio_projection = audio_embed.audio_projection

    print(f"  Audio encoder loaded")
    print(f"  Encoder type: {type(encoder).__name__}")
    print(f"  Projection type: {type(audio_projection).__name__}")

    return model, processor, encoder, audio_projection, audio_processor


def compute_phi4_embeddings(
    audio_files: List[str],
    model_name: str = "microsoft/Phi-4-multimodal-instruct",
    batch_size: int = 1,
    cache_path: str = None,
    use_projection: bool = True,
):
    """
    Compute audio embeddings using Phi-4's audio encoder.

    Args:
        audio_files: List of paths to audio files
        model_name: Phi-4 model name
        batch_size: Number of files to process at once
        cache_path: Optional path to cache embeddings
        use_projection: Whether to apply the audio projection layer

    Returns:
        embeddings: numpy array of shape (n_files, embedding_dim)
    """
    # Check for cached embeddings
    if cache_path and os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_embeddings = data['embeddings']
        print(f"  Loaded embeddings with shape {cached_embeddings.shape}")
        # Ensure 2D shape (n_samples, embedding_dim)
        if cached_embeddings.ndim == 1:
            print(f"  Warning: embeddings are 1D, reshaping to (1, {cached_embeddings.shape[0]})")
            cached_embeddings = cached_embeddings.reshape(1, -1)
        return cached_embeddings

    if cache_path:
        print(f"[CACHE MISS] Will compute embeddings and save to {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Phi-4 model and processor
    model, processor, encoder, audio_projection, audio_feature_extractor = load_phi4_encoder(model_name, device)

    embeddings = []
    sample_rate = 16000

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Computing Phi-4 embeddings"):
            try:
                # Load audio
                audio = load_audio(audio_path, sr=sample_rate)

                # Use audio_feature_extractor directly to get mel features
                # This bypasses the full processor's text tokenization validation
                audio_inputs = audio_feature_extractor(
                    audios=[(audio, sample_rate)],
                    return_tensors='pt'
                )

                # Get the mel spectrogram features
                input_audio_embeds = audio_inputs.input_audio_embeds.to(device, dtype=torch.bfloat16)
                audio_attention_mask = audio_inputs.get('audio_attention_mask', None)
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(device)

                # Reshape mel features for encoder input
                # input_audio_embeds shape: (batch, num_audio, seq_len, mel_dim) or (batch, seq_len, mel_dim)
                if input_audio_embeds.dim() == 4:
                    # (batch, num_audio, seq_len, mel_dim) -> (num_audio, seq_len, mel_dim)
                    mel_features = input_audio_embeds.squeeze(0)
                elif input_audio_embeds.dim() == 3:
                    # (batch, seq_len, mel_dim) -> keep as is
                    mel_features = input_audio_embeds
                else:
                    mel_features = input_audio_embeds.unsqueeze(0)

                # Create attention mask if not provided
                if audio_attention_mask is None:
                    # Create a mask of all True (no padding)
                    audio_attention_mask = torch.ones(
                        mel_features.shape[:-1], dtype=torch.long, device=device
                    )

                # Pass through ConformerEncoder to get contextualized audio features
                # encoder expects (batch, seq_len, mel_dim) and returns (batch, seq_len, encoder_dim)
                encoder_output, masks = encoder(mel_features, audio_attention_mask)

                # Apply audio projection to map to LLM embedding space
                if use_projection and audio_projection is not None:
                    # audio_projection is a ModuleDict with 'speech' and 'vision' keys
                    if isinstance(audio_projection, torch.nn.ModuleDict):
                        audio_features = audio_projection['speech'](encoder_output)
                    else:
                        audio_features = audio_projection(encoder_output)
                else:
                    audio_features = encoder_output

                # Mean pooling across time dimension (seq_len)
                # audio_features shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
                if audio_features.dim() == 3:
                    embedding = audio_features.squeeze(0).mean(dim=0).float().cpu().numpy()
                else:
                    embedding = audio_features.mean(dim=0).float().cpu().numpy()
                embeddings.append(embedding)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                # Add zero embedding for failed samples
                if len(embeddings) > 0:
                    embed_dim = embeddings[-1].shape[-1]
                else:
                    embed_dim = 3584  # Phi-4's hidden dimension
                embeddings.append(np.zeros(embed_dim))

    embeddings = np.array(embeddings)

    # Cache embeddings if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, files=np.array(audio_files, dtype=object))
        print(f"Cached embeddings to {cache_path}")

    # Clean up GPU memory
    del model, processor, encoder, audio_projection, audio_feature_extractor
    torch.cuda.empty_cache()

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
    # Ensure embeddings are 2D (n_samples, embedding_dim)
    if voicebench_embeddings.ndim == 1:
        voicebench_embeddings = voicebench_embeddings.reshape(1, -1)
    if advbench_embeddings.ndim == 1:
        advbench_embeddings = advbench_embeddings.reshape(1, -1)

    print(f"  VoiceBench embeddings shape: {voicebench_embeddings.shape}")
    print(f"  AdvBench embeddings shape: {advbench_embeddings.shape}")

    if metric == "cosine":
        # Normalize embeddings
        vb_norm = voicebench_embeddings / (np.linalg.norm(voicebench_embeddings, axis=1, keepdims=True) + 1e-8)
        ab_norm = advbench_embeddings / (np.linalg.norm(advbench_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarity = vb_norm @ ab_norm.T  # (n_voicebench, n_advbench)

        # Convert to distance (1 - similarity)
        distances = 1 - similarity

    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        distances = cdist(voicebench_embeddings, advbench_embeddings, metric='euclidean')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Get minimum distance to any AdvBench sample
    min_distances = distances.min(axis=1)

    return min_distances, distances


def filter_samples(
    voicebench_json: str,
    advbench_audio_dir: str,
    output_json: str,
    threshold: float = None,
    top_k: int = None,
    metric: str = "cosine",
    cache_dir: str = None,
    model_name: str = "microsoft/Phi-4-multimodal-instruct",
    filter_closest: bool = False,
):
    """
    Filter VoiceBench samples based on distance to AdvBench samples.

    By default, keeps samples FARTHEST from AdvBench (most benign).
    Use filter_closest=True to keep samples closest to AdvBench.

    Args:
        voicebench_json: Path to VoiceBench samples JSON
        advbench_audio_dir: Directory containing AdvBench audio files
        output_json: Path to save filtered VoiceBench samples
        threshold: Distance threshold
        top_k: Keep top-k samples
        metric: Distance metric to use
        cache_dir: Directory to cache embeddings
        model_name: Phi-4 model name
        filter_closest: If True, keep closest samples; if False, keep farthest (default)
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
    if len(voicebench_audio_files) > 0:
        print(f"First audio path: {voicebench_audio_files[0]}")

    # Load AdvBench audio files
    advbench_audio_dir = Path(advbench_audio_dir)
    advbench_audio_files = sorted(list(advbench_audio_dir.glob("*.mp3")) +
                                 list(advbench_audio_dir.glob("*.wav")))
    advbench_audio_files = [str(f) for f in advbench_audio_files]
    print(f"Found {len(advbench_audio_files)} AdvBench samples")

    # Set up cache paths
    vb_cache = None
    ab_cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vb_cache = str(cache_dir / "voicebench_embeddings_phi4.npz")
        ab_cache = str(cache_dir / "advbench_embeddings_phi4.npz")

    # Compute embeddings using Phi-4
    print(f"\n=== Computing VoiceBench embeddings using Phi-4 encoder ===")
    voicebench_embeddings = compute_phi4_embeddings(
        voicebench_audio_files,
        model_name=model_name,
        cache_path=vb_cache
    )

    print(f"\n=== Computing AdvBench embeddings using Phi-4 encoder ===")
    advbench_embeddings = compute_phi4_embeddings(
        advbench_audio_files,
        model_name=model_name,
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

    # Print percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(min_distances, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Filter samples
    if top_k is not None:
        if filter_closest:
            # Keep top-k samples with SMALLEST distance (closest to AdvBench)
            sorted_indices = np.argsort(min_distances)[:top_k]
            print(f"\n=== Keeping top {top_k} CLOSEST samples ===")
        else:
            # Keep top-k samples with LARGEST distance (farthest from AdvBench = most benign)
            sorted_indices = np.argsort(min_distances)[-top_k:][::-1]
            print(f"\n=== Keeping top {top_k} FARTHEST samples (most benign) ===")

        filtered_indices = sorted_indices
        print(f"  Min distance in selected: {min_distances[sorted_indices].min():.4f}")
        print(f"  Max distance in selected: {min_distances[sorted_indices].max():.4f}")
        print(f"  Mean distance in selected: {min_distances[sorted_indices].mean():.4f}")

    elif threshold is not None:
        if filter_closest:
            # Keep samples with distance <= threshold (close to AdvBench)
            filtered_indices = np.where(min_distances <= threshold)[0]
            print(f"\n=== Filtering with threshold <= {threshold} (closest) ===")
        else:
            # Keep samples with distance >= threshold (far from AdvBench = benign)
            filtered_indices = np.where(min_distances >= threshold)[0]
            print(f"\n=== Filtering with threshold >= {threshold} (most benign) ===")

        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    else:
        # Auto-select threshold (75th percentile for benign samples)
        threshold = np.percentile(min_distances, 75)
        filtered_indices = np.where(min_distances >= threshold)[0]
        print(f"\n=== Auto-selected threshold: {threshold:.4f} (75th percentile, most benign) ===")
        print(f"  Kept {len(filtered_indices)} / {len(voicebench_data)} samples ({100*len(filtered_indices)/len(voicebench_data):.1f}%)")

    # Create filtered dataset
    filtered_data = [voicebench_data[i] for i in filtered_indices]

    # Add distance metadata
    for i, idx in enumerate(filtered_indices):
        filtered_data[i]["min_distance_to_advbench"] = float(min_distances[idx])
        closest_advbench_idx = np.argmin(all_distances[idx])
        filtered_data[i]["closest_advbench_file"] = advbench_audio_files[closest_advbench_idx]

    # Sort by distance (farthest first for benign, closest first for harmful)
    filtered_data.sort(key=lambda x: x["min_distance_to_advbench"], reverse=not filter_closest)

    # Save filtered data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\n=== Saved filtered dataset ===")
    print(f"  Output: {output_json}")
    print(f"  Samples: {len(filtered_data)}")

    # Save distance analysis
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.npz"
    np.savez(
        analysis_path,
        min_distances=min_distances,
        all_distances=all_distances,
        filtered_indices=filtered_indices,
        voicebench_embeddings=voicebench_embeddings,
        advbench_embeddings=advbench_embeddings,
        voicebench_files=np.array(voicebench_audio_files, dtype=object),
        advbench_files=np.array(advbench_audio_files, dtype=object)
    )
    print(f"  Distance analysis: {analysis_path}")

    # Print some examples
    filter_type = "closest" if filter_closest else "farthest (most benign)"
    print(f"\n=== Top 10 {filter_type} VoiceBench samples ===")
    for i in range(min(10, len(filtered_data))):
        sample = filtered_data[i]
        print(f"  {i+1}. Distance: {sample['min_distance_to_advbench']:.4f}")
        if 'conversations' in sample and len(sample['conversations']) > 0:
            question = sample['conversations'][0].get('value', '')
            if len(question) > 80:
                question = question[:80] + "..."
            print(f"     Question: {question}")

    return str(output_json)


def main():
    parser = argparse.ArgumentParser(description="Filter VoiceBench samples using Phi-4 embeddings")
    parser.add_argument("--voicebench_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_full.json",
                       help="Path to VoiceBench samples JSON file")
    parser.add_argument("--advbench_audio_dir", type=str,
                       default="data/advbench/en",
                       help="Directory containing AdvBench audio files")
    parser.add_argument("--output_json", type=str,
                       default="data/voicebench/sd-qa/sd_qa_filtered_phi4.json",
                       help="Path to save filtered VoiceBench samples")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Distance threshold for filtering")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Keep top-k samples")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--cache_dir", type=str, default="data/embedding_cache_phi4",
                       help="Directory to cache embeddings")
    parser.add_argument("--model_name", type=str, default="/datasets/ai/phi/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
                       help="Phi-4 model name")
    parser.add_argument("--filter_closest", action="store_true",
                       help="Keep samples closest to AdvBench (default: keep farthest/most benign)")

    args = parser.parse_args()

    filter_samples(
        voicebench_json=args.voicebench_json,
        advbench_audio_dir=args.advbench_audio_dir,
        output_json=args.output_json,
        threshold=args.threshold,
        top_k=args.top_k,
        metric=args.metric,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        filter_closest=args.filter_closest,
    )


if __name__ == "__main__":
    main()
