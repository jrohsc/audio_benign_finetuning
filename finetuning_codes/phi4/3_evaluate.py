#!/usr/bin/env python3
"""
Evaluate Phi-4 Multimodal on AdvBench jailbreaking dataset.

Compares responses from:
1. Pretrained model
2. Finetuned model(s) from checkpoints

Usage:
    python 3_evaluate.py \
        --harmful_data_dir ../harmful_data/advbench_gtts/en \
        --output_dir results/advbench_eval \
        --pretrained_path phi4_model_complete \
        --finetuned_path checkpoints/phi4_voicebench_finetuned_thresh_0.25
"""

import argparse
import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
from glob import glob

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
import logging
import librosa

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def load_model(model_path: str, device: str = "cuda", use_flash_attention: bool = False):
    """Load Phi-4 model and processor from path."""
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        audio_enabled=True
    )

    # Set attention implementation
    if use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Load generation config from model
    generation_config = GenerationConfig.from_pretrained(model_path)
    logger.info(f"Loaded generation config: {generation_config}")

    model.eval()

    return model, processor, generation_config


def load_finetuned_model(
    base_model_path: str,
    finetuned_path: str,
    device: str = "cuda",
    use_flash_attention: bool = False
):
    """Load finetuned Phi-4 model.

    The finetuned model saves only the audio components, so we need to:
    1. Load the base model
    2. Load the finetuned weights for audio components
    """
    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading finetuned weights from {finetuned_path}")

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        audio_enabled=True
    )

    if use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Check if finetuned path contains model files
    finetuned_path = Path(finetuned_path)

    # Look for safetensors or pytorch model files
    safetensor_files = list(finetuned_path.glob("*.safetensors"))
    pytorch_files = list(finetuned_path.glob("pytorch_model*.bin"))

    if safetensor_files or pytorch_files:
        logger.info("Loading finetuned weights...")

        # Load state dict from finetuned checkpoint
        from safetensors.torch import load_file

        state_dict = {}
        for sf_file in safetensor_files:
            state_dict.update(load_file(sf_file))

        # Load the finetuned weights (only audio components)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded finetuned weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        if len(missing) > 0:
            logger.debug(f"Missing keys (first 10): {missing[:10]}")
    else:
        logger.warning(f"No model files found in {finetuned_path}, using base model")

    # Load generation config from base model
    generation_config = GenerationConfig.from_pretrained(base_model_path)
    logger.info(f"Loaded generation config: {generation_config}")

    model.eval()
    return model, processor, generation_config


def get_audio_files(harmful_data_dir: str) -> List[str]:
    """Get all audio files from directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob(os.path.join(harmful_data_dir, ext)))

    audio_files = sorted(audio_files)
    logger.info(f"Found {len(audio_files)} audio files in {harmful_data_dir}")
    return audio_files


def load_benchmark_prompts(benchmark_csv: str) -> Dict[int, str]:
    """Load text prompts from benchmark CSV file.

    Supports:
    - advbench.csv: uses 'goal' column
    - safetybench.csv: uses 'question' column

    Returns a dict mapping audio file index to text prompt.
    """
    prompts = {}

    with open(benchmark_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Determine which column contains the prompt
        fieldnames = reader.fieldnames
        if 'goal' in fieldnames:
            prompt_column = 'goal'
        elif 'question' in fieldnames:
            prompt_column = 'question'
        else:
            raise ValueError(f"Could not find 'goal' or 'question' column in {benchmark_csv}. "
                           f"Available columns: {fieldnames}")

        logger.info(f"Using '{prompt_column}' column from {benchmark_csv}")

        for idx, row in enumerate(reader):
            prompts[idx] = row[prompt_column]

    logger.info(f"Loaded {len(prompts)} text prompts from {benchmark_csv}")
    return prompts


def get_audio_index(audio_filename: str) -> Optional[int]:
    """Extract numeric index from audio filename (e.g., 'advbench_00042.mp3' -> 42)."""
    stem = Path(audio_filename).stem
    # Handle formats like "advbench_00042" or just "42"
    try:
        # Try extracting number after underscore
        if '_' in stem:
            num_str = stem.split('_')[-1]
            return int(num_str)
        else:
            return int(stem)
    except ValueError:
        return None


def run_inference(
    model,
    processor,
    audio_path: str,
    generation_config,
    prompt: str = "Please respond to the request.",
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """Run inference on a single audio file."""
    # Load audio
    audio = load_audio(audio_path, sr=16000)

    # Create user message with audio placeholder (following VoiceBench format)
    user_message = {
        'role': 'user',
        'content': f'<|audio_1|>',
    }

    # Apply chat template
    chat_prompt = processor.tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )

    # IMPORTANT: Remove trailing <|endoftext|> if present - it's for training, not inference
    if chat_prompt.endswith('<|endoftext|>'):
        chat_prompt = chat_prompt.rstrip('<|endoftext|>')

    # Process with audio
    inputs = processor(
        text=chat_prompt,
        audios=[(audio, 16000)],
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_logits_to_keep=0,  # Memory efficiency
            generation_config=generation_config,
        )

    # Decode only the generated tokens
    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[:, input_len:]
    response = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response.strip()


def evaluate_model(
    model,
    processor,
    generation_config,
    audio_files: List[str],
    prompt: str,
    max_new_tokens: int = 512,
    benchmark_prompts: Optional[Dict[int, str]] = None,
    output_file: Optional[Path] = None,
    device: str = "cuda",
) -> List[Dict]:
    """Evaluate model on all audio files.

    Args:
        model: The model to evaluate
        processor: The processor for the model
        generation_config: The generation config for the model
        audio_files: List of audio file paths
        prompt: The inference prompt to use
        max_new_tokens: Maximum tokens to generate
        benchmark_prompts: Optional dict mapping audio index to text prompt from benchmark CSV
        output_file: Optional path to write results incrementally after each sample
        device: Device to use for inference
    """
    results = []

    for audio_path in tqdm(audio_files, desc="Evaluating"):
        audio_name = os.path.basename(audio_path)

        # Get the corresponding text prompt from benchmark if available
        text_prompt = None
        if benchmark_prompts is not None:
            audio_idx = get_audio_index(audio_name)
            if audio_idx is not None and audio_idx in benchmark_prompts:
                text_prompt = benchmark_prompts[audio_idx]

        try:
            response = run_inference(
                model, processor, audio_path, generation_config, prompt, max_new_tokens, device
            )
            result_entry = {
                "audio_file": audio_name,
                "audio_path": audio_path,
                "prompt": prompt,
                "response": response,
                "error": None,
            }
            if text_prompt is not None:
                result_entry["text_prompt"] = text_prompt
            results.append(result_entry)
        except Exception as e:
            logger.error(f"Error processing {audio_name}: {e}")
            result_entry = {
                "audio_file": audio_name,
                "audio_path": audio_path,
                "prompt": prompt,
                "response": None,
                "error": str(e),
            }
            if text_prompt is not None:
                result_entry["text_prompt"] = text_prompt
            results.append(result_entry)

        # Write results incrementally after each sample
        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phi-4 Multimodal on AdvBench jailbreaking dataset"
    )

    # Data arguments
    parser.add_argument(
        "--harmful_data_dir",
        type=str,
        default="../harmful_data/advbench_gtts/en",
        help="Directory containing AdvBench audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/advbench_eval",
        help="Directory to save evaluation results"
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="phi4_model_complete",
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint (or comma-separated list of paths)"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["both", "pretrained_only", "finetuned_only"],
        default="both",
        help="Evaluation mode: 'pretrained_only', 'finetuned_only', or 'both' (default)"
    )

    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please respond to the request.",
        help="Prompt to use for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default="../harmful_data/advbench.csv",
        help="Path to benchmark CSV file to include text prompts in results"
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable Flash Attention 2"
    )

    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio files
    audio_files = get_audio_files(args.harmful_data_dir)

    if args.max_samples:
        audio_files = audio_files[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    if not audio_files:
        logger.error(f"No audio files found in {args.harmful_data_dir}")
        return

    # Load benchmark prompts if CSV provided
    benchmark_prompts = None
    if args.benchmark_csv and os.path.exists(args.benchmark_csv):
        benchmark_prompts = load_benchmark_prompts(args.benchmark_csv)
    elif args.benchmark_csv:
        logger.warning(f"Benchmark CSV not found: {args.benchmark_csv}")

    # Save evaluation config
    config = {
        "harmful_data_dir": args.harmful_data_dir,
        "pretrained_path": args.pretrained_path,
        "finetuned_path": args.finetuned_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": len(audio_files),
        "benchmark_csv": args.benchmark_csv,
        "eval_mode": args.eval_mode,
        "use_flash_attention": not args.no_flash_attention,
    }
    with open(output_dir / "eval_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    all_results = {}
    use_flash_attention = not args.no_flash_attention

    # Determine what to evaluate
    eval_pretrained = args.eval_mode in ["both", "pretrained_only"]
    eval_finetuned = args.eval_mode in ["both", "finetuned_only"]

    if eval_finetuned and not args.finetuned_path:
        logger.warning("Finetuned evaluation requested but --finetuned_path not provided. Skipping.")
        eval_finetuned = False

    logger.info(f"Evaluation mode: pretrained={eval_pretrained}, finetuned={eval_finetuned}")

    # Clear cached model modules
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules/phi4_model_complete"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        logger.info("Cleared cached model modules")

    # Evaluate pretrained model
    if eval_pretrained:
        logger.info("\n=== Evaluating Pretrained Model ===")
        model, processor, generation_config = load_model(args.pretrained_path, device, use_flash_attention)

        pretrained_output_file = output_dir / "pretrained_responses.json"
        pretrained_results = evaluate_model(
            model, processor, generation_config, audio_files, args.prompt, args.max_new_tokens,
            benchmark_prompts=benchmark_prompts,
            output_file=pretrained_output_file,
            device=device
        )

        all_results["pretrained"] = pretrained_results
        logger.info(f"Saved pretrained results to {pretrained_output_file}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Evaluate finetuned model(s)
    if eval_finetuned and args.finetuned_path:
        finetuned_paths = [p.strip() for p in args.finetuned_path.split(',')]

        for ft_path in finetuned_paths:
            if not os.path.exists(ft_path):
                logger.warning(f"Finetuned path not found: {ft_path}")
                continue

            logger.info(f"\n=== Evaluating Finetuned Model: {ft_path} ===")

            try:
                model, processor, generation_config = load_finetuned_model(
                    args.pretrained_path, ft_path, device, use_flash_attention
                )
            except Exception as e:
                logger.error(f"Failed to load finetuned model: {e}")
                continue

            # Create safe filename from path
            ft_name = Path(ft_path).name
            finetuned_output_file = output_dir / f"finetuned_{ft_name}_responses.json"

            finetuned_results = evaluate_model(
                model, processor, generation_config, audio_files, args.prompt, args.max_new_tokens,
                benchmark_prompts=benchmark_prompts,
                output_file=finetuned_output_file,
                device=device
            )

            all_results[f"finetuned_{ft_name}"] = finetuned_results
            logger.info(f"Saved finetuned results to {finetuned_output_file}")

            # Free memory
            del model
            torch.cuda.empty_cache()

    # Save combined results
    with open(output_dir / "all_responses.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n=== Evaluation Complete ===")
    logger.info(f"Results saved to: {output_dir}")

    # Print summary
    for model_name, results in all_results.items():
        num_success = sum(1 for r in results if r['error'] is None)
        num_error = sum(1 for r in results if r['error'] is not None)
        logger.info(f"{model_name}: {num_success} successful, {num_error} errors")


if __name__ == "__main__":
    main()
