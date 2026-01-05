#!/usr/bin/env python3
"""
Evaluate Audio Flamingo 3 on AdvBench dataset.

Compares responses from:
1. Pretrained model
2. Finetuned model(s) from checkpoints

Usage:
    python 3_evaluate_advbench.py \
        --HARMFUL_DATA_DIR advbench/en \
        --output_dir results/advbench_eval \
        --pretrained_path /path/to/pretrained \
        --finetuned_path checkpoints/af3_finetuned_thresh_0.025/best_model
"""

import argparse
import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
from glob import glob

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda"):
    """Load model and processor from path."""
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


def load_finetuned_model(base_model_path: str, adapter_path: str, device: str = "cuda"):
    """Load finetuned model with LoRA adapters."""
    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapters from {adapter_path}")

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Check if this is a LoRA checkpoint
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftModel
        logger.info("Loading LoRA adapters")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Full model checkpoint - load directly
        logger.info("Loading full model checkpoint")
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            adapter_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, processor


def get_audio_files(HARMFUL_DATA_DIR: str) -> List[str]:
    """Get all audio files from advbench directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob(os.path.join(HARMFUL_DATA_DIR, ext)))

    audio_files = sorted(audio_files)
    logger.info(f"Found {len(audio_files)} audio files in {HARMFUL_DATA_DIR}")
    return audio_files


def load_benchmark_prompts(benchmark_csv: str) -> Dict[int, str]:
    """Load text prompts from benchmark CSV file.

    Supports:
    - safetybench.csv: uses 'question' column
    - advbench.csv: uses 'goal' column

    Returns a dict mapping audio file index to text prompt.
    """
    prompts = {}

    with open(benchmark_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Determine which column contains the prompt
        fieldnames = reader.fieldnames
        if 'question' in fieldnames:
            prompt_column = 'question'
        elif 'goal' in fieldnames:
            prompt_column = 'goal'
        else:
            raise ValueError(f"Could not find 'question' or 'goal' column in {benchmark_csv}. "
                           f"Available columns: {fieldnames}")

        logger.info(f"Using '{prompt_column}' column from {benchmark_csv}")

        for idx, row in enumerate(reader):
            prompts[idx] = row[prompt_column]

    logger.info(f"Loaded {len(prompts)} text prompts from {benchmark_csv}")
    return prompts


def get_audio_index(audio_filename: str) -> Optional[int]:
    """Extract numeric index from audio filename (e.g., 'advbench_00042.mp3' -> 42, '42.mp3' -> 42)."""
    stem = Path(audio_filename).stem
    try:
        # Try extracting number after underscore (e.g., advbench_00042, safetybench_00000)
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
    prompt: str = "What is the person saying? Please respond to their request.",
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a single audio file."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to model device with correct dtype
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v.to(model.device) if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return response.strip()


def evaluate_model(
    model,
    processor,
    audio_files: List[str],
    prompt: str,
    max_new_tokens: int = 512,
    benchmark_prompts: Optional[Dict[int, str]] = None,
    output_file: Optional[Path] = None,
) -> List[Dict]:
    """Evaluate model on all audio files.

    Args:
        model: The model to evaluate
        processor: The processor for the model
        audio_files: List of audio file paths
        prompt: The inference prompt to use
        max_new_tokens: Maximum tokens to generate
        benchmark_prompts: Optional dict mapping audio index to text prompt from benchmark CSV
        output_file: Optional path to write results incrementally after each sample
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
                model, processor, audio_path, prompt, max_new_tokens
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
            logger.info(f"Updated {output_file} ({len(results)}/{len(audio_files)} samples)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Audio Flamingo 3 on AdvBench"
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
        default="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7",
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
        "--model_name",
        type=str,
        default=None,
        help="Name to use for output file (overrides using finetuned_path basename)"
    )

    args = parser.parse_args()

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
    if args.benchmark_csv:
        if os.path.exists(args.benchmark_csv):
            benchmark_prompts = load_benchmark_prompts(args.benchmark_csv)
        else:
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
    }
    with open(output_dir / "eval_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    all_results = {}

    # Determine what to evaluate
    eval_pretrained = args.eval_mode in ["both", "pretrained_only"]
    eval_finetuned = args.eval_mode in ["both", "finetuned_only"]

    if eval_finetuned and not args.finetuned_path:
        logger.warning("Finetuned evaluation requested but --finetuned_path not provided. Skipping.")
        eval_finetuned = False

    logger.info(f"Evaluation mode: pretrained={eval_pretrained}, finetuned={eval_finetuned}")

    # Evaluate pretrained model
    if eval_pretrained:
        logger.info("\n=== Evaluating Pretrained Model ===")
        model, processor = load_model(args.pretrained_path)

        pretrained_output_file = output_dir / "pretrained_responses.json"
        pretrained_results = evaluate_model(
            model, processor, audio_files, args.prompt, args.max_new_tokens,
            benchmark_prompts=benchmark_prompts,
            output_file=pretrained_output_file
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
                model, processor = load_finetuned_model(args.pretrained_path, ft_path)
            except Exception as e:
                logger.error(f"Failed to load finetuned model: {e}")
                # Try loading as full model
                try:
                    model, processor = load_model(ft_path)
                except Exception as e2:
                    logger.error(f"Also failed to load as full model: {e2}")
                    continue

            # Create safe filename from path (use --model_name if provided)
            ft_name = args.model_name if args.model_name else Path(ft_path).name
            finetuned_output_file = output_dir / f"finetuned_{ft_name}_responses.json"

            finetuned_results = evaluate_model(
                model, processor, audio_files, args.prompt, args.max_new_tokens,
                benchmark_prompts=benchmark_prompts,
                output_file=finetuned_output_file
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
