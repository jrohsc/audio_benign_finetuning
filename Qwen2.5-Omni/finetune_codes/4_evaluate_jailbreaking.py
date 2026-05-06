#!/usr/bin/env python3
"""
Evaluate Qwen2.5-Omni on AdvBench/SafetyBench datasets.

Compares responses from:
1. Pretrained model
2. Finetuned model(s) from checkpoints

Usage:
    python 4_evaluate_jailbreaking.py \
        --harmful_data_dir ../harmful_data/advbench_gtts/en \
        --output_dir results/advbench_eval \
        --pretrained_path /path/to/pretrained \
        --finetuned_path checkpoints/finetuned_model

    # Evaluate only finetuned model
    python 4_evaluate_jailbreaking.py \
        --harmful_data_dir ../harmful_data/advbench_gtts/en \
        --finetuned_path checkpoints/finetuned_model \
        --eval_mode finetuned_only
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda", use_flash_attn: bool = True):
    """Load Qwen2.5-Omni model and processor."""
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            logger.warning("flash_attn not installed, falling back to sdpa attention")
            load_kwargs["attn_implementation"] = "sdpa"

    # Try loading as Thinker model first (for text-only output, skips Talker/Token2Wav)
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        logger.info("Loaded as Thinker-only model (no Talker/Token2Wav overhead)")
    except Exception as e:
        logger.info(f"Could not load as Thinker model, trying full Omni model: {e}")
        # Disable Talker and audio output to save ~5GB VRAM (not needed for text inference)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.enable_audio_output = False
        config.enable_talker = False
        logger.info("Disabled Talker/audio output in config to save VRAM")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, config=config, **load_kwargs
        )

    model.eval()
    logger.info("Model loaded successfully!")

    return model, processor


def load_finetuned_model(base_model_path: str, adapter_path: str, device: str = "cuda", use_flash_attn: bool = True):
    """Load finetuned model with LoRA adapters."""
    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapters from {adapter_path}")

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            logger.warning("flash_attn not installed, falling back to sdpa attention")
            load_kwargs["attn_implementation"] = "sdpa"

    # Try loading as Thinker model first
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base_model_path, **load_kwargs
        )
        logger.info("Loaded base as Thinker-only model")
    except Exception as e:
        logger.info(f"Could not load as Thinker model, trying full Omni model: {e}")
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        config.enable_audio_output = False
        config.enable_talker = False
        logger.info("Disabled Talker/audio output in config to save VRAM")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            base_model_path, config=config, **load_kwargs
        )

    # Check if this is a LoRA checkpoint
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftModel
        logger.info("Loading LoRA adapters")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Full/merged model checkpoint - free the base model first, then load directly
        logger.info("Loading merged model checkpoint (freeing base model first)")
        del model
        torch.cuda.empty_cache()
        model = load_model(adapter_path, device=device, use_flash_attn=use_flash_attn)[0]
        # Reload processor from the finetuned path if available, else keep base processor
        try:
            processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            pass  # Keep base processor

    model.eval()
    return model, processor


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


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file at specified sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def postprocess_response(response: str) -> str:
    """Clean up model response."""
    # Extract only the assistant response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
        if response.startswith(":"):
            response = response[1:].strip()

    # Truncate at "Human:" if model generates fake conversation turns (common overfitting artifact)
    if "\nHuman:" in response:
        response = response.split("\nHuman:")[0].strip()
    elif "Human:" in response:
        response = response.split("Human:")[0].strip()

    return response.strip()


def run_inference(
    model,
    processor,
    audio_path: str,
    prompt: str = "Please respond to the request.",
    max_new_tokens: int = 512,
    system_prompt: str = None,
    temperature: float = None,
) -> str:
    """Run inference on a single audio file."""
    # Load audio at 16kHz
    audio = load_audio(audio_path, sr=16000)

    # Prepare conversation
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    conversation.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt}
        ]
    })

    # Process with the model
    text_input = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Prepare inputs
    inputs = processor(
        text=text_input,
        audio=audio,
        return_tensors="pt",
        padding=True
    )

    # Move inputs to model device
    inputs = {
        k: v.to(model.device) if hasattr(v, 'to') else v
        for k, v in inputs.items()
    }

    # Generate
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_audio_in_video=False,
    )
    if temperature is not None and temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    # Decode
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return postprocess_response(response)


def run_batch_inference(
    model,
    processor,
    audio_paths: List[str],
    prompt: str = "Please respond to the request.",
    max_new_tokens: int = 512,
    system_prompt: str = None,
    temperature: float = None,
) -> List[str]:
    """Run inference on a batch of audio files.

    Note: Qwen2.5-Omni processes each audio separately but we can parallelize
    the audio loading and reduce overhead by batching the generation loop.
    """
    import concurrent.futures

    # Pre-load all audio files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        audios = list(executor.map(lambda p: load_audio(p, sr=16000), audio_paths))

    responses = []

    # Process each sample (Qwen2.5-Omni doesn't support true batch audio processing)
    # But we've already pre-loaded the audio, which saves time
    for audio_path, audio in zip(audio_paths, audios):
        try:
            # Prepare conversation
            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt}
                ]
            })

            # Process with the model
            text_input = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            # Prepare inputs
            inputs = processor(
                text=text_input,
                audio=audio,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to model device
            inputs = {
                k: v.to(model.device) if hasattr(v, 'to') else v
                for k, v in inputs.items()
            }

            # Generate
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_audio_in_video=False,
            )
            if temperature is not None and temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = 0.9
            else:
                generate_kwargs["do_sample"] = False

            with torch.inference_mode():
                outputs = model.generate(**generate_kwargs)

            # Decode
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            responses.append(postprocess_response(response))
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            responses.append(None)

    return responses


def evaluate_model(
    model,
    processor,
    audio_files: List[str],
    prompt: str,
    max_new_tokens: int = 512,
    benchmark_prompts: Optional[Dict[int, str]] = None,
    output_file: Optional[Path] = None,
    batch_size: int = 1,
    resume: bool = False,
    system_prompt: str = None,
    temperature: float = None,
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
        batch_size: Number of samples to process in parallel (pre-loads audio)
        resume: If True, resume from existing partial results
        system_prompt: Optional system prompt for safety defense
    """
    results = []
    start_idx = 0

    # Handle resume logic
    if resume and output_file is not None and output_file.exists():
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            start_idx = len(results)
            logger.info(f"[RESUME] Loaded {start_idx} existing results from: {output_file}")
            logger.info(f"         Continuing from sample {start_idx}")

            if start_idx >= len(audio_files):
                logger.info(f"[COMPLETE] All {len(audio_files)} samples already processed!")
                return results
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}. Starting fresh.")
            results = []
            start_idx = 0

    # Get remaining audio files
    remaining_audio_files = audio_files[start_idx:]

    if not remaining_audio_files:
        logger.info("No remaining samples to process.")
        return results

    logger.info(f"Processing {len(remaining_audio_files)} samples (starting from {start_idx})")

    # Process in batches
    num_batches = (len(remaining_audio_files) + batch_size - 1) // batch_size

    with tqdm(total=len(audio_files), initial=start_idx, desc="Evaluating") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_audio_files))
            batch_audio_files = remaining_audio_files[batch_start:batch_end]

            # Run batch inference (pre-loads audio in parallel)
            batch_responses = run_batch_inference(
                model, processor, batch_audio_files, prompt, max_new_tokens,
                system_prompt=system_prompt, temperature=temperature,
            )

            # Process results
            for i, (audio_path, response) in enumerate(zip(batch_audio_files, batch_responses)):
                audio_name = os.path.basename(audio_path)

                # Get the corresponding text prompt from benchmark if available
                text_prompt = None
                if benchmark_prompts is not None:
                    audio_idx = get_audio_index(audio_name)
                    if audio_idx is not None and audio_idx in benchmark_prompts:
                        text_prompt = benchmark_prompts[audio_idx]

                if response is not None:
                    result_entry = {
                        "audio_file": audio_name,
                        "audio_path": audio_path,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": response,
                        "error": None,
                    }
                else:
                    result_entry = {
                        "audio_file": audio_name,
                        "audio_path": audio_path,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": None,
                        "error": "Inference failed",
                    }

                if text_prompt is not None:
                    result_entry["text_prompt"] = text_prompt
                results.append(result_entry)

            pbar.update(len(batch_audio_files))

            # Write results incrementally after each batch
            if output_file is not None:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Updated {output_file} ({len(results)}/{len(audio_files)} samples)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni on AdvBench/SafetyBench"
    )

    # Data arguments
    parser.add_argument(
        "--harmful_data_dir",
        type=str,
        default="../../harmful_data/advbench_gtts/en",
        help="Directory containing harmful audio files"
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
        default="/project/anon/Qwen2.5-Omni-7B",
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
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (pre-loads audio in parallel, default: 8)"
    )
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default="../../harmful_data/advbench.csv",
        help="Path to benchmark CSV file to include text prompts in results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name to use for output file (overrides using finetuned_path basename)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial results if available"
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention 2 (enabled by default for faster inference)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for faster repeated generation (adds startup time)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt for safety defense (inserted as system message)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (None=greedy, e.g. 0.7 for sampling)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()
    use_flash_attn = not args.no_flash_attn

    if args.seed is not None:
        set_seed(args.seed)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled TF32 and cuDNN benchmark for faster computation")

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
        "batch_size": args.batch_size,
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

        pretrained_output_file = output_dir / "pretrained_responses.json"

        # Check if we can skip (already complete)
        if args.resume and pretrained_output_file.exists():
            try:
                with open(pretrained_output_file, 'r') as f:
                    existing_results = json.load(f)
                if len(existing_results) >= len(audio_files):
                    logger.info(f"[SKIP] Pretrained evaluation already complete ({len(existing_results)}/{len(audio_files)})")
                    all_results["pretrained"] = existing_results
                    eval_pretrained = False  # Skip loading model
            except Exception:
                pass

        if eval_pretrained:
            model, processor = load_model(args.pretrained_path, use_flash_attn=use_flash_attn)
            if args.compile:
                logger.info("Compiling model with torch.compile()...")
                model = torch.compile(model)

            pretrained_results = evaluate_model(
                model, processor, audio_files, args.prompt, args.max_new_tokens,
                benchmark_prompts=benchmark_prompts,
                output_file=pretrained_output_file,
                batch_size=args.batch_size,
                resume=args.resume,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
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

            # Create safe filename from path (use --model_name if provided)
            ft_name = args.model_name if args.model_name else Path(ft_path).name
            finetuned_output_file = output_dir / f"finetuned_{ft_name}_responses.json"

            # Check if we can skip (already complete)
            skip_this_model = False
            if args.resume and finetuned_output_file.exists():
                try:
                    with open(finetuned_output_file, 'r') as f:
                        existing_results = json.load(f)
                    if len(existing_results) >= len(audio_files):
                        logger.info(f"[SKIP] Finetuned evaluation already complete ({len(existing_results)}/{len(audio_files)})")
                        all_results[f"finetuned_{ft_name}"] = existing_results
                        skip_this_model = True
                except Exception:
                    pass

            if skip_this_model:
                continue

            try:
                model, processor = load_finetuned_model(args.pretrained_path, ft_path, use_flash_attn=use_flash_attn)
            except Exception as e:
                logger.error(f"Failed to load finetuned model: {e}")
                # Try loading as full model
                try:
                    model, processor = load_model(ft_path, use_flash_attn=use_flash_attn)
                except Exception as e2:
                    logger.error(f"Also failed to load as full model: {e2}")
                    continue
            if args.compile:
                logger.info("Compiling model with torch.compile()...")
                model = torch.compile(model)

            finetuned_results = evaluate_model(
                model, processor, audio_files, args.prompt, args.max_new_tokens,
                benchmark_prompts=benchmark_prompts,
                output_file=finetuned_output_file,
                batch_size=args.batch_size,
                resume=args.resume,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
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
