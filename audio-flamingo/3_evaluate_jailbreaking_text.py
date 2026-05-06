#!/usr/bin/env python3
"""
Evaluate text LLM on harmful prompts (text-only, no audio).

This is the text-only counterpart to 3_evaluate_jailbreaking.py.
Instead of evaluating on audio prompts, this evaluates on text prompts directly.

Compares responses from:
1. Pretrained model
2. Finetuned model(s) from checkpoints_text_bft/

Usage:
    python 3_evaluate_jailbreaking_text.py \
        --benchmark_csv ../harmful_data/advbench.csv \
        --output_dir results_text/advbench_eval \
        --pretrained_path /path/to/pretrained \
        --finetuned_path checkpoints_text_bft/llm_finetuned_percentage_50/best_model
"""

import argparse
import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from tqdm import tqdm
import logging

# Try to import AudioFlamingo3 model class
try:
    from transformers import AudioFlamingo3ForConditionalGeneration
    HAS_AUDIOFLAMINGO_IMPORT = True
except ImportError:
    HAS_AUDIOFLAMINGO_IMPORT = False
    AudioFlamingo3ForConditionalGeneration = None
    import transformers
    print(f"Warning: AudioFlamingo3 not available in transformers {transformers.__version__}. "
          "This model requires transformers 5.0.0.dev0 or later. "
          "Install with: pip install git+https://github.com/huggingface/transformers.git")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda", use_flash_attention: bool = True):
    """Load model and tokenizer from path."""
    logger.info(f"Loading model from {model_path}")

    # Check if it's a local path
    is_local = model_path.startswith('/') or os.path.isdir(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": is_local,
    }

    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    return model, tokenizer


def load_finetuned_model(base_model_path: str, adapter_path: str, device: str = "cuda", use_flash_attention: bool = True):
    """Load finetuned model with LoRA adapters."""

    # Check if this is a LoRA checkpoint and get base model from adapter config
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    is_audioflamingo = False

    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        # Use base model from adapter config if available
        adapter_base_model = adapter_config.get("base_model_name_or_path")
        if adapter_base_model and os.path.isdir(adapter_base_model):
            logger.info(f"Using base model from adapter config: {adapter_base_model}")
            base_model_path = adapter_base_model
            # Check if this is an AudioFlamingo model
            if "audio-flamingo" in adapter_base_model.lower() or "audioflamingo" in adapter_base_model.lower():
                is_audioflamingo = True

    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapters from {adapter_path}")

    # Check if it's a local path
    is_local = base_model_path.startswith('/') or os.path.isdir(base_model_path)

    # Use appropriate model class based on model type
    if is_audioflamingo:
        if not HAS_AUDIOFLAMINGO_IMPORT:
            raise ImportError(
                f"AudioFlamingo3 model detected but AudioFlamingo3ForConditionalGeneration is not available. "
                f"Your transformers version does not support AudioFlamingo3. "
                f"Please upgrade transformers: pip install git+https://github.com/huggingface/transformers.git"
            )

        logger.info("Detected AudioFlamingo model, loading with trust_remote_code=True")
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA adapters if config exists
        if adapter_config_path.exists():
            from peft import PeftModel
            logger.info("Loading LoRA adapters")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model.eval()
        # Return processor's tokenizer for compatibility
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    # Fall back to standard AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=is_local,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": is_local,
    }

    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    # Apply LoRA adapters if config exists
    if adapter_config_path.exists():
        from peft import PeftModel
        logger.info("Loading LoRA adapters")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Full model checkpoint - load directly
        logger.info("Loading full model checkpoint")
        adapter_is_local = adapter_path.startswith('/') or os.path.isdir(adapter_path)
        model_kwargs["local_files_only"] = adapter_is_local
        model = AutoModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)

    model.eval()
    return model, tokenizer


def load_benchmark_prompts(benchmark_csv: str) -> List[Dict[str, str]]:
    """Load text prompts from benchmark CSV file.

    Supports:
    - safetybench.csv: uses 'question' column
    - advbench.csv: uses 'goal' column

    Returns a list of dicts with 'id' and 'prompt' keys.
    """
    prompts = []

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
            prompts.append({
                'id': f"{Path(benchmark_csv).stem}_{idx:05d}",
                'prompt': row[prompt_column]
            })

    logger.info(f"Loaded {len(prompts)} text prompts from {benchmark_csv}")
    return prompts


def run_inference(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a single text prompt."""
    # Format as chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback formatting
        text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return response.strip()


def evaluate_model(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    system_prompt: str,
    max_new_tokens: int = 512,
    output_file: Optional[Path] = None,
) -> List[Dict]:
    """Evaluate model on all text prompts.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        prompts: List of dicts with 'id' and 'prompt' keys
        system_prompt: System prompt to use
        max_new_tokens: Maximum tokens to generate
        output_file: Optional path to write results incrementally after each sample
    """
    results = []

    for item in tqdm(prompts, desc="Evaluating"):
        prompt_id = item['id']
        prompt = item['prompt']

        try:
            response = run_inference(
                model, tokenizer, prompt, system_prompt, max_new_tokens
            )
            result_entry = {
                "id": prompt_id,
                "text_prompt": prompt,
                "system_prompt": system_prompt,
                "response": response,
                "error": None,
            }
            results.append(result_entry)
        except Exception as e:
            logger.error(f"Error processing {prompt_id}: {e}")
            result_entry = {
                "id": prompt_id,
                "text_prompt": prompt,
                "system_prompt": system_prompt,
                "response": None,
                "error": str(e),
            }
            results.append(result_entry)

        # Write results incrementally after each sample
        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text LLM on harmful prompts (text-only)"
    )

    # Data arguments
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default="../harmful_data/advbench.csv",
        help="Path to benchmark CSV file with text prompts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_text/advbench_eval",
        help="Directory to save evaluation results"
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/datasets/ai/llama/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
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
        "--system_prompt",
        type=str,
        default="You are a helpful assistant who tries to help answer the user's question.",
        help="System prompt to use for inference"
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
        "--model_name",
        type=str,
        default=None,
        help="Name to use for output file (overrides using finetuned_path basename)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    if not os.path.exists(args.benchmark_csv):
        logger.error(f"Benchmark CSV not found: {args.benchmark_csv}")
        return

    prompts = load_benchmark_prompts(args.benchmark_csv)

    if args.max_samples:
        prompts = prompts[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    if not prompts:
        logger.error(f"No prompts loaded from {args.benchmark_csv}")
        return

    # Save evaluation config
    config = {
        "benchmark_csv": args.benchmark_csv,
        "pretrained_path": args.pretrained_path,
        "finetuned_path": args.finetuned_path,
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": len(prompts),
        "eval_mode": args.eval_mode,
        "modality": "text",
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
        model, tokenizer = load_model(args.pretrained_path)

        pretrained_output_file = output_dir / "pretrained_responses.json"
        pretrained_results = evaluate_model(
            model, tokenizer, prompts, args.system_prompt, args.max_new_tokens,
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
                model, tokenizer = load_finetuned_model(args.pretrained_path, ft_path)
            except Exception as e:
                logger.error(f"Failed to load finetuned model: {e}")
                # Try loading as full model
                try:
                    model, tokenizer = load_model(ft_path)
                except Exception as e2:
                    logger.error(f"Also failed to load as full model: {e2}")
                    continue

            # Create safe filename from path (use --model_name if provided)
            ft_name = args.model_name if args.model_name else Path(ft_path).name
            finetuned_output_file = output_dir / f"finetuned_{ft_name}_responses.json"

            finetuned_results = evaluate_model(
                model, tokenizer, prompts, args.system_prompt, args.max_new_tokens,
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
