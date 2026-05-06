#!/usr/bin/env python3
"""
Merge LoRA adapters with base Qwen2.5-Omni model.

This script merges a trained LoRA adapter with the base model to create
a standalone model that can be used without PEFT.

Usage:
    python 3_merge_lora.py \
        --base_model_path /datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/xxx \
        --adapter_path saves/qwen25-omni-7b/lora/sft \
        --output_path saves/qwen25-omni-7b/merged
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoProcessor


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device: str = "cuda",
):
    """
    Merge LoRA adapter with base model.

    Args:
        base_model_path: Path to the base Qwen2.5-Omni model
        adapter_path: Path to the trained LoRA adapter
        output_path: Path to save the merged model
    """
    print(f"\n{'='*60}")
    print("Merging LoRA Adapter")
    print(f"{'='*60}")
    print(f"Base model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")

    # Check if adapter exists
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"ERROR: adapter_config.json not found in {adapter_path}")
        print("This directory may not contain a valid LoRA adapter.")
        return 1

    try:
        from peft import PeftModel
    except ImportError:
        print("ERROR: peft not installed. Install with: pip install peft")
        return 1

    # Try loading as different model types
    model = None
    model_class_name = None

    # Try Thinker model first (text-only output)
    try:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        print("\nLoading base model as Qwen2_5OmniThinkerForConditionalGeneration...")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model_class_name = "Qwen2_5OmniThinkerForConditionalGeneration"
    except Exception as e:
        print(f"Could not load as Thinker model: {e}")

    # Try full Omni model
    if model is None:
        try:
            from transformers import Qwen2_5OmniModel
            print("\nLoading base model as Qwen2_5OmniModel...")
            model = Qwen2_5OmniModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model_class_name = "Qwen2_5OmniModel"
        except Exception as e:
            print(f"Could not load as Omni model: {e}")

    # Try AutoModel
    if model is None:
        try:
            from transformers import AutoModelForCausalLM
            print("\nLoading base model as AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model_class_name = "AutoModelForCausalLM"
        except Exception as e:
            print(f"Could not load as AutoModelForCausalLM: {e}")
            print("ERROR: Failed to load base model with any known model class.")
            return 1

    print(f"Loaded base model using {model_class_name}")

    # Load LoRA adapter
    print("\nLoading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge and unload
    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy processor/tokenizer from base model
    print("Copying processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    # Save merge info
    merge_info = {
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "model_class": model_class_name,
        "merge_timestamp": datetime.now().isoformat(),
        "torch_dtype": "bfloat16",
    }

    with open(os.path.join(output_path, "merge_info.json"), 'w') as f:
        json.dump(merge_info, f, indent=2)

    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"{'='*60}")
    print(f"Merged model saved to: {output_path}")

    # Verify the merged model can be loaded
    print("\nVerifying merged model...")
    try:
        if model_class_name == "Qwen2_5OmniThinkerForConditionalGeneration":
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration
            test_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                output_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        elif model_class_name == "Qwen2_5OmniModel":
            from transformers import Qwen2_5OmniModel
            test_model = Qwen2_5OmniModel.from_pretrained(
                output_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            from transformers import AutoModelForCausalLM
            test_model = AutoModelForCausalLM.from_pretrained(
                output_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        del test_model
        print("Verification successful!")
    except Exception as e:
        print(f"Warning: Could not verify merged model: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base Qwen2.5-Omni model"
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/project/anon/Qwen2.5-Omni-7B",
        help="Path to the base Qwen2.5-Omni model"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the merged model (default: adapter_path + '_merged')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for merging"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output_path is None:
        args.output_path = f"{args.adapter_path.rstrip('/')}_merged"

    return merge_lora_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
