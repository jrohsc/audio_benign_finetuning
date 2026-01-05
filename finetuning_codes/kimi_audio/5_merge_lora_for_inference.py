"""
Merge LoRA adapter with base model for inference.

This script:
1. Loads the base Kimi-Audio model
2. Loads the LoRA adapter weights
3. Manually merges the LoRA weights into base model
4. Exports a full model compatible with kimia_infer

Usage:
    python 5_merge_lora_for_inference.py \
        --lora_path output/finetuned_lora_voicebench_0.005 \
        --output_path output/finetuned_lora_voicebench_merged_0.005
"""

import os
import gc
import json
import shutil
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

# Base model path
BASE_MODEL_PATH = "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"


def merge_lora_weights(base_state_dict, lora_state_dict, lora_config):
    """Manually merge LoRA weights into base model weights."""

    alpha = lora_config.get("lora_alpha", 16)
    r = lora_config.get("r", 16)
    scaling = alpha / r

    print(f"  LoRA config: r={r}, alpha={alpha}, scaling={scaling}")

    merged_count = 0
    for key in list(lora_state_dict.keys()):
        if "lora_A" in key:
            # Find corresponding lora_B
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")

            if lora_b_key not in lora_state_dict:
                continue

            # Get base weight key
            # lora key: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            # base key: model.layers.0.self_attn.q_proj.weight
            base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")

            if base_key not in base_state_dict:
                print(f"  Warning: base key not found: {base_key}")
                continue

            lora_a = lora_state_dict[lora_a_key]
            lora_b = lora_state_dict[lora_b_key]
            base_weight = base_state_dict[base_key]

            # Compute delta: lora_B @ lora_A * scaling
            # lora_A: (r, in_features), lora_B: (out_features, r)
            delta = (lora_b @ lora_a) * scaling

            # Add delta to base weight
            base_state_dict[base_key] = base_weight + delta.to(base_weight.dtype)
            merged_count += 1

    print(f"  Merged {merged_count} LoRA weight pairs")
    return base_state_dict


def merge_lora_model(lora_path: str, output_path: str, base_model_path: str = BASE_MODEL_PATH):
    """Merge LoRA adapter with base model and export for inference."""

    print("="*60)
    print("LoRA Merge for Inference")
    print("="*60)
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output: {output_path}")
    print("="*60)

    # Check paths
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA adapter not found: {lora_path}")

    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise ValueError(f"adapter_config.json not found in {lora_path}")

    os.makedirs(output_path, exist_ok=True)

    # Load LoRA config
    with open(adapter_config_path, "r") as f:
        lora_config = json.load(f)
    print(f"\nLoRA config: r={lora_config.get('r')}, alpha={lora_config.get('lora_alpha')}")

    # Step 1: Load base model
    print("\n[1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Base model loaded: {type(base_model).__name__}")

    # Step 2: Load LoRA adapter weights
    print("\n[2/5] Loading LoRA adapter weights...")
    lora_weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(lora_weights_path):
        lora_state_dict = load_file(lora_weights_path)
    else:
        lora_weights_path = os.path.join(lora_path, "adapter_model.bin")
        lora_state_dict = torch.load(lora_weights_path, map_location="cpu")
    print(f"Loaded {len(lora_state_dict)} LoRA tensors")

    # Step 3: Merge weights manually
    print("\n[3/5] Merging LoRA weights into base model...")
    base_state_dict = base_model.state_dict()
    merged_state_dict = merge_lora_weights(base_state_dict, lora_state_dict, lora_config)
    base_model.load_state_dict(merged_state_dict)
    print("Weights merged successfully")

    # Step 4: Save merged model
    print("\n[4/5] Saving merged model...")
    base_model.save_pretrained(output_path, safe_serialization=True)
    print(f"Merged model saved to {output_path}")

    # Free memory
    del base_model
    del lora_state_dict
    del merged_state_dict
    gc.collect()

    # Step 5: Copy required files from base model
    print("\n[5/5] Copying additional required files...")

    # Copy configuration files (all files needed for inference)
    files_to_copy = [
        "configuration_moonshot_kimia.py",
        "modeling_moonshot_kimia.py",
        "tokenization_kimia.py",  # Required for tokenizer
        "tiktoken.model",  # Required for tokenizer vocab
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    for filename in files_to_copy:
        src = os.path.join(base_model_path, filename)
        dst = os.path.join(output_path, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")

    # Symlink whisper-large-v3 directory (required for inference) - avoids copying ~3GB
    whisper_src = os.path.join(base_model_path, "whisper-large-v3")
    whisper_dst = os.path.join(output_path, "whisper-large-v3")
    if os.path.exists(whisper_src) and not os.path.exists(whisper_dst):
        print("  Symlinking whisper-large-v3 directory...")
        os.symlink(whisper_src, whisper_dst)
        print("  Symlinked: whisper-large-v3/ -> " + whisper_src)

    print("\n" + "="*60)
    print("Merge complete!")
    print(f"Output saved to: {output_path}")
    print("="*60)

    # List output files
    print("\nOutput directory contents:")
    for f in sorted(os.listdir(output_path)):
        path = os.path.join(output_path, f)
        if os.path.isdir(path):
            print(f"  {f}/")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for merged model")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH,
                        help="Base model path (default: pretrained Kimi-Audio)")

    args = parser.parse_args()

    merge_lora_model(args.lora_path, args.output_path, args.base_model)


if __name__ == "__main__":
    main()
