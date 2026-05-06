#!/usr/bin/env python3
"""
Finetune Qwen2.5-Omni on filtered benign audio dataset.

This script wraps LlamaFactory CLI for Qwen2.5-Omni finetuning, providing
a similar interface to the Audio Flamingo and Kimi-Audio finetuning scripts.

Usage:
    # Using YAML config
    python 2_finetune_qwen_omni.py --config qwen25_omni_7b_lora_sft.yaml

    # Using command line arguments
    python 2_finetune_qwen_omni.py \
        --dataset_json data/voicebench_audio_semantic_percentage_50.json \
        --output_dir saves/qwen25-omni-7b/lora/sft_audio_semantic_50 \
        --model_path /project/anon/Qwen2.5-Omni-7B \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --use_lora
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path


def prepare_dataset_for_llamafactory(
    input_json: str,
    output_dir: str,
    dataset_name: str = "benign_audio_dataset"
) -> str:
    """Convert dataset to LlamaFactory format and register it."""
    from pathlib import Path

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load input data
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Convert to LlamaFactory format
    converted_data = []
    for item in data:
        converted_item = {"messages": [], "audios": []}

        # Extract audio path
        audio_path = item.get('audio', item.get('audio_path', ''))

        # Extract conversation content
        if 'conversations' in item:
            for conv in item['conversations']:
                role = "user" if conv.get('from') == 'human' else "assistant"
                content = conv.get('value', '')

                # Add <audio> tag to first user message
                if role == "user" and not converted_item["messages"]:
                    # Remove existing <audio> tag if present
                    content = content.replace('<audio>\n', '').replace('<audio>', '').strip()
                    content = f"<audio>{content}"

                converted_item["messages"].append({
                    "role": role,
                    "content": content
                })
        else:
            # Generic format
            question = item.get('question', item.get('prompt', item.get('instruction', '')))
            answer = item.get('answer', item.get('response', item.get('output', '')))

            converted_item["messages"] = [
                {"role": "user", "content": f"<audio>{question}"},
                {"role": "assistant", "content": answer}
            ]

        if audio_path:
            converted_item["audios"] = [audio_path]

        if converted_item["messages"] and converted_item["audios"]:
            converted_data.append(converted_item)

    # Save converted dataset
    output_json = os.path.join(output_dir, f"{dataset_name}.json")
    with open(output_json, 'w') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted_data)} samples to {output_json}")

    # Create/update dataset_info.json
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    dataset_info = {}

    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

    dataset_info[dataset_name] = {
        "file_name": f"{dataset_name}.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "audios": "audios"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }

    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Updated dataset_info.json")

    return output_json


def create_training_config(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    warmup_ratio: float = 0.1,
    cutoff_len: int = 3072,
    dataset_dir: str = None,
) -> str:
    """Create YAML config for LlamaFactory training."""

    config = {
        # Model
        "model_name_or_path": model_path,
        "trust_remote_code": True,

        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora" if use_lora else "full",

        # Dataset
        "dataset": dataset_name,
        "template": "qwen2_omni",
        "cutoff_len": cutoff_len,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 4,

        # Output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 3,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",

        # Training
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        "bf16": True,
        "ddp_timeout": 180000000,
        "gradient_checkpointing": True,
        "freeze_vision_tower": True,

        # Evaluation
        "val_size": 0.05,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500,
    }

    # Add LoRA specific config
    if use_lora:
        config["lora_rank"] = lora_rank
        config["lora_alpha"] = lora_alpha
        config["lora_target"] = "all"
        config["lora_dropout"] = 0.05

    # Add dataset directory if specified
    if dataset_dir:
        config["dataset_dir"] = dataset_dir

    # Save config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(output_dir, f"training_config_{timestamp}.yaml")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created training config: {config_path}")

    return config_path


def run_llamafactory_training(config_path: str, llamafactory_dir: str = None):
    """Run LlamaFactory training using the config file."""

    # Find LlamaFactory directory
    if llamafactory_dir is None:
        # Try to find LlamaFactory in common locations
        possible_paths = [
            "../LlamaFactory",
            "../../LlamaFactory",
            os.path.join(os.path.dirname(__file__), "..", "LlamaFactory"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                llamafactory_dir = os.path.abspath(path)
                break

    if llamafactory_dir is None or not os.path.exists(llamafactory_dir):
        print("ERROR: LlamaFactory directory not found.")
        print("Please specify --llamafactory_dir or ensure LlamaFactory is in the expected location.")
        sys.exit(1)

    print(f"Using LlamaFactory from: {llamafactory_dir}")

    # Run training
    cmd = [
        sys.executable, "-m", "llamafactory.train",
        config_path
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Working directory: {llamafactory_dir}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{llamafactory_dir}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        cmd,
        cwd=llamafactory_dir,
        env=env,
    )

    return result.returncode


def merge_lora_adapter(
    model_path: str,
    adapter_path: str,
    output_path: str,
    llamafactory_dir: str = None
):
    """Merge LoRA adapter with base model."""

    # Create merge config
    merge_config = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": adapter_path,
        "template": "qwen2_omni",
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 2,
        "export_device": "cuda",
        "export_legacy_format": False,
    }

    config_path = os.path.join(adapter_path, "merge_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(merge_config, f)

    print(f"Created merge config: {config_path}")

    # Find LlamaFactory
    if llamafactory_dir is None:
        possible_paths = [
            "../LlamaFactory",
            "../../LlamaFactory",
            os.path.join(os.path.dirname(__file__), "..", "LlamaFactory"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                llamafactory_dir = os.path.abspath(path)
                break

    if llamafactory_dir is None:
        print("ERROR: LlamaFactory directory not found for merging.")
        return 1

    # Run merge
    cmd = [
        sys.executable, "-m", "llamafactory.export",
        config_path
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{llamafactory_dir}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, cwd=llamafactory_dir, env=env)

    if result.returncode == 0:
        print(f"\nMerged model saved to: {output_path}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Qwen2.5-Omni on filtered benign dataset"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other arguments)"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_json",
        type=str,
        default=None,
        help="Path to filtered dataset JSON"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="benign_audio_dataset",
        help="Dataset name for LlamaFactory"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory containing dataset_info.json"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/project/anon/Qwen2.5-Omni-7B",
        help="Path to pretrained Qwen2.5-Omni model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saves/qwen25-omni-7b/lora/sft",
        help="Output directory for checkpoints"
    )

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--cutoff_len", type=int, default=3072)

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # LlamaFactory path
    parser.add_argument(
        "--llamafactory_dir",
        type=str,
        default=None,
        help="Path to LlamaFactory directory"
    )

    # Merge LoRA
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA adapter after training"
    )
    parser.add_argument(
        "--merge_output_dir",
        type=str,
        default=None,
        help="Output directory for merged model"
    )

    args = parser.parse_args()

    # If config file provided, use it directly
    if args.config:
        if os.path.exists(args.config):
            print(f"Using config file: {args.config}")
            return run_llamafactory_training(args.config, args.llamafactory_dir)
        else:
            print(f"ERROR: Config file not found: {args.config}")
            return 1

    # Otherwise, build config from arguments
    if args.dataset_json is None:
        print("ERROR: Either --config or --dataset_json must be provided")
        return 1

    # Prepare dataset
    if args.dataset_dir is None:
        args.dataset_dir = os.path.join(os.path.dirname(args.output_dir), "data")

    print(f"\n{'='*60}")
    print("Preparing dataset for LlamaFactory...")
    print(f"{'='*60}")
    prepare_dataset_for_llamafactory(
        args.dataset_json,
        args.dataset_dir,
        args.dataset_name
    )

    # Create training config
    print(f"\n{'='*60}")
    print("Creating training configuration...")
    print(f"{'='*60}")
    config_path = create_training_config(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        warmup_ratio=args.warmup_ratio,
        cutoff_len=args.cutoff_len,
        dataset_dir=args.dataset_dir,
    )

    # Run training
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    ret = run_llamafactory_training(config_path, args.llamafactory_dir)

    if ret != 0:
        print(f"Training failed with return code: {ret}")
        return ret

    # Merge LoRA if requested
    if args.merge_lora and args.use_lora:
        print(f"\n{'='*60}")
        print("Merging LoRA adapter...")
        print(f"{'='*60}")

        merge_output = args.merge_output_dir
        if merge_output is None:
            merge_output = f"{args.output_dir}_merged"

        ret = merge_lora_adapter(
            args.model_path,
            args.output_dir,
            merge_output,
            args.llamafactory_dir
        )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    return ret


if __name__ == "__main__":
    sys.exit(main())
