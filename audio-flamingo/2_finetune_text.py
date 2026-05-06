#!/usr/bin/env python3
# Copyright (c) 2025
# Text-only finetuning script for LLM (Benign Finetuning without Audio)

"""
Finetune a text-only LLM on filtered VoiceBench dataset (text only, no audio).

This is the text-only counterpart to 2_finetune_audio_flamingo.py.
Instead of audio+text, this uses text-only Q&A pairs for benign finetuning.

Usage:
    python 2_finetune_text.py \
        --dataset_json data_semantic/filtered_voicebench/voicebench_filtered_closest_percentage_50_text.json \
        --output_dir checkpoints_text_bft/llm_finetuned_percentage_50 \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --num_epochs 3 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5

This script:
1. Loads the text-only filtered VoiceBench dataset
2. Loads a text LLM (e.g., Llama, Mistral)
3. Finetunes using the conversation format
4. Saves checkpoints to checkpoints_text_bft/
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextConversationDataset(Dataset):
    """Dataset for text-only conversations."""

    def __init__(self, data_json: str, tokenizer, max_length: int = 2048, max_samples: int = None):
        """
        Args:
            data_json: Path to JSON file with text-only conversations
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to use (for debugging)
        """
        logger.info(f"Loading dataset from {data_json}")
        with open(data_json) as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for debugging")

        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.data[idx]
        conversations = sample['conversations']

        # Format as chat messages
        messages = []
        for turn in conversations:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            messages.append({"role": role, "content": content})

        return {
            'id': sample.get('id', str(idx)),
            'messages': messages,
        }


def collate_fn(batch, tokenizer, max_length=2048):
    """Collate batch of samples for training."""
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sample in batch:
        messages = sample['messages']

        # Apply chat template
        try:
            # Use chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                # Fallback: simple formatting
                text = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        text += f"User: {msg['content']}\n"
                    else:
                        text += f"Assistant: {msg['content']}\n"

            # Tokenize
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # For causal LM, labels = input_ids (shifted internally by the model)
            labels = input_ids.copy()

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            continue

    if not all_input_ids:
        return None

    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for input_ids, attention_mask, labels in zip(all_input_ids, all_attention_masks, all_labels):
        padding_length = max_len - len(input_ids)

        # Pad on the right
        padded_input_ids.append(input_ids + [pad_token_id] * padding_length)
        padded_attention_masks.append(attention_mask + [0] * padding_length)
        # Use -100 for padding in labels (ignored in loss)
        padded_labels.append(labels + [-100] * padding_length)

    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long),
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        if batch is None:
            continue

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        try:
            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}" if num_batches > 0 else "N/A",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        except Exception as e:
            logger.error(f"Error in training step {step}: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def finetune(
    dataset_json: str,
    output_dir: str,
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    max_length: int = 2048,
    max_samples: int = None,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    local_model_path: str = None,
    resume_from_checkpoint: str = None,
    use_flash_attention: bool = True,
):
    """
    Finetune a text LLM on text-only dataset.

    Args:
        dataset_json: Path to text-only dataset JSON
        output_dir: Directory to save checkpoints
        model_id: HuggingFace model ID
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Ratio of warmup steps
        max_grad_norm: Max gradient norm for clipping
        max_length: Maximum sequence length
        max_samples: Max samples for debugging
        use_lora: Whether to use LoRA for efficient finetuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        local_model_path: Local path to model checkpoint (alternative to model_id)
        resume_from_checkpoint: Path to checkpoint directory to resume training from
        use_flash_attention: Whether to use Flash Attention 2
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer and model
    model_source = local_model_path if local_model_path else model_id
    # Check if it's a local path (absolute path starting with / or existing directory)
    is_local = model_source.startswith('/') or os.path.isdir(model_source)
    logger.info(f"Loading tokenizer from {model_source} (local={is_local})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=is_local,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model from {model_source}")

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": is_local,
    }

    # Check if flash_attn is available before enabling it
    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            logger.warning("flash_attn not installed, using default attention")

    if not use_lora:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

    # Setup LoRA if requested
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            logger.info(f"Setting up LoRA with r={lora_r}, alpha={lora_alpha}")

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            model = model.to(device)

        except ImportError:
            logger.error("peft not installed. Install with: pip install peft")
            logger.info("Continuing without LoRA")
            use_lora = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load dataset
    dataset = TextConversationDataset(dataset_json, tokenizer, max_length=max_length, max_samples=max_samples)

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        num_workers=0,
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )

    num_training_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        training_state_path = checkpoint_path / "training_state.pt"

        if training_state_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

            # Load model weights
            if use_lora:
                from peft import PeftModel
                logger.info("Loading LoRA adapter weights...")
                model = PeftModel.from_pretrained(model, checkpoint_path)
                model = model.to(device)
            else:
                logger.info("Loading model weights from checkpoint...")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            # Load training state
            logger.info("Loading training state...")
            training_state = torch.load(training_state_path, map_location=device)

            if training_state.get('optimizer_state_dict'):
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
            if training_state.get('scheduler_state_dict'):
                scheduler.load_state_dict(training_state['scheduler_state_dict'])

            start_epoch = training_state['epoch'] + 1
            logger.info(f"Resumed from epoch {training_state['epoch']}, starting from epoch {start_epoch}")
        else:
            logger.warning(f"training_state.pt not found in {checkpoint_path}, starting from scratch")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        "model_id": model_id,
        "local_model_path": local_model_path,
        "dataset_json": dataset_json,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "max_length": max_length,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "num_samples": len(dataset),
        "modality": "text",
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")

        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )

        logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        if use_lora:
            model.save_pretrained(checkpoint_dir)
        else:
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_dir / "training_state.pt")

        # Track best model
        if avg_loss < best_loss:
            # Delete previous non-best checkpoints
            for prev_checkpoint in output_dir.glob("checkpoint-epoch-*"):
                if prev_checkpoint != checkpoint_dir:
                    logger.info(f"Removing non-best checkpoint: {prev_checkpoint}")
                    shutil.rmtree(prev_checkpoint)

            best_loss = avg_loss
            best_dir = output_dir / "best_model"
            logger.info(f"New best model! Saving to {best_dir}")

            if use_lora:
                model.save_pretrained(best_dir)
            else:
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, best_dir / "training_state.pt")

            with open(output_dir / "best_epoch.txt", 'w') as f:
                f.write(f"{epoch}")
        else:
            logger.info(f"Epoch {epoch} loss ({avg_loss:.4f}) not better than best ({best_loss:.4f})")
            logger.info(f"Removing non-best checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)

    logger.info(f"\n=== Training complete ===")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune text LLM on filtered VoiceBench (text-only)"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to text-only dataset JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_text_bft/llm_finetuned",
        help="Directory to save checkpoints"
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to model checkpoint (alternative to model_id)"
    )

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (for debugging)")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA (default: True)")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Flash attention
    parser.add_argument("--use_flash_attention", action="store_true", default=True, help="Use Flash Attention 2")
    parser.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false")

    # Resume arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )

    args = parser.parse_args()

    finetune(
        dataset_json=args.dataset_json,
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        max_samples=args.max_samples,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        local_model_path=args.local_model_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_flash_attention=args.use_flash_attention,
    )


if __name__ == "__main__":
    main()
