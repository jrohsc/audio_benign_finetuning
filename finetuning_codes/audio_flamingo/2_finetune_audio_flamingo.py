#!/usr/bin/env python3
# Copyright (c) 2025
# Finetuning script for Audio Flamingo 3

"""
Finetune Audio Flamingo 3 on filtered VoiceBench dataset.

Usage:
    python 2_finetune_audio_flamingo.py \
        --dataset_json data/filtered_voicebench/voicebench_filtered_thresh0.025_hf.json \
        --output_dir checkpoints/audio_flamingo_finetuned \
        --num_epochs 3 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5

This script:
1. Loads the filtered VoiceBench dataset
2. Loads Audio Flamingo 3 model
3. Finetunes using the conversation format
4. Saves checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor,
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


class AudioConversationDataset(Dataset):
    """Dataset for audio conversations."""

    def __init__(self, data_json: str, processor, max_samples: int = None):
        """
        Args:
            data_json: Path to JSON file with HuggingFace format conversations
            processor: AudioFlamingo3 processor
            max_samples: Maximum number of samples to use (for debugging)
        """
        logger.info(f"Loading dataset from {data_json}")
        with open(data_json) as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for debugging")

        self.processor = processor
        logger.info(f"Loaded {len(self.data)} samples")

        # Verify audio files exist
        missing = 0
        for sample in self.data:
            audio_path = sample.get('audio_path', '')
            if not os.path.exists(audio_path):
                missing += 1

        if missing > 0:
            logger.warning(f"{missing} audio files not found!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.data[idx]
        return {
            'id': sample.get('id', str(idx)),
            'conversations': sample['conversations'],
            'audio_path': sample.get('audio_path', ''),
        }


def collate_fn(batch, processor):
    """Collate batch of samples for training."""
    # Format conversations for the processor
    conversations = []
    for sample in batch:
        conversations.append(sample['conversations'])

    # Use processor's apply_chat_template to prepare inputs
    try:
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,  # Don't add generation prompt for training
            return_dict=True,
            output_labels=True,
            return_tensors="pt",
        )
        return inputs
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        # Return None to skip this batch
        return None


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

        # Move batch to device and convert floating point tensors to bfloat16
        batch = {
            k: v.to(device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(device) if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

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
    model_id: str = "nvidia/audio-flamingo-3-hf",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    save_steps: int = 500,
    max_samples: int = None,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    freeze_audio_encoder: bool = True,
    local_model_path: str = None,
    resume_from_checkpoint: str = None,
):
    """
    Finetune Audio Flamingo 3.

    Args:
        dataset_json: Path to HuggingFace format dataset JSON
        output_dir: Directory to save checkpoints
        model_id: HuggingFace model ID
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Ratio of warmup steps
        max_grad_norm: Max gradient norm for clipping
        save_steps: Save checkpoint every N steps
        max_samples: Max samples for debugging
        use_lora: Whether to use LoRA for efficient finetuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        freeze_audio_encoder: Whether to freeze the audio encoder
        local_model_path: Local path to model checkpoint (alternative to model_id)
        resume_from_checkpoint: Path to checkpoint directory to resume training from
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load processor and model
    model_source = local_model_path if local_model_path else model_id
    logger.info(f"Loading processor from {model_source}")
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)

    logger.info(f"Loading model from {model_source}")
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto" if not use_lora else None,
        trust_remote_code=True,
    )

    # Optionally freeze audio encoder
    if freeze_audio_encoder:
        logger.info("Freezing audio encoder")
        for param in model.audio_tower.parameters():
            param.requires_grad = False

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
    dataset = AudioConversationDataset(dataset_json, processor, max_samples=max_samples)

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=0,  # Audio processing may not be thread-safe
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
                # For LoRA, load the adapter weights
                logger.info("Loading LoRA adapter weights...")
                model = PeftModel.from_pretrained(model, checkpoint_path)
                model = model.to(device)
            else:
                # For full finetuning, load from the checkpoint directory
                logger.info("Loading model weights from checkpoint...")
                model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            # Load training state (optimizer, scheduler, epoch)
            logger.info("Loading training state...")
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
            start_epoch = training_state['epoch'] + 1
            previous_loss = training_state.get('loss', 'N/A')

            logger.info(f"Resumed from epoch {training_state['epoch']} (loss: {previous_loss})")
            logger.info(f"Starting from epoch {start_epoch}")
        else:
            logger.warning(f"training_state.pt not found in {checkpoint_path}")
            logger.warning("Starting training from scratch")

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
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "freeze_audio_encoder": freeze_audio_encoder,
        "num_samples": len(dataset),
        "resumed_from_checkpoint": resume_from_checkpoint,
        "start_epoch": start_epoch,
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
            processor.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_dir / "training_state.pt")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = output_dir / "best_model"
            logger.info(f"New best model! Saving to {best_dir}")
            if use_lora:
                model.save_pretrained(best_dir)
            else:
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)

    logger.info(f"\n=== Training complete ===")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Audio Flamingo 3 on filtered VoiceBench"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to HuggingFace format dataset JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/audio_flamingo_finetuned",
        help="Directory to save checkpoints"
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/audio-flamingo-3-hf",
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
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (for debugging)")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Freezing arguments
    parser.add_argument("--freeze_audio_encoder", action="store_true", help="Freeze audio encoder")
    parser.add_argument("--no_freeze_audio_encoder", dest="freeze_audio_encoder", action="store_false")
    parser.set_defaults(freeze_audio_encoder=True)

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
        save_steps=args.save_steps,
        max_samples=args.max_samples,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_audio_encoder=args.freeze_audio_encoder,
        local_model_path=args.local_model_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
