#!/usr/bin/env python3
"""
Finetune Phi-4 Multimodal on filtered VoiceBench dataset.

Based on the approach from finetune_Phi4_mm_asr_turkish_unf_public.ipynb,
adapted for VoiceBench SD-QA dataset.

Usage:
    python 2_finetune_phi4.py \
        --dataset_json data/phi4_filtered_voicebench/voicebench_phi4_benign_0.5.json \
        --output_dir checkpoints/phi4_voicebench_finetuned \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 1e-4

This script:
1. Loads the filtered VoiceBench dataset
2. Loads Phi-4 multimodal model
3. Unfreezes only speech components (audio_embed, encoder, audio_projection)
4. Finetunes using the conversation format
5. Saves checkpoints
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BatchFeature,
)
from tqdm import tqdm
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


class VoiceBenchDataset(Dataset):
    """Dataset for VoiceBench SD-QA samples formatted for Phi-4."""

    def __init__(self, data_json: str, processor, training: bool = True, max_samples: int = None):
        """
        Args:
            data_json: Path to JSON file with Phi-4 format data
            processor: Phi-4 processor
            training: Whether this is for training (affects label handling)
            max_samples: Maximum number of samples (for debugging)
        """
        logger.info(f"Loading dataset from {data_json}")
        with open(data_json) as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples")

        self.processor = processor
        self.training = training
        logger.info(f"Loaded {len(self.data)} samples (training={training})")

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
        """Get a single sample in Phi-4 format."""
        sample = self.data[idx]

        # Load audio
        audio_path = sample['audio_path']
        try:
            audio = load_audio(audio_path, sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return a short silent audio as fallback
            audio = np.zeros(16000, dtype=np.float32)

        # Get instruction and response
        instruction = sample.get('instruction', 'Describe this audio.')
        response = sample.get('response', '')

        # Create user message with audio placeholder
        user_message = {
            'role': 'user',
            'content': f'<|audio_1|>\n{instruction}',
        }

        # Apply chat template to get the prompt
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        # Process with audio
        inputs = self.processor(
            text=prompt,
            audios=[(audio, 16000)],
            return_tensors='pt'
        )

        # Create answer with suffix
        answer = f"{response}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids

        if self.training:
            # Concatenate prompt and answer for training
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            # Mask all tokens except the answer
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1]:] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """Pad sequences to the same length."""
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """Concatenate tensors with padding."""
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), 'All tensors must have the same number of dimensions'
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    return output


def voicebench_collate_fn(batch):
    """Collate function for VoiceBench dataset."""
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []

    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full(
                (inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool
            )
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=_IGNORE_INDEX)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1 else None
        )
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        raise

    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'input_audio_embeds': input_audio_embeds,
        'audio_embed_sizes': audio_embed_sizes,
        'audio_attention_mask': audio_attention_mask,
        'input_mode': 2,  # speech mode
    })


def unfreeze_speech_components(model):
    """
    Unfreeze only the speech-related components of Phi-4.

    Components to unfreeze:
    - audio_embed: The audio embedding module
    - encoder: The audio encoder
    - audio_projection: The audio projection layer
    """
    # Access audio components
    audio_embed = model.model.embed_tokens_extend.audio_embed
    audio_encoder = audio_embed.encoder
    audio_projection = audio_embed.audio_projection

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only audio components
    for component in [audio_embed, audio_encoder, audio_projection]:
        for param in component.parameters():
            param.requires_grad = True

    return model


def verify_unfrozen_components(model):
    """Verify that the correct components are unfrozen."""
    encoder_params = list(model.model.embed_tokens_extend.audio_embed.encoder.parameters())
    proj_params = list(model.model.embed_tokens_extend.audio_embed.audio_projection.parameters())

    encoder_unfrozen = any(p.requires_grad for p in encoder_params)
    proj_unfrozen = any(p.requires_grad for p in proj_params)

    if not encoder_unfrozen:
        logger.warning("Encoder params are frozen!")
    if not proj_unfrozen:
        logger.warning("Projection params are frozen!")

    if encoder_unfrozen and proj_unfrozen:
        logger.info("Audio components properly unfrozen")

    return encoder_unfrozen and proj_unfrozen


def finetune(
    dataset_json: str,
    output_dir: str,
    model_name: str = "microsoft/Phi-4-multimodal-instruct",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    use_flash_attention: bool = True,
    max_samples: int = None,
    save_steps: int = 500,
    logging_steps: int = 100,
    eval_dataset_json: str = None,
    resume_from_checkpoint: str = None,
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    Finetune Phi-4 on VoiceBench dataset.

    Args:
        dataset_json: Path to Phi-4 format dataset JSON
        output_dir: Directory to save checkpoints
        model_name: Phi-4 model name
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        max_grad_norm: Max gradient norm for clipping
        use_flash_attention: Whether to use Flash Attention 2
        max_samples: Max samples for debugging
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        eval_dataset_json: Optional evaluation dataset JSON
        resume_from_checkpoint: Path to checkpoint to resume from
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load processor
    logger.info(f"Loading processor from {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model with audio enabled
    logger.info(f"Loading model from {model_name}")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        audio_enabled=True
    )

    if use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    # Unfreeze only speech components
    logger.info("Unfreezing speech components...")
    model = unfreeze_speech_components(model)

    # Verify unfreezing
    verify_unfrozen_components(model)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    # Print unfrozen parameter names
    logger.info("Unfrozen components:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  - {name}")

    # Load datasets
    train_dataset = VoiceBenchDataset(dataset_json, processor, training=True, max_samples=max_samples)

    eval_dataset = None
    if eval_dataset_json:
        eval_dataset = VoiceBenchDataset(eval_dataset_json, processor, training=False, max_samples=max_samples)

    # Setup training arguments
    # Always use bf16 since model is loaded in bfloat16
    fp16 = False
    bf16 = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-7,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type='cosine',
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy='steps',
        save_total_limit=3,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=True,
        report_to="tensorboard",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
    )

    # Save training config
    config_dict = {
        "model_name": model_name,
        "dataset_json": dataset_json,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "use_flash_attention": use_flash_attention,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "num_samples": len(train_dataset),
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=voicebench_collate_fn,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(str(output_dir / "final_model"))
    # Workaround for transformers compatibility - Phi4MMProcessor lacks audio_tokenizer attr
    if not hasattr(processor, 'audio_tokenizer'):
        processor.audio_tokenizer = None
    processor.save_pretrained(str(output_dir / "final_model"))

    if push_to_hub:
        logger.info("Pushing to HuggingFace Hub...")
        trainer.push_to_hub(commit_message="Final finetuned model")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Phi-4 on filtered VoiceBench"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to Phi-4 format dataset JSON"
    )
    parser.add_argument(
        "--eval_dataset_json",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/phi4_voicebench_finetuned",
        help="Directory to save checkpoints"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Phi-4 model name"
    )

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.005, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (for debugging)")

    # Flash attention
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable Flash Attention 2")

    # Checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HuggingFace Hub model ID")

    args = parser.parse_args()

    finetune(
        dataset_json=args.dataset_json,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        use_flash_attention=not args.no_flash_attention,
        max_samples=args.max_samples,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_dataset_json=args.eval_dataset_json,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
