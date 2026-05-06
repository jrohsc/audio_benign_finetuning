# LoRA Fine-tuning for Kimi-Audio
# Based on the original finetune.py but with PEFT/LoRA support for single GPU training

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from huggingface_hub import snapshot_download

from peft import LoraConfig, get_peft_model, TaskType

from model import KimiAudioModel
from sft_dataset import LazySupervisedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_data = all_data[:int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio):]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    train_dataset = dataset_cls(
        train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer,
        max_len=max_len, kimia_token_offset=kimia_token_offset
    )

    if eval_data:
        eval_dataset = dataset_cls(
            eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer,
            max_len=max_len, kimia_token_offset=kimia_token_offset
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def compute_loss(outputs, labels, num_items_in_batch=None):
    # Model returns (text_logits, audio_logits) - see modeling_kimia.py line 911-913
    text_logits, audio_logits = outputs.logits

    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, "we only support micro batch size 1"

    # Ensure masks are float tensors for proper multiplication
    audio_loss_mask = audio_loss_mask.float()
    text_loss_mask = text_loss_mask.float()

    audio_loss = torch.nn.functional.cross_entropy(
        audio_logits.view(-1, audio_logits.shape[-1]),
        audio_labels.view(-1),
        reduction="none"
    )
    text_loss = torch.nn.functional.cross_entropy(
        text_logits.view(-1, text_logits.shape[-1]),
        text_labels.view(-1),
        reduction="none"
    )

    # Compute mask sums for normalization
    audio_mask_sum = audio_loss_mask.view(-1).sum()
    text_mask_sum = text_loss_mask.view(-1).sum()

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_mask_sum + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_mask_sum + 1e-4)
    loss = audio_loss + text_loss

    # Log losses for debugging
    if (local_rank == 0 or local_rank is None) and torch.rand(1).item() < 0.1:
        print(f"[Loss] audio: {audio_loss.item():.4f} (mask: {audio_mask_sum.item():.0f}), "
              f"text: {text_loss.item():.4f} (mask: {text_mask_sum.item():.0f}), total: {loss.item():.4f}")

    return loss


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    logger.info("Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        cache_path = model_args.model_name_or_path
    else:
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")

    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    # Load model
    model = KimiAudioModel.from_pretrained(
        model_args.model_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.cuda()

    # Store whisper_model reference before PEFT wrapping
    whisper_model_ref = model.whisper_model
    model_config = model.config

    # Apply LoRA
    rank0_print("Applying LoRA...")
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    rank0_print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=whisper_model_ref,
        text_tokenizer=text_tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model_config.kimia_token_offset
    )

    # Start trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    # Save LoRA adapter
    rank0_print(f"Saving LoRA adapter to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    rank0_print("Done!")


if __name__ == "__main__":
    train()
