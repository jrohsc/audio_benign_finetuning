# Qwen2.5-Omni Finetuning

This directory contains the setup for finetuning Qwen2.5-Omni models (3B and 7B) using LlamaFactory.

## Model Locations

- **Qwen2.5-Omni-7B**: `/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00`
- **Qwen2.5-Omni-3B**: `/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e`

## Setup

### 1. Environment Setup

The conda environment `qwen-omni` has been created. To set up dependencies:

```bash
cd finetune_codes
./0_setup_environment.sh
```

Or manually:
```bash
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate qwen-omni

# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install LlamaFactory
cd LlamaFactory
pip install -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes]"

# Additional dependencies
pip install transformers>=4.49.0 accelerate>=0.31.0 peft>=0.12.0 flash-attn --no-build-isolation
pip install soundfile librosa fire
```

### 2. Prepare Dataset

Convert your filtered benign dataset to LlamaFactory format:

```bash
python finetune_codes/1_prepare_dataset.py \
    --input path/to/filtered_data.json \
    --output data/benign_audio_llamafactory.json \
    --update_dataset_info \
    --dataset_info_dir LlamaFactory/data/
```

### 3. Finetune

```bash
cd finetune_codes

# Finetune 7B model with 50% of filtered voicebench data
./2_finetune_qwen_omni.sh --model 7b --percentage 50

# Finetune 3B model
./2_finetune_qwen_omni.sh --model 3b --percentage 50

# With different dataset and filter type
./2_finetune_qwen_omni.sh --model 7b --benign_dataset heysquad_accents --filter_type audio_acoustic --percentage 50
```

### 4. Merge LoRA Weights

After training, merge the LoRA weights for inference:

```bash
./3_merge_lora.sh --model 7b --lora_path checkpoints/qwen_omni_7b_lora_...
```

### 5. Evaluate

```bash
python 4_evaluate_jailbreaking.py \
    --model_path path/to/merged_model \
    --dataset_path path/to/harmful_audio.json \
    --output_path responses.json
```

## Dataset Format

LlamaFactory expects audio datasets in this format:

```json
[
  {
    "messages": [
      {"role": "user", "content": "<audio>What does the person say?"},
      {"role": "assistant", "content": "The response text here."}
    ],
    "audios": ["path/to/audio.wav"]
  }
]
```

The `<audio>` tag indicates where the audio content should be inserted.

## Configuration Files

- `qwen25_omni_7b_lora_sft.yaml`: Configuration for 7B model LoRA training
- `qwen25_omni_3b_lora_sft.yaml`: Configuration for 3B model LoRA training

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lora_rank | 8 | LoRA rank |
| lora_alpha | 16 | LoRA alpha |
| learning_rate | 1e-4 | Learning rate |
| num_train_epochs | 3 | Number of epochs |
| batch_size | 1 | Per-device batch size |
| gradient_accumulation | 8 | Gradient accumulation steps |

## Notes

- Only the "Thinker" part (audio encoder, vision encoder, LLM backbone) is finetuned
- The "Talker" part (audio decoder) is preserved from the original model
- Use `qwen_omni_merge.py` to properly merge LoRA weights back into the full model
