#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Define your parameter options
# You can customize these lists for your specific use case
model_paths = [
    "output/pretrained_kimi"
]

datasets = [
    "benign_close_to_harmful_0_93_semantic_codes.jsonl"
]

# Alternatively, you can use the structure similar to your example
# models = ['ultravox']
# datasets = ['safetybench']
# langs = ['kenya', 'nigeria', 'south_africa', 'philippines', 'australia', 'singapore']

# Template for the sbatch script
script_template = """#!/bin/bash
#SBATCH -c 4                 # Number of CPU cores per task
#SBATCH --mem=1000000         # Requested Memory in MB (64GB)
#SBATCH -p gpu-preempt  # Partition
#SBATCH --gpus-per-node=a100:3
#SBATCH -t 24:00:00          # Requested time (24 hours)
#SBATCH -o logs/finetune_{model_name}_{dataset_name}_%j.out  # Output file with job ID and parameters
#SBATCH -J kimi_{model_name}  # Job name with model identifier

# load CUDA
module load cuda/12.6

# Environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Directory setup
DIR=`pwd`

# GPU and distributed training setup
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# Model settings
MODEL="moonshotai/Kimi-Audio-7B"
PRETRAINED_MODEL_PATH="{model_path}"
DATA="{dataset}"

# Validate paths
if [ ! -f "$DATA" ]; then
    echo "Error: DATA file does not exist: $DATA"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist: $PRETRAINED_MODEL_PATH"
    exit 1
fi

echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA"

# Build distributed arguments
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "Starting finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

# Run the training command
torchrun $DISTRIBUTED_ARGS finetune.py \\
    --model_name_or_path $MODEL \\
    --model_path $PRETRAINED_MODEL_PATH \\
    --data_path $DATA \\
    --eval_ratio 0.05 \\
    --bf16 True \\
    --output_dir output/finetuned_kimi_{model_name}_{dataset_name} \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 1 \\
    --per_device_eval_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --save_strategy "steps" \\
    --save_steps 1000 \\
    --save_total_limit 10 \\
    --learning_rate 1e-3 \\
    --weight_decay 0.1 \\
    --adam_beta2 0.95 \\
    --warmup_ratio 0.01 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --report_to "none" \\
    --model_max_length 128 \\
    --gradient_checkpointing True \\
    --lazy_preprocess True \\
    --deepspeed ds_config_zero3.json \\
    --dataloader_num_workers 0 \\
    --dataloader_pin_memory False
"""

# Function to get filename without path and extension
def get_name(path):
    return os.path.basename(path).split('.')[0]

# Loop through all combinations of parameters and submit jobs
for model_path in model_paths:
    model_name = get_name(model_path)
    
    for dataset in datasets:
        dataset_name = get_name(dataset)
        
        # print(f"Submitting job for MODEL: {model_name}, DATASET: {dataset_name}")
        
        # Format the script with current parameters
        script_content = script_template.format(
            model_path=model_path,
            dataset=dataset,
            model_name=model_name,
            dataset_name=dataset_name
        )
        
        # Submit the job via sbatch using standard input
        subprocess.run(["sbatch"], input=script_content.encode(), check=True)
        
        print(f"Job submitted for MODEL: {model_name}, DATASET: {dataset_name}")

print("All jobs submitted!")