#!/bin/bash
#
# Finetune Qwen-Audio on filtered VoiceBench dataset.
#

set -e

# Load required modules
module load cuda/11.8
module load ffmpeg/7.0.2
export CUDA_HOME=$CUDA_ROOT

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=$(pwd)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Number of GPUs per GPU worker
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

# Threshold values (for filtering CLOSEST to harmful):
#   0.1062 = 10%, 0.1195 = 25%, 0.1377 = 50%, 0.1595 = 75%, 0.1855 = 90%

# Default values
MODEL="Qwen/Qwen-Audio-Chat"
PERCENTAGE="50"
DATA="data/qwen_filtered_voicebench/voicebench_qwen_closest_percentage_${PERCENTAGE}.jsonl"
EVAL_DATA=""
SAVE="checkpoints/qwen_voicebench_finetuned_percentage_${PERCENTAGE}"
DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA="True"
Q_LORA="False"

# Training hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.1"
WARMUP_RATIO="0.01"
MODEL_MAX_LENGTH=2000
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--data)
            DATA="$2"
            shift 2
            ;;
        --eval_data)
            EVAL_DATA="$2"
            shift 2
            ;;
        -o|--output_dir)
            SAVE="$2"
            shift 2
            ;;
        --deepspeed)
            DS_CONFIG_PATH="$2"
            shift 2
            ;;
        --use_lora)
            USE_LORA="$2"
            shift 2
            ;;
        --q_lora)
            Q_LORA="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            DATA="data/qwen_filtered_voicebench/voicebench_qwen_closest_percentage_${PERCENTAGE}.jsonl"
            SAVE="checkpoints/qwen_voicebench_finetuned_percentage_${PERCENTAGE}"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            DATA="data/qwen_filtered_voicebench/voicebench_qwen_closest_n${NUM_SAMPLES}.jsonl"
            SAVE="checkpoints/qwen_voicebench_finetuned_n${NUM_SAMPLES}"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            DATA="data/qwen_filtered_voicebench/voicebench_qwen_closest_${THRESHOLD}.jsonl"
            SAVE="checkpoints/qwen_voicebench_finetuned_thresh_${THRESHOLD}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --model PATH                    Qwen-Audio model path (default: Qwen/Qwen-Audio-Chat)"
            echo "  -d, --data PATH                     Training data path (JSONL format)"
            echo "  --eval_data PATH                    Evaluation data path (optional)"
            echo "  -o, --output_dir PATH               Output directory for checkpoints"
            echo "  --deepspeed PATH                    DeepSpeed config file (default: ds_config_zero2.json)"
            echo "  --use_lora BOOL                     Use LoRA (default: True)"
            echo "  --q_lora BOOL                       Use QLoRA (default: False)"
            echo "  --num_epochs N                      Number of epochs (default: 5)"
            echo "  --batch_size N                      Batch size per device (default: 2)"
            echo "  --gradient_accumulation_steps N     Gradient accumulation steps (default: 8)"
            echo "  --learning_rate LR                  Learning rate (default: 1e-4)"
            echo "  --gpus N                            Number of GPUs (default: 4)"
            echo "  --percentage VALUE                  Percentage of dataset for auto-selecting data/output paths"
            echo "  --num_samples VALUE                 Number of samples for auto-selecting data/output paths"
            echo "  --threshold VALUE                   Distance threshold for auto-selecting data/output paths"
            echo "  -h, --help                          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build distributed training arguments
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Build training command
CMD="torchrun $DISTRIBUTED_ARGS 2_finetune_qwen.py"
CMD="$CMD --model_name_or_path $MODEL"
CMD="$CMD --data_path $DATA"
CMD="$CMD --bf16 True"
CMD="$CMD --output_dir $SAVE"
CMD="$CMD --dataloader_num_workers 4"
CMD="$CMD --num_train_epochs $NUM_EPOCHS"
CMD="$CMD --per_device_train_batch_size $BATCH_SIZE"
CMD="$CMD --per_device_eval_batch_size 1"
CMD="$CMD --gradient_accumulation_steps $GRAD_ACCUM"
CMD="$CMD --evaluation_strategy no"
CMD="$CMD --save_strategy steps"
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --save_total_limit $SAVE_TOTAL_LIMIT"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --adam_beta1 0.9"
CMD="$CMD --adam_beta2 0.95"
CMD="$CMD --warmup_ratio $WARMUP_RATIO"
CMD="$CMD --lr_scheduler_type cosine"
CMD="$CMD --logging_steps 1"
CMD="$CMD --report_to tensorboard"
CMD="$CMD --model_max_length $MODEL_MAX_LENGTH"
CMD="$CMD --gradient_checkpointing True"
CMD="$CMD --lazy_preprocess True"
CMD="$CMD --use_lora $USE_LORA"
CMD="$CMD --q_lora $Q_LORA"
CMD="$CMD --deepspeed $DS_CONFIG_PATH"

if [ -n "$EVAL_DATA" ]; then
    CMD="$CMD --eval_data_path $EVAL_DATA"
fi

echo "Running: $CMD"
cd "$SCRIPT_DIR"
eval $CMD
