#!/bin/bash

# LoRA Fine-tuning for Kimi-Audio on VoiceBench
# Single GPU training - no DeepSpeed required

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Paths
# NOTE: You need to first prepare the pretrained model by running:
#   cd /work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/Kimi-Audio/finetune_codes
#   CUDA_VISIBLE_DEVICES=0 python -m model --model_name "moonshotai/Kimi-Audio-7B-Instruct" --output_dir "output/pretrained_kimi_instruct"
#
# Threshold values (with --center in step 0):
#   0.15 = 10%, 0.25 = 25%, 0.36 = 50%, 0.52 = 75%, 0.68 = 90%
HARMFUL_SOURCE="advbench"
THRESHOLD=0.25
PRETRAINED_MODEL_PATH="output/pretrained_kimi_instruct"
DATA_PATH="data/voicebench_filtered_${HARMFUL_SOURCE}_${THRESHOLD}_semantic_codes.jsonl"
OUTPUT_DIR="output/finetuned_lora_voicebench_${THRESHOLD}"

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training parameters
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_SEQ_LENGTH=128
EVAL_RATIO=0.05

# Parse command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model_path )
            shift
            PRETRAINED_MODEL_PATH=$1
            ;;
        -d | --data )
            shift
            DATA_PATH=$1
            ;;
        -o | --output_dir )
            shift
            OUTPUT_DIR=$1
            ;;
        -r | --lora_r )
            shift
            LORA_R=$1
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# Validate inputs
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: DATA file does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist: $PRETRAINED_MODEL_PATH"
    exit 1
fi

echo "============================================"
echo "LoRA Fine-tuning Configuration"
echo "============================================"
echo "Model path: $PRETRAINED_MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "LoRA rank: $LORA_R"
echo "LoRA alpha: $LORA_ALPHA"
echo "Learning rate: $LEARNING_RATE"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python 3_finetune_lora.py \
    --model_name_or_path "moonshotai/Kimi-Audio-7B-Instruct" \
    --model_path "$PRETRAINED_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --eval_ratio $EVAL_RATIO \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --model_max_length $MAX_SEQ_LENGTH \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to "none" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT
