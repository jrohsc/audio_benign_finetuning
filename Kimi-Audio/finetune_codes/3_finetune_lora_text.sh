#!/bin/bash

# Text-Only LoRA Fine-tuning for Kimi-Audio
# This script trains Kimi-Audio on TEXT-ONLY conversations (no audio).
#
# Key differences from 3_finetune_lora.sh:
# - Uses 3_finetune_lora_text.py instead of 3_finetune_lora.py
# - Reads from data_semantic_text/ folder (text-only format)
# - Outputs to output_text_bft/ folder
# - No semantic code extraction needed (no audio tokens)

set -e

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
HARMFUL_SOURCE="advbench"
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""

# Paths
PRETRAINED_MODEL_PATH="/project/anon/BFT_models/pretrained_kimi_instruct"

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
        --harmful_source)
            shift
            HARMFUL_SOURCE=$1
            ;;
        --percentage)
            shift
            PERCENTAGE=$1
            ;;
        --threshold)
            shift
            THRESHOLD=$1
            ;;
        --num_samples)
            shift
            NUM_SAMPLES=$1
            ;;
        -m | --model_path)
            shift
            PRETRAINED_MODEL_PATH=$1
            ;;
        -r | --lora_r)
            shift
            LORA_R=$1
            ;;
        --lr)
            shift
            LEARNING_RATE=$1
            ;;
        --epochs)
            shift
            NUM_EPOCHS=$1
            ;;
        -h | --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Text-Only LoRA fine-tuning for Kimi-Audio on filtered VoiceBench data."
            echo "This trains on TEXT-ONLY conversations (no audio processing)."
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME   Harmful source (advbench or safetybench)"
            echo "  --percentage VALUE      Percentage used in filtering"
            echo "  --threshold VALUE       Threshold used in filtering"
            echo "  --num_samples VALUE     Num samples used in filtering"
            echo "  -m, --model_path PATH   Pretrained model path"
            echo "  -r, --lora_r VALUE      LoRA rank (default: 16)"
            echo "  --lr VALUE              Learning rate (default: 2e-4)"
            echo "  --epochs VALUE          Number of epochs (default: 3)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 50"
            echo "  $0 --num_samples 500 --epochs 5"
            exit 0
            ;;
        *)
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# Data comes from data_semantic_text/ (text-only format)
DATA_DIR="data_semantic_text"

# Output goes to output_text_bft/
OUTPUT_BASE="output_text_bft"

# Determine filename suffix
if [ -n "$NUM_SAMPLES" ]; then
    FILE_SUFFIX="n${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    FILE_SUFFIX="percentage_${PERCENTAGE}"
elif [ -n "$THRESHOLD" ]; then
    FILE_SUFFIX="${THRESHOLD}"
else
    FILE_SUFFIX="auto"
fi

# Set data and output paths
# Note: Text-only format, no _semantic_codes suffix needed
DATA_PATH="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}_text.jsonl"
OUTPUT_DIR="${OUTPUT_BASE}/finetuned_lora_text_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}"

# Validate inputs
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: DATA file does not exist: $DATA_PATH"
    echo ""
    echo "Please run the text-only dataset preparation first:"
    echo "  bash 0_filter_voicebench_semantic_text.sh --percentage $PERCENTAGE"
    echo ""
    echo "Or convert existing semantic-filtered data to text-only format:"
    echo "  python 1_convert_to_text_only.py --percentage $PERCENTAGE"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist: $PRETRAINED_MODEL_PATH"
    echo ""
    echo "Please first prepare the pretrained model by running:"
    echo "  CUDA_VISIBLE_DEVICES=0 python -m model --model_name \"moonshotai/Kimi-Audio-7B-Instruct\" --output_dir \"/project/anon/BFT_models/pretrained_kimi_instruct\""
    exit 1
fi

echo "============================================"
echo "Text-Only LoRA Fine-tuning Configuration"
echo "============================================"
echo "Mode:            TEXT-ONLY (no audio)"
echo "Data directory:  $DATA_DIR"
echo "Model path:      $PRETRAINED_MODEL_PATH"
echo "Data path:       $DATA_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "LoRA rank:       $LORA_R"
echo "LoRA alpha:      $LORA_ALPHA"
echo "Learning rate:   $LEARNING_RATE"
echo "Epochs:          $NUM_EPOCHS"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python 3_finetune_lora_text.py \
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

echo ""
echo "============================================"
echo "Text-Only Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"

# Clean up intermediate checkpoints to save space (keep only final adapter)
echo ""
echo "Cleaning up intermediate checkpoints..."
for checkpoint_dir in "$OUTPUT_DIR"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        echo "Removing $checkpoint_dir"
        rm -rf "$checkpoint_dir"
    fi
done
echo "Cleanup complete."

echo ""
echo "Next steps:"
echo "  1. Merge LoRA weights: bash 5_merge_lora_for_inference_text.sh --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
echo "  2. Evaluate: bash 6_evaluate_harmful_audio.sh --filter_type text --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
