#!/bin/bash
#
# Finetune Phi-4 Multimodal on filtered VoiceBench dataset.
#

set -e

module load cuda/12.6

# Default values
NUM_SAMPLES="100"
DATASET_JSON=""
EVAL_DATASET_JSON=""
OUTPUT_DIR=""
# Use the complete model directory with all processor files (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${SCRIPT_DIR}/phi4_model_complete"
NUM_EPOCHS=1
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.005"
WARMUP_RATIO="0.1"
MAX_GRAD_NORM="1.0"
SAVE_STEPS=500
LOGGING_STEPS=100
MAX_SAMPLES=""
# Flash attention disabled by default (requires flash_attn package)
NO_FLASH_ATTENTION="--no_flash_attention"
RESUME_FROM_CHECKPOINT=""
PUSH_TO_HUB=""
HUB_MODEL_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --dataset_json)
            DATASET_JSON="$2"
            shift 2
            ;;
        --eval_dataset_json)
            EVAL_DATASET_JSON="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
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
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --max_grad_norm)
            MAX_GRAD_NORM="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --no_flash_attention)
            NO_FLASH_ATTENTION="--no_flash_attention"
            shift
            ;;
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB="--push_to_hub"
            shift
            ;;
        --hub_model_id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Data options:"
            echo "  --num_samples VALUE             Number of samples used for filtering (default: 100)"
            echo "  --dataset_json PATH             Path to Phi-4 format dataset JSON (overrides num_samples)"
            echo ""
            echo "Options:"
            echo "  --eval_dataset_json PATH        Path to evaluation dataset JSON"
            echo "  --output_dir PATH               Directory to save checkpoints (auto-set if not provided)"
            echo "  --model_name NAME               Phi-4 model name"
            echo "  --num_epochs N                  Number of epochs (default: 3)"
            echo "  --batch_size N                  Batch size per device (default: 4)"
            echo "  --gradient_accumulation_steps N Gradient accumulation steps (default: 2)"
            echo "  --learning_rate LR              Learning rate (default: 1e-4)"
            echo "  --weight_decay WD               Weight decay (default: 0.005)"
            echo "  --warmup_ratio RATIO            Warmup ratio (default: 0.1)"
            echo "  --max_grad_norm NORM            Max gradient norm (default: 1.0)"
            echo "  --save_steps N                  Save every N steps (default: 500)"
            echo "  --logging_steps N               Log every N steps (default: 100)"
            echo "  --max_samples N                 Max samples for debugging"
            echo "  --no_flash_attention            Disable Flash Attention 2"
            echo "  --resume_from_checkpoint PATH   Path to checkpoint to resume from"
            echo "  --push_to_hub                   Push to HuggingFace Hub"
            echo "  --hub_model_id ID               HuggingFace Hub model ID"
            echo "  -h, --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Derive dataset_json and output_dir if not explicitly provided
if [ -z "$DATASET_JSON" ]; then
    DATASET_JSON="data/phi4_filtered_voicebench/voicebench_phi4_closest_${NUM_SAMPLES}.json"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="checkpoints/phi4_voicebench_finetuned_${NUM_SAMPLES}_epoch_${NUM_EPOCHS}"
fi

# Check dataset exists
if [ ! -f "$DATASET_JSON" ]; then
    echo "Warning: Dataset file not found: $DATASET_JSON"
    echo "Make sure to run 1_prepare_phi4_dataset.sh first"
fi

# Build command
CMD="python 2_finetune_phi4.py"
CMD="$CMD --dataset_json $DATASET_JSON"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --warmup_ratio $WARMUP_RATIO"
CMD="$CMD --max_grad_norm $MAX_GRAD_NORM"
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --logging_steps $LOGGING_STEPS"

if [ -n "$EVAL_DATASET_JSON" ]; then
    CMD="$CMD --eval_dataset_json $EVAL_DATASET_JSON"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$NO_FLASH_ATTENTION" ]; then
    CMD="$CMD $NO_FLASH_ATTENTION"
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
fi

if [ -n "$PUSH_TO_HUB" ]; then
    CMD="$CMD $PUSH_TO_HUB"
fi

if [ -n "$HUB_MODEL_ID" ]; then
    CMD="$CMD --hub_model_id $HUB_MODEL_ID"
fi

echo "Running: $CMD"
cd "$(dirname "$0")"

# Clear cached model modules to ensure fresh load
rm -rf ~/.cache/huggingface/modules/transformers_modules/phi4_model_complete 2>/dev/null || true

$CMD
