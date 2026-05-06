#!/bin/bash
# Merge LoRA adapter with base model for inference (TEXT-ONLY training)
#
# This is for models trained with 3_finetune_lora_text.sh
# Reads from output_text_bft/, outputs to output_text_bft/
#
# Usage:
#   ./5_merge_lora_for_inference_text.sh --percentage 50 --num_epochs 3

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default settings
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""
NUM_EPOCHS=3

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --percentage) PERCENTAGE="$2"; THRESHOLD=""; NUM_SAMPLES=""; shift ;;
        --threshold) THRESHOLD="$2"; PERCENTAGE=""; NUM_SAMPLES=""; shift ;;
        --num_samples) NUM_SAMPLES="$2"; PERCENTAGE=""; THRESHOLD=""; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --lora_path) LORA_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Merge LoRA adapter with base model for inference (TEXT-ONLY training)."
            echo ""
            echo "Options:"
            echo "  --percentage VALUE      Percentage used in filtering (default: 50)"
            echo "  --threshold VALUE       Threshold used in filtering"
            echo "  --num_samples VALUE     Num samples used in filtering"
            echo "  --num_epochs N          Number of epochs used during training (default: 3)"
            echo "  --lora_path PATH        Override LoRA adapter path"
            echo "  --output_path PATH      Override merged output path"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 50 --num_epochs 3"
            echo "  $0 --num_samples 500 --num_epochs 5"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

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

# Set default paths if not specified
LORA_PATH="${LORA_PATH:-${OUTPUT_BASE}/finetuned_lora_text_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_BASE}/finetuned_lora_text_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}_merged}"

echo "============================================"
echo "LoRA Merge for Inference (TEXT-ONLY)"
echo "============================================"
echo "Mode:          TEXT-ONLY training"
echo "LoRA adapter:  $LORA_PATH"
echo "Output path:   $OUTPUT_PATH"
echo "============================================"
echo ""

# Check if LoRA adapter exists
if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA adapter not found: $LORA_PATH"
    exit 1
fi

# Run merge (uses the same merge script)
CUDA_VISIBLE_DEVICES=0 python 5_merge_lora_for_inference.py \
    --lora_path "$LORA_PATH" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "Done! Merged model saved to: $OUTPUT_PATH"
echo ""
echo "To evaluate on harmful audio:"
echo "  bash 6_evaluate_harmful_audio.sh --filter_type text --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
