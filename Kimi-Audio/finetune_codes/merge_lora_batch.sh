#!/bin/bash
# Merge LoRA adapters with base model for all thresholds
#
# This script merges each LoRA adapter with the base model for inference.

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Paths
PRETRAINED_MODEL_PATH="/project/anon/BFT_models/pretrained_kimi_instruct"
HARMFUL_SOURCE="advbench"

# Use distance mode for consistency with Qwen-Audio
USE_DISTANCE=true

if [ "$USE_DISTANCE" = true ]; then
    THRESHOLDS=(0.060 0.010 0.008 0.006 0.004 0.002)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_dist_"
else
    THRESHOLDS=(0.940 0.988 0.990 0.992 0.994 0.996)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_"
fi

echo "============================================"
echo "Batch LoRA Merge"
echo "============================================"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "============================================"

for THRESHOLD in "${THRESHOLDS[@]}"; do
    LORA_PATH="output/finetuned_lora_${FILE_PREFIX}${THRESHOLD}"
    OUTPUT_PATH="output/finetuned_lora_${FILE_PREFIX}${THRESHOLD}_merged"

    if [ ! -d "$LORA_PATH" ]; then
        echo "Skipping $THRESHOLD: LoRA not found: $LORA_PATH"
        continue
    fi

    if [ -d "$OUTPUT_PATH" ]; then
        echo "Skipping $THRESHOLD: Merged model already exists: $OUTPUT_PATH"
        continue
    fi

    echo ""
    echo "Merging threshold $THRESHOLD..."
    echo "  LoRA: $LORA_PATH"
    echo "  Output: $OUTPUT_PATH"

    CUDA_VISIBLE_DEVICES=0 python 5_merge_lora_for_inference.py \
        --model_path "$PRETRAINED_MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --output_path "$OUTPUT_PATH"

    if [ $? -eq 0 ]; then
        echo "  Merge complete!"
    else
        echo "  Error merging $THRESHOLD"
    fi
done

echo ""
echo "============================================"
echo "Batch merge complete!"
echo "============================================"
