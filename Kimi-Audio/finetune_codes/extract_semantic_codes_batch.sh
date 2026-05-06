#!/bin/bash
# Extract semantic codes for all filtered VoiceBench datasets
#
# This script processes all filtered JSONL files and adds audio_tokens to each.
# These tokens are required for training.

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Use distance mode for consistency with Qwen-Audio
USE_DISTANCE=true
HARMFUL_SOURCE="advbench"

if [ "$USE_DISTANCE" = true ]; then
    # Distance mode thresholds (lower = closer to harmful)
    # 0.002:   30 samples (0.5%)
    # 0.004:  403 samples (6.6%)
    # 0.006: 1004 samples (16.5%)
    # 0.008: 2563 samples (42.1%)
    # 0.010: 4920 samples (80.9%)
    # 0.060: 6083 samples (100%)
    THRESHOLDS=(0.002 0.004 0.006 0.008 0.010 0.060)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_dist_"
else
    # Similarity mode thresholds (higher = closer to harmful)
    THRESHOLDS=(0.940 0.988 0.990 0.992 0.994 0.996)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_"
fi

echo "============================================"
echo "Batch Semantic Codes Extraction"
echo "============================================"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "============================================"

for THRESHOLD in "${THRESHOLDS[@]}"; do
    INPUT_FILE="data/${FILE_PREFIX}${THRESHOLD}.jsonl"
    OUTPUT_FILE="data/${FILE_PREFIX}${THRESHOLD}_semantic_codes.jsonl"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping $THRESHOLD: Input file not found: $INPUT_FILE"
        continue
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $THRESHOLD: Output file already exists: $OUTPUT_FILE"
        continue
    fi

    echo ""
    echo "Processing threshold $THRESHOLD..."
    echo "  Input:  $INPUT_FILE"
    echo "  Output: $OUTPUT_FILE"

    CUDA_VISIBLE_DEVICES=0 python 1_extract_semantic_codes.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "  Done!"
    else
        echo "  Error processing $THRESHOLD"
    fi
done

echo ""
echo "============================================"
echo "Batch extraction complete!"
echo "============================================"
echo ""
echo "Next step: Run finetune_batch.sh to finetune models"
