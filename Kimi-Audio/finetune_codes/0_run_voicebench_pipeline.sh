#!/bin/bash
# Complete VoiceBench Finetuning Pipeline
#
# This script runs the entire pipeline:
# 1. Filter VoiceBench by embedding similarity to harmful audio
# 2. Extract semantic codes for filtered data
# 3. Run finetuning (optional - controlled by RUN_FINETUNE flag)
#
# Usage:
#   ./run_voicebench_pipeline.sh                    # Run filtering + semantic codes only
#   ./run_voicebench_pipeline.sh --finetune         # Run full pipeline including finetuning
#   ./run_voicebench_pipeline.sh --threshold 0.85   # Use different threshold
#   ./run_voicebench_pipeline.sh --harmful safetybench  # Use SafetyBench instead of AdvBench

set -e  # Exit on error

# Default configuration
HARMFUL_SOURCE="advbench"
THRESHOLD=0.80
RUN_FINETUNE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --finetune) RUN_FINETUNE=true ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --harmful) HARMFUL_SOURCE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

HARMFUL_DIR="/work/anon/audio_benign_finetuning/harmful_data/${HARMFUL_SOURCE}_gtts/en"
FILTERED_FILE="voicebench_filtered_${HARMFUL_SOURCE}_${THRESHOLD}.jsonl"
SEMANTIC_CODES_FILE="voicebench_filtered_${HARMFUL_SOURCE}_${THRESHOLD}_semantic_codes.jsonl"
OUTPUT_DIR="output/finetuned_lora_voicebench_${HARMFUL_SOURCE}_${THRESHOLD}"

echo "========================================================"
echo "VoiceBench Finetuning Pipeline"
echo "========================================================"
echo "Harmful source: ${HARMFUL_SOURCE}_gtts/en"
echo "Threshold: ${THRESHOLD}"
echo "Run finetuning: ${RUN_FINETUNE}"
echo "========================================================"
echo ""

# Step 0: Filter VoiceBench by embedding similarity
echo "========================================================"
echo "Step 0: Filtering VoiceBench by embedding similarity"
echo "========================================================"
if [ -f "$FILTERED_FILE" ]; then
    echo "Filtered file already exists: $FILTERED_FILE"
    echo "Skipping filtering step. Delete the file to re-run."
else
    CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_by_embedding.py \
        --harmful_dir "$HARMFUL_DIR" \
        --threshold $THRESHOLD \
        --output "$FILTERED_FILE" \
        --cache_dir "embedding_cache" \
        --device cuda
fi
echo ""

# Check if filtering produced any samples
if [ ! -f "$FILTERED_FILE" ]; then
    echo "Error: Filtering did not produce output file"
    exit 1
fi

NUM_SAMPLES=$(wc -l < "$FILTERED_FILE")
echo "Filtered samples: $NUM_SAMPLES"

if [ "$NUM_SAMPLES" -eq 0 ]; then
    echo "Error: No samples passed the filter. Try lowering the threshold."
    exit 1
fi
echo ""

# Step 1: Extract semantic codes
echo "========================================================"
echo "Step 1: Extracting semantic codes"
echo "========================================================"
if [ -f "$SEMANTIC_CODES_FILE" ]; then
    echo "Semantic codes file already exists: $SEMANTIC_CODES_FILE"
    echo "Skipping extraction step. Delete the file to re-run."
else
    CUDA_VISIBLE_DEVICES=0 python -m extract_semantic_codes \
        --input_file "$FILTERED_FILE" \
        --output_file "$SEMANTIC_CODES_FILE"
fi
echo ""

# Check if semantic codes extraction succeeded
if [ ! -f "$SEMANTIC_CODES_FILE" ]; then
    echo "Error: Semantic codes extraction failed"
    exit 1
fi
echo ""

# Step 2 (optional): Run finetuning
if [ "$RUN_FINETUNE" = true ]; then
    echo "========================================================"
    echo "Step 2: Running LoRA finetuning"
    echo "========================================================"

    # Check pretrained model exists
    if [ ! -d "/project/anon/BFT_models/pretrained_kimi_instruct" ]; then
        echo "Error: Pretrained model not found at /project/anon/BFT_models/pretrained_kimi_instruct"
        echo "Run: python -m model --model_name 'moonshotai/Kimi-Audio-7B-Instruct' --output_dir '/project/anon/BFT_models/pretrained_kimi_instruct'"
        exit 1
    fi

    ./finetune_lora.sh \
        -d "$SEMANTIC_CODES_FILE" \
        -o "$OUTPUT_DIR"
else
    echo "========================================================"
    echo "Pipeline Complete (without finetuning)"
    echo "========================================================"
    echo ""
    echo "Filtered data: $FILTERED_FILE"
    echo "With semantic codes: $SEMANTIC_CODES_FILE"
    echo "Samples: $NUM_SAMPLES"
    echo ""
    echo "To run finetuning:"
    echo "  ./finetune_lora.sh -d $SEMANTIC_CODES_FILE -o $OUTPUT_DIR"
    echo ""
    echo "Or re-run this script with --finetune flag:"
    echo "  ./run_voicebench_pipeline.sh --finetune --threshold $THRESHOLD --harmful $HARMFUL_SOURCE"
fi

echo ""
echo "Done!"
