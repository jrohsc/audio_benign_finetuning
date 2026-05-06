#!/bin/bash
# =============================================================================
# TEXT-BASED SEMANTIC FILTERING - BBH (BIG-Bench Hard)
# =============================================================================
#
# Filter BBH samples by TEXT SEMANTIC embedding similarity to harmful texts.
# Uses sentence-transformers for text embeddings.
#
# Output: data_semantic_bbh/ folder

set -e

# Load conda and CUDA environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
HARMFUL_SOURCE="advbench"
HARMFUL_TEXTS_CSV="../../harmful_data/${HARMFUL_SOURCE}.csv"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Output directory
OUTPUT_DIR="data_semantic_bbh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --harmful_source)
            HARMFUL_SOURCE="$2"
            HARMFUL_TEXTS_CSV="../../harmful_data/${HARMFUL_SOURCE}.csv"
            shift 2
            ;;
        --harmful_texts_csv)
            HARMFUL_TEXTS_CSV="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter BBH by TEXT SEMANTIC embedding similarity to harmful texts."
            echo "Uses sentence-transformers for text embeddings."
            echo ""
            echo "Output: data_semantic_bbh/ folder"
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME      Harmful source (advbench or safetybench)"
            echo "  --harmful_texts_csv PATH   Path to CSV with harmful texts"
            echo "  --threshold VALUE          Distance threshold"
            echo "  --percentage VALUE         Keep top percentage of closest samples"
            echo "  --num_samples VALUE        Keep exact number of samples"
            echo "  --output_dir PATH          Output directory"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        --select_safest)
            SELECT_SAFEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine output filename
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_SUFFIX="n${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_SUFFIX="percentage_${PERCENTAGE}"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_SUFFIX="${THRESHOLD}"
else
    OUTPUT_SUFFIX="auto"
fi
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_safest"
fi
OUTPUT="${OUTPUT_DIR}/bbh_filtered_text_semantic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "BBH TEXT SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       TEXT SEMANTIC (sentence-transformers)"
echo "Harmful texts CSV: $HARMFUL_TEXTS_CSV"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_bbh_text_semantic.py"
CMD="$CMD --harmful_texts_csv \"$HARMFUL_TEXTS_CSV\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_text_semantic_bbh\""
CMD="$CMD --device cuda"

if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

if [ -n "$PERCENTAGE" ]; then
    CMD="$CMD --percentage $PERCENTAGE"
fi

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ "$SELECT_SAFEST" = true ]; then
    CMD="$CMD --select_safest"
fi

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "TEXT SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --benign_dataset bbh --filter_type text_semantic"
