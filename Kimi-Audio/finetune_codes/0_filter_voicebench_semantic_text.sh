#!/bin/bash

# Filter VoiceBench samples by SEMANTIC text similarity and output TEXT-ONLY format.
#
# This script:
# 1. Runs semantic filtering (text-to-text comparison)
# 2. Converts the output to text-only format (no audio messages)
#
# Output goes to data_semantic_text/ directory.

set -e

# Load CUDA for sentence-transformers
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""
HARMFUL_SOURCE="advbench"
HARMFUL_CSV="../../harmful_data/advbench.csv"

# Parse command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        --percentage)
            shift
            PERCENTAGE=$1
            NUM_SAMPLES=""
            THRESHOLD=""
            ;;
        --threshold)
            shift
            THRESHOLD=$1
            PERCENTAGE=""
            NUM_SAMPLES=""
            ;;
        --num_samples)
            shift
            NUM_SAMPLES=$1
            PERCENTAGE=""
            THRESHOLD=""
            ;;
        --harmful_source)
            shift
            HARMFUL_SOURCE=$1
            ;;
        --harmful_csv)
            shift
            HARMFUL_CSV=$1
            ;;
        -h | --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench samples by SEMANTIC text similarity."
            echo "Output is in TEXT-ONLY format (no audio messages)."
            echo ""
            echo "Options:"
            echo "  --percentage VALUE      Keep top percentage of closest samples (default: 50)"
            echo "  --threshold VALUE       Keep samples with distance <= threshold"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --harmful_source NAME   Name for output files (default: advbench)"
            echo "  --harmful_csv PATH      Path to harmful texts CSV"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 50"
            echo "  $0 --num_samples 500"
            exit 0
            ;;
        *)
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# Determine filter args and output suffix
if [ -n "$NUM_SAMPLES" ]; then
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
    FILE_SUFFIX="n${NUM_SAMPLES}"
elif [ -n "$THRESHOLD" ]; then
    FILTER_ARGS="--threshold $THRESHOLD"
    FILE_SUFFIX="${THRESHOLD}"
elif [ -n "$PERCENTAGE" ]; then
    FILTER_ARGS="--percentage $PERCENTAGE"
    FILE_SUFFIX="percentage_${PERCENTAGE}"
else
    echo "Error: Must specify --percentage, --threshold, or --num_samples"
    exit 1
fi

# Output directories
AUDIO_OUTPUT_DIR="data_semantic"
TEXT_OUTPUT_DIR="data_semantic_text"

mkdir -p "$AUDIO_OUTPUT_DIR"
mkdir -p "$TEXT_OUTPUT_DIR"

# Output file paths
AUDIO_OUTPUT="${AUDIO_OUTPUT_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}.jsonl"
TEXT_OUTPUT="${TEXT_OUTPUT_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}_text.jsonl"

echo "============================================"
echo "SEMANTIC Filtering + TEXT-ONLY Conversion"
echo "============================================"
echo "Filter args:      $FILTER_ARGS"
echo "Harmful CSV:      $HARMFUL_CSV"
echo "Audio output:     $AUDIO_OUTPUT"
echo "Text output:      $TEXT_OUTPUT"
echo "============================================"

# Step 1: Run semantic filtering (generates audio format)
echo ""
echo "Step 1: Running semantic filtering..."
python 0_filter_voicebench_semantic.py \
    --harmful_texts_csv "$HARMFUL_CSV" \
    --output "$AUDIO_OUTPUT" \
    $FILTER_ARGS

# Step 2: Convert to text-only format
echo ""
echo "Step 2: Converting to text-only format..."
if [ -n "$NUM_SAMPLES" ]; then
    python 1_convert_to_text_only.py --num_samples "$NUM_SAMPLES" --harmful_source "$HARMFUL_SOURCE"
elif [ -n "$PERCENTAGE" ]; then
    python 1_convert_to_text_only.py --percentage "$PERCENTAGE" --harmful_source "$HARMFUL_SOURCE"
elif [ -n "$THRESHOLD" ]; then
    python 1_convert_to_text_only.py --threshold "$THRESHOLD" --harmful_source "$HARMFUL_SOURCE"
fi

echo ""
echo "============================================"
echo "Filtering + Conversion Complete!"
echo "============================================"
echo "Audio format: $AUDIO_OUTPUT"
echo "Text format:  $TEXT_OUTPUT"
echo "============================================"
