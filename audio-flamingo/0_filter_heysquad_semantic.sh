#!/bin/bash
# Step 0 (Semantic): Filter HeySQuAD samples by SEMANTIC text embedding distance from harmful prompts
#
# This script filters HeySQuAD samples based on TEXT MEANING similarity.
# Uses sentence-transformers to compute text embeddings.
#
# HeySQuAD is similar to VoiceBench SD-QA - the QUESTION is spoken as audio.
# The existing semantic filtering script works directly since the JSON format is identical.
#
# Supports three filtering modes:
#   --threshold VALUE   Filter by distance threshold
#   --percentage VALUE  Filter by percentage of dataset (e.g., 50 for 50%)
#   --num_samples VALUE Filter by exact number of samples

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# HeySQuAD-specific paths
HEYSQUAD_JSON="data/heysquad/heysquad_human_train.json"
HARMFUL_TEXTS_CSV="../harmful_data/advbench.csv"
OUTPUT_DIR="data_semantic/heysquad"
CACHE_DIR="data_semantic/embedding_cache"

# Default values
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""
SELECT_SAFEST=""
MODEL="all-MiniLM-L6-v2"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_json)
            HEYSQUAD_JSON="$2"
            shift 2
            ;;
        --harmful_texts_csv)
            HARMFUL_TEXTS_CSV="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            THRESHOLD=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            PERCENTAGE=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            PERCENTAGE=""
            THRESHOLD=""
            shift 2
            ;;
        --select_safest)
            SELECT_SAFEST=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Filter HeySQuAD samples by semantic text embedding distance from harmful prompts."
            echo ""
            echo "Options:"
            echo "  --input_json PATH       HeySQuAD JSON file (default: data/heysquad/heysquad_human_train.json)"
            echo "  --harmful_texts_csv PATH  CSV with harmful texts (default: ../harmful_data/advbench.csv)"
            echo "  --output_dir PATH       Output directory (default: data_semantic/heysquad)"
            echo "  --percentage VALUE      Keep top percentage closest (default: 50)"
            echo "  --threshold VALUE       Keep samples with distance <= threshold"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --model NAME            Sentence transformer model (default: all-MiniLM-L6-v2)"
            echo "  --device DEVICE         Device to use (default: cuda)"
            echo "  --cache_dir PATH        Embedding cache directory"
            echo "  --select_safest          Select safest samples (furthest from harmful)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input
if [[ ! -f "$HEYSQUAD_JSON" ]]; then
    echo "ERROR: HeySQuAD JSON not found: $HEYSQUAD_JSON"
    echo "Run download_heysquad.py first to download the dataset."
    exit 1
fi

if [[ ! -f "$HARMFUL_TEXTS_CSV" ]]; then
    echo "ERROR: Harmful texts CSV not found: $HARMFUL_TEXTS_CSV"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine output filename
if [[ -n "$PERCENTAGE" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_semantic_percentage_${PERCENTAGE}.json"
    FILTER_ARG="--percentage $PERCENTAGE"
elif [[ -n "$THRESHOLD" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_semantic_threshold_${THRESHOLD}.json"
    FILTER_ARG="--threshold $THRESHOLD"
elif [[ -n "$NUM_SAMPLES" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_semantic_num_${NUM_SAMPLES}.json"
    FILTER_ARG="--num_samples $NUM_SAMPLES"
else
    echo "ERROR: Must specify --percentage, --threshold, or --num_samples"
    exit 1
fi

# Append _safest suffix if selecting safest samples
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_JSON="${OUTPUT_JSON%.json}_safest.json"
    MODE_DESC="${MODE_DESC}, select_safest"
fi

echo "=============================================="
echo "HeySQuAD Semantic Filtering"
echo "=============================================="
echo "Input JSON: $HEYSQUAD_JSON"
echo "Harmful texts: $HARMFUL_TEXTS_CSV"
echo "Output JSON: $OUTPUT_JSON"
echo "Filter: $FILTER_ARG"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "=============================================="

# Run the semantic filtering script (same as VoiceBench)
CMD="python 0_filter_voicebench_semantic.py"
CMD="$CMD --voicebench_json $HEYSQUAD_JSON"
CMD="$CMD --harmful_texts_csv $HARMFUL_TEXTS_CSV"
CMD="$CMD --output_json $OUTPUT_JSON"
CMD="$CMD $FILTER_ARG"

if [ "$SELECT_SAFEST" = true ]; then
    CMD="$CMD --select_safest"
fi

CMD="$CMD --cache_dir $CACHE_DIR"
CMD="$CMD --device $DEVICE"
CMD="$CMD --model $MODEL"

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Semantic filtering complete!"
echo "Output: $OUTPUT_JSON"
echo "=============================================="
