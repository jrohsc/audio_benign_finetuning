#!/bin/bash
# Step 0 (Acoustic): Filter HeySQuAD samples by ACOUSTIC embedding distance from AdvBench
#
# This script filters HeySQuAD samples based on AUDIO CHARACTERISTICS similarity.
# Uses Audio-Flamingo 3's encoder to compute audio embeddings.
#
# HeySQuAD is similar to VoiceBench SD-QA - the QUESTION is spoken as audio.
# The existing acoustic filtering script works directly since the JSON format is identical.
#
# Supports three filtering modes:
#   --threshold VALUE   Filter by distance threshold
#   --percentage VALUE  Filter by percentage of dataset (e.g., 50 for 50%)
#   --num_samples VALUE Filter by exact number of samples

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Model path
MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"

# HeySQuAD-specific paths
HEYSQUAD_JSON="data/heysquad/heysquad_human_train.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
OUTPUT_DIR="data/heysquad"
CACHE_DIR="data/embedding_cache_heysquad"

# Default values
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""
SELECT_SAFEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input_json)
            HEYSQUAD_JSON="$2"
            shift 2
            ;;
        --advbench_audio_dir)
            ADVBENCH_AUDIO_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            PERCENTAGE=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            THRESHOLD=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            THRESHOLD=""
            PERCENTAGE=""
            shift 2
            ;;
        --select_safest)
            SELECT_SAFEST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Filter HeySQuAD samples by acoustic audio embedding distance from harmful audio."
            echo ""
            echo "Options:"
            echo "  --model_path PATH       Audio Flamingo 3 model path"
            echo "  --input_json PATH       HeySQuAD JSON file (default: data/heysquad/heysquad_human_train.json)"
            echo "  --advbench_audio_dir PATH  AdvBench audio directory (default: ../harmful_data/advbench_gtts/en)"
            echo "  --output_dir PATH       Output directory (default: data/heysquad)"
            echo "  --percentage VALUE      Keep top percentage closest (default: 50)"
            echo "  --threshold VALUE       Keep samples with distance <= threshold"
            echo "  --num_samples VALUE     Keep exact number of samples"
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

if [[ ! -d "$ADVBENCH_AUDIO_DIR" ]]; then
    echo "ERROR: AdvBench audio directory not found: $ADVBENCH_AUDIO_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine output filename
if [[ -n "$PERCENTAGE" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_acoustic_percentage_${PERCENTAGE}.json"
    FILTER_ARG="--percentage $PERCENTAGE"
    MODE_DESC="percentage=${PERCENTAGE}%"
elif [[ -n "$THRESHOLD" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_acoustic_threshold_${THRESHOLD}.json"
    FILTER_ARG="--threshold $THRESHOLD"
    MODE_DESC="threshold=${THRESHOLD}"
elif [[ -n "$NUM_SAMPLES" ]]; then
    OUTPUT_JSON="${OUTPUT_DIR}/heysquad_acoustic_num_${NUM_SAMPLES}.json"
    FILTER_ARG="--num_samples $NUM_SAMPLES"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
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
echo "HeySQuAD Acoustic Filtering"
echo "=============================================="
echo "Input JSON: $HEYSQUAD_JSON"
echo "AdvBench audio: $ADVBENCH_AUDIO_DIR"
echo "Output JSON: $OUTPUT_JSON"
echo "Filter: $MODE_DESC"
echo "Model: $MODEL_PATH"
echo "Cache: $CACHE_DIR"
echo "=============================================="

# Run the acoustic filtering script (same as VoiceBench)
CMD="python 0_filter_closest_to_advbench.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --voicebench_json $HEYSQUAD_JSON"
CMD="$CMD --advbench_audio_dir $ADVBENCH_AUDIO_DIR"
CMD="$CMD --output_json $OUTPUT_JSON"
CMD="$CMD --cache_dir $CACHE_DIR"
CMD="$CMD $FILTER_ARG"

if [ "$SELECT_SAFEST" = true ]; then
    CMD="$CMD --select_safest"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Acoustic filtering complete!"
echo "Output: $OUTPUT_JSON"
echo "=============================================="
echo ""
echo "Next step: Run data preparation"
echo "  bash 1_prepare_heysquad_dataset.sh --filter_type acoustic --percentage $PERCENTAGE"
