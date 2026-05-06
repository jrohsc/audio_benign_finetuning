#!/bin/bash
# Step 0 (Audio-Semantic): Filter VoiceBench Noise variant samples by AUDIO SEMANTIC embedding distance
#
# This script filters VoiceBench Noisy variant samples based on SEMANTIC content.
# Supports: voicebench_cafe, voicebench_traffic, etc.
#
# Usage:
#   bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type cafe --percentage 50
#   bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type traffic --percentage 25

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
MODEL_PATH="/work/anon/audio-flamingo/checkpoints/af3_pretrained"

# Default values
NOISE_TYPE="cafe"  # cafe, traffic, etc.
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --noise_type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --advbench_audio_dir)
            ADVBENCH_AUDIO_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench Noisy variants by AUDIO SEMANTIC embedding similarity."
            echo "Uses AF3's AudioFlamingo3Encoder + MultiModalProjector."
            echo ""
            echo "Options:"
            echo "  --noise_type TYPE        Noise type: cafe, traffic, etc."
            echo "  --threshold VALUE        Filter by distance threshold"
            echo "  --percentage VALUE       Filter by percentage of dataset (e.g., 50 for 50%)"
            echo "  --num_samples VALUE      Filter by exact number of samples"
            echo "  --advbench_audio_dir PATH  AdvBench audio directory"
            echo "  --model_path PATH        Path to AF3 model checkpoint"
            echo "  --select_safest          Select safest samples (furthest from harmful)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --noise_type cafe --percentage 50"
            echo "  $0 --noise_type traffic --num_samples 500"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

DATASET_NAME="voicebench_${NOISE_TYPE}"
VOICEBENCH_JSON="data/${DATASET_NAME}/${DATASET_NAME}_full.json"
CACHE_DIR="data_${DATASET_NAME}/embedding_cache"

# Determine output file name
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_JSON="data_${DATASET_NAME}/filtered/${DATASET_NAME}_semantic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_${DATASET_NAME}/filtered/${DATASET_NAME}_semantic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_${DATASET_NAME}/filtered/${DATASET_NAME}_semantic_threshold_${THRESHOLD}.json"
    MODE_DESC="threshold=${THRESHOLD}"
    FILTER_ARGS="--threshold $THRESHOLD"
else
    echo "Error: Must specify --threshold, --percentage, or --num_samples"
    exit 1
fi

# Append _safest suffix if selecting safest samples
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_JSON="${OUTPUT_JSON%.json}_safest.json"
    MODE_DESC="${MODE_DESC}, select_safest"
fi

# Check if input JSON exists
if [ ! -f "$VOICEBENCH_JSON" ]; then
    echo "Error: Input JSON not found: $VOICEBENCH_JSON"
    echo "Please run generate_voicebench_noisy.py first with --noise_types $NOISE_TYPE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "${DATASET_NAME} AUDIO SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       AUDIO-SEMANTIC (model understanding)"
echo "Encoder:           AF3's AudioFlamingo3Encoder + MultiModalProjector"
echo "Input JSON:        $VOICEBENCH_JSON"
echo "AdvBench audio:    $ADVBENCH_AUDIO_DIR"
echo "Model path:        $MODEL_PATH"
echo "Mode:              $MODE_DESC"
echo "Output:            $OUTPUT_JSON"
echo "Cache:             $CACHE_DIR"
echo "============================================"

# Run filtering with AF3 encoder
python 0_filter_closest_to_advbench.py \
    --voicebench_json "$VOICEBENCH_JSON" \
    --advbench_audio_dir "$ADVBENCH_AUDIO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_json "$OUTPUT_JSON" \
    --model_path "$MODEL_PATH" \
    --use_af3_encoder \
    $FILTER_ARGS \
    ${SELECT_SAFEST:+--select_safest}

echo ""
echo "============================================"
echo "AUDIO SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "============================================"
