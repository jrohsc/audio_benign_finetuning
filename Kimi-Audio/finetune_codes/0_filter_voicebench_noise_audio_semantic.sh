#!/bin/bash
# =============================================================================
# AUDIO-BASED SEMANTIC FILTERING - VoiceBench Noisy Variants
# =============================================================================
#
# Filter VoiceBench Noisy variant samples by AUDIO SEMANTIC embedding similarity.
# Supports: voicebench_cafe, voicebench_traffic, etc.
#
# Usage:
#   bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type cafe --percentage 50
#   bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type traffic --percentage 25

set -e

# Load CUDA
module load cuda/12.6

# Activate conda environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
NOISE_TYPE="cafe"  # cafe, traffic, etc.
HARMFUL_SOURCE="advbench"
HARMFUL_DIR="../../harmful_data/${HARMFUL_SOURCE}_gtts/en"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# IMPORTANT: Center embeddings (required for Kimi-Audio semantic embeddings)
# Without centering, all distances are ~0 because embeddings share a large common baseline
CENTER=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --noise_type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --harmful_source)
            HARMFUL_SOURCE="$2"
            HARMFUL_DIR="../../harmful_data/${HARMFUL_SOURCE}_gtts/en"
            shift 2
            ;;
        --harmful_dir)
            HARMFUL_DIR="$2"
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
        --center)
            CENTER=true
            shift
            ;;
        --no_center)
            CENTER=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench Noisy variants by AUDIO SEMANTIC embedding similarity."
            echo "Uses GLM-4 Voice Tokenizer (WhisperVQEncoder)."
            echo ""
            echo "Options:"
            echo "  --noise_type TYPE       Noise type: cafe, traffic, etc."
            echo "  --harmful_source NAME   Harmful audio source (advbench or safetybench)"
            echo "  --harmful_dir PATH      Path to harmful audio directory"
            echo "  --threshold VALUE       Distance threshold"
            echo "  --percentage VALUE      Keep top percentage (e.g., 50 for 50%)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --center                Center embeddings before computing distance (default, REQUIRED)"
            echo "  --no_center             Don't center embeddings (NOT recommended - distances will be ~0)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --noise_type cafe --percentage 50"
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

DATASET_NAME="voicebench_${NOISE_TYPE}"
OUTPUT_DIR="data_${DATASET_NAME}"

# Determine output filename
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_SUFFIX="n${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_SUFFIX="percentage_${PERCENTAGE}"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_SUFFIX="thresh_${THRESHOLD}"
else
    OUTPUT_SUFFIX="auto"
fi
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_safest"
fi
OUTPUT="${OUTPUT_DIR}/${DATASET_NAME}_filtered_${HARMFUL_SOURCE}_audio_semantic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "${DATASET_NAME} AUDIO SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       AUDIO-SEMANTIC (model understanding)"
echo "Encoder:           GLM-4 Voice Tokenizer (WhisperVQEncoder)"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Center embeddings: $CENTER (REQUIRED for meaningful distances)"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_noise_audio_semantic.py"
CMD="$CMD --noise_type \"$NOISE_TYPE\""
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_${DATASET_NAME}\""
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

if [ "$CENTER" = true ]; then
    CMD="$CMD --center"
fi

if [ "$SELECT_SAFEST" = true ]; then
    CMD="$CMD --select_safest"
fi

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "AUDIO SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
