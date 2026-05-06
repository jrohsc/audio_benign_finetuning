#!/bin/bash
# Step 0 (Acoustic): Filter VoiceBench samples by ACOUSTIC embedding similarity to harmful audio
#
# This script filters VoiceBench samples based on AUDIO CHARACTERISTICS similarity.
# Uses GLM-4 Voice Tokenizer (WhisperVQEncoder) to extract semantic embeddings from audio.
#
# DIFFERENCE FROM SEMANTIC FILTERING:
#   - Acoustic (this script): filters by audio characteristics (voice, tone, prosody, etc.)
#   - Semantic (0_filter_voicebench_semantic.sh): filters by text meaning (what is being said)
#
# Output is saved to data/ folder (separate from semantic results in data_semantic/)
#
# IMPORTANT: Use --center flag to center embeddings!
# Without centering: distance range ~0.00001-0.0004 (all samples nearly identical)
# With centering: distance range ~0.01-1.1 (varied, meaningful distances)

set -e

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
HARMFUL_SOURCE="advbench"  # or "safetybench"
HARMFUL_DIR="../../harmful_data/${HARMFUL_SOURCE}_gtts/en"

# Use distance (1-similarity) instead of similarity for consistency with Qwen-Audio
USE_DISTANCE=true
CENTER=true

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""

# Output directory
OUTPUT_DIR="data"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use_distance)
            USE_DISTANCE=true
            shift
            ;;
        --no_distance)
            USE_DISTANCE=false
            shift
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
            echo "Filter VoiceBench by ACOUSTIC embedding similarity to harmful audio."
            echo "Results saved to data/ folder."
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME   Harmful audio source (advbench or safetybench)"
            echo "  --harmful_dir PATH      Path to harmful audio directory"
            echo "  --threshold VALUE       Distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE      Keep top percentage of closest samples (e.g., 50 for 50%)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --output_dir PATH       Output directory (default: data)"
            echo "  --use_distance          Use cosine distance (default)"
            echo "  --no_distance           Use cosine similarity instead"
            echo "  --center                Center embeddings (default, recommended)"
            echo "  --no_center             Don't center embeddings"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --percentage 10 --harmful_source advbench"
            exit 0
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
OUTPUT="${OUTPUT_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "VoiceBench ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       ACOUSTIC (audio characteristics)"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Use distance:      $USE_DISTANCE"
echo "Center embeddings: $CENTER"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_by_embedding.py"
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_semantic\""
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

if [ "$USE_DISTANCE" = true ]; then
    CMD="$CMD --use_distance"
fi

if [ "$CENTER" = true ]; then
    CMD="$CMD --center"
fi

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type acoustic"
echo "  bash 1_extract_semantic_tokens.sh --filter_type acoustic --percentage $PERCENTAGE"
