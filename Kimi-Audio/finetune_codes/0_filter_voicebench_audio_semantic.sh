#!/bin/bash
# =============================================================================
# AUDIO-BASED SEMANTIC FILTERING
# =============================================================================
#
# Filter VoiceBench samples by SEMANTIC embedding similarity to harmful audio.
# Uses GLM-4 Voice Tokenizer (WhisperVQEncoder) - captures WHAT is being said.
#
# KIMI-AUDIO HAS TWO ENCODERS:
#   1. WhisperVQEncoder (this script) -> SEMANTIC tokens (linguistic content)
#   2. Whisper-Large-V3 (0_filter_voicebench_acoustic.sh) -> ACOUSTIC features
#
# THREE FILTERING APPROACHES:
#   - Audio-Semantic (this script): SEMANTIC from AUDIO (WhisperVQEncoder)
#   - Text-Semantic (0_filter_voicebench_text_semantic.sh): SEMANTIC from TEXT
#   - Acoustic (0_filter_voicebench_acoustic.sh): ACOUSTIC from AUDIO (Whisper-LV3)
#
# Output: data/ folder
#
# IMPORTANT: Use --center flag (embeddings have 99.96% norm in global mean)

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
SELECT_SAFEST=""

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
        --select_safest)
            SELECT_SAFEST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench by AUDIO-SEMANTIC embedding similarity to harmful audio."
            echo "Uses WhisperVQEncoder to capture linguistic content from audio."
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
            echo "  --select_safest         Select samples FURTHEST from harmful (safest)"
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
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_safest"
fi
OUTPUT="${OUTPUT_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "VoiceBench AUDIO-SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       AUDIO-SEMANTIC (linguistic content from audio)"
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
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_audio_semantic.py"
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

if [ "$SELECT_SAFEST" = true ]; then
    CMD="$CMD --select_safest"
fi

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "AUDIO-SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type audio_semantic"
echo "  bash 1_extract_semantic_tokens.sh --filter_type audio_semantic --percentage $PERCENTAGE"
