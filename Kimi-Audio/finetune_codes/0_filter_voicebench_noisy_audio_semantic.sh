#!/bin/bash
# =============================================================================
# AUDIO-BASED SEMANTIC FILTERING (WhisperVQEncoder) - VoiceBench Noisy
# =============================================================================
#
# Filter VoiceBench Noisy samples by AUDIO SEMANTIC embedding similarity to harmful audio.
# Uses Kimi-Audio's WhisperVQEncoder - captures WHAT is being said (linguistic content).
#
# VoiceBench Noisy contains the original VoiceBench dataset augmented with realistic
# background noise (cafe, car, traffic, office, etc.) at configurable SNR levels.
#
# Output: data_voicebench_noisy/ folder

set -e

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
HARMFUL_SOURCE="advbench"  # or "safetybench"
HARMFUL_DIR="../../harmful_data/${HARMFUL_SOURCE}_gtts/en"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""

# Output directory
OUTPUT_DIR="data_voicebench_noisy"

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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench Noisy by AUDIO SEMANTIC embedding similarity to harmful audio."
            echo "Uses Kimi-Audio's WhisperVQEncoder (semantic content from audio)."
            echo "Results saved to data_voicebench_noisy/ folder."
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME   Harmful audio source (advbench or safetybench)"
            echo "  --harmful_dir PATH      Path to harmful audio directory"
            echo "  --threshold VALUE       Distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE      Keep top percentage of closest samples (e.g., 50 for 50%)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --output_dir PATH       Output directory (default: data_voicebench_noisy)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --percentage 50 --harmful_source advbench"
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
    OUTPUT_SUFFIX="thresh_${THRESHOLD}"
else
    OUTPUT_SUFFIX="auto"
fi
OUTPUT="${OUTPUT_DIR}/voicebench_noisy_filtered_${HARMFUL_SOURCE}_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "VoiceBench Noisy AUDIO SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       AUDIO-SEMANTIC (linguistic content)"
echo "Encoder:           Kimi-Audio's WhisperVQEncoder"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"
echo ""
echo "NOTE: This uses AUDIO SEMANTIC features from WhisperVQEncoder,"
echo "      which captures WHAT is being said (linguistic content),"
echo "      NOT voice characteristics (HOW it sounds)."
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_noisy_audio_semantic.py"
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_voicebench_noisy\""
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

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "AUDIO SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type audio_semantic --benign_dataset voicebench_noisy"
