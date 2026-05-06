#!/bin/bash
# =============================================================================
# ACOUSTIC FILTERING - MMSU (Multi-Subject Understanding)
# =============================================================================
#
# Filter MMSU samples by TRUE ACOUSTIC embedding similarity to harmful audio.
# Uses Whisper-Large-V3 encoder - captures HOW the audio sounds.
#
# Output: data_acoustic_mmsu/ folder

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
HARMFUL_DIR="../../harmful_data/${HARMFUL_SOURCE}_gtts/en"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Output directory
OUTPUT_DIR="data_acoustic_mmsu"

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
            echo "Filter MMSU by TRUE ACOUSTIC embedding similarity to harmful audio."
            echo "Uses Whisper-Large-V3 encoder for acoustic features."
            echo ""
            echo "Output: data_acoustic_mmsu/ folder"
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME   Harmful audio source (advbench or safetybench)"
            echo "  --harmful_dir PATH      Path to harmful audio directory"
            echo "  --threshold VALUE       Distance threshold"
            echo "  --percentage VALUE      Keep top percentage of closest samples"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --output_dir PATH       Output directory"
            echo "  -h, --help              Show this help message"
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
OUTPUT="${OUTPUT_DIR}/mmsu_filtered_acoustic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "MMSU ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       ACOUSTIC (Whisper-Large-V3)"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_mmsu_acoustic.py"
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_acoustic_mmsu\""
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
echo "ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --benign_dataset mmsu --filter_type audio_acoustic"
