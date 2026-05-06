#!/bin/bash
# Filter LibriSpeech samples by TRUE ACOUSTIC embedding distance
#
# Uses Whisper-Large-V3 encoder for acoustic features.
# Output: data_acoustic_librispeech/ folder

set -e

# Load conda and CUDA environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
LIBRISPEECH_JSONL="data_librispeech/librispeech_full.jsonl"
HARMFUL_DIR="../../harmful_data/advbench_gtts/en"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Output directory
OUTPUT_DIR="data_acoustic_librispeech"

# Whisper model
WHISPER_MODEL="openai/whisper-large-v3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --librispeech_jsonl)
            LIBRISPEECH_JSONL="$2"
            shift 2
            ;;
        --harmful_dir)
            HARMFUL_DIR="$2"
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter LibriSpeech by TRUE ACOUSTIC embedding similarity."
            echo "Results saved to data_acoustic_librispeech/ folder."
            echo ""
            echo "Options:"
            echo "  --librispeech_jsonl PATH    Path to LibriSpeech JSONL file"
            echo "  --harmful_dir PATH          Path to harmful audio directory"
            echo "  --threshold VALUE           Distance threshold"
            echo "  --percentage VALUE          Keep top percentage (e.g., 50 for 50%)"
            echo "  --num_samples VALUE         Keep exact number of samples"
            echo "  --output_dir PATH           Output directory"
            echo "  --model NAME                Whisper model"
            echo "  -h, --help                  Show this help message"
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
    OUTPUT_SUFFIX="thresh_${THRESHOLD}"
else
    OUTPUT_SUFFIX="auto"
fi
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_safest"
fi
OUTPUT="${OUTPUT_DIR}/librispeech_filtered_acoustic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "LibriSpeech TRUE ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       ACOUSTIC"
echo "Encoder:           Whisper-Large-V3"
echo "Input JSONL:       $LIBRISPEECH_JSONL"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_librispeech_acoustic.py"
CMD="$CMD --librispeech_jsonl \"$LIBRISPEECH_JSONL\""
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_acoustic_librispeech\""
CMD="$CMD --device cuda"
CMD="$CMD --model \"$WHISPER_MODEL\""

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
echo "Filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
