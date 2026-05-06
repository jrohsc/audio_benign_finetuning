#!/bin/bash
# =============================================================================
# AUDIO-BASED ACOUSTIC FILTERING (TRUE ACOUSTIC FEATURES)
# =============================================================================
#
# Filter VoiceBench samples by TRUE ACOUSTIC embedding similarity to harmful audio.
# Uses Whisper-Large-V3 encoder - captures HOW it sounds (voice, prosody, timbre).
#
# KIMI-AUDIO HAS TWO ENCODERS (as per the paper):
#   1. WhisperVQEncoder -> SEMANTIC tokens (linguistic content)
#   2. Whisper-Large-V3 (THIS SCRIPT) -> ACOUSTIC features (voice characteristics)
#
# THREE FILTERING APPROACHES:
#   - Audio-Semantic (0_filter_voicebench_audio_semantic.sh): SEMANTIC from AUDIO
#   - Text-Semantic (0_filter_voicebench_text_semantic.sh): SEMANTIC from TEXT
#   - Acoustic (THIS SCRIPT): TRUE ACOUSTIC from AUDIO (Whisper-Large-V3)
#
# Output: data_acoustic/ folder

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

# Output directory (separate from semantic results)
OUTPUT_DIR="data_acoustic"

# Whisper model for acoustic features
WHISPER_MODEL="openai/whisper-large-v3"

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
        --model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench by TRUE ACOUSTIC embedding similarity to harmful audio."
            echo "Uses Whisper-Large-V3 encoder (same as Kimi-Audio's continuous features)."
            echo "Results saved to data_acoustic/ folder."
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME   Harmful audio source (advbench or safetybench)"
            echo "  --harmful_dir PATH      Path to harmful audio directory"
            echo "  --threshold VALUE       Distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE      Keep top percentage of closest samples (e.g., 50 for 50%)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --output_dir PATH       Output directory (default: data_acoustic)"
            echo "  --model NAME            Whisper model (default: openai/whisper-large-v3)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --percentage 50 --harmful_source advbench"
            exit 0
            ;;
        --select_safest)
            SELECT_SAFEST="true"
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
if [ -n "$SELECT_SAFEST" ]; then
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_safest"
fi
OUTPUT="${OUTPUT_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_acoustic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "VoiceBench TRUE ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       ACOUSTIC (voice characteristics)"
echo "Encoder:           Whisper-Large-V3 (continuous features)"
echo "Harmful audio dir: $HARMFUL_DIR"
echo "Whisper model:     $WHISPER_MODEL"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "============================================"
echo ""
echo "NOTE: This uses TRUE ACOUSTIC features from Whisper-Large-V3,"
echo "      which captures voice characteristics (HOW it sounds),"
echo "      NOT semantic content (WHAT is being said)."
echo "============================================"

# Build command
CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_voicebench_acoustic.py"
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_acoustic\""
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

if [ -n "$SELECT_SAFEST" ]; then
    CMD="$CMD --select_safest"
fi

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "TRUE ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type acoustic"
echo "  bash 1_extract_semantic_tokens.sh --filter_type acoustic --percentage $PERCENTAGE"
