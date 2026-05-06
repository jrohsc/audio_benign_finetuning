#!/bin/bash
# =============================================================================
# TEXT-BASED SEMANTIC FILTERING
# =============================================================================
#
# Filter VoiceBench samples by SEMANTIC embedding similarity to harmful TEXT prompts.
# Uses sentence-transformers (all-MiniLM-L6-v2) to extract text embeddings.
#
# THREE FILTERING APPROACHES:
#   - Text-Semantic (this script): SEMANTIC from TEXT (sentence-transformers)
#   - Audio-Semantic (0_filter_voicebench_audio_semantic.sh): SEMANTIC from AUDIO
#   - Acoustic (0_filter_voicebench_acoustic.sh): ACOUSTIC from AUDIO (Whisper-LV3)
#
# Output: data_semantic/ folder

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
HARMFUL_SOURCE="advbench"  # or "safetybench"
HARMFUL_TEXTS_CSV="../../harmful_data/${HARMFUL_SOURCE}.csv"
TEXT_COLUMN="goal"  # Column name in CSV containing harmful texts

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Output directory (separate from acoustic filtering)
OUTPUT_DIR="data_semantic"

# Sentence transformer model
SEMANTIC_MODEL="all-MiniLM-L6-v2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --harmful_source)
            HARMFUL_SOURCE="$2"
            HARMFUL_TEXTS_CSV="../../harmful_data/${HARMFUL_SOURCE}.csv"
            shift 2
            ;;
        --harmful_texts_csv)
            HARMFUL_TEXTS_CSV="$2"
            shift 2
            ;;
        --text_column)
            TEXT_COLUMN="$2"
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
            SEMANTIC_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench by SEMANTIC (text) embedding similarity to harmful prompts."
            echo "Results saved to data_semantic/ folder."
            echo ""
            echo "Options:"
            echo "  --harmful_source NAME     Harmful source name (advbench or safetybench)"
            echo "  --harmful_texts_csv PATH  Path to CSV file with harmful texts"
            echo "  --text_column NAME        Column name in CSV for harmful texts (default: goal)"
            echo "  --threshold VALUE         Distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE        Keep top percentage of closest samples (e.g., 50 for 50%)"
            echo "  --num_samples VALUE       Keep exact number of samples"
            echo "  --output_dir PATH         Output directory (default: data_semantic)"
            echo "  --model NAME              Sentence transformer model (default: all-MiniLM-L6-v2)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --percentage 10 --harmful_source advbench"
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
    OUTPUT_SUFFIX="threshold_${THRESHOLD}"
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
echo "VoiceBench SEMANTIC (Text) Filtering"
echo "============================================"
echo "Filter type:       SEMANTIC (text meaning)"
echo "Harmful texts CSV: $HARMFUL_TEXTS_CSV"
echo "Text column:       $TEXT_COLUMN"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output directory:  $OUTPUT_DIR"
echo "Output file:       $OUTPUT"
echo "Semantic model:    $SEMANTIC_MODEL"
echo "============================================"

# Build command
CMD="python 0_filter_voicebench_text_semantic.py"
CMD="$CMD --harmful_texts_csv \"$HARMFUL_TEXTS_CSV\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --text_column \"$TEXT_COLUMN\""
CMD="$CMD --cache_dir \"embedding_cache_semantic_text\""
CMD="$CMD --model \"$SEMANTIC_MODEL\""
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
echo "SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type semantic"
echo "  bash 1_extract_semantic_tokens.sh --filter_type semantic --percentage $PERCENTAGE"
