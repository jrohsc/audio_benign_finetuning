#!/bin/bash
# Step 0 (Text-Semantic): Filter BBH samples by TEXT SEMANTIC embedding similarity
#
# This script filters BBH (BIG-Bench Hard) samples based on TEXT MEANING similarity to harmful prompts.
# Uses sentence-transformers to extract text embeddings.
#
# THREE FILTER TYPES AVAILABLE:
#   - audio_acoustic (0_filter_bbh_audio_acoustic.sh): TRUE acoustic features
#       Uses: Whisper-Large-V3 encoder only
#       Output: data_acoustic_bbh/
#
#   - audio_semantic (0_filter_bbh_audio_semantic.sh): Audio content as understood by model
#       Uses: AF3's AudioFlamingo3Encoder + MultiModalProjector
#       Output: data_bbh/
#
#   - text_semantic (this script): Text meaning similarity
#       Uses: sentence-transformers (all-MiniLM-L6-v2)
#       Output: data_semantic_bbh/
#
# Output is saved to data_semantic_bbh/ folder

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
BBH_JSON="data/bbh/bbh_full.json"
HARMFUL_TEXTS_CSV="../harmful_data/advbench.csv"
TEXT_COLUMN="goal"
CACHE_DIR="data_semantic_bbh/embedding_cache"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Sentence transformer model
SEMANTIC_MODEL="all-MiniLM-L6-v2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bbh_json)
            BBH_JSON="$2"
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
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --model)
            SEMANTIC_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter BBH by TEXT SEMANTIC embedding similarity to harmful prompts."
            echo "Uses sentence-transformers to compare text meaning."
            echo "Results saved to data_semantic_bbh/ folder."
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE    Distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE   Keep top percentage of closest samples (e.g., 10 for 10%)"
            echo "  --num_samples VALUE  Keep exact number of samples"
            echo ""
            echo "Other options:"
            echo "  --bbh_json PATH           Path to BBH JSON file"
            echo "  --harmful_texts_csv PATH  Path to CSV file with harmful texts"
            echo "  --text_column NAME        Column name in CSV for harmful texts (default: goal)"
            echo "  --cache_dir PATH          Directory to cache embeddings"
            echo "  --model NAME              Sentence transformer model (default: all-MiniLM-L6-v2)"
            echo "  --select_safest          Select safest samples (furthest from harmful)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 10        # Filter 10% closest samples"
            echo "  $0 --num_samples 500      # Filter 500 closest samples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine output filename and mode description
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_JSON="data_semantic_bbh/bbh_text_semantic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_semantic_bbh/bbh_text_semantic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_semantic_bbh/bbh_text_semantic_threshold_${THRESHOLD}.json"
    MODE_DESC="threshold=${THRESHOLD}"
else
    echo "Error: Must specify --threshold, --percentage, or --num_samples"
    exit 1
fi

# Append _safest suffix if selecting safest samples
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_JSON="${OUTPUT_JSON%.json}_safest.json"
    MODE_DESC="${MODE_DESC}, select_safest"
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "BBH TEXT-SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       TEXT-SEMANTIC (text meaning similarity)"
echo "Encoder:           sentence-transformers ($SEMANTIC_MODEL)"
echo "BBH JSON:          $BBH_JSON"
echo "Harmful texts CSV: $HARMFUL_TEXTS_CSV"
echo "Text column:       $TEXT_COLUMN"
echo "Mode:              $MODE_DESC"
echo "Output:            $OUTPUT_JSON"
echo "Cache directory:   $CACHE_DIR"
echo "============================================"

# Build command
CMD="python 0_filter_bbh_semantic.py"
CMD="$CMD --bbh_json \"$BBH_JSON\""
CMD="$CMD --harmful_texts_csv \"$HARMFUL_TEXTS_CSV\""
CMD="$CMD --output_json \"$OUTPUT_JSON\""
CMD="$CMD --text_column \"$TEXT_COLUMN\""
CMD="$CMD --cache_dir \"$CACHE_DIR\""
CMD="$CMD --model \"$SEMANTIC_MODEL\""

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

echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "TEXT-SEMANTIC filtering complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "============================================"
echo ""
echo "Next step: Run 1_prepare_filtered_dataset.sh with --filter_type text_semantic --benign_dataset bbh"
echo "  bash 1_prepare_filtered_dataset.sh --filter_type text_semantic --benign_dataset bbh --percentage $PERCENTAGE"
