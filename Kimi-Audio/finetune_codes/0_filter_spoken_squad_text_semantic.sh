#!/bin/bash
# Filter Spoken-SQuAD samples by TEXT SEMANTIC embedding distance
#
# Uses sentence-transformers for text embeddings.
# Output: data_semantic_spoken_squad/ folder

set -e

# Load conda and CUDA environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
SPOKEN_SQUAD_JSONL="data_spoken_squad/spoken_squad_full.jsonl"
HARMFUL_TEXTS_CSV="../../harmful_data/advbench.csv"
TEXT_COLUMN="goal"

# Filtering parameters
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Output directory
OUTPUT_DIR="data_semantic_spoken_squad"

# Model
SEMANTIC_MODEL="all-MiniLM-L6-v2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spoken_squad_jsonl)
            SPOKEN_SQUAD_JSONL="$2"
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
            echo "Filter Spoken-SQuAD by TEXT SEMANTIC embedding similarity."
            echo "Results saved to data_semantic_spoken_squad/ folder."
            echo ""
            echo "Options:"
            echo "  --spoken_squad_jsonl PATH   Path to Spoken-SQuAD JSONL file"
            echo "  --harmful_texts_csv PATH    Path to CSV file with harmful texts"
            echo "  --text_column NAME          Column name in CSV (default: goal)"
            echo "  --threshold VALUE           Distance threshold"
            echo "  --percentage VALUE          Keep top percentage (e.g., 50 for 50%)"
            echo "  --num_samples VALUE         Keep exact number of samples"
            echo "  --output_dir PATH           Output directory"
            echo "  --model NAME                Sentence transformer model"
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
OUTPUT="${OUTPUT_DIR}/spoken_squad_filtered_semantic_${OUTPUT_SUFFIX}.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Spoken-SQuAD TEXT SEMANTIC Filtering"
echo "============================================"
echo "Filter type:       TEXT SEMANTIC"
echo "Encoder:           sentence-transformers ($SEMANTIC_MODEL)"
echo "Input JSONL:       $SPOKEN_SQUAD_JSONL"
echo "Harmful texts CSV: $HARMFUL_TEXTS_CSV"
echo "Text column:       $TEXT_COLUMN"
echo "Threshold:         ${THRESHOLD:-not set}"
echo "Percentage:        ${PERCENTAGE:-not set}"
echo "Num samples:       ${NUM_SAMPLES:-not set}"
echo "Output file:       $OUTPUT"
echo "============================================"

# Build command
CMD="python 0_filter_spoken_squad_text_semantic.py"
CMD="$CMD --spoken_squad_jsonl \"$SPOKEN_SQUAD_JSONL\""
CMD="$CMD --harmful_texts_csv \"$HARMFUL_TEXTS_CSV\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --text_column \"$TEXT_COLUMN\""
CMD="$CMD --cache_dir \"embedding_cache_semantic_spoken_squad\""
CMD="$CMD --device cuda"
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

# Run filtering
echo "Running: $CMD"
eval $CMD

echo ""
echo "============================================"
echo "Filtering complete!"
echo "Output saved to: $OUTPUT"
echo "============================================"
