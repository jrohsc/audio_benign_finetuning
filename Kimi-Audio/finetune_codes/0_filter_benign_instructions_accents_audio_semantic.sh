#!/bin/bash
# Filter Benign Instructions Accents samples by AUDIO SEMANTIC embedding distance
#
# Uses WhisperVQEncoder for semantic audio understanding.
# Output: data_benign_instructions_accents/ folder

set -e

source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio
module load cuda/12.6

DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

INPUT_JSONL="data/benign_instructions_accents/benign_instructions_accents_full.jsonl"
HARMFUL_DIR="../../harmful_data/advbench_gtts/en"
OUTPUT_DIR="data_benign_instructions_accents"

THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold) THRESHOLD="$2"; PERCENTAGE=""; NUM_SAMPLES=""; shift 2 ;;
        --percentage) PERCENTAGE="$2"; THRESHOLD=""; NUM_SAMPLES=""; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; THRESHOLD=""; PERCENTAGE=""; shift 2 ;;
        --select_safest) SELECT_SAFEST=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--percentage VALUE | --num_samples VALUE | --threshold VALUE]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
OUTPUT="${OUTPUT_DIR}/benign_instructions_accents_filtered_audio_semantic_${OUTPUT_SUFFIX}.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Benign Instructions Accents AUDIO SEMANTIC Filtering"
echo "============================================"
echo "Input JSONL:   $INPUT_JSONL"
echo "Output file:   $OUTPUT"
echo "============================================"

CMD="CUDA_VISIBLE_DEVICES=0 python 0_filter_heysquad_audio_semantic.py"
CMD="$CMD --heysquad_jsonl \"$INPUT_JSONL\""
CMD="$CMD --harmful_dir \"$HARMFUL_DIR\""
CMD="$CMD --output \"$OUTPUT\""
CMD="$CMD --cache_dir \"embedding_cache_audio_semantic_benign_instructions_accents\""
CMD="$CMD --device cuda"

[ -n "$THRESHOLD" ] && CMD="$CMD --threshold $THRESHOLD"
[ -n "$PERCENTAGE" ] && CMD="$CMD --percentage $PERCENTAGE"
[ -n "$NUM_SAMPLES" ] && CMD="$CMD --num_samples $NUM_SAMPLES"
[ "$SELECT_SAFEST" = true ] && CMD="$CMD --select_safest"

echo "Running: $CMD"
eval $CMD

echo "Filtering complete! Output: $OUTPUT"
