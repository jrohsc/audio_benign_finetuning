#!/bin/bash
# Filter GammaCorpus USA samples by AUDIO SEMANTIC embedding distance
#
# Uses AF3's encoder + projector for semantic audio understanding.
# Results saved to data_gammacorpus_usa/ folder.

set -e

DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

INPUT_JSON="data/gammacorpus_usa/gammacorpus_usa_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data_gammacorpus_usa/embedding_cache"

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
            echo "Usage: $0 [--percentage VALUE | --num_samples VALUE | --threshold VALUE] [--select_safest]"
            echo "  --select_safest          Select safest samples (furthest from harmful)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_JSON="data_gammacorpus_usa/gammacorpus_usa_audio_semantic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_gammacorpus_usa/gammacorpus_usa_audio_semantic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_gammacorpus_usa/gammacorpus_usa_audio_semantic_threshold_${THRESHOLD}.json"
    MODE_DESC="threshold=${THRESHOLD}"
    FILTER_ARGS="--threshold $THRESHOLD"
else
    echo "Error: Must specify --threshold, --percentage, or --num_samples"
    exit 1
fi

# Append _safest suffix if selecting safest samples
if [ "$SELECT_SAFEST" = true ]; then
    OUTPUT_JSON="${OUTPUT_JSON%.json}_safest.json"
    MODE_DESC="${MODE_DESC}, select_safest"
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "GammaCorpus USA AUDIO SEMANTIC Filtering"
echo "============================================"
echo "Input JSON:    $INPUT_JSON"
echo "Output JSON:   $OUTPUT_JSON"
echo "============================================"

python 0_filter_heysquad_audio_semantic.py \
    --heysquad_json "$INPUT_JSON" \
    --advbench_audio_dir "$ADVBENCH_AUDIO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_json "$OUTPUT_JSON" \
    --dataset_name "GammaCorpus USA" \
    $FILTER_ARGS \
    ${SELECT_SAFEST:+--select_safest}

echo "Filtering complete! Output: $OUTPUT_JSON"
