#!/bin/bash
# Step 0 (Acoustic): Filter VoiceBench samples by ACOUSTIC embedding distance from AdvBench
#
# This script filters VoiceBench samples based on AUDIO CHARACTERISTICS similarity.
# Uses Audio-Flamingo 3's encoder to compute audio embeddings.
#
# DIFFERENCE FROM SEMANTIC FILTERING (0_filter_voicebench_semantic.sh):
#   - Acoustic (this script): filters by audio characteristics (voice, tone, prosody)
#   - Semantic: filters by text meaning (what is being said)
#
# Output is saved to data/ folder (separate from semantic results in data_semantic/)
#
# Supports three filtering modes:
#   --threshold VALUE   Filter by distance threshold (e.g., 0.0318)
#   --percentage VALUE  Filter by percentage of dataset (e.g., 10 for 10%)
#   --num_samples VALUE Filter by exact number of samples (e.g., 500)
#
# Threshold values reference (for filtering CLOSEST to harmful):
#   0.0226 = 10%, 0.0262 = 25%, 0.0318 = 50%, 0.0382 = 75%, 0.0445 = 90%

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
VOICEBENCH_JSON="data/voicebench/sd-qa/sd_qa_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data/embedding_cache"

# Default values (percentage mode with 65%)
THRESHOLD=""
PERCENTAGE="65"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --voicebench_json)
            VOICEBENCH_JSON="$2"
            shift 2
            ;;
        --advbench_audio_dir)
            ADVBENCH_AUDIO_DIR="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Filter VoiceBench by ACOUSTIC embedding similarity to harmful audio."
            echo "Results saved to data/ folder."
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE    Filter by distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE   Filter by percentage of dataset (e.g., 10 for 10%)"
            echo "  --num_samples VALUE  Filter by exact number of samples (e.g., 500)"
            echo ""
            echo "Other options:"
            echo "  --model_path PATH         Audio Flamingo model path"
            echo "  --voicebench_json PATH    VoiceBench JSON file"
            echo "  --advbench_audio_dir PATH AdvBench audio directory"
            echo "  --cache_dir PATH          Embedding cache directory"
            echo "  --select_safest          Select safest samples (furthest from harmful)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 10         # Filter 10% closest samples"
            echo "  $0 --num_samples 500       # Filter 500 closest samples"
            echo "  $0 --threshold 0.0318      # Filter by distance threshold"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine output file name and mode description based on filtering mode
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_JSON="data/voicebench/sd-qa/sd_qa_closest_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data/voicebench/sd-qa/sd_qa_closest_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data/voicebench/sd-qa/sd_qa_closest_threshold_${THRESHOLD}.json"
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

echo "============================================"
echo "VoiceBench ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       ACOUSTIC (audio characteristics)"
echo "Model:             $MODEL_PATH"
echo "VoiceBench JSON:   $VOICEBENCH_JSON"
echo "AdvBench audio:    $ADVBENCH_AUDIO_DIR"
echo "Mode:              $MODE_DESC"
echo "Output:            $OUTPUT_JSON"
echo "============================================"

python 0_filter_closest_to_advbench.py \
    --model_path "$MODEL_PATH" \
    --voicebench_json "$VOICEBENCH_JSON" \
    --advbench_audio_dir "$ADVBENCH_AUDIO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_json "$OUTPUT_JSON" \
    $FILTER_ARGS \
    ${SELECT_SAFEST:+--select_safest}

echo ""
echo "============================================"
echo "ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "============================================"
echo ""
echo "Next step: Run 1_prepare_filtered_dataset.sh with --filter_type acoustic"
echo "  bash 1_prepare_filtered_dataset.sh --filter_type acoustic --percentage $PERCENTAGE"
