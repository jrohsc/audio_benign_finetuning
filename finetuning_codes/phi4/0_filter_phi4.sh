#!/bin/bash
#
# Filter VoiceBench samples based on embedding distance from AdvBench samples.
# Uses Phi-4's audio encoder to compute audio embeddings.
#
# Threshold values (for filtering CLOSEST to harmful):
#   0.2301 = 10%, 0.2561 = 25%, 0.2906 = 50%, 0.3289 = 75%, 0.3658 = 90%

set -e

# Default values
VOICEBENCH_JSON="../audio-flamingo/data/voicebench/sd-qa/sd_qa_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data/embedding_cache_phi4"
MODEL_NAME="/datasets/ai/phi/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
METRIC="cosine"

# Threshold for 50% of dataset
THRESHOLD="0.3658"

# Output file (includes threshold in name)
OUTPUT_JSON="data/voicebench/sd-qa/sd_qa_filtered_phi4_${THRESHOLD}.json"

TOP_K=""
FILTER_CLOSEST="--filter_closest"  # Filter closest to harmful

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --voicebench_json)
            VOICEBENCH_JSON="$2"
            shift 2
            ;;
        --advbench_audio_dir)
            ADVBENCH_AUDIO_DIR="$2"
            shift 2
            ;;
        --output_json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --filter_closest)
            FILTER_CLOSEST="--filter_closest"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --voicebench_json PATH     Path to VoiceBench samples JSON"
            echo "  --advbench_audio_dir PATH  Directory containing AdvBench audio files"
            echo "  --output_json PATH         Path to save filtered samples"
            echo "  --cache_dir PATH           Directory to cache embeddings"
            echo "  --model_name NAME          Phi-4 model name"
            echo "  --metric METRIC            Distance metric (cosine or euclidean)"
            echo "  --threshold VALUE          Distance threshold for filtering"
            echo "  --top_k VALUE              Keep top-k samples"
            echo "  --filter_closest           Keep closest samples instead of farthest"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python 0_filter_phi4.py"
CMD="$CMD --voicebench_json $VOICEBENCH_JSON"
CMD="$CMD --advbench_audio_dir $ADVBENCH_AUDIO_DIR"
CMD="$CMD --output_json $OUTPUT_JSON"
CMD="$CMD --cache_dir $CACHE_DIR"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --metric $METRIC"

if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

if [ -n "$TOP_K" ]; then
    CMD="$CMD --top_k $TOP_K"
fi

if [ -n "$FILTER_CLOSEST" ]; then
    CMD="$CMD $FILTER_CLOSEST"
fi

echo "Running: $CMD"
cd "$(dirname "$0")"
$CMD
