#!/bin/bash
# Step 0 (Audio-Acoustic): Filter HeySQuAD Accents samples by TRUE ACOUSTIC embedding distance
#
# This script filters HeySQuAD Accents samples based on TRUE AUDIO ACOUSTIC characteristics.
# Uses original Whisper-Large-V3 encoder ONLY (without AF3's multi-modal projector).
#
# Supports three filtering modes:
#   --threshold VALUE   Filter by distance threshold
#   --percentage VALUE  Filter by percentage of dataset (e.g., 50 for 50%)
#   --num_samples VALUE Filter by exact number of samples

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

HEYSQUAD_JSON="data/heysquad_accents/heysquad_accents_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data_acoustic_heysquad_accents/embedding_cache"

# Default values (percentage mode with 50%)
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --heysquad_json)
            HEYSQUAD_JSON="$2"
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
            echo "Filter HeySQuAD Accents by TRUE ACOUSTIC embedding similarity to harmful audio."
            echo "Uses Whisper-Large-V3 encoder ONLY (without AF3's projector)."
            echo "Results saved to data_acoustic_heysquad_accents/ folder."
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE    Filter by distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE   Filter by percentage of dataset (e.g., 10 for 10%)"
            echo "  --num_samples VALUE  Filter by exact number of samples"
            echo ""
            echo "Other options:"
            echo "  --heysquad_json PATH      HeySQuAD Accents JSON file"
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
    OUTPUT_JSON="data_acoustic_heysquad_accents/heysquad_accents_acoustic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_acoustic_heysquad_accents/heysquad_accents_acoustic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_acoustic_heysquad_accents/heysquad_accents_acoustic_threshold_${THRESHOLD}.json"
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

# Create output directory
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$CACHE_DIR"

echo "============================================"
echo "HeySQuAD Accents TRUE ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:          AUDIO-ACOUSTIC (TRUE acoustic features)"
echo "Encoder:              Whisper-Large-V3 (original, no AF3 projector)"
echo "HeySQuAD Accents JSON: $HEYSQUAD_JSON"
echo "AdvBench audio:       $ADVBENCH_AUDIO_DIR"
echo "Mode:                 $MODE_DESC"
echo "Output:               $OUTPUT_JSON"
echo "Cache:                $CACHE_DIR"
echo "============================================"

# Run filtering (reuse heysquad filter script with heysquad_accents paths)
python 0_filter_heysquad.py \
    --heysquad_json "$HEYSQUAD_JSON" \
    --advbench_audio_dir "$ADVBENCH_AUDIO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_json "$OUTPUT_JSON" \
    $FILTER_ARGS \
    ${SELECT_SAFEST:+--select_safest}

echo ""
echo "============================================"
echo "TRUE ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "============================================"
echo ""
echo "Next step: Run 1_prepare_filtered_dataset.sh with --benign_dataset heysquad_accents --filter_type audio_acoustic"
