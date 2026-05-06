#!/bin/bash
# Step 0 (Audio-Acoustic): Filter VoiceBench Noisy samples by TRUE ACOUSTIC embedding distance
#
# This script filters VoiceBench Noisy samples based on TRUE AUDIO ACOUSTIC characteristics.
# Uses original Whisper-Large-V3 encoder ONLY (without AF3's multi-modal projector).
#
# VoiceBench Noisy contains the original VoiceBench dataset augmented with realistic
# background noise (cafe, car, traffic, office, etc.) at configurable SNR levels.

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

VOICEBENCH_JSON="data/voicebench_noisy/voicebench_noisy_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data_acoustic_voicebench_noisy/embedding_cache"

# Default values (percentage mode with 50%)
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
SELECT_SAFEST=""

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
            echo "Filter VoiceBench Noisy by TRUE ACOUSTIC embedding similarity to harmful audio."
            echo "Uses Whisper-Large-V3 encoder ONLY (without AF3's projector)."
            echo "Results saved to data_acoustic_voicebench_noisy/ folder."
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE    Filter by distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE   Filter by percentage of dataset (e.g., 10 for 10%)"
            echo "  --num_samples VALUE  Filter by exact number of samples"
            echo ""
            echo "Other options:"
            echo "  --voicebench_json PATH    VoiceBench Noisy JSON file"
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
    OUTPUT_JSON="data_acoustic_voicebench_noisy/filtered/voicebench_noisy_acoustic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_acoustic_voicebench_noisy/filtered/voicebench_noisy_acoustic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_acoustic_voicebench_noisy/filtered/voicebench_noisy_acoustic_threshold_${THRESHOLD}.json"
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
echo "VoiceBench Noisy TRUE ACOUSTIC Filtering"
echo "============================================"
echo "Filter type:       AUDIO-ACOUSTIC (TRUE acoustic features)"
echo "Encoder:           Whisper-Large-V3 (original, no AF3 projector)"
echo "VoiceBench JSON:   $VOICEBENCH_JSON"
echo "AdvBench audio:    $ADVBENCH_AUDIO_DIR"
echo "Mode:              $MODE_DESC"
echo "Output:            $OUTPUT_JSON"
echo "Cache:             $CACHE_DIR"
echo "============================================"

# Run filtering with --use_original_whisper flag (no AF3 encoder)
python 0_filter_closest_to_advbench.py \
    --voicebench_json "$VOICEBENCH_JSON" \
    --advbench_audio_dir "$ADVBENCH_AUDIO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_json "$OUTPUT_JSON" \
    --use_original_whisper \
    $FILTER_ARGS \
    ${SELECT_SAFEST:+--select_safest}

echo ""
echo "============================================"
echo "TRUE ACOUSTIC filtering complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "============================================"
echo ""
echo "Next step: Run 1_prepare_filtered_dataset.sh with --filter_type audio_acoustic --benign_dataset voicebench_noisy"
