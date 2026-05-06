#!/bin/bash
# Step 0 (Audio-Acoustic): Filter VoiceBench samples by TRUE ACOUSTIC embedding distance
#
# This script filters VoiceBench samples based on TRUE AUDIO ACOUSTIC characteristics.
# Uses original Whisper-Large-V3 encoder ONLY (without AF3's multi-modal projector).
#
# THREE FILTER TYPES AVAILABLE:
#   - audio_acoustic (this script): TRUE acoustic features (voice, prosody, timbre)
#       Uses: Whisper-Large-V3 encoder only
#       Output: data_acoustic/
#
#   - audio_semantic (0_filter_voicebench_audio_semantic.sh): Audio content as understood by model
#       Uses: AF3's AudioFlamingo3Encoder + MultiModalProjector
#       Output: data/
#
#   - text_semantic (0_filter_voicebench_text_semantic.sh): Text meaning similarity
#       Uses: sentence-transformers (all-MiniLM-L6-v2)
#       Output: data_semantic/
#
# Supports three filtering modes:
#   --threshold VALUE   Filter by distance threshold
#   --percentage VALUE  Filter by percentage of dataset (e.g., 50 for 50%)
#   --num_samples VALUE Filter by exact number of samples

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

VOICEBENCH_JSON="data/voicebench/sd-qa/sd_qa_full.json"
ADVBENCH_AUDIO_DIR="../harmful_data/advbench_gtts/en"
CACHE_DIR="data_acoustic/embedding_cache"

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
            echo "Filter VoiceBench by TRUE ACOUSTIC embedding similarity to harmful audio."
            echo "Uses Whisper-Large-V3 encoder ONLY (without AF3's projector)."
            echo "Results saved to data_acoustic/ folder."
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE    Filter by distance threshold (keep samples <= threshold)"
            echo "  --percentage VALUE   Filter by percentage of dataset (e.g., 10 for 10%)"
            echo "  --num_samples VALUE  Filter by exact number of samples"
            echo ""
            echo "Other options:"
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
    OUTPUT_JSON="data_acoustic/voicebench/sd-qa/sd_qa_acoustic_n${NUM_SAMPLES}.json"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSON="data_acoustic/voicebench/sd-qa/sd_qa_acoustic_percentage_${PERCENTAGE}.json"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
elif [ -n "$THRESHOLD" ]; then
    OUTPUT_JSON="data_acoustic/voicebench/sd-qa/sd_qa_acoustic_threshold_${THRESHOLD}.json"
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
echo "VoiceBench TRUE ACOUSTIC Filtering"
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
echo "Next step: Run 1_prepare_filtered_dataset.sh with --filter_type audio_acoustic"
echo "  bash 1_prepare_filtered_dataset.sh --filter_type audio_acoustic --percentage $PERCENTAGE"
