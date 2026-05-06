#!/bin/bash
# =============================================================================
# RANDOM K% SAMPLING (BASELINE FOR COMPARISON)
# =============================================================================
#
# Randomly select k% of VoiceBench samples WITHOUT considering any distance
# to harmful audio. This serves as a baseline to compare against filtering approaches.
#
# FOUR FILTERING APPROACHES:
#   - audio_acoustic: TRUE acoustic features (voice, prosody, timbre)
#   - audio_semantic: Audio content as understood by model
#   - text_semantic: Text meaning similarity
#   - random (THIS SCRIPT): No filtering - just random k% selection
#
# Output: data_random/ folder

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
PERCENTAGE="50"
NUM_SAMPLES=""
SEED="42"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --percentage)
            PERCENTAGE="$2"
            NUM_SAMPLES=""
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            PERCENTAGE=""
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --select_safest)
            # Ignored for random filtering (no distance to sort by)
            echo "Warning: --select_safest is ignored for random filtering"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Randomly sample k% of VoiceBench data (baseline for comparison)."
            echo "No distance-based filtering - purely random selection."
            echo "Results saved to data_random/ folder."
            echo ""
            echo "Options:"
            echo "  --percentage VALUE       Percentage of samples to keep (e.g., 25 for 25%)"
            echo "  --num_samples VALUE      Exact number of samples to keep"
            echo "  --seed VALUE             Random seed for reproducibility (default: 42)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 25                    # Random 25%"
            echo "  $0 --percentage 50 --seed 123         # Random 50% with different seed"
            echo "  $0 --num_samples 1000                 # Exactly 1000 random samples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine output file name based on filtering mode
if [ -n "$NUM_SAMPLES" ]; then
    OUTPUT_JSONL="data_random/voicebench_filtered_random_n${NUM_SAMPLES}.jsonl"
    MODE_DESC="num_samples=${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$PERCENTAGE" ]; then
    OUTPUT_JSONL="data_random/voicebench_filtered_random_percentage_${PERCENTAGE}.jsonl"
    MODE_DESC="percentage=${PERCENTAGE}%"
    FILTER_ARGS="--percentage $PERCENTAGE"
else
    echo "Error: Must specify --percentage or --num_samples"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_JSONL")"

echo "============================================"
echo "VoiceBench RANDOM Sampling (Baseline)"
echo "============================================"
echo "Filter type:       RANDOM (no distance filtering)"
echo "Mode:              $MODE_DESC"
echo "Random seed:       $SEED"
echo "Output:            $OUTPUT_JSONL"
echo "============================================"

# Run random filtering
python 0_filter_random.py \
    --benign_dataset voicebench \
    --output "$OUTPUT_JSONL" \
    --seed "$SEED" \
    $FILTER_ARGS

echo ""
echo "============================================"
echo "RANDOM sampling complete!"
echo "Output saved to: $OUTPUT_JSONL"
echo "============================================"
echo ""
echo "Next step: Run 1_extract_semantic_tokens.sh with --filter_type random"
echo "  bash 1_extract_semantic_tokens.sh --filter_type random --benign_dataset voicebench --percentage $PERCENTAGE"
