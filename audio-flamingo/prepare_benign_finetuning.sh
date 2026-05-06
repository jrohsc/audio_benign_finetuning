#!/bin/bash
# Complete pipeline for preparing benign finetuning data
# This script:
# 1. Downloads VoiceBench (sd-qa subset)
# 2. Filters benign samples using embedding distance from harmful samples
# 3. Prepares dataset for finetuning

set -e

echo "=== Benign Finetuning Data Preparation Pipeline ==="
echo ""

# Configuration
HARMFUL_AUDIO_DIR="advbench/en"
BENIGN_JSON="data/voicebench/sd-qa/sd_qa_full.json"
FILTERED_JSON="data/voicebench/sd-qa/sd_qa_filtered.json"

# Options for filtering
DISTANCE_METRIC=${DISTANCE_METRIC:-"cosine"}  # cosine or euclidean
FILTER_MODE=${FILTER_MODE:-"threshold"}       # threshold or top_k

# Threshold mode: keep samples with distance > threshold
DISTANCE_THRESHOLD=${DISTANCE_THRESHOLD:-"0.5"}

# Top-k mode: keep top K samples with largest distance
TOP_K=${TOP_K:-"1000"}

echo "Configuration:"
echo "  Harmful audio: $HARMFUL_AUDIO_DIR"
echo "  Distance metric: $DISTANCE_METRIC"
echo "  Filter mode: $FILTER_MODE"
if [ "$FILTER_MODE" = "threshold" ]; then
    echo "  Distance threshold: $DISTANCE_THRESHOLD"
else
    echo "  Top-K samples: $TOP_K"
fi
echo ""

# Step 1: Download VoiceBench dataset
echo "=== Step 1: Downloading VoiceBench (sd-qa) ==="
if [ -f "$BENIGN_JSON" ]; then
    echo "Dataset already exists at $BENIGN_JSON"
    echo "Skip download? (y/n)"
    read -r skip_download
    if [ "$skip_download" != "y" ]; then
        python download_voicebench.py
    fi
else
    python download_voicebench.py
fi

echo ""
echo "=== Step 2: Filtering benign samples ==="
if [ "$FILTER_MODE" = "threshold" ]; then
    python filter_by_embedding_distance.py \
        --benign_json "$BENIGN_JSON" \
        --harmful_audio_dir "$HARMFUL_AUDIO_DIR" \
        --output_json "$FILTERED_JSON" \
        --threshold "$DISTANCE_THRESHOLD" \
        --metric "$DISTANCE_METRIC"
else
    python filter_by_embedding_distance.py \
        --benign_json "$BENIGN_JSON" \
        --harmful_audio_dir "$HARMFUL_AUDIO_DIR" \
        --output_json "$FILTERED_JSON" \
        --top_k "$TOP_K" \
        --metric "$DISTANCE_METRIC"
fi

echo ""
echo "=== Step 3: Dataset Configuration ==="
echo "Update llava/data/datasets_mixture.py to include:"
echo ""
echo "    sd_qa_filtered = Dataset("
echo "        dataset_name=\"sd_qa_filtered\","
echo "        dataset_type=\"torch\","
echo "        data_path=\"$FILTERED_JSON\","
echo "    )"
echo "    add_dataset(sd_qa_filtered)"
echo ""

echo "=== Pipeline Complete! ==="
echo "Filtered dataset ready at: $FILTERED_JSON"
echo ""
echo "Next steps:"
echo "1. Update datasets_mixture.py with the above configuration"
echo "2. Run finetuning with: bash scripts/stage3_af3.sh <checkpoint_path> sd_qa_filtered"
