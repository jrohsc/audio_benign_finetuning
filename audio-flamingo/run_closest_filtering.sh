#!/bin/bash
#
# Run the full pipeline to finetune Audio Flamingo 3 on benign data CLOSEST to harmful data.
#
# Usage:
#   ./run_closest_filtering.sh --top_k 1000        # Keep 1000 closest samples (default)
#   ./run_closest_filtering.sh --threshold 0.025   # Keep samples with distance <= 0.025
#

set -e

# Load required modules
module load cuda/11.8
module load ffmpeg/7.0.2

cd "$(dirname "$0")"

# Default values
THRESHOLD="0.02"
TOP_K=""
SKIP_FILTER="false"
SKIP_PREPARE="false"
SKIP_FINETUNE="false"

# Model path
MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD="$2"
            TOP_K=""  # Clear top_k when threshold is specified
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            THRESHOLD=""  # Clear threshold when top_k is specified
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --skip_filter)
            SKIP_FILTER="true"
            shift
            ;;
        --skip_prepare)
            SKIP_PREPARE="true"
            shift
            ;;
        --skip_finetune)
            SKIP_FINETUNE="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "This script finetunes Audio Flamingo 3 on benign data CLOSEST to harmful data."
            echo ""
            echo "Options:"
            echo "  --top_k VALUE        Keep top-k closest samples (default: 1000)"
            echo "  --threshold VALUE    Distance threshold for filtering"
            echo "                       Keeps samples with distance <= threshold"
            echo "                       Suggested values:"
            echo "                         0.020 -> ~175 samples (2.9%)"
            echo "                         0.025 -> ~1205 samples (19.8%)"
            echo "                         0.030 -> ~2595 samples (42.7%)"
            echo "                         0.035 -> ~3822 samples (62.8%)"
            echo "                         0.040 -> ~4890 samples (80.4%)"
            echo "  --model_path PATH    Path to Audio Flamingo 3 model"
            echo "  --skip_filter        Skip filtering step (use cached analysis)"
            echo "  --skip_prepare       Skip dataset preparation step"
            echo "  --skip_finetune      Skip finetuning step (only filter and prepare)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set output paths based on threshold or top_k
if [ -n "$TOP_K" ]; then
    FILTER_SUFFIX="top${TOP_K}"
else
    FILTER_SUFFIX="thresh_${THRESHOLD}"
fi

OUTPUT_DIR="data/filtered_voicebench_closest"
CHECKPOINT_DIR="checkpoints/af3_finetuned_closest_${FILTER_SUFFIX}"

echo "============================================================"
echo "Audio Flamingo 3 Finetuning Pipeline (CLOSEST to harmful)"
echo "============================================================"
echo ""
echo "Configuration:"
if [ -n "$TOP_K" ]; then
    echo "  Mode:       top_k = $TOP_K (closest samples)"
else
    echo "  Mode:       threshold <= $THRESHOLD (closest samples)"
fi
echo "  Model:      $MODEL_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo ""

# Step 1: Filter VoiceBench samples (closest to AdvBench)
if [ "$SKIP_FILTER" != "true" ]; then
    echo "============================================================"
    echo "Step 1: Filtering VoiceBench samples (CLOSEST to AdvBench)"
    echo "============================================================"

    FILTER_CMD="python 0_filter_closest_to_advbench.py"
    FILTER_CMD="$FILTER_CMD --model_path $MODEL_PATH"
    FILTER_CMD="$FILTER_CMD --voicebench_json data/voicebench/sd-qa/sd_qa_full.json"
    FILTER_CMD="$FILTER_CMD --advbench_audio_dir ../harmful_data/advbench_gtts/en"
    FILTER_CMD="$FILTER_CMD --output_json data/voicebench/sd-qa/sd_qa_closest_to_advbench.json"
    FILTER_CMD="$FILTER_CMD --cache_dir data/embedding_cache"

    if [ -n "$TOP_K" ]; then
        FILTER_CMD="$FILTER_CMD --top_k $TOP_K"
    else
        FILTER_CMD="$FILTER_CMD --threshold $THRESHOLD"
    fi

    echo "Running: $FILTER_CMD"
    eval $FILTER_CMD
    echo ""
else
    echo "Step 1: Skipped (using cached analysis)"
fi

# Step 2: Prepare dataset in HuggingFace format
if [ "$SKIP_PREPARE" != "true" ]; then
    echo "============================================================"
    echo "Step 2: Preparing dataset for Audio Flamingo 3 finetuning"
    echo "============================================================"

    PREPARE_CMD="python 1_prepare_filtered_dataset.py"
    PREPARE_CMD="$PREPARE_CMD --analysis_file data/voicebench/sd-qa/sd_qa_closest_to_advbench_analysis.npz"
    PREPARE_CMD="$PREPARE_CMD --voicebench_json data/voicebench/sd-qa/sd_qa_full.json"
    PREPARE_CMD="$PREPARE_CMD --output_dir $OUTPUT_DIR"
    # Default is already closest filtering, no flag needed

    if [ -n "$TOP_K" ]; then
        PREPARE_CMD="$PREPARE_CMD --top_k $TOP_K"
    else
        PREPARE_CMD="$PREPARE_CMD --threshold $THRESHOLD"
    fi

    echo "Running: $PREPARE_CMD"
    eval $PREPARE_CMD
    echo ""
else
    echo "Step 2: Skipped"
fi

# Step 3: Finetune Audio Flamingo 3
if [ "$SKIP_FINETUNE" != "true" ]; then
    echo "============================================================"
    echo "Step 3: Finetuning Audio Flamingo 3 on closest-to-harmful data"
    echo "============================================================"

    # Determine the data file path
    if [ -n "$TOP_K" ]; then
        DATA_FILE="$OUTPUT_DIR/voicebench_filtered_thresh_top${TOP_K}_hf.json"
    else
        DATA_FILE="$OUTPUT_DIR/voicebench_filtered_thresh_${THRESHOLD}_hf.json"
    fi

    FINETUNE_CMD="python 2_finetune_audio_flamingo.py"
    FINETUNE_CMD="$FINETUNE_CMD --dataset_json $DATA_FILE"
    FINETUNE_CMD="$FINETUNE_CMD --output_dir $CHECKPOINT_DIR"
    FINETUNE_CMD="$FINETUNE_CMD --local_model_path $MODEL_PATH"
    FINETUNE_CMD="$FINETUNE_CMD --num_epochs 3"
    FINETUNE_CMD="$FINETUNE_CMD --batch_size 1"
    FINETUNE_CMD="$FINETUNE_CMD --gradient_accumulation_steps 8"
    FINETUNE_CMD="$FINETUNE_CMD --learning_rate 2e-5"
    FINETUNE_CMD="$FINETUNE_CMD --freeze_audio_encoder"

    echo "Running: $FINETUNE_CMD"
    eval $FINETUNE_CMD
else
    echo "Step 3: Skipped"
    echo ""
    echo "To run finetuning manually:"
    if [ -n "$TOP_K" ]; then
        echo "  python 2_finetune_audio_flamingo.py \\"
        echo "    --dataset_json $OUTPUT_DIR/voicebench_filtered_thresh_top${TOP_K}_hf.json \\"
        echo "    --output_dir $CHECKPOINT_DIR \\"
        echo "    --local_model_path $MODEL_PATH \\"
        echo "    --freeze_audio_encoder"
    else
        echo "  python 2_finetune_audio_flamingo.py \\"
        echo "    --dataset_json $OUTPUT_DIR/voicebench_filtered_thresh_${THRESHOLD}_hf.json \\"
        echo "    --output_dir $CHECKPOINT_DIR \\"
        echo "    --local_model_path $MODEL_PATH \\"
        echo "    --freeze_audio_encoder"
    fi
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
