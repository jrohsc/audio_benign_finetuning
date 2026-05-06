#!/bin/bash
#
# Finetune Audio Flamingo 3 on filtered VoiceBench dataset.
#

set -e

# Default values
THRESHOLD=""
TOP_K=""
PERCENTAGE="5"
NUM_SAMPLES=""

MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-5
FREEZE_AUDIO_ENCODER="--freeze_audio_encoder"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--data)
            DATASET_JSON="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --freeze_audio_encoder)
            FREEZE_AUDIO_ENCODER="--freeze_audio_encoder"
            shift
            ;;
        --no_freeze_audio_encoder)
            FREEZE_AUDIO_ENCODER=""
            shift
            ;;
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
        --threshold)
            THRESHOLD="$2"
            PERCENTAGE=""
            NUM_SAMPLES=""
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --model PATH                    Audio Flamingo model path"
            echo "  -d, --data PATH                     Training data path (JSON format)"
            echo "  -o, --output_dir PATH               Output directory for checkpoints"
            echo "  --num_epochs N                      Number of epochs (default: 1)"
            echo "  --batch_size N                      Batch size per device (default: 1)"
            echo "  --gradient_accumulation_steps N     Gradient accumulation steps (default: 8)"
            echo "  --learning_rate LR                  Learning rate (default: 2e-5)"
            echo "  --freeze_audio_encoder              Freeze audio encoder (default)"
            echo "  --no_freeze_audio_encoder           Don't freeze audio encoder"
            echo "  --percentage VALUE                  Percentage of dataset for auto-selecting data/output paths"
            echo "  --num_samples VALUE                 Number of samples for auto-selecting data/output paths"
            echo "  --threshold VALUE                   Distance threshold for auto-selecting data/output paths"
            echo "  -h, --help                          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-select data and output paths based on filtering mode
if [ -z "$DATASET_JSON" ]; then
    if [ -n "$NUM_SAMPLES" ]; then
        DATASET_JSON="data/filtered_voicebench/voicebench_filtered_closest_n${NUM_SAMPLES}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/af3_finetuned_n${NUM_SAMPLES}}"
    elif [ -n "$PERCENTAGE" ]; then
        DATASET_JSON="data/filtered_voicebench/voicebench_filtered_closest_percentage_${PERCENTAGE}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/af3_finetuned_percentage_${PERCENTAGE}}"
    elif [ -n "$THRESHOLD" ]; then
        DATASET_JSON="data/filtered_voicebench/voicebench_filtered_closest_${THRESHOLD}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/af3_finetuned_thresh_${THRESHOLD}}"
    else
        echo "Error: Must specify --data, --num_samples, --percentage, or --threshold"
        exit 1
    fi
fi

# Set default output dir if not specified
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/af3_finetuned}"

echo "============================================"
echo "Audio Flamingo 3 Finetuning Configuration"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_JSON"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Learning rate: $LEARNING_RATE"
echo "Freeze audio encoder: ${FREEZE_AUDIO_ENCODER:-no}"
echo "============================================"

# Build command
CMD="python 2_finetune_audio_flamingo.py"
CMD="$CMD --dataset_json \"$DATASET_JSON\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --local_model_path \"$MODEL_PATH\""
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
CMD="$CMD --learning_rate $LEARNING_RATE"

if [ -n "$FREEZE_AUDIO_ENCODER" ]; then
    CMD="$CMD $FREEZE_AUDIO_ENCODER"
fi

echo "Running: $CMD"
cd "$(dirname "$0")"
eval $CMD
