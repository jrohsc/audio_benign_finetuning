#!/bin/bash
#
# Finetune Audio Flamingo 3 on TEXT-ONLY data (Text BFT).
#
# This uses the SAME Audio Flamingo 3 model as audio finetuning,
# but trains on TEXT-ONLY conversations (no audio).
#
# Output goes to checkpoints_text_bft/

set -e

# Default values
PERCENTAGE="50"
NUM_SAMPLES=""

# Default model: Audio Flamingo 3
MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-5
MAX_LENGTH=2048
USE_LORA="--use_lora"
LORA_R=16
LORA_ALPHA=32

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
        --use_lora)
            USE_LORA="--use_lora"
            shift
            ;;
        --no_lora)
            USE_LORA="--no_lora"
            shift
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
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
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Finetune Audio Flamingo 3 on TEXT-ONLY data (Text BFT)."
            echo "Uses the same AF3 model as audio finetuning, but with text-only conversations."
            echo ""
            echo "Options:"
            echo "  -m, --model PATH                    Audio Flamingo 3 model path"
            echo "  -d, --data PATH                     Training data path (JSON format)"
            echo "  -o, --output_dir PATH               Output directory for checkpoints"
            echo "  --num_epochs N                      Number of epochs (default: 3)"
            echo "  --batch_size N                      Batch size per device (default: 1)"
            echo "  --gradient_accumulation_steps N     Gradient accumulation steps (default: 8)"
            echo "  --learning_rate LR                  Learning rate (default: 2e-5)"
            echo "  --use_lora                          Use LoRA for efficient finetuning (default)"
            echo "  --no_lora                           Don't use LoRA (full finetuning)"
            echo "  --lora_r N                          LoRA rank (default: 16)"
            echo "  --lora_alpha N                      LoRA alpha (default: 32)"
            echo "  --percentage VALUE                  Percentage of dataset for auto-selecting data/output paths"
            echo "  --num_samples VALUE                 Number of samples for auto-selecting data/output paths"
            echo "  --max_samples N                     Max samples for debugging"
            echo "  -h, --help                          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --percentage 50"
            echo "  $0 --num_samples 500 --num_epochs 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Data comes from data_semantic/ (semantic text filtering)
DATA_DIR="data_semantic"

# Output always goes to checkpoints_text_bft/
CHECKPOINT_DIR="checkpoints_text_bft"

# Auto-select data and output paths based on filtering mode
# Uses *_text.json files (text-only format)
if [ -z "$DATASET_JSON" ]; then
    if [ -n "$NUM_SAMPLES" ]; then
        DATASET_JSON="${DATA_DIR}/filtered_voicebench/voicebench_filtered_closest_n${NUM_SAMPLES}_text.json"
        OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_text_finetuned_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}}"
    elif [ -n "$PERCENTAGE" ]; then
        DATASET_JSON="${DATA_DIR}/filtered_voicebench/voicebench_filtered_closest_percentage_${PERCENTAGE}_text.json"
        OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_text_finetuned_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}}"
    else
        echo "Error: Must specify --data, --num_samples, or --percentage"
        exit 1
    fi
fi

# Set default output dir if not specified
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_text_finetuned_epoch_${NUM_EPOCHS}}"

echo "============================================"
echo "Audio Flamingo 3 TEXT-ONLY Finetuning"
echo "============================================"
echo "Model:                Audio Flamingo 3"
echo "Modality:             TEXT ONLY (no audio)"
echo "Model path:           $MODEL_PATH"
echo "Dataset:              $DATASET_JSON"
echo "Output:               $OUTPUT_DIR"
echo "Epochs:               $NUM_EPOCHS"
echo "Batch size:           $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Learning rate:        $LEARNING_RATE"
echo "LoRA:                 ${USE_LORA}"
echo "============================================"

# Check if dataset file exists
if [ ! -f "$DATASET_JSON" ]; then
    echo ""
    echo "Error: Dataset file not found: $DATASET_JSON"
    echo ""
    echo "Please run the dataset preparation step first:"
    echo "  bash 1_prepare_filtered_dataset.sh --filter_type semantic --percentage $PERCENTAGE"
    echo ""
    echo "This will generate both audio and text format datasets."
    exit 1
fi

# Check for existing checkpoint to resume from
RESUME_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -n1)
    if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT/training_state.pt" ]; then
        echo ""
        echo "Found existing checkpoint: $LATEST_CHECKPOINT"
        RESUME_CHECKPOINT="$LATEST_CHECKPOINT"
    elif [ -f "$OUTPUT_DIR/best_model/training_state.pt" ]; then
        echo ""
        echo "Found existing checkpoint in best_model"
        RESUME_CHECKPOINT="$OUTPUT_DIR/best_model"
    fi
fi

# Build command
CMD="python 2_finetune_af3_text.py"
CMD="$CMD --dataset_json \"$DATASET_JSON\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --local_model_path \"$MODEL_PATH\""
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD $USE_LORA"
CMD="$CMD --lora_r $LORA_R"
CMD="$CMD --lora_alpha $LORA_ALPHA"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint \"$RESUME_CHECKPOINT\""
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Running: $CMD"
cd "$(dirname "$0")"
eval $CMD

echo ""
echo "Training complete. Best model saved to $OUTPUT_DIR/best_model"
