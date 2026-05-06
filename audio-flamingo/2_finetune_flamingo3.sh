#!/bin/bash
#
# Finetune Audio Flamingo 3 on filtered benign dataset.
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset
#   - spoken_squad:          Spoken-SQuAD dataset
#   - librispeech:           LibriSpeech dataset
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#   - heysquad_accents:      HeySQuAD with 11 accent variations (TTS)
#   - gammacorpus_accents:   GammaCorpus-Fact-QA with 11 accent variations (TTS)
#
# THREE FILTER TYPES SUPPORTED:
#   --filter_type audio_acoustic  -> data_acoustic[_dataset]/filtered/ -> checkpoints_acoustic[_dataset]/
#   --filter_type audio_semantic  -> data[_dataset]/filtered/ -> checkpoints[_dataset]/
#   --filter_type text_semantic   -> data_semantic[_dataset]/filtered/ -> checkpoints_semantic[_dataset]/

set -e

# Default values
BENIGN_DATASET="voicebench"  # "voicebench", "spoken_squad", or "librispeech"
FILTER_TYPE="audio_acoustic"  # "audio_acoustic", "audio_semantic", or "text_semantic"
THRESHOLD=""
TOP_K=""
PERCENTAGE="50"
NUM_SAMPLES=""

MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
MODEL_BASE_DIR="/project/anon/BFT_models"
SELECT_SAFEST=""  # Set to "true" to use safest (furthest from harmful) samples
NUM_EPOCHS=10
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-5
FREEZE_AUDIO_ENCODER="--freeze_audio_encoder"
USE_LORA=""
USE_AF_THINK=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benign_dataset)
            BENIGN_DATASET="$2"
            shift 2
            ;;
        --filter_type)
            FILTER_TYPE="$2"
            shift 2
            ;;
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
        --use_lora)
            USE_LORA="--use_lora"
            shift
            ;;
        --use_af_think)
            USE_AF_THINK="--use_af_think"
            shift
            ;;
        --model_base_dir)
            MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --select_safest)
            SELECT_SAFEST="true"
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
            echo "Finetune Audio Flamingo 3 on filtered benign dataset."
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo "    voicebench           VoiceBench SD-QA dataset (default)"
            echo "    spoken_squad         Spoken-SQuAD dataset"
            echo "    librispeech          LibriSpeech dataset"
            echo "    heysquad_accents     HeySQuAD with 11 accent variations"
            echo "    gammacorpus_accents  GammaCorpus-Fact-QA with 11 accent variations"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  -> data_acoustic[_dataset]/filtered/ -> checkpoints_acoustic[_dataset]/"
            echo "    audio_semantic  -> data[_dataset]/filtered/ -> checkpoints[_dataset]/"
            echo "    text_semantic   -> data_semantic[_dataset]/filtered/ -> checkpoints_semantic[_dataset]/"
            echo ""
            echo "Training options:"
            echo "  -m, --model PATH                    Audio Flamingo model path"
            echo "  -d, --data PATH                     Training data path (JSON format)"
            echo "  -o, --output_dir PATH               Output directory for checkpoints"
            echo "  --num_epochs N                      Number of epochs (default: 10)"
            echo "  --batch_size N                      Batch size per device (default: 1)"
            echo "  --gradient_accumulation_steps N     Gradient accumulation steps (default: 8)"
            echo "  --learning_rate LR                  Learning rate (default: 2e-5)"
            echo "  --freeze_audio_encoder              Freeze audio encoder (default)"
            echo "  --no_freeze_audio_encoder           Don't freeze audio encoder"
            echo ""
            echo "Filtering parameters:"
            echo "  --percentage VALUE      Percentage of dataset"
            echo "  --num_samples VALUE     Number of samples"
            echo "  --threshold VALUE       Distance threshold"
            echo ""
            echo "Examples:"
            echo "  $0 --benign_dataset voicebench --filter_type audio_acoustic --percentage 50"
            echo "  $0 --benign_dataset spoken_squad --filter_type audio_acoustic --percentage 50"
            echo "  $0 --benign_dataset librispeech --filter_type text_semantic --percentage 25"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine dataset suffix
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    DATASET_SUFFIX=""
    FILTERED_SUBDIR="filtered_voicebench"
    FILE_PREFIX="voicebench_filtered"
else
    DATASET_SUFFIX="_${BENIGN_DATASET}"
    FILTERED_SUBDIR="filtered"
    FILE_PREFIX="${BENIGN_DATASET}_filtered"
fi

# Set directories based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_audio_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_audio_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_audio_semantic"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_audio_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_text_semantic"
    else
        DATA_DIR="data_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_text_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        CHECKPOINT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_random_${BENIGN_DATASET}"
    fi
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'."
    echo "Valid options: audio_acoustic, audio_semantic, text_semantic, random"
    exit 1
fi

# Format learning rate for directory name (keep as-is, safe for directory names)
LR_SUFFIX="_lr${LEARNING_RATE}"

# Determine filter direction label for file naming
# When select_safest is true, prepare script uses "benign" instead of "closest"
if [ "$SELECT_SAFEST" = "true" ]; then
    FILTER_DIRECTION="benign"
    SAFEST_SUFFIX="_safest"
else
    FILTER_DIRECTION="closest"
    SAFEST_SUFFIX=""
fi

# Auto-select data and output paths based on filtering mode
if [ -z "$DATASET_JSON" ]; then
    if [ "$FILTER_TYPE" = "random" ]; then
        # Random filter uses pre-filtered naming convention
        DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_random_hf.json"
        if [ -n "$NUM_SAMPLES" ]; then
            OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
        elif [ -n "$PERCENTAGE" ]; then
            OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
        else
            OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
        fi
    elif [ -n "$NUM_SAMPLES" ]; then
        DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${FILTER_DIRECTION}_n${NUM_SAMPLES}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
    elif [ -n "$PERCENTAGE" ]; then
        DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${FILTER_DIRECTION}_percentage_${PERCENTAGE}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
    elif [ -n "$THRESHOLD" ]; then
        DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${FILTER_DIRECTION}_${THRESHOLD}_hf.json"
        OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_thresh_${THRESHOLD}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"
    else
        echo "Error: Must specify --data, --num_samples, --percentage, or --threshold"
        exit 1
    fi
fi

# Set default output dir if not specified
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/af3_finetuned_${BENIGN_DATASET}_${FILTER_TYPE}_epoch_${NUM_EPOCHS}${LR_SUFFIX}${SAFEST_SUFFIX}}"

# Append _think suffix if using AF-Think adapter
if [ -n "$USE_AF_THINK" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_think"
fi

echo "============================================"
echo "Audio Flamingo 3 Finetuning Configuration"
echo "============================================"
echo "Benign dataset:       $BENIGN_DATASET"
echo "Filter type:          $FILTER_TYPE"
echo "Model:                $MODEL_PATH"
echo "Dataset:              $DATASET_JSON"
echo "Output:               $OUTPUT_DIR"
echo "Epochs:               $NUM_EPOCHS"
echo "Batch size:           $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Learning rate:        $LEARNING_RATE"
echo "Freeze audio encoder: ${FREEZE_AUDIO_ENCODER:-no}"
echo "Select safest:        ${SELECT_SAFEST:-false}"
echo "============================================"

# Check for existing checkpoint to resume from
RESUME_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    # First, look for the latest checkpoint-epoch-* directory with training_state.pt
    LATEST_CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -n1)
    if [ -n "$LATEST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT/training_state.pt" ]; then
        echo ""
        echo "Found existing checkpoint: $LATEST_CHECKPOINT"
        RESUME_CHECKPOINT="$LATEST_CHECKPOINT"
    # Fallback: check best_model for training_state.pt
    elif [ -f "$OUTPUT_DIR/best_model/training_state.pt" ]; then
        echo ""
        echo "Found existing checkpoint in best_model"
        RESUME_CHECKPOINT="$OUTPUT_DIR/best_model"
    fi
fi

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

if [ -n "$USE_LORA" ]; then
    CMD="$CMD $USE_LORA"
fi

if [ -n "$USE_AF_THINK" ]; then
    CMD="$CMD $USE_AF_THINK"
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

# Clean up intermediate checkpoints, keep only best_model
if [ -d "$OUTPUT_DIR/best_model" ]; then
    echo "Cleaning up intermediate checkpoints..."
    for ckpt in "$OUTPUT_DIR"/checkpoint-epoch-*; do
        if [ -d "$ckpt" ]; then
            echo "  Removing $ckpt"
            rm -rf "$ckpt"
        fi
    done
    echo "Cleanup complete. Only best_model retained."
else
    echo "Warning: best_model not found in $OUTPUT_DIR, skipping checkpoint cleanup."
fi
