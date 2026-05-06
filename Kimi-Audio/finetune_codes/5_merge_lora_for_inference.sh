#!/bin/bash
# Merge LoRA adapter with base model for inference
#
# THREE FILTER TYPES SUPPORTED:
#   --filter_type audio_acoustic  -> output_acoustic/
#   --filter_type audio_semantic  -> output/
#   --filter_type text_semantic   -> output_semantic/
#
# Usage:
#   ./5_merge_lora_for_inference.sh --filter_type audio_acoustic --percentage 50 --num_epochs 3

# Load CUDA
module load cuda/12.6

# Activate conda environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default settings
FILTER_TYPE="audio_acoustic"  # "audio_acoustic", "audio_semantic", or "text_semantic"
BENIGN_DATASET="voicebench"   # "voicebench", "spoken_squad", or "librispeech"
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""
NUM_EPOCHS=3

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --filter_type) FILTER_TYPE="$2"; shift ;;
        --benign_dataset) BENIGN_DATASET="$2"; shift ;;
        --percentage) PERCENTAGE="$2"; THRESHOLD=""; NUM_SAMPLES=""; shift ;;
        --threshold) THRESHOLD="$2"; PERCENTAGE=""; NUM_SAMPLES=""; shift ;;
        --num_samples) NUM_SAMPLES="$2"; PERCENTAGE=""; THRESHOLD=""; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --lora_path) LORA_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Merge LoRA adapter with base model for inference."
            echo ""
            echo "Options:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  -> output_acoustic/"
            echo "    audio_semantic  -> output/"
            echo "    text_semantic   -> output_semantic/"
            echo ""
            echo "  --benign_dataset NAME   Benign dataset: voicebench (default), spoken_squad, librispeech, heysquad, heysquad_accents, gammacorpus_accents, gammacorpus_usa, benign_instructions_usa, benign_instructions_accents, mmsu, bbh, voicebench_cafe, voicebench_traffic"
            echo "  --percentage VALUE      Percentage used in filtering (default: 50)"
            echo "  --threshold VALUE       Threshold used in filtering"
            echo "  --num_samples VALUE     Num samples used in filtering"
            echo "  --num_epochs N          Number of epochs used during training (default: 3)"
            echo "  --lora_path PATH        Override LoRA adapter path"
            echo "  --output_path PATH      Override merged output path"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --filter_type audio_acoustic --percentage 50 --num_epochs 3"
            echo "  $0 --benign_dataset librispeech --filter_type audio_acoustic --percentage 50 --num_epochs 3"
            echo "  $0 --filter_type text_semantic --percentage 25 --num_epochs 3"
            exit 0
            ;;
        --select_safest) SELECT_SAFEST="true" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Determine directories based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        OUTPUT_BASE="output_acoustic"
    else
        OUTPUT_BASE="output_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        OUTPUT_BASE="output"
    else
        OUTPUT_BASE="output_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        OUTPUT_BASE="output_random"
    else
        OUTPUT_BASE="output_random_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        OUTPUT_BASE="output_semantic"
    else
        OUTPUT_BASE="output_semantic_${BENIGN_DATASET}"
    fi
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'."
    echo "Valid options: audio_acoustic, audio_semantic, text_semantic, random"
    exit 1
fi

# Determine filename suffix
if [ -n "$NUM_SAMPLES" ]; then
    FILE_SUFFIX="n${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    FILE_SUFFIX="percentage_${PERCENTAGE}"
elif [ -n "$THRESHOLD" ]; then
    FILE_SUFFIX="thresh_${THRESHOLD}"
else
    FILE_SUFFIX="auto"
fi
if [ -n "$SELECT_SAFEST" ]; then
    FILE_SUFFIX="${FILE_SUFFIX}_safest"
fi

# Persistent storage for merged models
PROJECT_MODELS="/project/anon/BFT_models"

# Set default paths if not specified
LORA_PATH="${LORA_PATH:-${OUTPUT_BASE}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_MODELS}/Kimi-Audio_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}_merged}"

echo "============================================"
echo "LoRA Merge for Inference"
echo "============================================"
echo "Benign dataset: $BENIGN_DATASET"
echo "Filter type:    $FILTER_TYPE"
echo "LoRA adapter:   $LORA_PATH"
echo "Output path:    $OUTPUT_PATH"
echo "============================================"
echo ""

# Check if LoRA adapter exists
if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA adapter not found: $LORA_PATH"
    exit 1
fi

# Run merge
CUDA_VISIBLE_DEVICES=0 python 5_merge_lora_for_inference.py \
    --lora_path "$LORA_PATH" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "Done! Merged model saved to: $OUTPUT_PATH"
echo ""
echo "To evaluate:"
echo "  bash 6_evaluate_harmful_audio.sh --benign_dataset $BENIGN_DATASET --filter_type $FILTER_TYPE --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
