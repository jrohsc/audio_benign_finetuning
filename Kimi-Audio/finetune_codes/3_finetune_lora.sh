#!/bin/bash

# LoRA Fine-tuning for Kimi-Audio on VoiceBench
# Single GPU training - no DeepSpeed required
#
# THREE FILTER TYPES SUPPORTED:
#   --filter_type audio_acoustic  -> data_acoustic/ -> output_acoustic/
#   --filter_type audio_semantic  -> data/ -> output/
#   --filter_type text_semantic   -> data_semantic/ -> output_semantic/

set -e

# Load CUDA
module load cuda/12.6

# Activate conda environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
FILTER_TYPE="audio_acoustic"  # "audio_acoustic", "audio_semantic", or "text_semantic"
BENIGN_DATASET="voicebench"   # "voicebench", "spoken_squad", or "librispeech"
HARMFUL_SOURCE="advbench"
PERCENTAGE="50"
THRESHOLD=""
NUM_SAMPLES=""

# Paths
# PRETRAINED_MODEL_PATH="moonshotai/Kimi-Audio-7B-Instruct"
PRETRAINED_MODEL_PATH="/project/anon/BFT_models/pretrained_kimi_instruct"

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training parameters
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_SEQ_LENGTH=128
EVAL_RATIO=0.05

# Parse command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        --filter_type)
            shift
            FILTER_TYPE=$1
            ;;
        --benign_dataset)
            shift
            BENIGN_DATASET=$1
            ;;
        --harmful_source)
            shift
            HARMFUL_SOURCE=$1
            ;;
        --percentage)
            shift
            PERCENTAGE=$1
            ;;
        --threshold)
            shift
            THRESHOLD=$1
            ;;
        --num_samples)
            shift
            NUM_SAMPLES=$1
            ;;
        -m | --model_path)
            shift
            PRETRAINED_MODEL_PATH=$1
            ;;
        -r | --lora_r)
            shift
            LORA_R=$1
            ;;
        --lr)
            shift
            LEARNING_RATE=$1
            ;;
        --epochs)
            shift
            NUM_EPOCHS=$1
            ;;
        -h | --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "LoRA fine-tuning for Kimi-Audio on filtered benign data."
            echo ""
            echo "Options:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  -> data_acoustic/ -> output_acoustic/"
            echo "    audio_semantic  -> data/ -> output/"
            echo "    text_semantic   -> data_semantic/ -> output_semantic/"
            echo ""
            echo "  --benign_dataset NAME   Benign dataset: voicebench (default), spoken_squad, librispeech, heysquad, heysquad_accents, gammacorpus_accents, gammacorpus_usa, benign_instructions_usa, benign_instructions_accents, mmsu, bbh, voicebench_cafe, voicebench_traffic"
            echo "  --harmful_source NAME   Harmful source (advbench or safetybench)"
            echo "  --percentage VALUE      Percentage used in filtering"
            echo "  --threshold VALUE       Threshold used in filtering"
            echo "  --num_samples VALUE     Num samples used in filtering"
            echo "  -m, --model_path PATH   Pretrained model path"
            echo "  -r, --lora_r VALUE      LoRA rank (default: 16)"
            echo "  --lr VALUE              Learning rate (default: 2e-4)"
            echo "  --epochs VALUE          Number of epochs (default: 3)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --filter_type audio_acoustic --percentage 50"
            echo "  $0 --benign_dataset librispeech --filter_type audio_acoustic --percentage 50"
            echo "  $0 --filter_type text_semantic --percentage 10"
            exit 0
            ;;
        --select_safest)
            SELECT_SAFEST="true"
            ;;
        *)
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# Determine directories based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
        OUTPUT_BASE="output_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
        OUTPUT_BASE="output_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        OUTPUT_BASE="output"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
        OUTPUT_BASE="output_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        OUTPUT_BASE="output_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        OUTPUT_BASE="output_random_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
        OUTPUT_BASE="output_semantic"
    else
        DATA_DIR="data_semantic_${BENIGN_DATASET}"
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

# Set data and output paths based on benign dataset
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    # VoiceBench naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    elif [ "$FILTER_TYPE" = "random" ]; then
        DATA_PATH="${DATA_DIR}/voicebench_filtered_random_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
    # Spoken-SQuAD naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/spoken_squad_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/spoken_squad_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "librispeech" ]; then
    # LibriSpeech naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/librispeech_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/librispeech_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "heysquad" ]; then
    # HeySQuAD naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/heysquad_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/heysquad_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
    # HeySQuAD Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/heysquad_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/heysquad_accents_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
    # GammaCorpus Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/gammacorpus_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/gammacorpus_accents_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "gammacorpus_usa" ]; then
    # GammaCorpus USA naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/gammacorpus_usa_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/gammacorpus_usa_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "benign_instructions_usa" ]; then
    # Benign Instructions USA naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/benign_instructions_usa_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/benign_instructions_usa_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "benign_instructions_accents" ]; then
    # Benign Instructions Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/benign_instructions_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/benign_instructions_accents_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "mmsu" ]; then
    # MMSU (VoiceBench Multi-Subject Understanding) naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/mmsu_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/mmsu_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "bbh" ]; then
    # BBH (BIG-Bench Hard) naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/bbh_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/bbh_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "voicebench_cafe" ] || [ "$BENIGN_DATASET" = "voicebench_traffic" ]; then
    # VoiceBench Noisy variants naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        DATA_PATH="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        DATA_PATH="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
else
    echo "Error: Invalid benign_dataset '$BENIGN_DATASET'."
    echo "Valid options: voicebench, spoken_squad, librispeech, heysquad, heysquad_accents, gammacorpus_accents, gammacorpus_usa, benign_instructions_usa, benign_instructions_accents, mmsu, bbh, voicebench_cafe, voicebench_traffic"
    exit 1
fi
OUTPUT_DIR="${OUTPUT_BASE}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${FILE_SUFFIX}_epoch_${NUM_EPOCHS}"

# Validate inputs
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: DATA file does not exist: $DATA_PATH"
    echo ""
    echo "Please run the extraction script first:"
    echo "  bash 1_extract_semantic_tokens.sh --benign_dataset $BENIGN_DATASET --filter_type $FILTER_TYPE --percentage $PERCENTAGE"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist: $PRETRAINED_MODEL_PATH"
    echo ""
    echo "Please first prepare the pretrained model by running:"
    echo "  CUDA_VISIBLE_DEVICES=0 python -m model --model_name \"moonshotai/Kimi-Audio-7B-Instruct\" --output_dir \"output/pretrained_kimi_instruct\""
    exit 1
fi

echo "============================================"
echo "LoRA Fine-tuning Configuration"
echo "============================================"
echo "Benign dataset:  $BENIGN_DATASET"
echo "Filter type:     $FILTER_TYPE"
echo "Data directory:  $DATA_DIR"
echo "Model path:      $PRETRAINED_MODEL_PATH"
echo "Data path:       $DATA_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "LoRA rank:       $LORA_R"
echo "LoRA alpha:      $LORA_ALPHA"
echo "Learning rate:   $LEARNING_RATE"
echo "Epochs:          $NUM_EPOCHS"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python 3_finetune_lora.py \
    --model_name_or_path "moonshotai/Kimi-Audio-7B-Instruct" \
    --model_path "$PRETRAINED_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --eval_ratio $EVAL_RATIO \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --model_max_length $MAX_SEQ_LENGTH \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to "none" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT

echo ""
echo "============================================"
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"

# Clean up intermediate checkpoints to save space (keep only final adapter)
echo ""
echo "Cleaning up intermediate checkpoints..."
for checkpoint_dir in "$OUTPUT_DIR"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        echo "Removing $checkpoint_dir"
        rm -rf "$checkpoint_dir"
    fi
done
echo "Cleanup complete."

echo ""
echo "Next steps:"
echo "  1. Merge LoRA weights: bash 5_merge_lora_for_inference.sh --benign_dataset $BENIGN_DATASET --filter_type $FILTER_TYPE --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
echo "  2. Evaluate: bash 6_evaluate_harmful_audio.sh --benign_dataset $BENIGN_DATASET --filter_type $FILTER_TYPE --percentage $PERCENTAGE --num_epochs $NUM_EPOCHS"
