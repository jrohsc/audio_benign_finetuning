#!/bin/bash
#
# Finetune Qwen2.5-Omni on filtered benign dataset using LlamaFactory.
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset
#   - spoken_squad:          Spoken-SQuAD dataset
#   - librispeech:           LibriSpeech dataset
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#   - heysquad_accents:      HeySQuAD with 11 accent variations (TTS)
#   - gammacorpus_accents:   GammaCorpus-Fact-QA with 11 accent variations (TTS)
#
# FOUR FILTER TYPES SUPPORTED:
#   --filter_type random          -> data_random[_dataset]/filtered/
#   --filter_type audio_acoustic  -> data_acoustic[_dataset]/filtered/
#   --filter_type audio_semantic  -> data[_dataset]/filtered/
#   --filter_type sbert_audio_semantic   -> data_semantic[_dataset]/filtered/

set -e

# Load CUDA module and set library paths for PyTorch
# PyTorch 2.6.0+cu124 bundles its own CUDA libraries via pip nvidia-* packages
module load cuda/12.4.1 2>/dev/null || true
CUDA_BASE="/modules/spack/packages/linux-ubuntu24.04-x86_64_v3/gcc-13.2.0/cuda-12.4.1-r5m2xtnxqs7cmmqvdozajqynuvriqdkb"
export LD_LIBRARY_PATH="$CUDA_BASE/lib64:$CUDA_BASE/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="$CUDA_BASE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_FACTORY_DIR="$SCRIPT_DIR/../LlamaFactory"
CONDA_PATH="/work/anon/miniconda3"

# Default values
MODEL_SIZE="7b"  # "3b" or "7b"
BENIGN_DATASET="voicebench"
FILTER_TYPE="audio_acoustic"
PERCENTAGE="50"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=3
LEARNING_RATE="1e-4"
LORA_RANK=8
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
SELECT_SAFEST=""
RESUME_FROM_CHECKPOINT=""
CUSTOM_OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --benign_dataset)
            BENIGN_DATASET="$2"
            shift 2
            ;;
        --filter_type)
            FILTER_TYPE="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            NUM_SAMPLES=""
            THRESHOLD=""
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            PERCENTAGE=""
            THRESHOLD=""
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            PERCENTAGE=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --select_safest)
            SELECT_SAFEST="true"
            shift
            ;;
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Finetune Qwen2.5-Omni on filtered benign dataset."
            echo ""
            echo "Model options:"
            echo "  --model SIZE            Model size: 3b or 7b (default: 7b)"
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    random, audio_acoustic, audio_semantic, sbert_audio_semantic"
            echo ""
            echo "Training options:"
            echo "  --num_epochs N          Number of epochs (default: 3)"
            echo "  --learning_rate LR      Learning rate (default: 1e-4)"
            echo "  --lora_rank R           LoRA rank (default: 8)"
            echo "  --batch_size N          Batch size per device (default: 1)"
            echo "  --gradient_accumulation N  Gradient accumulation steps (default: 8)"
            echo ""
            echo "Filtering parameters (mutually exclusive):"
            echo "  --percentage VALUE      Percentage of dataset"
            echo "  --num_samples VALUE     Number of samples"
            echo "  --threshold VALUE       Distance threshold"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set model path based on size
if [ "$MODEL_SIZE" = "7b" ]; then
    MODEL_PATH="/project/anon/Qwen2.5-Omni-7B"
    CONFIG_FILE="qwen25_omni_7b_lora_sft.yaml"
elif [ "$MODEL_SIZE" = "3b" ]; then
    MODEL_PATH="/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    CONFIG_FILE="qwen25_omni_3b_lora_sft.yaml"
else
    echo "Error: Invalid model size '$MODEL_SIZE'. Use '3b' or '7b'."
    exit 1
fi

# Determine dataset suffix and paths
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
        CHECKPOINT_DIR="checkpoints_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        CHECKPOINT_DIR="checkpoints"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "sbert_audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_sbert_audio_semantic"
        CHECKPOINT_DIR="checkpoints_sbert_audio_semantic"
    else
        DATA_DIR="data_sbert_audio_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_sbert_audio_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
    # Text-only: reuses sbert_audio_semantic data dir, separate checkpoints
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_sbert_audio_semantic"
        CHECKPOINT_DIR="checkpoints_sbert_text_semantic"
    else
        DATA_DIR="data_sbert_audio_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_sbert_text_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        CHECKPOINT_DIR="checkpoints_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_random_${BENIGN_DATASET}"
    fi
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'."
    exit 1
fi

# Determine selection tag and safest suffix
if [ -n "$SELECT_SAFEST" ]; then
    SELECTION_TAG="safest"
    SAFEST_SUFFIX="_safest"
else
    SELECTION_TAG="closest"
    SAFEST_SUFFIX=""
fi

# Determine llamafactory suffix (text_only for sbert_text_semantic)
if [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
    LF_SUFFIX="_text_only_llamafactory.json"
else
    LF_SUFFIX="_llamafactory.json"
fi

# Determine dataset file and output directory
if [ -n "$NUM_SAMPLES" ]; then
    DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_n${NUM_SAMPLES}${SAFEST_SUFFIX}${LF_SUFFIX}"
    OUTPUT_DIR="${CHECKPOINT_DIR}/qwen_omni_${MODEL_SIZE}_lora_${BENIGN_DATASET}_${FILTER_TYPE}_n${NUM_SAMPLES}${SAFEST_SUFFIX}_epoch_${NUM_EPOCHS}"
elif [ -n "$PERCENTAGE" ]; then
    DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_percentage_${PERCENTAGE}${SAFEST_SUFFIX}${LF_SUFFIX}"
    OUTPUT_DIR="${CHECKPOINT_DIR}/qwen_omni_${MODEL_SIZE}_lora_${BENIGN_DATASET}_${FILTER_TYPE}_percentage_${PERCENTAGE}${SAFEST_SUFFIX}_epoch_${NUM_EPOCHS}"
elif [ -n "$THRESHOLD" ]; then
    DATASET_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_${THRESHOLD}${SAFEST_SUFFIX}${LF_SUFFIX}"
    OUTPUT_DIR="${CHECKPOINT_DIR}/qwen_omni_${MODEL_SIZE}_lora_${BENIGN_DATASET}_${FILTER_TYPE}_thresh_${THRESHOLD}${SAFEST_SUFFIX}_epoch_${NUM_EPOCHS}"
else
    echo "Error: Must specify --num_samples, --percentage, or --threshold"
    exit 1
fi

# Override output directory if explicitly provided (e.g. by run_full_pipeline.sh)
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    FINAL_OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
else
    FINAL_OUTPUT_DIR="$SCRIPT_DIR/../$OUTPUT_DIR"
fi

echo "============================================"
echo "Qwen2.5-Omni LoRA Finetuning Configuration"
echo "============================================"
echo "Model size:           $MODEL_SIZE"
echo "Model path:           $MODEL_PATH"
echo "Benign dataset:       $BENIGN_DATASET"
echo "Filter type:          $FILTER_TYPE"
echo "Dataset:              $DATASET_JSON"
echo "Output:               $OUTPUT_DIR"
echo "Epochs:               $NUM_EPOCHS"
echo "Learning rate:        $LEARNING_RATE"
echo "LoRA rank:            $LORA_RANK"
echo "Batch size:           $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "============================================"

# Initialize conda
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate qwen-omni

# Change to LlamaFactory directory
cd "$LLAMA_FACTORY_DIR"

# Create dataset info entry if needed
DATASET_INFO_FILE="$LLAMA_FACTORY_DIR/data/dataset_info.json"
DATASET_NAME="benign_audio_dataset"

# Build training command
TRAIN_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --trust_remote_code true
    --stage sft
    --do_train true
    --finetuning_type lora
    --lora_rank "$LORA_RANK"
    --lora_alpha $((LORA_RANK * 2))
    --lora_target all
    --lora_dropout 0.05
    --dataset_dir "$SCRIPT_DIR/../"
    --dataset "$DATASET_NAME"
    --template qwen2_omni
    --cutoff_len 3072
    --preprocessing_num_workers 8
    --output_dir "$FINAL_OUTPUT_DIR"
    --logging_steps 10
    --save_steps 500
    --save_total_limit 3
    --plot_loss true
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$NUM_EPOCHS"
    --lr_scheduler_type cosine
    --warmup_ratio 0.1
    --bf16 true
    --ddp_timeout 180000000
    --gradient_checkpointing true
    --freeze_vision_tower true
    --report_to none
)

# Handle resume vs fresh training
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resuming training from checkpoint: $RESUME_FROM_CHECKPOINT"
    TRAIN_ARGS+=(--overwrite_output_dir false)
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
else
    TRAIN_ARGS+=(--overwrite_output_dir true)
fi

# Run training using llamafactory-cli
llamafactory-cli train "${TRAIN_ARGS[@]}"

echo ""
echo "Training complete. Model saved to $OUTPUT_DIR"
