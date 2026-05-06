#!/bin/bash
#
# Full Pipeline: Filter -> Prepare -> Finetune -> Merge -> Evaluate -> ASR
#
# This script runs the complete benign finetuning pipeline for Qwen2.5-Omni:
#   0. Filter benign dataset samples by embedding distance
#   1. Prepare filtered dataset in LlamaFactory format
#   2. Finetune Qwen2.5-Omni LoRA using LlamaFactory
#   3. Merge LoRA weights with base model
#   4. Evaluate the finetuned model on harmful audio datasets
#   5. Run ASR evaluation
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset (speech QA)
#   - spoken_squad:          Spoken-SQuAD dataset (speech QA)
#   - librispeech:           LibriSpeech dataset (transcription task)
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#   - heysquad_accents:      HeySQuAD with 11 accent variations (TTS-generated)
#   - gammacorpus_accents:   GammaCorpus-Fact-QA with 11 accent variations (TTS)
#   - mmsu:                  VoiceBench MMSU (Multi-Subject Understanding) ~3k samples
#   - bbh:                   VoiceBench BBH (BIG-Bench Hard) ~1k samples
#
# FILTER TYPES SUPPORTED:
#   - audio_acoustic: ACOUSTIC features
#   - audio_semantic: SEMANTIC (Whisper-Large-V3 - Qwen2.5-Omni's internal encoder)
#   - sbert_audio_semantic:  SBERT AUDIO SEMANTIC (sentence-BERT text filtering, audio training)
#   - sbert_text_semantic:   SBERT TEXT SEMANTIC (sentence-BERT text filtering, TEXT-ONLY training)
#
# Usage:
#   ./run_full_pipeline.sh                                                    # Defaults (7b, voicebench, audio_semantic, 50%)
#   ./run_full_pipeline.sh --model 3b --benign_dataset voicebench --percentage 50
#   ./run_full_pipeline.sh --benign_dataset heysquad_accents --filter_type audio_acoustic --percentage 50
#   ./run_full_pipeline.sh --resume                                            # Auto-detect and resume from checkpoint
#   ./run_full_pipeline.sh --skip_filter --skip_prepare                       # Resume from finetuning (manual)
#   ./run_full_pipeline.sh --learning_rate 5e-5 --num_epochs 5               # Custom training config

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Project directories
PROJECT_DIR="$SCRIPT_DIR/.."
LLAMA_FACTORY_DIR="$PROJECT_DIR/LlamaFactory"
BENIGN_DATA_ROOT="$SCRIPT_DIR/../../benign_data"
HARMFUL_DATA_ROOT="$SCRIPT_DIR/../../harmful_data"
SHARED_AUDIO_ROOT="$SCRIPT_DIR/../../shared_audio"
CONDA_PATH="/work/anon/miniconda3"

# Default values
MODEL_SIZE="7b"
BENIGN_DATASET="mmsu"
FILTER_TYPE="audio_semantic" # audio_semantic, audio_acoustic, random
PERCENTAGE="25"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=3
LEARNING_RATE="1e-4"
LORA_RANK=8
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
DATASET="both"
EVAL_MODE="finetuned_only"
SELECT_SAFEST=""
SAVE_TO="project"  # "project" -> /project/.../BFT_models, "local" -> Qwen2.5-Omni/checkpoints

# Skip options
SKIP_FILTER=""
SKIP_PREPARE=""
SKIP_FINETUNE=""
SKIP_MERGE=""
SKIP_EVALUATE=""
SKIP_ASR=""
DRY_RUN=""
RESUME=""
RESUME_CHECKPOINT=""


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
        --learning_rate|--lr)
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
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --eval_mode)
            EVAL_MODE="$2"
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
        --skip_merge)
            SKIP_MERGE="true"
            shift
            ;;
        --skip_evaluate)
            SKIP_EVALUATE="true"
            shift
            ;;
        --skip_asr)
            SKIP_ASR="true"
            shift
            ;;
        --select_safest)
            SELECT_SAFEST="true"
            shift
            ;;
        --save_to)
            SAVE_TO="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run the full benign finetuning pipeline for Qwen2.5-Omni."
            echo ""
            echo "Pipeline steps:"
            echo "  0. Filter benign dataset samples by embedding distance"
            echo "  1. Prepare filtered dataset in LlamaFactory format"
            echo "  2. Finetune Qwen2.5-Omni LoRA"
            echo "  3. Merge LoRA weights with base model"
            echo "  4. Evaluate on harmful audio datasets"
            echo "  5. Run ASR evaluation"
            echo ""
            echo "Model options:"
            echo "  --model SIZE            Model size: 3b or 7b (default: 7b)"
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo "    voicebench           VoiceBench SD-QA dataset (default)"
            echo "    spoken_squad         Spoken-SQuAD dataset"
            echo "    librispeech          LibriSpeech dataset"
            echo "    heysquad             HeySQuAD dataset (human-spoken QA)"
            echo "    heysquad_accents     HeySQuAD with 11 accent variations (TTS)"
            echo "    gammacorpus_accents  GammaCorpus with 11 accent variations (TTS)"
            echo "    mmsu                 VoiceBench MMSU (~3k multi-subject QA)"
            echo "    bbh                  VoiceBench BBH (~1k reasoning)"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: random)"
            echo "    random          RANDOM sampling (baseline - no similarity filtering)"
            echo "    audio_acoustic  ACOUSTIC features"
            echo "    audio_semantic  SEMANTIC (Whisper-Large-V3 internal encoder)"
            echo "    sbert_audio_semantic   SBERT AUDIO SEMANTIC (sentence-BERT text filtering, audio training)"
            echo "    sbert_text_semantic    SBERT TEXT SEMANTIC (sentence-BERT text filtering, TEXT-ONLY training)"
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --percentage VALUE      Keep top percentage of closest samples (default: 50)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --threshold VALUE       Distance threshold"
            echo "  --select_safest         Select SAFEST samples (furthest from harmful) instead of closest"
            echo ""
            echo "Training options:"
            echo "  --num_epochs N          Number of training epochs (default: 3)"
            echo "  --learning_rate LR      Learning rate (default: 1e-4)"
            echo "  --lr LR                 Alias for --learning_rate"
            echo "  --lora_rank R           LoRA rank (default: 8)"
            echo "  --batch_size N          Batch size per device (default: 1)"
            echo "  --gradient_accumulation N  Gradient accumulation steps (default: 8)"
            echo ""
            echo "Evaluation options:"
            echo "  --dataset NAME          Evaluation dataset: 'advbench', 'safetybench', or 'both' (default: both)"
            echo "  --eval_mode MODE        Evaluation mode: 'finetuned_only', 'pretrained_only', or 'both' (default: finetuned_only)"
            echo ""
            echo "Skip options (to resume from a specific step):"
            echo "  --skip_filter           Skip step 0 (filtering)"
            echo "  --skip_prepare          Skip step 1 (dataset preparation)"
            echo "  --skip_finetune         Skip step 2 (finetuning)"
            echo "  --skip_merge            Skip step 3 (LoRA merging)"
            echo "  --skip_evaluate         Skip step 4 (evaluation)"
            echo "  --skip_asr              Skip step 5 (ASR evaluation)"
            echo ""
            echo "Resume options:"
            echo "  --resume                Auto-detect completed steps and resume from where pipeline left off"
            echo "                          Skips filter/prepare if outputs exist, resumes training from"
            echo "                          latest checkpoint, skips merge if merged model exists"
            echo ""
            echo "Other options:"
            echo "  --save_to LOCATION      Where to save merged model: 'project' or 'local' (default: project)"
            echo "                          project -> /project/anon/BFT_models/"
            echo "                          local   -> Qwen2.5-Omni/<checkpoint_dir>/"
            echo "  --dry-run               Show what would be done without executing"
            echo ""
            echo "Examples:"
            echo "  $0                                                         # Default: 7b, voicebench, audio_semantic, 50%"
            echo "  $0 --model 3b --benign_dataset voicebench --percentage 50  # 3b model"
            echo "  $0 --benign_dataset heysquad_accents --filter_type audio_acoustic"
            echo "  $0 --learning_rate 5e-5 --num_epochs 5                     # Custom training"
            echo "  $0 --skip_filter --skip_prepare                            # Resume from finetuning"
            echo "  $0 --resume                                                    # Auto-detect and resume"
            echo "  $0 --skip_filter --skip_prepare --skip_finetune --skip_merge  # Evaluate only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================
# Validate inputs
# ============================================
if [ "$MODEL_SIZE" != "3b" ] && [ "$MODEL_SIZE" != "7b" ]; then
    echo "Error: Invalid model size '$MODEL_SIZE'. Use '3b' or '7b'."
    exit 1
fi

if [ "$MODEL_SIZE" = "7b" ]; then
    MODEL_PATH="/project/anon/Qwen2.5-Omni-7B"
elif [ "$MODEL_SIZE" = "3b" ]; then
    MODEL_PATH="/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
fi

VALID_DATASETS="voicebench spoken_squad librispeech heysquad heysquad_accents gammacorpus_accents gammacorpus_usa benign_instructions_usa benign_instructions_accents mmsu bbh voicebench_cafe voicebench_traffic"
IS_VALID=""
for valid in $VALID_DATASETS; do
    if [ "$BENIGN_DATASET" = "$valid" ]; then
        IS_VALID="true"
        break
    fi
done
if [ -z "$IS_VALID" ]; then
    echo "Error: Invalid benign_dataset '$BENIGN_DATASET'"
    echo "Valid options: $VALID_DATASETS"
    exit 1
fi

if [ "$FILTER_TYPE" != "audio_acoustic" ] && [ "$FILTER_TYPE" != "audio_semantic" ] && [ "$FILTER_TYPE" != "sbert_audio_semantic" ] && [ "$FILTER_TYPE" != "sbert_text_semantic" ] && [ "$FILTER_TYPE" != "random" ]; then
    echo "Error: Invalid filter_type '$FILTER_TYPE'. Use 'audio_acoustic', 'audio_semantic', 'sbert_audio_semantic', 'sbert_text_semantic', or 'random'."
    exit 1
fi

# ============================================
# Compute paths and naming conventions
# ============================================

# Filtering mode
if [ -n "$NUM_SAMPLES" ]; then
    MODE_DESC="n${NUM_SAMPLES}"
    MODE_SUFFIX="n${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$THRESHOLD" ]; then
    MODE_DESC="thresh_${THRESHOLD}"
    MODE_SUFFIX="thresh_${THRESHOLD}"
    FILTER_ARGS="--threshold $THRESHOLD"
elif [ -n "$PERCENTAGE" ]; then
    MODE_DESC="percentage_${PERCENTAGE}"
    MODE_SUFFIX="percentage_${PERCENTAGE}"
    FILTER_ARGS="--percentage $PERCENTAGE"
else
    echo "Error: Must specify --percentage, --num_samples, or --threshold"
    exit 1
fi

# Append _safest suffix if selecting safest samples
if [ -n "$SELECT_SAFEST" ]; then
    MODE_DESC="${MODE_DESC}_safest"
    MODE_SUFFIX="${MODE_SUFFIX}_safest"
    FILTER_ARGS="$FILTER_ARGS --select_safest"
fi

# Directories based on filter type and benign dataset (matches 2_finetune_qwen_omni.sh)
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
        CHECKPOINT_DIR="checkpoints_acoustic"
        RESULTS_DIR="results_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_acoustic_${BENIGN_DATASET}"
        RESULTS_DIR="results_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        CHECKPOINT_DIR="checkpoints"
        RESULTS_DIR="results"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_${BENIGN_DATASET}"
        RESULTS_DIR="results_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "sbert_audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_sbert_audio_semantic"
        CHECKPOINT_DIR="checkpoints_sbert_audio_semantic"
        RESULTS_DIR="results_sbert_audio_semantic"
    else
        DATA_DIR="data_sbert_audio_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_sbert_audio_semantic_${BENIGN_DATASET}"
        RESULTS_DIR="results_sbert_audio_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
    # Text-only training: reuses sbert_audio_semantic filtered data, but trains without audio
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_sbert_audio_semantic"
        CHECKPOINT_DIR="checkpoints_sbert_text_semantic"
        RESULTS_DIR="results_sbert_text_semantic"
    else
        DATA_DIR="data_sbert_audio_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_sbert_text_semantic_${BENIGN_DATASET}"
        RESULTS_DIR="results_sbert_text_semantic_${BENIGN_DATASET}"
    fi
else  # random
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        CHECKPOINT_DIR="checkpoints_random"
        RESULTS_DIR="results_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_random_${BENIGN_DATASET}"
        RESULTS_DIR="results_random_${BENIGN_DATASET}"
    fi
fi

# File naming (matches 2_finetune_qwen_omni.sh)
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    FILTERED_SUBDIR="filtered_voicebench"
    FILE_PREFIX="voicebench_filtered"
else
    FILTERED_SUBDIR="filtered"
    FILE_PREFIX="${BENIGN_DATASET}_filtered"
fi

# Filtered dataset paths
if [ -n "$SELECT_SAFEST" ]; then
    SELECTION_TAG="safest"
else
    SELECTION_TAG="closest"
fi
FILTERED_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_${MODE_SUFFIX}.json"
if [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
    LLAMAFACTORY_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_${MODE_SUFFIX}_text_only_llamafactory.json"
else
    LLAMAFACTORY_JSON="${DATA_DIR}/${FILTERED_SUBDIR}/${FILE_PREFIX}_${SELECTION_TAG}_${MODE_SUFFIX}_llamafactory.json"
fi

# Checkpoint paths
CHECKPOINT_NAME="qwen_omni_${MODEL_SIZE}_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_SUFFIX}_epoch_${NUM_EPOCHS}"

# Model save locations (LoRA + merged)
if [ "$SAVE_TO" = "local" ]; then
    LORA_OUTPUT_DIR="${PROJECT_DIR}/${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
    MERGED_OUTPUT_DIR="${PROJECT_DIR}/${CHECKPOINT_DIR}/${CHECKPOINT_NAME}_merged"
elif [ "$SAVE_TO" = "project" ]; then
    BFT_MODELS_DIR="/project/anon/BFT_models"
    LORA_OUTPUT_DIR="${BFT_MODELS_DIR}/${CHECKPOINT_NAME}"
    MERGED_OUTPUT_DIR="${BFT_MODELS_DIR}/${CHECKPOINT_NAME}_merged"
else
    echo "Error: Invalid --save_to value '$SAVE_TO'. Use 'project' or 'local'."
    exit 1
fi

# Dataset info for LlamaFactory
DATASET_INFO_PATH="${PROJECT_DIR}/dataset_info.json"
DATASET_FILE_RELATIVE="finetune_codes/${LLAMAFACTORY_JSON}"

# Map benign datasets to their source JSON files and shared_audio directories.
# JSON metadata lives in audio-flamingo/data/; audio files in shared_audio/.
AUDIO_FLAMINGO_DATA="$SCRIPT_DIR/../../audio-flamingo/data"
case $BENIGN_DATASET in
    voicebench)              BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/voicebench/sd-qa/sd_qa_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/voicebench" ;;
    spoken_squad)            BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/spoken_squad/spoken_squad_full.json"
                             SHARED_AUDIO_DIR="" ;;
    librispeech)             BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/librispeech/librispeech_full.json"
                             SHARED_AUDIO_DIR="" ;;
    heysquad)                BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/heysquad/heysquad_full.json"
                             SHARED_AUDIO_DIR="" ;;
    heysquad_accents)        BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/heysquad_accents/heysquad_accents_full.json"
                             SHARED_AUDIO_DIR="" ;;
    gammacorpus_accents)     BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/gammacorpus_accents/gammacorpus_accents_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/gammacorpus_accents" ;;
    gammacorpus_usa)         BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/gammacorpus_usa/gammacorpus_usa_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/gammacorpus_usa" ;;
    benign_instructions_usa) BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/benign_instructions_usa/benign_instructions_usa_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/benign_instructions_usa" ;;
    benign_instructions_accents) BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/benign_instructions_accents/benign_instructions_accents_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/benign_instructions_accents" ;;
    mmsu)                    BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/mmsu/mmsu_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/mmsu" ;;
    bbh)                     BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/bbh/bbh_full.json"
                             SHARED_AUDIO_DIR="" ;;
    voicebench_cafe)         BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/voicebench_cafe/voicebench_cafe_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/voicebench_cafe" ;;
    voicebench_traffic)      BENIGN_JSON="${AUDIO_FLAMINGO_DATA}/voicebench_traffic/voicebench_traffic_full.json"
                             SHARED_AUDIO_DIR="${SHARED_AUDIO_ROOT}/voicebench_traffic" ;;
esac

# ============================================
# Print configuration
# ============================================
echo "=============================================="
echo "   QWEN2.5-OMNI FULL PIPELINE"
echo "=============================================="
echo "Model size:       $MODEL_SIZE"
echo "Model path:       $MODEL_PATH"
echo "Benign dataset:   $BENIGN_DATASET"
echo "Filter type:      $FILTER_TYPE"
echo "Mode:             $MODE_DESC"
echo "Training epochs:  $NUM_EPOCHS"
echo "Learning rate:    $LEARNING_RATE"
echo "LoRA rank:        $LORA_RANK"
echo "Eval dataset:     $DATASET"
echo "Eval mode:        $EVAL_MODE"
echo "Data directory:   $DATA_DIR"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
echo "Results dir:      $RESULTS_DIR"
echo "Save to:          $SAVE_TO ($MERGED_OUTPUT_DIR)"
echo "Resume mode:      ${RESUME:-false}"
echo "Dry run:          ${DRY_RUN:-false}"
echo "=============================================="
echo ""

# Helper function for dry-run support
run_cmd() {
    if [ -n "$DRY_RUN" ]; then
        echo "[DRY-RUN] Would execute: $*"
    else
        eval "$@"
    fi
}

# Dataset sizes for evaluation (number of audio files)
SAFETYBENCH_TOTAL=939
ADVBENCH_TOTAL=520

# Helper function to count responses in a JSON file
count_responses_in_json() {
    local json_file="$1"
    if [ -f "$json_file" ]; then
        python3 -c "import json; print(len(json.load(open('$json_file'))))" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Helper function to check if LoRA adapter exists
check_lora_adapter_exists() {
    local lora_path="$1"
    if [ -d "$lora_path" ]; then
        if [ -f "$lora_path/adapter_config.json" ] || [ -f "$lora_path/adapter_model.safetensors" ] || [ -f "$lora_path/adapter_model.bin" ]; then
            return 0
        fi
    fi
    return 1
}

# Helper function to check if merged model exists
check_merged_model_exists() {
    local model_path="$1"
    if [ -d "$model_path" ]; then
        if [ -f "$model_path/config.json" ] && ([ -f "$model_path/model.safetensors" ] || [ -f "$model_path/pytorch_model.bin" ] || ls "$model_path"/model-*.safetensors 1>/dev/null 2>&1); then
            return 0
        fi
    fi
    return 1
}

# Helper function to find the latest training checkpoint in a directory
find_latest_checkpoint() {
    local output_dir="$1"
    if [ ! -d "$output_dir" ]; then
        echo ""
        return
    fi
    # LlamaFactory saves checkpoints as checkpoint-<step>
    local latest=""
    local max_step=0
    for ckpt_dir in "$output_dir"/checkpoint-*; do
        if [ -d "$ckpt_dir" ] && [ -f "$ckpt_dir/trainer_state.json" ]; then
            local step
            step=$(basename "$ckpt_dir" | sed 's/checkpoint-//')
            if [ "$step" -gt "$max_step" ] 2>/dev/null; then
                max_step="$step"
                latest="$ckpt_dir"
            fi
        fi
    done
    echo "$latest"
}

# ============================================
# Auto-resume: detect completed steps
# ============================================
if [ -n "$RESUME" ]; then
    echo "=============================================="
    echo "   RESUME MODE: Detecting completed steps"
    echo "=============================================="

    # Check if filtered data exists -> skip filter
    if [ -f "$FILTERED_JSON" ]; then
        echo "[DONE] Filtered data exists: $FILTERED_JSON"
        SKIP_FILTER="true"
    else
        echo "[TODO] Filtered data not found, will run filtering"
        SKIP_FILTER=""
    fi

    # Check if LlamaFactory dataset exists -> skip prepare
    if [ -f "$LLAMAFACTORY_JSON" ]; then
        echo "[DONE] LlamaFactory dataset exists: $LLAMAFACTORY_JSON"
        SKIP_PREPARE="true"
    else
        echo "[TODO] LlamaFactory dataset not found, will run preparation"
        SKIP_PREPARE=""
    fi

    # Check finetuning status
    if check_lora_adapter_exists "$LORA_OUTPUT_DIR"; then
        echo "[DONE] LoRA adapter exists: $LORA_OUTPUT_DIR"
        SKIP_FINETUNE="true"
    elif check_merged_model_exists "$MERGED_OUTPUT_DIR"; then
        echo "[DONE] Merged model exists (training completed): $MERGED_OUTPUT_DIR"
        SKIP_FINETUNE="true"
    else
        LATEST_CKPT=$(find_latest_checkpoint "$LORA_OUTPUT_DIR")
        if [ -n "$LATEST_CKPT" ]; then
            echo "[RESUME] Found training checkpoint: $LATEST_CKPT"
            echo "         Will resume training from this checkpoint"
            RESUME_CHECKPOINT="$LATEST_CKPT"
        else
            echo "[TODO] No training checkpoint found, will start fresh"
        fi
        SKIP_FINETUNE=""
    fi

    # Check merge status
    if check_merged_model_exists "$MERGED_OUTPUT_DIR"; then
        echo "[DONE] Merged model exists: $MERGED_OUTPUT_DIR"
        SKIP_MERGE="true"
    else
        echo "[TODO] Merged model not found, will run merge"
        SKIP_MERGE=""
    fi

    echo "=============================================="
    echo ""
fi

# Helper function to get expected total samples for a dataset
get_expected_total() {
    local dataset="$1"
    if [ "$dataset" = "safetybench" ]; then
        echo "$SAFETYBENCH_TOTAL"
    else
        echo "$ADVBENCH_TOTAL"
    fi
}

# ============================================
# Step 0: Filter benign dataset
# ============================================
if [ -z "$SKIP_FILTER" ]; then
    echo "=============================================="
    echo "STEP 0: Filtering $BENIGN_DATASET samples ($FILTER_TYPE)"
    echo "=============================================="

    if [ -z "$DRY_RUN" ] && [ ! -f "$BENIGN_JSON" ]; then
        echo "Error: Benign dataset JSON not found: $BENIGN_JSON"
        echo "Please ensure the benign dataset is available, or use --skip_filter."
        exit 1
    fi

    mkdir -p "$(dirname "$FILTERED_JSON")"

    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        # Audio-Acoustic: reference acoustic encoder for filtering.
        echo "Filter: Audio-Acoustic"
        EMBED_CACHE_DIR="${DATA_DIR}/embedding_cache"
        FILTER_CMD="python 0_filter_voicebench_audio_acoustic.py"
        FILTER_CMD="$FILTER_CMD --voicebench_json \"$BENIGN_JSON\""
        FILTER_CMD="$FILTER_CMD --harmful_audio_dir \"${HARMFUL_DATA_ROOT}/advbench_gtts/en\""
        FILTER_CMD="$FILTER_CMD --output_json \"$FILTERED_JSON\""
        FILTER_CMD="$FILTER_CMD --cache_dir \"$EMBED_CACHE_DIR\""
        if [ -n "$SHARED_AUDIO_DIR" ]; then
            FILTER_CMD="$FILTER_CMD --audio_base_dir \"$SHARED_AUDIO_DIR\""
        fi
        FILTER_CMD="$FILTER_CMD $FILTER_ARGS"
        run_cmd $FILTER_CMD
    elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
        # Audio-Semantic: Use Whisper-Large-V3 (Qwen2.5-Omni's own internal encoder).
        echo "Filter: Audio-Semantic (Whisper-Large-V3 - Qwen2.5-Omni's internal encoder)"
        EMBED_CACHE_DIR="${DATA_DIR}/embedding_cache"
        FILTER_CMD="python 0_filter_voicebench_audio_semantic.py"
        FILTER_CMD="$FILTER_CMD --voicebench_json \"$BENIGN_JSON\""
        FILTER_CMD="$FILTER_CMD --harmful_audio_dir \"${HARMFUL_DATA_ROOT}/advbench_gtts/en\""
        FILTER_CMD="$FILTER_CMD --output_json \"$FILTERED_JSON\""
        FILTER_CMD="$FILTER_CMD --cache_dir \"$EMBED_CACHE_DIR\""
        if [ -n "$SHARED_AUDIO_DIR" ]; then
            FILTER_CMD="$FILTER_CMD --audio_base_dir \"$SHARED_AUDIO_DIR\""
        fi
        FILTER_CMD="$FILTER_CMD $FILTER_ARGS"
        run_cmd $FILTER_CMD
    elif [ "$FILTER_TYPE" = "random" ]; then
        echo "Filter: Random sampling (baseline - no similarity filtering)"
        run_cmd python 0_filter_random.py \
            --input_json "\"$BENIGN_JSON\"" \
            --output_json "\"$FILTERED_JSON\"" \
            --seed 42 \
            $FILTER_ARGS
    elif [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
        # Text-only variant: uses same SBERT filtering as sbert_audio_semantic
        # but prepares text-only data (no audio). Filter step is identical.
        echo "Filter: SBERT Text-Semantic (sentence-BERT text filtering, TEXT-ONLY finetuning)"
        EMBED_CACHE_DIR="${DATA_DIR}/embedding_cache"
        HARMFUL_CSV="${HARMFUL_DATA_ROOT}/advbench.csv"
        FILTER_CMD="python 0_filter_voicebench_sbert_audio_semantic.py"
        FILTER_CMD="$FILTER_CMD --voicebench_json \"$BENIGN_JSON\""
        FILTER_CMD="$FILTER_CMD --harmful_texts_csv \"$HARMFUL_CSV\""
        FILTER_CMD="$FILTER_CMD --output_json \"$FILTERED_JSON\""
        FILTER_CMD="$FILTER_CMD --cache_dir \"$EMBED_CACHE_DIR\""
        if [ -n "$SHARED_AUDIO_DIR" ]; then
            FILTER_CMD="$FILTER_CMD --audio_base_dir \"$SHARED_AUDIO_DIR\""
        fi
        FILTER_CMD="$FILTER_CMD $FILTER_ARGS"
        run_cmd $FILTER_CMD
    elif [ "$FILTER_TYPE" = "sbert_audio_semantic" ]; then
        # Text-Semantic: Use sentence-transformers (all-MiniLM-L6-v2) for text similarity.
        # Filters based on text meaning of questions, not audio characteristics.
        echo "Filter: SBERT Audio-Semantic (sentence-BERT text filtering, audio finetuning)"
        EMBED_CACHE_DIR="${DATA_DIR}/embedding_cache"
        HARMFUL_CSV="${HARMFUL_DATA_ROOT}/advbench.csv"
        FILTER_CMD="python 0_filter_voicebench_sbert_audio_semantic.py"
        FILTER_CMD="$FILTER_CMD --voicebench_json \"$BENIGN_JSON\""
        FILTER_CMD="$FILTER_CMD --harmful_texts_csv \"$HARMFUL_CSV\""
        FILTER_CMD="$FILTER_CMD --output_json \"$FILTERED_JSON\""
        FILTER_CMD="$FILTER_CMD --cache_dir \"$EMBED_CACHE_DIR\""
        if [ -n "$SHARED_AUDIO_DIR" ]; then
            FILTER_CMD="$FILTER_CMD --audio_base_dir \"$SHARED_AUDIO_DIR\""
        fi
        FILTER_CMD="$FILTER_CMD $FILTER_ARGS"
        run_cmd $FILTER_CMD
    else
        echo "Error: Unknown filter type '$FILTER_TYPE'"
        exit 1
    fi

    echo ""
    echo "Step 0 completed successfully!"
    echo "Filtered data saved to: $FILTERED_JSON"
    echo ""
else
    echo "Skipping Step 0 (filtering)..."
fi

# ============================================
# Step 1: Prepare dataset for LlamaFactory
# ============================================
if [ -z "$SKIP_PREPARE" ]; then
    echo "=============================================="
    echo "STEP 1: Preparing dataset for LlamaFactory"
    echo "=============================================="

    if [ -z "$DRY_RUN" ] && [ ! -f "$FILTERED_JSON" ]; then
        echo "Error: Filtered dataset not found: $FILTERED_JSON"
        echo "Run step 0 first, or check the path."
        exit 1
    fi

    # Convert to LlamaFactory format
    # Audio paths in the source JSON are relative to audio-flamingo dir, so pass it as base path
    AUDIO_BASE_PATH="$SCRIPT_DIR/../../audio-flamingo"
    PREPARE_CMD="python 1_prepare_dataset.py --input \"$FILTERED_JSON\" --output \"$LLAMAFACTORY_JSON\" --audio_base_path \"$AUDIO_BASE_PATH\""
    if [ "$FILTER_TYPE" = "sbert_text_semantic" ]; then
        PREPARE_CMD="$PREPARE_CMD --text_only"
    fi
    run_cmd $PREPARE_CMD

    # Create/update dataset_info.json for LlamaFactory
    echo "Updating dataset_info.json for LlamaFactory..."
    if [ -z "$DRY_RUN" ]; then
        python3 << EOF
import json, os
info_path = '${DATASET_INFO_PATH}'
filter_type = '${FILTER_TYPE}'
if os.path.exists(info_path):
    with open(info_path) as f:
        info = json.load(f)
else:
    info = {}
entry = {
    'file_name': '${DATASET_FILE_RELATIVE}',
    'formatting': 'sharegpt',
    'columns': {'messages': 'messages'},
    'tags': {
        'role_tag': 'role',
        'content_tag': 'content',
        'user_tag': 'user',
        'assistant_tag': 'assistant'
    }
}
if filter_type != 'sbert_text_semantic':
    entry['columns']['audios'] = 'audios'
info['benign_audio_dataset'] = entry
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print(f'Updated {info_path}')
EOF
    else
        echo "[DRY-RUN] Would update dataset_info.json"
    fi

    echo ""
    echo "Step 1 completed successfully!"
    echo "LlamaFactory dataset: $LLAMAFACTORY_JSON"
    echo "Dataset info updated: $DATASET_INFO_PATH"
    echo ""
else
    echo "Skipping Step 1 (dataset preparation)..."
fi

# ============================================
# Step 2: Finetune Qwen2.5-Omni LoRA
# ============================================
if [ -z "$SKIP_FINETUNE" ]; then
    echo "=============================================="
    echo "STEP 2: Finetuning Qwen2.5-Omni ${MODEL_SIZE} LoRA"
    echo "=============================================="

    # Check if LoRA adapter or merged model already exists
    if check_lora_adapter_exists "$LORA_OUTPUT_DIR"; then
        echo "[SKIP] LoRA adapter already exists: $LORA_OUTPUT_DIR"
        echo "       To re-train, delete this directory first."
    elif check_merged_model_exists "$MERGED_OUTPUT_DIR"; then
        echo "[SKIP] Merged model already exists: $MERGED_OUTPUT_DIR"
        echo "       Training already completed. To re-train, delete the merged model first."
    else
        # Build finetune command
        FINETUNE_CMD="bash 2_finetune_qwen_omni.sh"
        FINETUNE_CMD="$FINETUNE_CMD --model \"$MODEL_SIZE\""
        FINETUNE_CMD="$FINETUNE_CMD --benign_dataset \"$BENIGN_DATASET\""
        FINETUNE_CMD="$FINETUNE_CMD --filter_type \"$FILTER_TYPE\""
        FINETUNE_CMD="$FINETUNE_CMD --num_epochs \"$NUM_EPOCHS\""
        FINETUNE_CMD="$FINETUNE_CMD --learning_rate \"$LEARNING_RATE\""
        FINETUNE_CMD="$FINETUNE_CMD --lora_rank \"$LORA_RANK\""
        FINETUNE_CMD="$FINETUNE_CMD --batch_size \"$BATCH_SIZE\""
        FINETUNE_CMD="$FINETUNE_CMD --gradient_accumulation \"$GRADIENT_ACCUMULATION\""
        FINETUNE_CMD="$FINETUNE_CMD --output_dir \"$LORA_OUTPUT_DIR\""
        FINETUNE_CMD="$FINETUNE_CMD $FILTER_ARGS"

        # Resume from checkpoint if available
        if [ -n "$RESUME_CHECKPOINT" ]; then
            echo "Resuming finetuning from checkpoint: $RESUME_CHECKPOINT"
            FINETUNE_CMD="$FINETUNE_CMD --resume_from_checkpoint \"$RESUME_CHECKPOINT\""
        else
            echo "LoRA adapter not found at: $LORA_OUTPUT_DIR"
            echo "Running finetuning..."
        fi

        run_cmd $FINETUNE_CMD

        echo ""
        echo "Step 2 completed successfully!"
        echo "LoRA checkpoint: $LORA_OUTPUT_DIR"
    fi
    echo ""
else
    echo "Skipping Step 2 (finetuning)..."
fi

# ============================================
# Step 3: Merge LoRA weights
# ============================================
if [ -z "$SKIP_MERGE" ]; then
    echo "=============================================="
    echo "STEP 3: Merging LoRA weights"
    echo "=============================================="

    # Check if merged model already exists
    if check_merged_model_exists "$MERGED_OUTPUT_DIR"; then
        echo "[SKIP] Merged model already exists: $MERGED_OUTPUT_DIR"
        echo "       To re-merge, delete this directory first."
    else
        if [ -z "$DRY_RUN" ] && [ ! -d "$LORA_OUTPUT_DIR" ]; then
            echo "Error: LoRA checkpoint not found: $LORA_OUTPUT_DIR"
            echo "Run step 2 first, or check the path."
            exit 1
        fi

        echo "Merged model not found at: $MERGED_OUTPUT_DIR"
        echo "Running LoRA merge..."
        run_cmd bash 3_merge_lora.sh \
            --model "$MODEL_SIZE" \
            --lora_path "$LORA_OUTPUT_DIR" \
            --output_path "$MERGED_OUTPUT_DIR"

        echo ""
        echo "Step 3 completed successfully!"
        echo "Merged model: $MERGED_OUTPUT_DIR"
    fi
    echo ""
else
    echo "Skipping Step 3 (LoRA merging)..."
fi

# ============================================
# Step 4: Evaluate on harmful datasets
# ============================================
if [ -z "$SKIP_EVALUATE" ]; then
    echo "=============================================="
    echo "STEP 4: Evaluating on $DATASET"
    echo "=============================================="

    # Activate conda environment for evaluation
    if [ -z "$DRY_RUN" ]; then
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate qwen-omni
    fi

    # Determine evaluation datasets
    if [ "$DATASET" = "both" ]; then
        EVAL_DATASETS="advbench safetybench"
    else
        EVAL_DATASETS="$DATASET"
    fi

    for EVAL_DATASET in $EVAL_DATASETS; do
        echo ""
        echo "--- Checking $EVAL_DATASET evaluation ---"

        # Set harmful data paths
        if [ "$EVAL_DATASET" = "advbench" ]; then
            HARMFUL_DIR="${HARMFUL_DATA_ROOT}/advbench_gtts/en"
            BENCHMARK_CSV="${HARMFUL_DATA_ROOT}/advbench.csv"
        elif [ "$EVAL_DATASET" = "safetybench" ]; then
            HARMFUL_DIR="${HARMFUL_DATA_ROOT}/safetybench_gtts/en"
            BENCHMARK_CSV="${HARMFUL_DATA_ROOT}/safetybench.csv"
        fi

        if [ -z "$DRY_RUN" ] && [ ! -d "$HARMFUL_DIR" ]; then
            echo "Warning: Harmful data directory not found: $HARMFUL_DIR"
            echo "Skipping $EVAL_DATASET evaluation."
            continue
        fi

        EVAL_OUTPUT_DIR="${RESULTS_DIR}/${EVAL_DATASET}_eval"
        EXPECTED_TOTAL=$(get_expected_total "$EVAL_DATASET")

        # Check for existing finetuned results
        # Determine model name based on which model will be used (merged preferred)
        if [ -d "$MERGED_OUTPUT_DIR" ]; then
            CHECK_MODEL_NAME="${CHECKPOINT_NAME}_merged"
        else
            CHECK_MODEL_NAME="${CHECKPOINT_NAME}"
        fi
        FINETUNED_RESULTS_FILE="${EVAL_OUTPUT_DIR}/finetuned_${CHECK_MODEL_NAME}_responses.json"
        PRETRAINED_RESULTS_FILE="${EVAL_OUTPUT_DIR}/pretrained_responses.json"

        echo "Expected total samples: $EXPECTED_TOTAL"

        # Check finetuned results
        FINETUNED_COUNT=0
        if [ -f "$FINETUNED_RESULTS_FILE" ]; then
            FINETUNED_COUNT=$(count_responses_in_json "$FINETUNED_RESULTS_FILE")
            echo "Found finetuned results: $FINETUNED_RESULTS_FILE"
            echo "Finetuned responses: $FINETUNED_COUNT / $EXPECTED_TOTAL"
        fi

        # Check pretrained results
        PRETRAINED_COUNT=0
        if [ -f "$PRETRAINED_RESULTS_FILE" ]; then
            PRETRAINED_COUNT=$(count_responses_in_json "$PRETRAINED_RESULTS_FILE")
            echo "Found pretrained results: $PRETRAINED_RESULTS_FILE"
            echo "Pretrained responses: $PRETRAINED_COUNT / $EXPECTED_TOTAL"
        fi

        # Determine if we need to run evaluation
        NEED_FINETUNED_EVAL="false"
        NEED_PRETRAINED_EVAL="false"

        if [ "$EVAL_MODE" = "finetuned_only" ] || [ "$EVAL_MODE" = "both" ]; then
            if [ "$FINETUNED_COUNT" -lt "$EXPECTED_TOTAL" ]; then
                NEED_FINETUNED_EVAL="true"
            fi
        fi

        if [ "$EVAL_MODE" = "pretrained_only" ] || [ "$EVAL_MODE" = "both" ]; then
            if [ "$PRETRAINED_COUNT" -lt "$EXPECTED_TOTAL" ]; then
                NEED_PRETRAINED_EVAL="true"
            fi
        fi

        # Check if both are complete
        if [ "$NEED_FINETUNED_EVAL" = "false" ] && [ "$NEED_PRETRAINED_EVAL" = "false" ]; then
            echo "[SKIP] Evaluation complete for $EVAL_DATASET"
            if [ "$EVAL_MODE" = "finetuned_only" ] || [ "$EVAL_MODE" = "both" ]; then
                echo "       Finetuned: $FINETUNED_COUNT/$EXPECTED_TOTAL responses"
            fi
            if [ "$EVAL_MODE" = "pretrained_only" ] || [ "$EVAL_MODE" = "both" ]; then
                echo "       Pretrained: $PRETRAINED_COUNT/$EXPECTED_TOTAL responses"
            fi
            continue
        fi

        # Report what needs to be done
        if [ "$NEED_FINETUNED_EVAL" = "true" ] && [ "$FINETUNED_COUNT" -gt 0 ]; then
            echo "[RESUME] Incomplete finetuned evaluation ($FINETUNED_COUNT/$EXPECTED_TOTAL)"
            echo "         Will continue from sample $FINETUNED_COUNT"
        elif [ "$NEED_FINETUNED_EVAL" = "true" ]; then
            echo "Starting fresh finetuned evaluation..."
        fi

        if [ "$NEED_PRETRAINED_EVAL" = "true" ] && [ "$PRETRAINED_COUNT" -gt 0 ]; then
            echo "[RESUME] Incomplete pretrained evaluation ($PRETRAINED_COUNT/$EXPECTED_TOTAL)"
            echo "         Will continue from sample $PRETRAINED_COUNT"
        elif [ "$NEED_PRETRAINED_EVAL" = "true" ]; then
            echo "Starting fresh pretrained evaluation..."
        fi

        # Build evaluation command
        EVAL_CMD="python 4_evaluate_jailbreaking.py"
        EVAL_CMD="$EVAL_CMD --harmful_data_dir \"$HARMFUL_DIR\""
        EVAL_CMD="$EVAL_CMD --output_dir \"$EVAL_OUTPUT_DIR\""
        EVAL_CMD="$EVAL_CMD --pretrained_path \"$MODEL_PATH\""
        EVAL_CMD="$EVAL_CMD --eval_mode \"$EVAL_MODE\""

        if [ -f "$BENCHMARK_CSV" ]; then
            EVAL_CMD="$EVAL_CMD --benchmark_csv \"$BENCHMARK_CSV\""
        fi

        # Add finetuned model path (prefer merged, fall back to LoRA)
        # Also set the model_name to match the path basename for consistent file naming
        EVAL_MODEL_NAME="$CHECKPOINT_NAME"
        if [ -d "$MERGED_OUTPUT_DIR" ]; then
            EVAL_CMD="$EVAL_CMD --finetuned_path \"$MERGED_OUTPUT_DIR\""
            EVAL_MODEL_NAME="${CHECKPOINT_NAME}_merged"
        elif [ -d "$LORA_OUTPUT_DIR" ]; then
            echo "Note: Merged model not found, using LoRA checkpoint directly."
            EVAL_CMD="$EVAL_CMD --finetuned_path \"$LORA_OUTPUT_DIR\""
        else
            if [ "$EVAL_MODE" != "pretrained_only" ]; then
                echo "Warning: No finetuned model found. Evaluating pretrained only."
                EVAL_CMD="$EVAL_CMD --eval_mode pretrained_only"
            fi
        fi

        EVAL_CMD="$EVAL_CMD --model_name \"$EVAL_MODEL_NAME\""

        # Add resume flag if we have partial results
        if [ "$FINETUNED_COUNT" -gt 0 ] || [ "$PRETRAINED_COUNT" -gt 0 ]; then
            EVAL_CMD="$EVAL_CMD --resume"
        fi

        run_cmd $EVAL_CMD

        echo "Results saved to: $EVAL_OUTPUT_DIR"
    done

    echo ""
    echo "Step 4 completed successfully!"
    echo ""
else
    echo "Skipping Step 4 (evaluation)..."
fi

# ============================================
# Step 5: Run ASR evaluation
# ============================================
if [ -z "$SKIP_ASR" ]; then
    echo "=============================================="
    echo "STEP 5: Running ASR Evaluation"
    echo "=============================================="

    ASR_SCRIPT="$SCRIPT_DIR/../../calculate_asr.py"
    ASR_OUTPUT_DIR="$SCRIPT_DIR/../../asr_results/Qwen2.5-Omni"
    mkdir -p "$ASR_OUTPUT_DIR"

    if [ ! -f "$ASR_SCRIPT" ]; then
        echo "Warning: ASR calculation script not found: $ASR_SCRIPT"
        echo "Skipping ASR evaluation."
    else
        # Determine evaluation datasets
        if [ "$DATASET" = "both" ]; then
            EVAL_DATASETS="advbench safetybench"
        else
            EVAL_DATASETS="$DATASET"
        fi

        for EVAL_DATASET in $EVAL_DATASETS; do
            echo ""
            echo "--- ASR evaluation for $EVAL_DATASET ---"

            EVAL_OUTPUT_DIR="${RESULTS_DIR}/${EVAL_DATASET}_eval"

            # Determine correct filename (prefer merged model name)
            if [ -d "$MERGED_OUTPUT_DIR" ]; then
                ASR_MODEL_NAME="${CHECKPOINT_NAME}_merged"
            else
                ASR_MODEL_NAME="${CHECKPOINT_NAME}"
            fi

            # Evaluate finetuned responses
            FINETUNED_RESPONSES="${EVAL_OUTPUT_DIR}/finetuned_${ASR_MODEL_NAME}_responses.json"
            if [ -f "$FINETUNED_RESPONSES" ]; then
                echo "Processing finetuned responses: $FINETUNED_RESPONSES"
                run_cmd python3 "\"$ASR_SCRIPT\"" \
                    --input "\"$FINETUNED_RESPONSES\"" \
                    --output-dir "\"$ASR_OUTPUT_DIR\"" \
                    --model "\"Qwen2.5-Omni\"" \
                    --dataset "\"$EVAL_DATASET\"" \
                    --batch-size 8
            else
                echo "Warning: Finetuned responses not found: $FINETUNED_RESPONSES"
            fi

            # Evaluate pretrained responses if available
            PRETRAINED_RESPONSES="${EVAL_OUTPUT_DIR}/pretrained_responses.json"
            if [ -f "$PRETRAINED_RESPONSES" ]; then
                echo "Processing pretrained responses: $PRETRAINED_RESPONSES"
                run_cmd python3 "\"$ASR_SCRIPT\"" \
                    --input "\"$PRETRAINED_RESPONSES\"" \
                    --output-dir "\"$ASR_OUTPUT_DIR\"" \
                    --model "\"Qwen2.5-Omni\"" \
                    --dataset "\"$EVAL_DATASET\"" \
                    --batch-size 8
            fi
        done
    fi

    echo ""
    echo "Step 5 completed successfully!"
    echo ""
else
    echo "Skipping Step 5 (ASR evaluation)..."
fi

echo "=============================================="
echo "       PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Model:           Qwen2.5-Omni ${MODEL_SIZE}"
echo "  Benign dataset:  $BENIGN_DATASET"
echo "  Filter type:     $FILTER_TYPE"
echo "  Mode:            $MODE_DESC"
if [ -n "$SELECT_SAFEST" ]; then
echo "  Selection:       SAFEST (furthest from harmful)"
else
echo "  Selection:       CLOSEST to harmful"
fi
echo "  Epochs:          $NUM_EPOCHS"
echo "  Learning rate:   $LEARNING_RATE"
echo "  LoRA rank:       $LORA_RANK"
echo "  Eval dataset:    $DATASET"
echo "  Eval mode:       $EVAL_MODE"
echo ""
echo "Output locations:"
echo "  Filtered data:   $DATA_DIR/$FILTERED_SUBDIR/"
echo "  LoRA checkpoint: $LORA_OUTPUT_DIR"
echo "  Merged model:    $MERGED_OUTPUT_DIR"
echo "  Results:         $RESULTS_DIR/"
echo "  ASR results:     ../../asr_results/Qwen2.5-Omni/"
echo "=============================================="
