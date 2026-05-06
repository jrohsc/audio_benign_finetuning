#!/bin/bash
#
# Full Pipeline: Filter -> Extract -> Finetune -> Merge -> Evaluate
#
# This script runs the complete benign finetuning pipeline for Kimi-Audio:
#   0. Filter benign samples by embedding distance
#   1. Extract semantic codes for each dataset
#   2. Finetune LoRA models
#   3. Merge LoRA weights
#   4. Evaluate on harmful audio datasets
#   5. Analyze results
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset
#   - spoken_squad:          Spoken-SQuAD dataset
#   - librispeech:           LibriSpeech dataset
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#   - heysquad_accents:      HeySQuAD with 11 accent variations (TTS-generated)
#   - gammacorpus_accents:   GammaCorpus with 11 accent variations (TTS-generated)
#   - mmsu:                  VoiceBench MMSU (Multi-Subject Understanding) ~3k samples
#   - bbh:                   VoiceBench BBH (BIG-Bench Hard) ~1k samples
#
# FILTER TYPES SUPPORTED:
#   - audio_acoustic: TRUE ACOUSTIC features (Whisper-Large-V3)
#                     Captures HOW audio sounds (voice, prosody, timbre)
#
#   - audio_semantic: SEMANTIC from AUDIO (WhisperVQEncoder)
#                     Captures WHAT is said from audio (linguistic content)
#
#   - text_semantic:  SEMANTIC from TEXT (sentence-transformers)
#                     Captures text meaning similarity
#
#   - random:         RANDOM baseline (no distance filtering)
#                     Randomly selects k% of samples for comparison
#
# Usage:
#   ./run_full_pipeline.sh                                           # Defaults (voicebench, audio_semantic, 50%)
#   ./run_full_pipeline.sh --benign_dataset spoken_squad --percentage 50
#   ./run_full_pipeline.sh --benign_dataset librispeech --filter_type audio_acoustic --percentage 50
#   ./run_full_pipeline.sh --filter_type text_semantic --percentage 25 --num_epochs 5

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load CUDA
module load cuda/12.6

# Default values
BENIGN_DATASET="voicebench_cafe"  # "voicebench", "spoken_squad", or "librispeech"
FILTER_TYPE="audio_semantic"  # "audio_acoustic", "audio_semantic", or "text_semantic"
PERCENTAGE="25"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=5
DATASET="both"  # advbench, safetybench, or both for evaluation
SKIP_FILTER=""
SKIP_EXTRACT=""
SKIP_FINETUNE=""
SKIP_MERGE=""
SKIP_EVALUATE=""
SKIP_ANALYZE=""
SKIP_ASR=""
RUN_ASR="true"  # Run ASR evaluation by default
DRY_RUN=""
SELECT_SAFEST=""

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
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --skip_filter)
            SKIP_FILTER="true"
            shift
            ;;
        --skip_extract)
            SKIP_EXTRACT="true"
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
        --skip_analyze)
            SKIP_ANALYZE="true"
            shift
            ;;
        --skip_asr)
            SKIP_ASR="true"
            RUN_ASR=""
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --select_safest)
            SELECT_SAFEST="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run the full benign finetuning pipeline for Kimi-Audio."
            echo ""
            echo "Pipeline steps:"
            echo "  0. Filter benign samples by embedding distance"
            echo "  1. Extract semantic codes for each dataset"
            echo "  2. Finetune LoRA models"
            echo "  3. Merge LoRA weights"
            echo "  4. Evaluate on harmful audio datasets"
            echo "  5. Analyze results"
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo "    voicebench        VoiceBench SD-QA dataset (default)"
            echo "    spoken_squad      Spoken-SQuAD dataset"
            echo "    librispeech       LibriSpeech dataset"
            echo "    heysquad          HeySQuAD dataset (human-spoken QA)"
            echo "    heysquad_accents  HeySQuAD with 11 accent variations (TTS)"
            echo "    gammacorpus_accents  GammaCorpus with 11 accent variations (TTS)"
            echo "    mmsu              VoiceBench MMSU (~3k multi-subject QA samples)"
            echo "    bbh               VoiceBench BBH (~1k reasoning samples)"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: audio_semantic)"
            echo "    audio_acoustic  TRUE ACOUSTIC features (Whisper-Large-V3)"
            echo "    audio_semantic  SEMANTIC from AUDIO (WhisperVQEncoder)"
            echo "    text_semantic   SEMANTIC from TEXT (sentence-transformers)"
            echo "    random          RANDOM baseline (no distance filtering)"
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --percentage VALUE      Keep top percentage of closest samples (default: 50)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --threshold VALUE       Distance threshold"
            echo ""
            echo "Training options:"
            echo "  --num_epochs N          Number of training epochs (default: 5)"
            echo ""
            echo "Evaluation options:"
            echo "  --dataset NAME          Evaluation dataset: 'advbench', 'safetybench', or 'both' (default: both)"
            echo ""
            echo "Skip options (to resume from a specific step):"
            echo "  --skip_filter           Skip step 0 (filtering)"
            echo "  --skip_extract          Skip step 1 (semantic code extraction)"
            echo "  --skip_finetune         Skip step 2 (finetuning)"
            echo "  --skip_merge            Skip step 3 (LoRA merging)"
            echo "  --skip_evaluate         Skip step 4 (evaluation)"
            echo "  --skip_analyze          Skip step 5 (analysis)"
            echo "  --skip_asr              Skip step 6 (ASR evaluation)"
            echo ""
            echo "Other options:"
            echo "  --select_safest         Select samples FURTHEST from harmful (safest)"
            echo "                          instead of closest to harmful (default)"
            echo "  --dry-run               Show what would be done without executing"
            echo ""
            echo "Examples:"
            echo "  $0                                                    # Default: voicebench, audio_semantic, 50%, 5 epochs"
            echo "  $0 --benign_dataset spoken_squad --percentage 50     # Spoken-SQuAD at 50%"
            echo "  $0 --benign_dataset librispeech --percentage 50      # LibriSpeech at 50%"
            echo "  $0 --filter_type audio_acoustic --percentage 65      # TRUE acoustic filtering at 65%"
            echo "  $0 --filter_type text_semantic --num_samples 500     # Text semantic with 500 samples"
            echo "  $0 --skip_filter --skip_extract                      # Resume from finetuning"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine filtering mode description
if [ -n "$NUM_SAMPLES" ]; then
    MODE_DESC="n${NUM_SAMPLES}"
    FILTER_ARGS="--num_samples $NUM_SAMPLES"
elif [ -n "$THRESHOLD" ]; then
    MODE_DESC="thresh_${THRESHOLD}"
    FILTER_ARGS="--threshold $THRESHOLD"
elif [ -n "$PERCENTAGE" ]; then
    MODE_DESC="percentage_${PERCENTAGE}"
    FILTER_ARGS="--percentage $PERCENTAGE"
else
    echo "Error: Must specify --percentage, --num_samples, or --threshold"
    exit 1
fi

# Add safest suffix to mode description if selecting safest samples
if [ -n "$SELECT_SAFEST" ]; then
    MODE_DESC="${MODE_DESC}_safest"
    FILTER_ARGS="$FILTER_ARGS --select_safest"
fi

# Validate benign_dataset
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

# Determine dataset suffix for directory naming
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    DATASET_SUFFIX=""
else
    DATASET_SUFFIX="_${BENIGN_DATASET}"
fi

# Set data and output directories based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
        OUTPUT_DIR="output_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
        OUTPUT_DIR="output_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        OUTPUT_DIR="output"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
        OUTPUT_DIR="output_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        OUTPUT_DIR="output_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        OUTPUT_DIR="output_random_${BENIGN_DATASET}"
    fi
else  # text_semantic
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
        OUTPUT_DIR="output_semantic"
    else
        DATA_DIR="data_semantic_${BENIGN_DATASET}"
        OUTPUT_DIR="output_semantic_${BENIGN_DATASET}"
    fi
fi

SELECTION_MODE="CLOSEST to harmful"
if [ -n "$SELECT_SAFEST" ]; then
    SELECTION_MODE="SAFEST (furthest from harmful)"
fi

echo "=============================================="
echo "       KIMI-AUDIO FULL PIPELINE"
echo "=============================================="
echo "Benign dataset:   $BENIGN_DATASET"
echo "Filter type:      $FILTER_TYPE"
echo "Selection:        $SELECTION_MODE"
echo "Mode:             $MODE_DESC"
echo "Training epochs:  $NUM_EPOCHS"
echo "Eval dataset:     $DATASET"
echo "Data directory:   $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
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
        # Count array elements in JSON
        python3 -c "import json; data=json.load(open('$json_file')); print(len(data))" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Helper function to find the most recent evaluation result file for a model
find_latest_eval_file() {
    local eval_dir="$1"
    local model_name="$2"
    local dataset="$3"

    if [ -d "$eval_dir" ]; then
        # Find files matching pattern: {dataset}_en_{model_name}_*.json
        local pattern="${dataset}_en_${model_name}_*.json"
        local latest_file=$(ls -t "$eval_dir"/$pattern 2>/dev/null | head -1)
        echo "$latest_file"
    fi
}

# Helper function to check if merged model exists
check_merged_model_exists() {
    local model_path="$1"
    if [ -d "$model_path" ]; then
        # Check for essential model files
        if [ -f "$model_path/config.json" ] || [ -f "$model_path/model.safetensors" ] || [ -f "$model_path/pytorch_model.bin" ]; then
            return 0  # Model exists
        fi
    fi
    return 1  # Model does not exist
}

# Helper function to check if LoRA adapter exists
check_lora_adapter_exists() {
    local lora_path="$1"
    if [ -d "$lora_path" ]; then
        # Check for LoRA adapter files
        if [ -f "$lora_path/adapter_config.json" ] || [ -f "$lora_path/adapter_model.safetensors" ] || [ -f "$lora_path/adapter_model.bin" ]; then
            return 0  # LoRA adapter exists
        fi
    fi
    return 1  # LoRA adapter does not exist
}

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
# Auto-detect: skip steps whose outputs already exist
# ============================================
# Persistent storage for merged models
PROJECT_MODELS="/project/anon/BFT_models"

echo "Checking for existing artifacts..."

LORA_ADAPTER_PATH="${OUTPUT_DIR}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}"
# Check both /project and local for merged model
MERGED_MODEL_PATH_PROJECT="${PROJECT_MODELS}/Kimi-Audio_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}_merged"
MERGED_MODEL_PATH_LOCAL="${OUTPUT_DIR}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}_merged"

# Prefer /project path
if check_merged_model_exists "$MERGED_MODEL_PATH_PROJECT"; then
    MERGED_MODEL_PATH="$MERGED_MODEL_PATH_PROJECT"
elif check_merged_model_exists "$MERGED_MODEL_PATH_LOCAL"; then
    MERGED_MODEL_PATH="$MERGED_MODEL_PATH_LOCAL"
else
    MERGED_MODEL_PATH="$MERGED_MODEL_PATH_PROJECT"
fi

if check_merged_model_exists "$MERGED_MODEL_PATH"; then
    echo "[AUTO-SKIP] Merged model found: $MERGED_MODEL_PATH"
    echo "            Skipping steps 0-3, jumping to evaluation."
    SKIP_FILTER="true"
    SKIP_EXTRACT="true"
    SKIP_FINETUNE="true"
    SKIP_MERGE="true"
elif check_lora_adapter_exists "$LORA_ADAPTER_PATH"; then
    echo "[AUTO-SKIP] LoRA adapter found: $LORA_ADAPTER_PATH"
    echo "            Skipping steps 0-2, jumping to merge."
    SKIP_FILTER="true"
    SKIP_EXTRACT="true"
    SKIP_FINETUNE="true"
else
    SEMANTIC_CODES_CHECK=$(ls -1 "${DATA_DIR}"/*"${MODE_DESC}"*_semantic_codes.jsonl 2>/dev/null || true)
    if [ -n "$SEMANTIC_CODES_CHECK" ]; then
        echo "[AUTO-SKIP] Semantic codes found in ${DATA_DIR}/"
        echo "            Skipping steps 0-1, jumping to finetuning."
        SKIP_FILTER="true"
        SKIP_EXTRACT="true"
    else
        FILTERED_DATA_CHECK=$(ls -1 "${DATA_DIR}"/*"${MODE_DESC}"*.jsonl 2>/dev/null | grep -v semantic_codes || true)
        if [ -n "$FILTERED_DATA_CHECK" ]; then
            echo "[AUTO-SKIP] Filtered data found in ${DATA_DIR}/"
            echo "            Skipping step 0, jumping to extraction."
            SKIP_FILTER="true"
        fi
    fi
fi
echo ""

# ============================================
# Step 0: Filter benign samples
# ============================================
if [ -z "$SKIP_FILTER" ]; then
    echo "=============================================="
    echo "STEP 0: Filtering $BENIGN_DATASET samples"
    echo "=============================================="

    # Auto-detect: skip if filtered data already exists
    STEP0_SKIPPED=""
    FILTERED_FILES=$(ls -1 "${DATA_DIR}"/*"${MODE_DESC}"*.jsonl 2>/dev/null | grep -v semantic_codes || true)
    if [ -n "$FILTERED_FILES" ]; then
        echo "[SKIP] Filtered data already exists in ${DATA_DIR}/:"
        for f in $FILTERED_FILES; do echo "       $(basename $f)"; done
        echo "       To re-filter, delete these files first."
        STEP0_SKIPPED="true"
    fi

    if [ -z "$STEP0_SKIPPED" ]; then

    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_voicebench_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_voicebench_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            run_cmd bash 0_filter_voicebench_text_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM (baseline - no distance filtering)"
            run_cmd bash 0_filter_voicebench_random.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_spoken_squad_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_spoken_squad_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            run_cmd bash 0_filter_spoken_squad_text_semantic.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "librispeech" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_librispeech_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_librispeech_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            run_cmd bash 0_filter_librispeech_text_semantic.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "heysquad" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_heysquad_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_heysquad_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for HeySQuAD"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_heysquad_accents_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_heysquad_accents_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for HeySQuAD Accents"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_gammacorpus_accents_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_gammacorpus_accents_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for GammaCorpus Accents"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "gammacorpus_usa" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_gammacorpus_usa_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_gammacorpus_usa_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for GammaCorpus USA"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "benign_instructions_usa" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_benign_instructions_usa_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_benign_instructions_usa_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for Benign Instructions USA"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "benign_instructions_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_benign_instructions_accents_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_benign_instructions_accents_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for Benign Instructions Accents"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "mmsu" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_mmsu_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_mmsu_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            run_cmd bash 0_filter_mmsu_text_semantic.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "bbh" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_bbh_acoustic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_bbh_audio_semantic.sh $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            run_cmd bash 0_filter_bbh_text_semantic.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "voicebench_cafe" ] || [ "$BENIGN_DATASET" = "voicebench_traffic" ]; then
        # Extract noise type from dataset name (e.g., voicebench_cafe -> cafe)
        NOISE_TYPE="${BENIGN_DATASET#voicebench_}"
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            run_cmd bash 0_filter_voicebench_noise_acoustic.sh --noise_type "$NOISE_TYPE" $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (WhisperVQEncoder semantic from audio)"
            run_cmd bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type "$NOISE_TYPE" $FILTER_ARGS
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers semantic from text)"
            echo "Warning: text_semantic filter not yet implemented for VoiceBench Noisy"
            exit 1
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    fi

    fi  # end of STEP0_SKIPPED check

    if [ -z "$STEP0_SKIPPED" ]; then
        echo ""
        echo "Step 0 completed successfully!"
    fi
    echo ""
else
    echo "Skipping Step 0 (filtering)..."
fi

# ============================================
# Step 1: Extract semantic codes
# ============================================
if [ -z "$SKIP_EXTRACT" ]; then
    echo "=============================================="
    echo "STEP 1: Extracting semantic codes"
    echo "=============================================="

    # Auto-detect: skip if semantic codes file already exists
    SEMANTIC_CODES_FILES=$(ls -1 "${DATA_DIR}"/*"${MODE_DESC}"*_semantic_codes.jsonl 2>/dev/null || true)
    if [ -n "$SEMANTIC_CODES_FILES" ]; then
        echo "[SKIP] Semantic codes already exist in ${DATA_DIR}/:"
        for f in $SEMANTIC_CODES_FILES; do echo "       $(basename $f)"; done
        echo "       To re-extract, delete these files first."
    else
        run_cmd bash 1_extract_semantic_tokens.sh \
            --filter_type "$FILTER_TYPE" \
            --benign_dataset "$BENIGN_DATASET" \
            $FILTER_ARGS

        echo ""
        echo "Step 1 completed successfully!"
    fi
    echo ""
else
    echo "Skipping Step 1 (semantic code extraction)..."
fi

# ============================================
# Step 2: Finetune LoRA models
# ============================================
if [ -z "$SKIP_FINETUNE" ]; then
    echo "=============================================="
    echo "STEP 2: Finetuning LoRA models"
    echo "=============================================="

    # Determine LoRA adapter path
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            LORA_OUTPUT_BASE="output_acoustic"
        else
            LORA_OUTPUT_BASE="output_acoustic_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            LORA_OUTPUT_BASE="output"
        else
            LORA_OUTPUT_BASE="output_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "random" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            LORA_OUTPUT_BASE="output_random"
        else
            LORA_OUTPUT_BASE="output_random_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "text_semantic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            LORA_OUTPUT_BASE="output_semantic"
        else
            LORA_OUTPUT_BASE="output_semantic_${BENIGN_DATASET}"
        fi
    else
        LORA_OUTPUT_BASE="output"
    fi

    LORA_ADAPTER_PATH="${LORA_OUTPUT_BASE}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}"

    # Check if LoRA adapter already exists
    if check_lora_adapter_exists "$LORA_ADAPTER_PATH"; then
        echo "[SKIP] LoRA adapter already exists: $LORA_ADAPTER_PATH"
        echo "       To re-train, delete this directory first."
    else
        echo "LoRA adapter not found at: $LORA_ADAPTER_PATH"
        echo "Running finetuning..."
        run_cmd bash 3_finetune_lora.sh \
            --filter_type "$FILTER_TYPE" \
            --benign_dataset "$BENIGN_DATASET" \
            --epochs "$NUM_EPOCHS" \
            $FILTER_ARGS

        echo ""
        echo "Step 2 completed successfully!"
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

    # Determine merged model path
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            MERGE_OUTPUT_BASE="output_acoustic"
        else
            MERGE_OUTPUT_BASE="output_acoustic_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            MERGE_OUTPUT_BASE="output"
        else
            MERGE_OUTPUT_BASE="output_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "random" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            MERGE_OUTPUT_BASE="output_random"
        else
            MERGE_OUTPUT_BASE="output_random_${BENIGN_DATASET}"
        fi
    elif [ "$FILTER_TYPE" = "text_semantic" ]; then
        if [ "$BENIGN_DATASET" = "voicebench" ]; then
            MERGE_OUTPUT_BASE="output_semantic"
        else
            MERGE_OUTPUT_BASE="output_semantic_${BENIGN_DATASET}"
        fi
    else
        MERGE_OUTPUT_BASE="output"
    fi

    MERGED_MODEL_PATH="${PROJECT_MODELS}/Kimi-Audio_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}_merged"
    MERGED_MODEL_PATH_LOCAL="${MERGE_OUTPUT_BASE}/finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}_merged"

    # Check if merged model already exists (check both /project and local)
    if check_merged_model_exists "$MERGED_MODEL_PATH" || check_merged_model_exists "$MERGED_MODEL_PATH_LOCAL"; then
        echo "[SKIP] Merged model already exists: $MERGED_MODEL_PATH"
        echo "       To re-merge, delete this directory first."
    else
        echo "Merged model not found at: $MERGED_MODEL_PATH"
        echo "Running LoRA merge..."
        run_cmd bash 5_merge_lora_for_inference.sh \
            --filter_type "$FILTER_TYPE" \
            --benign_dataset "$BENIGN_DATASET" \
            --num_epochs "$NUM_EPOCHS" \
            $FILTER_ARGS

        echo ""
        echo "Step 3 completed successfully!"
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

    # Function to check and run evaluation for a single dataset
    check_and_run_eval() {
        local eval_dataset="$1"
        local eval_dir="response_log/${eval_dataset}_eval"
        local expected_total=$(get_expected_total "$eval_dataset")

        # Get model name for file matching
        local model_name="finetuned_lora_${BENIGN_DATASET}_${FILTER_TYPE}_${MODE_DESC}_epoch_${NUM_EPOCHS}_merged"

        echo ""
        echo "--- Checking $eval_dataset evaluation ---"
        echo "Expected total samples: $expected_total"

        # Find the most recent evaluation file for this model
        local latest_file=$(find_latest_eval_file "$eval_dir" "$model_name" "$eval_dataset")

        if [ -n "$latest_file" ] && [ -f "$latest_file" ]; then
            local current_count=$(count_responses_in_json "$latest_file")
            echo "Found existing results: $latest_file"
            echo "Current responses: $current_count / $expected_total"

            if [ "$current_count" -ge "$expected_total" ]; then
                echo "[SKIP] Evaluation complete for $eval_dataset ($current_count/$expected_total responses)"
                echo "       File: $latest_file"
                return 0
            else
                echo "[RESUME] Incomplete evaluation for $eval_dataset ($current_count/$expected_total responses)"
                echo "         Will continue from sample $current_count"
                echo "         Resuming from file: $latest_file"

                # Run evaluation with resume flag
                run_cmd bash 6_evaluate_harmful_audio.sh \
                    --filter_type "$FILTER_TYPE" \
                    --benign_dataset "$BENIGN_DATASET" \
                    --dataset "$eval_dataset" \
                    --num_epochs "$NUM_EPOCHS" \
                    --resume_from "$latest_file" \
                    $FILTER_ARGS
                return $?
            fi
        else
            echo "No existing results found for $eval_dataset"
            echo "Starting fresh evaluation..."

            # Run evaluation from scratch
            run_cmd bash 6_evaluate_harmful_audio.sh \
                --filter_type "$FILTER_TYPE" \
                --benign_dataset "$BENIGN_DATASET" \
                --dataset "$eval_dataset" \
                --num_epochs "$NUM_EPOCHS" \
                $FILTER_ARGS
            return $?
        fi
    }

    if [ "$DATASET" = "both" ]; then
        check_and_run_eval "advbench"
        check_and_run_eval "safetybench"
    else
        check_and_run_eval "$DATASET"
    fi

    echo ""
    echo "Step 4 completed successfully!"
    echo ""
else
    echo "Skipping Step 4 (evaluation)..."
fi

# ============================================
# Step 5: Analyze results
# ============================================
if [ -z "$SKIP_ANALYZE" ]; then
    echo "=============================================="
    echo "STEP 5: Analyzing results"
    echo "=============================================="

    if [ "$DATASET" = "both" ]; then
        run_cmd python analyze_results.py \
            --response_dir "response_log/advbench_eval"
        run_cmd python analyze_results.py \
            --response_dir "response_log/safetybench_eval"
    else
        run_cmd python analyze_results.py \
            --response_dir "response_log/${DATASET}_eval"
    fi

    echo ""
    echo "Step 5 completed successfully!"
    echo ""
else
    echo "Skipping Step 5 (analysis)..."
fi

# ============================================
# Step 6: Run ASR evaluation
# ============================================
if [ -z "$SKIP_ASR" ]; then
    echo "=============================================="
    echo "STEP 6: Running ASR Evaluation"
    echo "=============================================="

    ASR_OUTPUT_ARGS=""

    if [ "$DATASET" = "both" ]; then
        echo "Running ASR evaluation for advbench..."
        run_cmd bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset advbench \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET" \
            $ASR_OUTPUT_ARGS

        echo ""
        echo "Running ASR evaluation for safetybench..."
        run_cmd bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset safetybench \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET" \
            $ASR_OUTPUT_ARGS
    else
        run_cmd bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset "$DATASET" \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET" \
            $ASR_OUTPUT_ARGS
    fi

    echo ""
    echo "Step 6 completed successfully!"
    echo ""
else
    echo "Skipping Step 6 (ASR evaluation)..."
fi

echo "=============================================="
echo "       PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Benign dataset: $BENIGN_DATASET"
echo "  Filter type: $FILTER_TYPE"
echo "  Mode: $MODE_DESC"
echo "  Epochs: $NUM_EPOCHS"
echo "  Eval dataset: $DATASET"
echo ""
echo "Data saved to: $DATA_DIR/"
echo "Model saved to: $OUTPUT_DIR/"
echo "Response logs: response_log/${DATASET}_eval/"
echo "ASR results saved to: ../../asr_results/Kimi-Audio/"
echo "=============================================="
