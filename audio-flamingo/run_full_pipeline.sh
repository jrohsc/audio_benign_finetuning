#!/bin/bash
#
# Full Pipeline: Filter -> Prepare -> Finetune -> Evaluate
#
# This script runs the complete benign finetuning pipeline for Audio Flamingo 3:
#   0. Filter benign dataset samples by embedding distance
#   1. Prepare filtered dataset in HuggingFace format
#   2. Finetune Audio Flamingo 3 on the filtered dataset
#   3. Evaluate the finetuned model on harmful audio datasets
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset (speech QA)
#   - spoken_squad:          Spoken-SQuAD dataset (speech QA)
#   - librispeech:           LibriSpeech dataset (transcription task)
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#   - heysquad_accents:      HeySQuAD with 11 accent variations (TTS-generated)
#   - mmsu:                  VoiceBench MMSU (Multi-Subject Understanding) ~3k samples
#   - bbh:                   VoiceBench BBH (BIG-Bench Hard) ~1k samples
#
# FOUR FILTER TYPES SUPPORTED:
#   - audio_acoustic: TRUE ACOUSTIC features (Whisper-Large-V3 only)
#                     Captures HOW audio sounds (voice, prosody, timbre)
#
#   - audio_semantic: AUDIO SEMANTIC (AF3's encoder + projector)
#                     Captures how model "understands" audio content
#
#   - text_semantic:  TEXT SEMANTIC (sentence-transformers)
#                     Captures text meaning similarity
#
#   - random:         RANDOM BASELINE (no distance filtering)
#                     Randomly samples k% of data for comparison
#
# OUTPUT DIRECTORIES:
#   VoiceBench (default):    data_acoustic/, checkpoints_acoustic/, results_acoustic/
#   Spoken-SQuAD:            data_acoustic_spoken_squad/, checkpoints_acoustic_spoken_squad/, etc.
#   LibriSpeech:             data_acoustic_librispeech/, checkpoints_acoustic_librispeech/, etc.
#
# Usage:
#   ./run_full_pipeline.sh                                                    # Defaults (voicebench, audio_acoustic, 50%)
#   ./run_full_pipeline.sh --benign_dataset spoken_squad --percentage 50      # Spoken-SQuAD
#   ./run_full_pipeline.sh --benign_dataset librispeech --percentage 50       # LibriSpeech
#   ./run_full_pipeline.sh --filter_type text_semantic --percentage 25        # Text semantic filtering

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
BENIGN_DATASET="mmsu"  # "gammacorpus_accents", "voicebench", "spoken_squad", or "librispeech", "mmsu", "bbh", "heysquad"
FILTER_TYPE="audio_semantic"  # "audio_acoustic", "audio_semantic", or "text_semantic", "random"
PERCENTAGE="50"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=3
LEARNING_RATE="2e-5"  # Learning rate for finetuning
DATASET="both"  # advbench, safetybench, or both for evaluation
SKIP_FILTER=""
SKIP_PREPARE=""
SKIP_FINETUNE=""
SKIP_EVALUATE=""
SKIP_ASR=""
RUN_ASR="true"  # Run ASR evaluation by default
STORAGE="project"  # "work" or "project" - controls where models are saved
SELECT_SAFEST=""  # Set to "true" to select SAFEST (furthest from harmful) samples instead of closest
USE_LORA="true"  # Use LoRA for memory-efficient finetuning (required for L40S)
USE_AF_THINK=""  # Set to "true" to load AF-Think adapter before finetuning

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
        --learning_rate|--lr)
            LEARNING_RATE="$2"
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
        --skip_prepare)
            SKIP_PREPARE="true"
            shift
            ;;
        --skip_finetune)
            SKIP_FINETUNE="true"
            shift
            ;;
        --skip_evaluate)
            SKIP_EVALUATE="true"
            shift
            ;;
        --skip_asr)
            SKIP_ASR="true"
            RUN_ASR=""
            shift
            ;;
        --storage)
            STORAGE="$2"
            shift 2
            ;;
        --select_safest)
            SELECT_SAFEST="true"
            shift
            ;;
        --no_select_safest)
            SELECT_SAFEST=""
            shift
            ;;
        --use_lora)
            USE_LORA="true"
            shift
            ;;
        --no_lora)
            USE_LORA=""
            shift
            ;;
        --use_af_think)
            USE_AF_THINK="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run the full benign finetuning pipeline for Audio Flamingo 3."
            echo ""
            echo "Pipeline steps:"
            echo "  0. Filter benign dataset samples by embedding distance"
            echo "  1. Prepare filtered dataset in HuggingFace format"
            echo "  2. Finetune Audio Flamingo 3"
            echo "  3. Evaluate on harmful audio datasets"
            echo "  4. Run ASR evaluation"
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo "    voicebench           VoiceBench SD-QA dataset (speech QA) - DEFAULT"
            echo "    spoken_squad         Spoken-SQuAD dataset (speech QA)"
            echo "    librispeech          LibriSpeech dataset (transcription task)"
            echo "    heysquad             HeySQuAD dataset (human-spoken QA)"
            echo "    heysquad_accents     HeySQuAD with 11 accent variations (TTS)"
            echo "    gammacorpus_accents  GammaCorpus-Fact-QA with 11 accent variations (TTS)"
            echo "    mmsu                 VoiceBench MMSU (~3k multi-subject QA samples)"
            echo "    bbh                  VoiceBench BBH (~1k reasoning samples)"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  TRUE ACOUSTIC features (Whisper-Large-V3 only)"
            echo "    audio_semantic  AUDIO SEMANTIC (AF3's encoder + projector)"
            echo "    text_semantic   TEXT SEMANTIC (sentence-transformers)"
            echo "    random          RANDOM BASELINE (no distance filtering)"
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --percentage VALUE      Keep top percentage of closest samples (default: 50)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --threshold VALUE       Distance threshold"
            echo ""
            echo "Training options:"
            echo "  --num_epochs N          Number of training epochs (default: 3)"
            echo "  --learning_rate LR      Learning rate for finetuning (default: 2e-5)"
            echo "  --lr LR                 Alias for --learning_rate"
            echo ""
            echo "Evaluation options:"
            echo "  --dataset NAME          Evaluation dataset: 'advbench', 'safetybench', or 'both' (default: both)"
            echo ""
            echo "Data selection:"
            echo "  --select_safest         Select SAFEST samples (furthest from harmful) instead of closest (default: on)"
            echo "  --no_select_safest      Select CLOSEST samples to harmful (disable safest mode)"
            echo ""
            echo "Storage options:"
            echo "  --storage DIR           Root storage: 'work' or 'project' (default: project)"
            echo "                          Models saved to /DIR/anon/BFT_models/"
            echo ""
            echo "Skip options (to resume from a specific step):"
            echo "  --skip_filter           Skip step 0 (filtering)"
            echo "  --skip_prepare          Skip step 1 (dataset preparation)"
            echo "  --skip_finetune         Skip step 2 (finetuning)"
            echo "  --skip_evaluate         Skip step 3 (evaluation)"
            echo "  --skip_asr              Skip step 4 (ASR evaluation)"
            echo ""
            echo "Examples:"
            echo "  $0                                                         # Default: voicebench, audio_acoustic, 50%, lr=2e-5"
            echo "  $0 --benign_dataset spoken_squad --percentage 50           # Spoken-SQuAD at 50%"
            echo "  $0 --benign_dataset librispeech --percentage 50            # LibriSpeech at 50%"
            echo "  $0 --filter_type text_semantic --percentage 25             # Text semantic filtering"
            echo "  $0 --learning_rate 5e-5 --percentage 50                    # Higher learning rate"
            echo "  $0 --lr 1e-4 --filter_type audio_acoustic                  # Aggressive LR with acoustic"
            echo "  $0 --storage work --percentage 50                             # Save models to /work/..."
            echo "  $0 --skip_filter --skip_prepare                            # Resume from finetuning"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Validate storage option and compute model base directory
if [ "$STORAGE" != "work" ] && [ "$STORAGE" != "project" ]; then
    echo "Error: Invalid storage '$STORAGE'. Must be 'work' or 'project'."
    exit 1
fi
if [ "$STORAGE" = "project" ]; then
    MODEL_BASE_DIR="/project/anon/BFT_models"
else
    MODEL_BASE_DIR="/work/anon/audio_benign_finetuning/audio-flamingo/checkpoints"
fi

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

# Determine directory suffix based on benign dataset
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    DATASET_SUFFIX=""
else
    DATASET_SUFFIX="_${BENIGN_DATASET}"
fi

# Determine data and checkpoint directories based on filter type and benign dataset
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
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
        CHECKPOINT_DIR="checkpoints_semantic"
        RESULTS_DIR="results_semantic"
    else
        DATA_DIR="data_semantic_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_semantic_${BENIGN_DATASET}"
        RESULTS_DIR="results_semantic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        CHECKPOINT_DIR="checkpoints_random"
        RESULTS_DIR="results_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
        CHECKPOINT_DIR="checkpoints_random_${BENIGN_DATASET}"
        RESULTS_DIR="results_random_${BENIGN_DATASET}"
    fi
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'"
    echo "Valid options: audio_acoustic, audio_semantic, text_semantic, random"
    exit 1
fi

# Format learning rate for directory name (replace scientific notation)
# e.g., 2e-5 -> 2e-5, 1e-4 -> 1e-4 (keep as-is, safe for directory names)
LR_SUFFIX="_lr${LEARNING_RATE}"

# Append learning rate to checkpoint and results directories
CHECKPOINT_DIR="${CHECKPOINT_DIR}${LR_SUFFIX}"
RESULTS_DIR="${RESULTS_DIR}${LR_SUFFIX}"

# Append _safest suffix if selecting safest samples
if [ "$SELECT_SAFEST" = "true" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR}_safest"
    RESULTS_DIR="${RESULTS_DIR}_safest"
fi

# Build select_safest argument for sub-scripts
SAFEST_ARG=""
if [ "$SELECT_SAFEST" = "true" ]; then
    SAFEST_ARG="--select_safest"
fi

echo "=============================================="
echo "       AUDIO FLAMINGO 3 FULL PIPELINE"
echo "=============================================="
echo "Benign dataset:   $BENIGN_DATASET"
echo "Filter type:      $FILTER_TYPE"
echo "Mode:             $MODE_DESC"
echo "Select safest:    ${SELECT_SAFEST:-false}"
echo "Training epochs:  $NUM_EPOCHS"
echo "Learning rate:    $LEARNING_RATE"
echo "Eval dataset:     $DATASET"
echo "Storage:          $STORAGE"
echo "Model base dir:   $MODEL_BASE_DIR"
echo "Data directory:   $DATA_DIR"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
echo "=============================================="
echo ""

# ============================================
# Step 0: Filter benign dataset samples
# ============================================
if [ -z "$SKIP_FILTER" ]; then
    echo "=============================================="
    echo "STEP 0: Filtering $BENIGN_DATASET samples"
    echo "=============================================="

    # Select the appropriate filter script based on benign_dataset and filter_type
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_voicebench_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_voicebench_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            bash 0_filter_voicebench_text_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            bash 0_filter_voicebench_random.sh $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_spoken_squad_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_spoken_squad_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            bash 0_filter_spoken_squad_text_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/spoken_squad/spoken_squad_full.json"
            OUTPUT_JSON="data_random/spoken_squad/spoken_squad_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "librispeech" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_librispeech_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_librispeech_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            bash 0_filter_librispeech_text_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/librispeech/librispeech_full.json"
            OUTPUT_JSON="data_random/librispeech/librispeech_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "heysquad" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_heysquad_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_heysquad_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for HeySQuAD"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/heysquad/heysquad_full.json"
            OUTPUT_JSON="data_random/heysquad/heysquad_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_heysquad_accents_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_heysquad_accents_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for HeySQuAD Accents"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/heysquad_accents/heysquad_accents_full.json"
            OUTPUT_JSON="data_random/heysquad_accents/heysquad_accents_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_gammacorpus_accents_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_gammacorpus_accents_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for GammaCorpus Accents"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/gammacorpus_accents/gammacorpus_accents_full.json"
            OUTPUT_JSON="data_random/gammacorpus_accents/gammacorpus_accents_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "gammacorpus_usa" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_gammacorpus_usa_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_gammacorpus_usa_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for GammaCorpus USA"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/gammacorpus_usa/gammacorpus_usa_full.json"
            OUTPUT_JSON="data_random/gammacorpus_usa/gammacorpus_usa_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "benign_instructions_usa" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_benign_instructions_usa_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_benign_instructions_usa_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for Benign Instructions USA"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/benign_instructions_usa/benign_instructions_usa_full.json"
            OUTPUT_JSON="data_random/benign_instructions_usa/benign_instructions_usa_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "benign_instructions_accents" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_benign_instructions_accents_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_benign_instructions_accents_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for Benign Instructions Accents"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/benign_instructions_accents/benign_instructions_accents_full.json"
            OUTPUT_JSON="data_random/benign_instructions_accents/benign_instructions_accents_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "mmsu" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_mmsu_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_mmsu_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            bash 0_filter_mmsu_text_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/mmsu/mmsu_full.json"
            OUTPUT_JSON="data_random/mmsu/mmsu_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "bbh" ]; then
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_bbh_audio_acoustic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_bbh_audio_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            bash 0_filter_bbh_text_semantic.sh $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/bbh/bbh_full.json"
            OUTPUT_JSON="data_random/bbh/bbh_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    elif [ "$BENIGN_DATASET" = "voicebench_cafe" ] || [ "$BENIGN_DATASET" = "voicebench_traffic" ]; then
        # Extract noise type from dataset name (e.g., voicebench_cafe -> cafe)
        NOISE_TYPE="${BENIGN_DATASET#voicebench_}"
        if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
            echo "Filter: Audio-Acoustic (Whisper-Large-V3 TRUE ACOUSTIC features)"
            bash 0_filter_voicebench_noise_audio_acoustic.sh --noise_type "$NOISE_TYPE" $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
            echo "Filter: Audio-Semantic (AF3's encoder + projector)"
            bash 0_filter_voicebench_noise_audio_semantic.sh --noise_type "$NOISE_TYPE" $FILTER_ARGS $SAFEST_ARG
        elif [ "$FILTER_TYPE" = "text_semantic" ]; then
            echo "Filter: Text-Semantic (sentence-transformers)"
            echo "Warning: text_semantic filter not yet implemented for VoiceBench Noisy"
            exit 1
        elif [ "$FILTER_TYPE" = "random" ]; then
            echo "Filter: RANDOM BASELINE (no distance filtering)"
            INPUT_JSON="data/voicebench_${NOISE_TYPE}/sd-qa/sd_qa_full.json"
            OUTPUT_JSON="data_random/voicebench_${NOISE_TYPE}/sd-qa/sd_qa_random_${MODE_DESC}.json"
            python 0_filter_random.py --input_json "$INPUT_JSON" --output_json "$OUTPUT_JSON" $FILTER_ARGS
        else
            echo "Error: Invalid filter_type '$FILTER_TYPE'"
            exit 1
        fi
    fi

    echo ""
    echo "Step 0 completed successfully!"
    echo ""
else
    echo "Skipping Step 0 (filtering)..."
fi

# ============================================
# Step 1: Prepare filtered dataset
# ============================================
if [ -z "$SKIP_PREPARE" ]; then
    echo "=============================================="
    echo "STEP 1: Preparing filtered dataset"
    echo "=============================================="

    # Use --filter_benign when selecting safest (furthest) samples
    PREPARE_FILTER_MODE=""
    if [ "$SELECT_SAFEST" = "true" ]; then
        PREPARE_FILTER_MODE="--filter_benign"
    fi

    bash 1_prepare_filtered_dataset.sh \
        --filter_type "$FILTER_TYPE" \
        --benign_dataset "$BENIGN_DATASET" \
        $FILTER_ARGS \
        $PREPARE_FILTER_MODE

    echo ""
    echo "Step 1 completed successfully!"
    echo ""
else
    echo "Skipping Step 1 (dataset preparation)..."
fi

# ============================================
# Step 2: Finetune Audio Flamingo 3
# ============================================
if [ -z "$SKIP_FINETUNE" ]; then
    echo "=============================================="
    echo "STEP 2: Finetuning Audio Flamingo 3"
    echo "=============================================="

    # Build LoRA argument
    LORA_ARG=""
    if [ "$USE_LORA" = "true" ]; then
        LORA_ARG="--use_lora"
    fi

    # Build AF-Think argument
    AF_THINK_ARG=""
    if [ "$USE_AF_THINK" = "true" ]; then
        AF_THINK_ARG="--use_af_think"
    fi

    bash 2_finetune_flamingo3.sh \
        --filter_type "$FILTER_TYPE" \
        --benign_dataset "$BENIGN_DATASET" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --model_base_dir "$MODEL_BASE_DIR" \
        $FILTER_ARGS \
        $SAFEST_ARG \
        $LORA_ARG \
        $AF_THINK_ARG

    echo ""
    echo "Step 2 completed successfully!"
    echo ""
else
    echo "Skipping Step 2 (finetuning)..."
fi

# ============================================
# Step 3: Evaluate on harmful datasets
# ============================================
if [ -z "$SKIP_EVALUATE" ]; then
    echo "=============================================="
    echo "STEP 3: Evaluating on $DATASET"
    echo "=============================================="

    bash 3_evaluate_jailbreaking.sh \
        --filter_type "$FILTER_TYPE" \
        --benign_dataset "$BENIGN_DATASET" \
        --dataset "$DATASET" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --model_base_dir "$MODEL_BASE_DIR" \
        $FILTER_ARGS \
        $SAFEST_ARG \
        $AF_THINK_ARG

    echo ""
    echo "Step 3 completed successfully!"
    echo ""
else
    echo "Skipping Step 3 (evaluation)..."
fi

# ============================================
# Step 4: Run ASR evaluation
# ============================================
if [ -z "$SKIP_ASR" ]; then
    echo "=============================================="
    echo "STEP 4: Running ASR Evaluation"
    echo "=============================================="

    if [ "$DATASET" = "both" ]; then
        echo "Running ASR evaluation for advbench..."
        bash ../run_asr_eval.sh \
            --model audio-flamingo \
            --dataset advbench \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET"

        echo ""
        echo "Running ASR evaluation for safetybench..."
        bash ../run_asr_eval.sh \
            --model audio-flamingo \
            --dataset safetybench \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET"
    else
        bash ../run_asr_eval.sh \
            --model audio-flamingo \
            --dataset "$DATASET" \
            --filter-type "$FILTER_TYPE" \
            --benign-dataset "$BENIGN_DATASET"
    fi

    echo ""
    echo "Step 4 completed successfully!"
    echo ""
else
    echo "Skipping Step 4 (ASR evaluation)..."
fi

echo "=============================================="
echo "       PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Benign dataset: $BENIGN_DATASET"
echo "  Filter type: $FILTER_TYPE"
echo "  Mode: $MODE_DESC"
echo "  Select safest: ${SELECT_SAFEST:-false}"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Eval dataset: $DATASET"
echo ""
echo "Data saved to: $DATA_DIR/"
echo "Checkpoint saved to: $CHECKPOINT_DIR/"
echo "Results saved to: $RESULTS_DIR/"
echo "ASR results saved to: ../asr_results/audio-flamingo/"
echo "=============================================="
