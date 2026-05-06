#!/bin/bash
# Step 1: Prepare filtered dataset for finetuning Audio Flamingo 3
#
# This script uses the pre-computed analysis file from step 0 to prepare
# the filtered dataset in HuggingFace format for training.
#
# BENIGN DATASETS SUPPORTED:
#   - voicebench (default):  VoiceBench SD-QA dataset
#   - spoken_squad:          Spoken-SQuAD dataset
#   - librispeech:           LibriSpeech dataset
#   - heysquad:              HeySQuAD dataset (human-spoken QA)
#
# FOUR FILTER TYPES SUPPORTED:
#   --filter_type audio_acoustic  -> data_acoustic[_dataset]/ -> data_acoustic[_dataset]/filtered/
#   --filter_type audio_semantic  -> data[_dataset]/ -> data[_dataset]/filtered/
#   --filter_type text_semantic   -> data_semantic[_dataset]/ -> data_semantic[_dataset]/filtered/
#   --filter_type random          -> data_random[_dataset]/ -> data_random[_dataset]/filtered/

set -e

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Default values
BENIGN_DATASET="voicebench"  # "voicebench", "spoken_squad", or "librispeech"
FILTER_TYPE="audio_acoustic"  # "audio_acoustic", "audio_semantic", or "text_semantic"
FILTER_CLOSEST="--filter_closest"  # Filter closest to harmful

# Default filtering mode (percentage)
THRESHOLD=""
PERCENTAGE="50"
NUM_SAMPLES=""
TOP_K=""

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
        --analysis_file)
            ANALYSIS_FILE="$2"
            shift 2
            ;;
        --source_json)
            SOURCE_JSON="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            PERCENTAGE=""
            NUM_SAMPLES=""
            TOP_K=""
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            PERCENTAGE=""
            THRESHOLD=""
            NUM_SAMPLES=""
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            THRESHOLD=""
            NUM_SAMPLES=""
            TOP_K=""
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            PERCENTAGE=""
            THRESHOLD=""
            TOP_K=""
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --filter_closest)
            FILTER_CLOSEST="--filter_closest"
            shift
            ;;
        --filter_benign)
            FILTER_CLOSEST="--filter_benign"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Prepare filtered dataset using pre-computed analysis."
            echo ""
            echo "Benign datasets:"
            echo "  --benign_dataset NAME   Benign dataset to use (default: voicebench)"
            echo "    voicebench     VoiceBench SD-QA dataset (default)"
            echo "    spoken_squad   Spoken-SQuAD dataset"
            echo "    librispeech    LibriSpeech dataset"
            echo ""
            echo "Filter types:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  -> data_acoustic[_dataset]/"
            echo "    audio_semantic  -> data[_dataset]/"
            echo "    text_semantic   -> data_semantic[_dataset]/"
            echo "    random          -> data_random[_dataset]/ (pre-filtered, no analysis needed)"
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --threshold VALUE       Distance threshold"
            echo "  --percentage VALUE      Keep top percentage of samples (e.g., 50 for 50%)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --top_k VALUE           Keep top-k samples"
            echo ""
            echo "Other options:"
            echo "  --analysis_file PATH    Path to analysis .npz file (auto-detected)"
            echo "  --source_json PATH      Path to original dataset JSON"
            echo "  --output_dir PATH       Output directory for filtered dataset"
            echo "  --filter_closest        Filter closest samples (default)"
            echo "  --filter_benign         Filter most benign (farthest) samples"
            echo "  -h, --help              Show this help message"
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

# Determine dataset suffix and file prefix
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    DATASET_SUFFIX=""
    FILE_PREFIX="voicebench_filtered"
else
    DATASET_SUFFIX="_${BENIGN_DATASET}"
    FILE_PREFIX="${BENIGN_DATASET}_filtered"
fi

# Set directories and paths based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
        DEFAULT_ANALYSIS_FILE="data_acoustic/voicebench/sd-qa/sd_qa_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/voicebench/sd-qa/sd_qa_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic/filtered_voicebench"
    elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
        DATA_DIR="data_acoustic_spoken_squad"
        DEFAULT_ANALYSIS_FILE="data_acoustic_spoken_squad/spoken_squad_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/spoken_squad/spoken_squad_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_spoken_squad/filtered"
    elif [ "$BENIGN_DATASET" = "librispeech" ]; then
        DATA_DIR="data_acoustic_librispeech"
        DEFAULT_ANALYSIS_FILE="data_acoustic_librispeech/librispeech_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/librispeech/librispeech_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_librispeech/filtered"
    elif [ "$BENIGN_DATASET" = "heysquad" ]; then
        DATA_DIR="data_acoustic_heysquad"
        DEFAULT_ANALYSIS_FILE="data_acoustic_heysquad/heysquad_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/heysquad/heysquad_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_heysquad/filtered"
    elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
        DATA_DIR="data_acoustic_heysquad_accents"
        DEFAULT_ANALYSIS_FILE="data_acoustic_heysquad_accents/heysquad_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/heysquad_accents/heysquad_accents_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_heysquad_accents/filtered"
    elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
        DATA_DIR="data_acoustic_gammacorpus_accents"
        DEFAULT_ANALYSIS_FILE="data_acoustic_gammacorpus_accents/gammacorpus_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/gammacorpus_accents/gammacorpus_accents_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_gammacorpus_accents/filtered"
    elif [ "$BENIGN_DATASET" = "mmsu" ]; then
        DATA_DIR="data_acoustic_mmsu"
        DEFAULT_ANALYSIS_FILE="data_acoustic_mmsu/mmsu_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/mmsu/mmsu_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_mmsu/filtered"
    elif [ "$BENIGN_DATASET" = "bbh" ]; then
        DATA_DIR="data_acoustic_bbh"
        DEFAULT_ANALYSIS_FILE="data_acoustic_bbh/bbh_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/bbh/bbh_full.json"
        DEFAULT_OUTPUT_DIR="data_acoustic_bbh/filtered"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
        DEFAULT_ANALYSIS_FILE="data/voicebench/sd-qa/sd_qa_closest_to_advbench_analysis.npz"
        DEFAULT_SOURCE_JSON="data/voicebench/sd-qa/sd_qa_full.json"
        DEFAULT_OUTPUT_DIR="data/filtered_voicebench"
    elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
        DATA_DIR="data_spoken_squad"
        DEFAULT_ANALYSIS_FILE="data/spoken_squad/spoken_squad_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/spoken_squad/spoken_squad_full.json"
        DEFAULT_OUTPUT_DIR="data_spoken_squad/filtered"
    elif [ "$BENIGN_DATASET" = "librispeech" ]; then
        DATA_DIR="data_librispeech"
        DEFAULT_ANALYSIS_FILE="data/librispeech/librispeech_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/librispeech/librispeech_full.json"
        DEFAULT_OUTPUT_DIR="data_librispeech/filtered"
    elif [ "$BENIGN_DATASET" = "heysquad" ]; then
        DATA_DIR="data_heysquad"
        DEFAULT_ANALYSIS_FILE="data_heysquad/heysquad_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/heysquad/heysquad_full.json"
        DEFAULT_OUTPUT_DIR="data_heysquad/filtered"
    elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
        DATA_DIR="data_heysquad_accents"
        DEFAULT_ANALYSIS_FILE="data_heysquad_accents/heysquad_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/heysquad_accents/heysquad_accents_full.json"
        DEFAULT_OUTPUT_DIR="data_heysquad_accents/filtered"
    elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
        DATA_DIR="data_gammacorpus_accents"
        DEFAULT_ANALYSIS_FILE="data_gammacorpus_accents/gammacorpus_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/gammacorpus_accents/gammacorpus_accents_full.json"
        DEFAULT_OUTPUT_DIR="data_gammacorpus_accents/filtered"
    elif [ "$BENIGN_DATASET" = "mmsu" ]; then
        DATA_DIR="data_mmsu"
        DEFAULT_ANALYSIS_FILE="data/mmsu/mmsu_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/mmsu/mmsu_full.json"
        DEFAULT_OUTPUT_DIR="data_mmsu/filtered"
    elif [ "$BENIGN_DATASET" = "bbh" ]; then
        DATA_DIR="data_bbh"
        DEFAULT_ANALYSIS_FILE="data/bbh/bbh_audio_semantic_analysis.npz"
        DEFAULT_SOURCE_JSON="data/bbh/bbh_full.json"
        DEFAULT_OUTPUT_DIR="data_bbh/filtered"
    else
        echo "Error: audio_semantic filter not yet implemented for $BENIGN_DATASET"
        exit 1
    fi
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
        # For text_semantic, find analysis file based on filtering mode
        # Append _safest suffix when using --filter_benign (safest mode)
        SAFEST_SUFFIX=""
        if [ "$FILTER_CLOSEST" = "--filter_benign" ]; then
            SAFEST_SUFFIX="_safest"
        fi
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic/voicebench/sd-qa/sd_qa_text_semantic_n${NUM_SAMPLES}${SAFEST_SUFFIX}_analysis.npz"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic/voicebench/sd-qa/sd_qa_text_semantic_percentage_${PERCENTAGE}${SAFEST_SUFFIX}_analysis.npz"
        else
            DEFAULT_ANALYSIS_FILE="data_semantic/voicebench/sd-qa/sd_qa_text_semantic_percentage_50${SAFEST_SUFFIX}_analysis.npz"
        fi
        DEFAULT_SOURCE_JSON="data/voicebench/sd-qa/sd_qa_full.json"
        DEFAULT_OUTPUT_DIR="data_semantic/filtered_voicebench"
    elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
        DATA_DIR="data_semantic_spoken_squad"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic_spoken_squad/spoken_squad_text_semantic_n${NUM_SAMPLES}_analysis.npz"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic_spoken_squad/spoken_squad_text_semantic_percentage_${PERCENTAGE}_analysis.npz"
        else
            DEFAULT_ANALYSIS_FILE="data_semantic_spoken_squad/spoken_squad_text_semantic_percentage_50_analysis.npz"
        fi
        DEFAULT_SOURCE_JSON="data/spoken_squad/spoken_squad_full.json"
        DEFAULT_OUTPUT_DIR="data_semantic_spoken_squad/filtered"
    elif [ "$BENIGN_DATASET" = "librispeech" ]; then
        DATA_DIR="data_semantic_librispeech"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic_librispeech/librispeech_text_semantic_n${NUM_SAMPLES}_analysis.npz"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_ANALYSIS_FILE="data_semantic_librispeech/librispeech_text_semantic_percentage_${PERCENTAGE}_analysis.npz"
        else
            DEFAULT_ANALYSIS_FILE="data_semantic_librispeech/librispeech_text_semantic_percentage_50_analysis.npz"
        fi
        DEFAULT_SOURCE_JSON="data/librispeech/librispeech_full.json"
        DEFAULT_OUTPUT_DIR="data_semantic_librispeech/filtered"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    # Random filter: data is already filtered, no analysis file needed
    SKIP_ANALYSIS="true"
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
        # For random, the filtered JSON is the source (already filtered by 0_filter_random.py)
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/voicebench/sd-qa/sd_qa_random_n${NUM_SAMPLES}_seed42.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/voicebench/sd-qa/sd_qa_random_percentage_${PERCENTAGE}_seed42.json"
        else
            DEFAULT_SOURCE_JSON="data_random/voicebench/sd-qa/sd_qa_random_percentage_50_seed42.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random/filtered_voicebench"
    elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
        DATA_DIR="data_random_gammacorpus_accents"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/gammacorpus_accents/gammacorpus_accents_random_n${NUM_SAMPLES}.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/gammacorpus_accents/gammacorpus_accents_random_percentage_${PERCENTAGE}.json"
        else
            DEFAULT_SOURCE_JSON="data_random/gammacorpus_accents/gammacorpus_accents_random_percentage_50.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random_gammacorpus_accents/filtered"
    elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
        DATA_DIR="data_random_heysquad_accents"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/heysquad_accents/heysquad_accents_random_n${NUM_SAMPLES}.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/heysquad_accents/heysquad_accents_random_percentage_${PERCENTAGE}.json"
        else
            DEFAULT_SOURCE_JSON="data_random/heysquad_accents/heysquad_accents_random_percentage_50.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random_heysquad_accents/filtered"
    elif [ "$BENIGN_DATASET" = "mmsu" ]; then
        DATA_DIR="data_random_mmsu"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/mmsu/mmsu_random_n${NUM_SAMPLES}.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/mmsu/mmsu_random_percentage_${PERCENTAGE}.json"
        else
            DEFAULT_SOURCE_JSON="data_random/mmsu/mmsu_random_percentage_50.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random_mmsu/filtered"
    elif [ "$BENIGN_DATASET" = "bbh" ]; then
        DATA_DIR="data_random_bbh"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/bbh/bbh_random_n${NUM_SAMPLES}.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/bbh/bbh_random_percentage_${PERCENTAGE}.json"
        else
            DEFAULT_SOURCE_JSON="data_random/bbh/bbh_random_percentage_50.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random_bbh/filtered"
    else
        # Generic fallback for other datasets
        DATA_DIR="data_random_${BENIGN_DATASET}"
        if [ -n "$NUM_SAMPLES" ]; then
            DEFAULT_SOURCE_JSON="data_random/${BENIGN_DATASET}/${BENIGN_DATASET}_random_n${NUM_SAMPLES}.json"
        elif [ -n "$PERCENTAGE" ]; then
            DEFAULT_SOURCE_JSON="data_random/${BENIGN_DATASET}/${BENIGN_DATASET}_random_percentage_${PERCENTAGE}.json"
        else
            DEFAULT_SOURCE_JSON="data_random/${BENIGN_DATASET}/${BENIGN_DATASET}_random_percentage_50.json"
        fi
        DEFAULT_OUTPUT_DIR="data_random_${BENIGN_DATASET}/filtered"
    fi
    DEFAULT_ANALYSIS_FILE=""  # Not needed for random
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'."
    echo "Valid options: audio_acoustic, audio_semantic, text_semantic, random"
    exit 1
fi

# Use provided or default values
ANALYSIS_FILE="${ANALYSIS_FILE:-$DEFAULT_ANALYSIS_FILE}"
SOURCE_JSON="${SOURCE_JSON:-$DEFAULT_SOURCE_JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

# Check if analysis file exists (skip for random and pre-filtered types)
if [ "$FILTER_TYPE" != "random" ] && [ "$SKIP_ANALYSIS" != "true" ]; then
    if [ ! -f "$ANALYSIS_FILE" ]; then
        echo "Error: Analysis file not found: $ANALYSIS_FILE"
        echo ""
        echo "Please run the appropriate filtering script first."
        exit 1
    fi
else
    # For random filter, check if the pre-filtered JSON exists
    if [ ! -f "$SOURCE_JSON" ]; then
        echo "Error: Pre-filtered random JSON not found: $SOURCE_JSON"
        echo ""
        echo "Please run the random filtering script first:"
        echo "  bash 0_filter_voicebench_random.sh --percentage $PERCENTAGE"
        exit 1
    fi
fi

# Determine mode description
if [ -n "$NUM_SAMPLES" ]; then
    MODE_DESC="num_samples=${NUM_SAMPLES}"
elif [ -n "$PERCENTAGE" ]; then
    MODE_DESC="percentage=${PERCENTAGE}%"
elif [ -n "$TOP_K" ]; then
    MODE_DESC="top_k=${TOP_K}"
elif [ -n "$THRESHOLD" ]; then
    MODE_DESC="threshold=${THRESHOLD}"
else
    echo "Error: Must specify --threshold, --percentage, --num_samples, or --top_k"
    exit 1
fi

echo "============================================"
echo "Prepare Filtered Dataset"
echo "============================================"
echo "Benign dataset: $BENIGN_DATASET"
echo "Filter type:    $FILTER_TYPE"
echo "Data directory: $DATA_DIR"
echo "Analysis file:  $ANALYSIS_FILE"
echo "Source JSON:    $SOURCE_JSON"
echo "Mode:           $MODE_DESC"
echo "Output dir:     $OUTPUT_DIR"
echo "============================================"

# Build and run command
if [ "$FILTER_TYPE" = "random" ] || [ "$SKIP_ANALYSIS" = "true" ]; then
    # For random/pre-filtered types, the JSON is already filtered - just prepare for training format
    echo "Using pre-filtered JSON directly"
    CMD="python 1_prepare_filtered_dataset.py"
    CMD="$CMD --voicebench_json $SOURCE_JSON"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --file_prefix $FILE_PREFIX"
    CMD="$CMD --skip_analysis"  # Skip analysis file requirement

    echo "Running: $CMD"
    $CMD
else
    # For distance-based filters, use analysis file
    CMD="python 1_prepare_filtered_dataset.py"
    CMD="$CMD --analysis_file $ANALYSIS_FILE"
    CMD="$CMD --voicebench_json $SOURCE_JSON"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --file_prefix $FILE_PREFIX"

    if [ -n "$THRESHOLD" ]; then
        CMD="$CMD --threshold $THRESHOLD"
    fi

    if [ -n "$TOP_K" ]; then
        CMD="$CMD --top_k $TOP_K"
    fi

    if [ -n "$PERCENTAGE" ]; then
        CMD="$CMD --percentage $PERCENTAGE"
    fi

    if [ -n "$NUM_SAMPLES" ]; then
        CMD="$CMD --num_samples $NUM_SAMPLES"
    fi

    if [ -n "$FILTER_CLOSEST" ]; then
        CMD="$CMD $FILTER_CLOSEST"
    fi

    echo "Running: $CMD"
    $CMD
fi

echo ""
echo "============================================"
echo "Dataset preparation complete!"
echo "============================================"
echo ""
echo "Next step: Run 2_finetune_flamingo3.sh"
echo "  bash 2_finetune_flamingo3.sh --filter_type $FILTER_TYPE --benign_dataset $BENIGN_DATASET --percentage $PERCENTAGE"
