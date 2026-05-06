#!/bin/bash
# Step 1: Extract semantic codes for filtered VoiceBench data
#
# This script adds audio_tokens to each audio message in the filtered JSONL file.
# These tokens are required for training.
#
# THREE FILTER TYPES SUPPORTED:
#   --filter_type audio_acoustic  -> reads from data_acoustic/ folder
#   --filter_type audio_semantic  -> reads from data/ folder
#   --filter_type text_semantic   -> reads from data_semantic/ folder

set -e

# Load conda and CUDA environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio
module load cuda/12.6

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --filter_type)
            FILTER_TYPE="$2"
            shift 2
            ;;
        --benign_dataset)
            BENIGN_DATASET="$2"
            shift 2
            ;;
        --harmful_source)
            HARMFUL_SOURCE="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Extract semantic codes for filtered benign data."
            echo ""
            echo "Options:"
            echo "  --filter_type TYPE      Filter type (default: audio_acoustic)"
            echo "    audio_acoustic  -> reads from data_acoustic/ folder"
            echo "    audio_semantic  -> reads from data/ folder"
            echo "    text_semantic   -> reads from data_semantic/ folder"
            echo ""
            echo "  --benign_dataset NAME   Benign dataset: voicebench (default), spoken_squad, librispeech, heysquad, heysquad_accents, gammacorpus_accents, gammacorpus_usa, benign_instructions_usa, benign_instructions_accents, bbh, mmsu, voicebench_cafe, voicebench_traffic"
            echo "  --harmful_source NAME   Harmful source (advbench or safetybench)"
            echo "  --percentage VALUE      Percentage used in filtering"
            echo "  --threshold VALUE       Threshold used in filtering"
            echo "  --num_samples VALUE     Num samples used in filtering"
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
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine input/output directories based on filter type and benign dataset
if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_acoustic"
    else
        DATA_DIR="data_acoustic_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "audio_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data"
    else
        DATA_DIR="data_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "random" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_random"
    else
        DATA_DIR="data_random_${BENIGN_DATASET}"
    fi
elif [ "$FILTER_TYPE" = "text_semantic" ]; then
    if [ "$BENIGN_DATASET" = "voicebench" ]; then
        DATA_DIR="data_semantic"
    else
        DATA_DIR="data_semantic_${BENIGN_DATASET}"
    fi
else
    echo "Error: Invalid filter_type '$FILTER_TYPE'."
    echo "Valid options: audio_acoustic, audio_semantic, text_semantic, random"
    exit 1
fi

# Determine filename suffix based on filter type and parameters
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

# Determine input/output files based on benign dataset
if [ "$BENIGN_DATASET" = "voicebench" ]; then
    # VoiceBench naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    elif [ "$FILTER_TYPE" = "random" ]; then
        INPUT_FILE="${DATA_DIR}/voicebench_filtered_random_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/voicebench_filtered_random_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/voicebench_filtered_${HARMFUL_SOURCE}_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "spoken_squad" ]; then
    # Spoken-SQuAD naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/spoken_squad_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/spoken_squad_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/spoken_squad_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/spoken_squad_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "librispeech" ]; then
    # LibriSpeech naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/librispeech_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/librispeech_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/librispeech_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/librispeech_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "heysquad" ]; then
    # HeySQuAD naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/heysquad_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/heysquad_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/heysquad_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/heysquad_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "heysquad_accents" ]; then
    # HeySQuAD Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/heysquad_accents_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/heysquad_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/heysquad_accents_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/heysquad_accents_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "gammacorpus_accents" ]; then
    # GammaCorpus Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/gammacorpus_accents_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/gammacorpus_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/gammacorpus_accents_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/gammacorpus_accents_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "gammacorpus_usa" ]; then
    # GammaCorpus USA naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/gammacorpus_usa_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/gammacorpus_usa_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/gammacorpus_usa_filtered_audio_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/gammacorpus_usa_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "benign_instructions_usa" ]; then
    # Benign Instructions USA naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/benign_instructions_usa_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/benign_instructions_usa_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/benign_instructions_usa_filtered_audio_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/benign_instructions_usa_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "benign_instructions_accents" ]; then
    # Benign Instructions Accents naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/benign_instructions_accents_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/benign_instructions_accents_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/benign_instructions_accents_filtered_audio_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/benign_instructions_accents_filtered_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "bbh" ]; then
    # BBH naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/bbh_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/bbh_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    elif [ "$FILTER_TYPE" = "text_semantic" ]; then
        INPUT_FILE="${DATA_DIR}/bbh_filtered_text_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/bbh_filtered_text_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/bbh_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/bbh_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "mmsu" ]; then
    # MMSU naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/mmsu_filtered_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/mmsu_filtered_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    elif [ "$FILTER_TYPE" = "text_semantic" ]; then
        INPUT_FILE="${DATA_DIR}/mmsu_filtered_text_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/mmsu_filtered_text_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/mmsu_filtered_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/mmsu_filtered_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
elif [ "$BENIGN_DATASET" = "voicebench_cafe" ] || [ "$BENIGN_DATASET" = "voicebench_traffic" ]; then
    # VoiceBench Noisy variants naming pattern
    if [ "$FILTER_TYPE" = "audio_acoustic" ]; then
        INPUT_FILE="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_acoustic_${FILE_SUFFIX}_semantic_codes.jsonl"
    else
        INPUT_FILE="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_audio_semantic_${FILE_SUFFIX}.jsonl"
        OUTPUT_FILE="${DATA_DIR}/${BENIGN_DATASET}_filtered_${HARMFUL_SOURCE}_audio_semantic_${FILE_SUFFIX}_semantic_codes.jsonl"
    fi
else
    echo "Error: Invalid benign_dataset '$BENIGN_DATASET'."
    echo "Valid options: voicebench, spoken_squad, librispeech, heysquad, heysquad_accents, gammacorpus_accents, gammacorpus_usa, benign_instructions_usa, benign_instructions_accents, bbh, mmsu, voicebench_cafe, voicebench_traffic"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo ""
    echo "Please run the appropriate filtering script first."
    echo ""
    echo "Available files in ${DATA_DIR}/:"
    ls -la "${DATA_DIR}"/*.jsonl 2>/dev/null || echo "  No .jsonl files found"
    exit 1
fi

echo "============================================"
echo "Semantic Codes Extraction"
echo "============================================"
echo "Benign dataset: $BENIGN_DATASET"
echo "Filter type:    $FILTER_TYPE"
echo "Data dir:       $DATA_DIR"
echo "Input file:     $INPUT_FILE"
echo "Output file:    $OUTPUT_FILE"
echo "============================================"

# Run extraction
CUDA_VISIBLE_DEVICES=0 python 1_extract_semantic_codes.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo ""
echo "============================================"
echo "Extraction complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "============================================"
echo ""
echo "Next step: Run 3_finetune_lora.sh"
echo "  bash 3_finetune_lora.sh --benign_dataset $BENIGN_DATASET --filter_type $FILTER_TYPE --percentage $PERCENTAGE"
