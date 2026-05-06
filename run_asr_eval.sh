#!/bin/bash
#
# Run ASR evaluation for audio model responses
# Usage: ./run_asr_eval.sh --model <model_name> --dataset <dataset_name> [--file <specific_file>]
#
# Models: audio-flamingo, Phi-4, QwenAudioSFT, Kimi-Audio, Qwen2.5-Omni
# Datasets: advbench, safetybench, both
# Filter types: audio_acoustic, audio_semantic, text_semantic
#

set -e

# Load conda environment
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

# Default values
MODEL="Qwen2.5-Omni"
DATASET="both"
FILTER_TYPE="audio_semantic"    # audio_acoustic, audio_semantic, or text_semantic
BENIGN_DATASET="gammacorpus_accents"     # voicebench, spoken_squad, or librispeech, mmsu, heysquad, bbh
SPECIFIC_FILE=""
BATCH_SIZE=8
OUTPUT_DIR="/work/anon/audio_benign_finetuning/asr_results_respond_to_question_in_audio"
SKIP_EXISTING=true

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALCULATE_ASR="${SCRIPT_DIR}/calculate_asr.py"

# Print usage
usage() {
    echo "Usage: $0 --model <model_name> --dataset <dataset_name> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --model, -m      Model name: audio-flamingo, Phi-4, QwenAudioSFT, Kimi-Audio, Qwen2.5-Omni"
    echo "  --dataset, -d    Dataset name: advbench, safetybench"
    echo ""
    echo "Optional arguments:"
    echo "  --filter-type, -t  Filter type: audio_acoustic, audio_semantic, or text_semantic (default: audio_semantic)"
    echo "  --benign-dataset   Benign dataset: voicebench (default), spoken_squad, librispeech, or heysquad"
    echo "  --file, -f         Specific JSON file to evaluate (default: all files in directory)"
    echo "  --batch-size, -b   Batch size for evaluation (default: 8)"
    echo "  --output, -o       Output directory for ASR results (default: ${OUTPUT_DIR})"
    echo "  --skip-existing    Skip if output JSON already exists (default)"
    echo "  --no-skip-existing Force re-evaluation even if output exists"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model audio-flamingo --dataset advbench"
    echo "  $0 --model Phi-4 --dataset safetybench --batch-size 4"
    echo "  $0 --model Kimi-Audio --dataset advbench --file pretrained_responses.json"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --filter-type|-t)
            FILTER_TYPE="$2"
            shift 2
            ;;
        --benign-dataset)
            BENIGN_DATASET="$2"
            shift 2
            ;;
        --file|-f)
            SPECIFIC_FILE="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" ]] || [[ -z "$DATASET" ]]; then
    echo "Error: Both --model and --dataset are required"
    usage
fi

# Validate filter type
if [[ "$FILTER_TYPE" != "audio_acoustic" ]] && [[ "$FILTER_TYPE" != "audio_semantic" ]] && [[ "$FILTER_TYPE" != "text_semantic" ]] && [[ "$FILTER_TYPE" != "random" ]] && [[ "$FILTER_TYPE" != "sbert_audio_semantic" ]]; then
    echo "Error: Filter type must be 'audio_acoustic', 'audio_semantic', 'text_semantic', 'random', or 'sbert_audio_semantic'"
    exit 1
fi

# Determine benign dataset suffix for directory naming
if [[ "$BENIGN_DATASET" == "voicebench" ]]; then
    BENIGN_SUFFIX=""
else
    BENIGN_SUFFIX="_${BENIGN_DATASET}"
fi

# Base paths for each model (set after argument parsing based on filter type and benign dataset)
declare -A MODEL_PATHS
if [[ "$FILTER_TYPE" == "audio_acoustic" ]]; then
    MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_acoustic${BENIGN_SUFFIX}"
elif [[ "$FILTER_TYPE" == "audio_semantic" ]]; then
    if [[ "$BENIGN_DATASET" == "voicebench" ]]; then
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_semantic"
    else
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_${BENIGN_DATASET}"
    fi
elif [[ "$FILTER_TYPE" == "text_semantic" ]]; then
    MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_semantic${BENIGN_SUFFIX}"
elif [[ "$FILTER_TYPE" == "random" ]]; then
    if [[ "$BENIGN_DATASET" == "voicebench" ]]; then
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_random"
    else
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_random_${BENIGN_DATASET}"
    fi
elif [[ "$FILTER_TYPE" == "sbert_audio_semantic" ]]; then
    if [[ "$BENIGN_DATASET" == "voicebench" ]]; then
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_sbert_audio_semantic"
    else
        MODEL_PATHS["audio-flamingo"]="/work/anon/audio_benign_finetuning/audio-flamingo/results_voicebench_sbert_audio_semantic_${BENIGN_DATASET}"
    fi
fi
MODEL_PATHS["Phi-4"]="/work/anon/audio_benign_finetuning/Phi-4/results"
MODEL_PATHS["QwenAudioSFT"]="/work/anon/audio_benign_finetuning/QwenAudioSFT/results"
# Kimi-Audio: all files are in the same response_log directory (benign dataset is in filename)
MODEL_PATHS["Kimi-Audio"]="/work/anon/audio_benign_finetuning/Kimi-Audio/finetune_codes/response_log"
# Qwen2.5-Omni: directory depends on filter type (matches run_full_pipeline.sh conventions)
if [[ "$FILTER_TYPE" == "audio_acoustic" ]]; then
    MODEL_PATHS["Qwen2.5-Omni"]="/work/anon/audio_benign_finetuning/Qwen2.5-Omni/finetune_codes/results_acoustic_${BENIGN_DATASET}"
elif [[ "$FILTER_TYPE" == "random" ]]; then
    MODEL_PATHS["Qwen2.5-Omni"]="/work/anon/audio_benign_finetuning/Qwen2.5-Omni/finetune_codes/results_random"
elif [[ "$FILTER_TYPE" == "sbert_audio_semantic" ]]; then
    MODEL_PATHS["Qwen2.5-Omni"]="/work/anon/audio_benign_finetuning/Qwen2.5-Omni/finetune_codes/results_sbert_audio_semantic_${BENIGN_DATASET}"
else
    MODEL_PATHS["Qwen2.5-Omni"]="/work/anon/audio_benign_finetuning/Qwen2.5-Omni/finetune_codes/results_${BENIGN_DATASET}"
fi

# Validate model name
if [[ -z "${MODEL_PATHS[$MODEL]}" ]]; then
    echo "Error: Unknown model '$MODEL'"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

# Validate dataset name
if [[ "$DATASET" != "advbench" ]] && [[ "$DATASET" != "safetybench" ]] && [[ "$DATASET" != "both" ]]; then
    echo "Error: Dataset must be 'advbench', 'safetybench', or 'both'"
    exit 1
fi

# If dataset is "both", run for both advbench and safetybench
if [[ "$DATASET" == "both" ]]; then
    echo "Running evaluation for both advbench and safetybench..."
    for DS in advbench safetybench; do
        echo ""
        echo "========================================"
        echo "Processing dataset: $DS"
        echo "========================================"
        "$0" --model "$MODEL" --dataset "$DS" --filter-type "$FILTER_TYPE" --benign-dataset "$BENIGN_DATASET" --batch-size "$BATCH_SIZE" --output "$OUTPUT_DIR" $(if [[ "$SKIP_EXISTING" == "true" ]]; then echo "--skip-existing"; else echo "--no-skip-existing"; fi)
    done
    exit 0
fi

# Construct the results directory path
RESULTS_DIR="${MODEL_PATHS[$MODEL]}/${DATASET}_eval"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Create model-specific output directory
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL}"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "========================================"
echo "ASR Evaluation"
echo "========================================"
echo "Model:         $MODEL"
echo "Dataset:       $DATASET"
echo "Filter type:   $FILTER_TYPE"
echo "Benign dataset: $BENIGN_DATASET"
echo "Results:       $RESULTS_DIR"
echo "Output:        $MODEL_OUTPUT_DIR"
echo "Batch size:    $BATCH_SIZE"
echo "Skip existing: $SKIP_EXISTING"
echo "========================================"
echo ""

# Find JSON files to process
if [[ -n "$SPECIFIC_FILE" ]]; then
    # Process specific file
    JSON_FILE="${RESULTS_DIR}/${SPECIFIC_FILE}"
    if [[ ! -f "$JSON_FILE" ]]; then
        echo "Error: File not found: $JSON_FILE"
        exit 1
    fi
    JSON_FILES=("$JSON_FILE")
else
    # Process all JSON files except config files and aggregate files (all_responses.json)
    # For Kimi-Audio, filter based on benign dataset and filter type in filename
    if [[ "$MODEL" == "Kimi-Audio" ]] || [[ "$MODEL" == "Qwen2.5-Omni" ]]; then
        if [[ "$BENIGN_DATASET" == "voicebench" ]]; then
            # voicebench files don't have benign dataset name, but have audio_acoustic or audio_semantic
            # Exclude files that have other benign dataset names (heysquad, mmsu, bbh, librispeech, spoken_squad)
            JSON_FILES=($(find "$RESULTS_DIR" -name "*${FILTER_TYPE}*.json" -type f ! -name "*config*" ! -name "all_responses.json" \
                ! -name "*heysquad*" ! -name "*mmsu*" ! -name "*bbh*" ! -name "*librispeech*" ! -name "*spoken_squad*" | sort))
        else
            # For other benign datasets, filter by benign dataset name and filter type
            JSON_FILES=($(find "$RESULTS_DIR" -name "*${BENIGN_DATASET}*${FILTER_TYPE}*.json" -type f ! -name "*config*" ! -name "all_responses.json" | sort))
        fi
    else
        JSON_FILES=($(find "$RESULTS_DIR" -name "*.json" -type f ! -name "*config*" ! -name "all_responses.json" | sort))
    fi
fi

if [[ ${#JSON_FILES[@]} -eq 0 ]]; then
    echo "No JSON files found in $RESULTS_DIR"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON file(s) to process:"
for f in "${JSON_FILES[@]}"; do
    echo "  - $(basename "$f")"
done
echo ""

# Track completed files count
COMPLETED=0
TOTAL=${#JSON_FILES[@]}

# Process each JSON file
for JSON_FILE in "${JSON_FILES[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "[$COMPLETED/$TOTAL] Processing: $(basename "$JSON_FILE")"
    echo "----------------------------------------"

    # Build command
    CMD="python3 \"$CALCULATE_ASR\""
    CMD="$CMD --input \"$JSON_FILE\""
    CMD="$CMD --output-dir \"$MODEL_OUTPUT_DIR\""
    CMD="$CMD --model \"$MODEL\""
    CMD="$CMD --dataset \"$DATASET\""
    CMD="$CMD --filter-type \"$FILTER_TYPE\""
    CMD="$CMD --batch-size $BATCH_SIZE"

    if [[ "$SKIP_EXISTING" == "true" ]]; then
        CMD="$CMD --skip-existing"
    fi

    eval $CMD

    echo ""
    echo "[CHECKPOINT] $COMPLETED/$TOTAL files completed. Results saved to $MODEL_OUTPUT_DIR/asr_summary.csv"
    echo ""
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $MODEL_OUTPUT_DIR"
echo "========================================"
echo ""

# Print summary table from CSV
SUMMARY_CSV="$MODEL_OUTPUT_DIR/asr_summary.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo "Summary for $MODEL - $DATASET - $FILTER_TYPE:"
    echo "----------------------------------------------------"

    # Check if CSV has filter_type column (new format)
    if head -1 "$SUMMARY_CSV" | grep -q "filter_type"; then
        # New format: timestamp,model,dataset,filter_type,threshold,input_file,...
        echo "Filter    | Threshold | Samples | Harmful | ASR"
        echo "----------|-----------|---------|---------|-------"
        grep "$MODEL,$DATASET,$FILTER_TYPE" "$SUMMARY_CSV" | while IFS=',' read -r ts model ds filter thresh file samples harmful asr; do
            printf "%-10s| %-10s| %-7s | %-7s | %s\n" "$filter" "$thresh" "$samples" "$harmful" "$asr"
        done
    else
        # Old format: timestamp,model,dataset,threshold,input_file,...
        # Filter by filename containing filter type keyword
        echo "Threshold | Samples | Harmful | ASR     | File"
        echo "----------|---------|---------|---------|------------------"
        if [[ "$FILTER_TYPE" == "text_semantic" ]]; then
            grep "$MODEL,$DATASET" "$SUMMARY_CSV" | grep "text_semantic\|text_finetuned" | while IFS=',' read -r ts model ds thresh file samples harmful asr; do
                printf "%-10s| %-7s | %-7s | %-7s | %s\n" "$thresh" "$samples" "$harmful" "$asr" "$file"
            done
        elif [[ "$FILTER_TYPE" == "audio_semantic" ]]; then
            grep "$MODEL,$DATASET" "$SUMMARY_CSV" | grep "audio_semantic\|_semantic_" | grep -v "text_semantic" | while IFS=',' read -r ts model ds thresh file samples harmful asr; do
                printf "%-10s| %-7s | %-7s | %-7s | %s\n" "$thresh" "$samples" "$harmful" "$asr" "$file"
            done
        else
            # audio_acoustic - match acoustic patterns or files without semantic/text in name
            grep "$MODEL,$DATASET" "$SUMMARY_CSV" | grep -E "audio_acoustic|_acoustic_" | while IFS=',' read -r ts model ds thresh file samples harmful asr; do
                printf "%-10s| %-7s | %-7s | %-7s | %s\n" "$thresh" "$samples" "$harmful" "$asr" "$file"
            done
        fi
    fi
    echo "----------------------------------------------------"
fi
