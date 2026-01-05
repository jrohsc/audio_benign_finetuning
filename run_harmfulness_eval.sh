#!/bin/bash
#
# Run Harmfulness Score evaluation for audio model responses
# Uses PKU-Alignment's Beaver Cost Model for continuous harmfulness scoring
#
# Usage: ./run_harmfulness_eval.sh --model <model_name> --dataset <dataset_name> [--file <specific_file>]
#
# Models: audio-flamingo, Phi-4, QwenAudioSFT, Kimi-Audio
# Datasets: advbench, safetybench
#

set -e

# Load conda and CUDA environment
source /work/pi_ahoumansadr_umass_edu/jroh/miniconda3/etc/profile.d/conda.sh
conda activate harmbench
module load cuda/12.6
export CUDA_HOME=$CUDA_PATH

# Default values
MODEL="Kimi-Audio"
DATASET="safetybench"
SPECIFIC_FILE=""
BATCH_SIZE=4  # Lower than ASR due to larger cost model
OUTPUT_DIR="/work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/harmfulness_results"
SKIP_EXISTING=true
COST_MODEL="PKU-Alignment/beaver-7b-v1.0-cost"

# Base paths for each model
declare -A MODEL_PATHS
MODEL_PATHS["audio-flamingo"]="/work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/audio-flamingo/results"
MODEL_PATHS["Phi-4"]="/work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/Phi-4/results"
MODEL_PATHS["QwenAudioSFT"]="/work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/QwenAudioSFT/results"
MODEL_PATHS["Kimi-Audio"]="/work/pi_ahoumansadr_umass_edu/jroh/audio_benign_finetuning/Kimi-Audio/finetune_codes/response"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALCULATE_HARMFULNESS="${SCRIPT_DIR}/calculate_harmfulness.py"

# Print usage
usage() {
    echo "Usage: $0 --model <model_name> --dataset <dataset_name> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --model, -m         Model name: audio-flamingo, Phi-4, QwenAudioSFT, Kimi-Audio"
    echo "  --dataset, -d       Dataset name: advbench, safetybench"
    echo ""
    echo "Optional arguments:"
    echo "  --file, -f          Specific JSON file to evaluate (default: all files in directory)"
    echo "  --batch-size, -b    Batch size for evaluation (default: 4)"
    echo "  --output, -o        Output directory for harmfulness results (default: ${OUTPUT_DIR})"
    echo "  --cost-model, -c    PKU-Alignment cost model (default: ${COST_MODEL})"
    echo "  --skip-existing     Skip if output JSON already exists (default)"
    echo "  --no-skip-existing  Force re-evaluation even if output exists"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Cost Model Options:"
    echo "  PKU-Alignment/beaver-7b-v1.0-cost (default)"
    echo "  PKU-Alignment/beaver-7b-v3.0-cost (newer version)"
    echo ""
    echo "Examples:"
    echo "  $0 --model audio-flamingo --dataset advbench"
    echo "  $0 --model Phi-4 --dataset safetybench --batch-size 2"
    echo "  $0 --model Kimi-Audio --dataset advbench --file pretrained_responses.json"
    echo "  $0 --model Phi-4 --dataset advbench --cost-model PKU-Alignment/beaver-7b-v3.0-cost"
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
        --cost-model|-c)
            COST_MODEL="$2"
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

# Validate model name
if [[ -z "${MODEL_PATHS[$MODEL]}" ]]; then
    echo "Error: Unknown model '$MODEL'"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

# Validate dataset name
if [[ "$DATASET" != "advbench" ]] && [[ "$DATASET" != "safetybench" ]]; then
    echo "Error: Dataset must be 'advbench' or 'safetybench'"
    exit 1
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
echo "Harmfulness Score Evaluation"
echo "========================================"
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Cost Model:   $COST_MODEL"
echo "Results:      $RESULTS_DIR"
echo "Output:       $MODEL_OUTPUT_DIR"
echo "Batch size:   $BATCH_SIZE"
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
    JSON_FILES=($(find "$RESULTS_DIR" -name "*.json" -type f ! -name "*config*" ! -name "all_responses.json" | sort))
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
    CMD="python3 \"$CALCULATE_HARMFULNESS\""
    CMD="$CMD --input \"$JSON_FILE\""
    CMD="$CMD --output-dir \"$MODEL_OUTPUT_DIR\""
    CMD="$CMD --model \"$MODEL\""
    CMD="$CMD --dataset \"$DATASET\""
    CMD="$CMD --cost-model \"$COST_MODEL\""
    CMD="$CMD --batch-size $BATCH_SIZE"

    if [[ "$SKIP_EXISTING" == "true" ]]; then
        CMD="$CMD --skip-existing"
    fi

    eval $CMD

    echo ""
    echo "[CHECKPOINT] $COMPLETED/$TOTAL files completed. Results saved to $MODEL_OUTPUT_DIR/harmfulness_summary.csv"
    echo ""
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $MODEL_OUTPUT_DIR"
echo "========================================"
echo ""

# Print summary table from CSV
SUMMARY_CSV="$MODEL_OUTPUT_DIR/harmfulness_summary.csv"
if [[ -f "$SUMMARY_CSV" ]]; then
    echo "Summary for $MODEL - $DATASET:"
    echo "--------------------------------------------------------------------------------"
    echo "Threshold  | Samples | Mean     | Std      | Min      | Max      | Median"
    echo "-----------|---------|----------|----------|----------|----------|----------"
    grep "$MODEL,$DATASET" "$SUMMARY_CSV" | while IFS=',' read -r ts model ds thresh file samples mean std min max median; do
        printf "%-10s | %-7s | %-8s | %-8s | %-8s | %-8s | %s\n" "$thresh" "$samples" "$mean" "$std" "$min" "$max" "$median"
    done
    echo "--------------------------------------------------------------------------------"
fi
