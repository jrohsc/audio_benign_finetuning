#!/bin/bash
#
# Evaluate text LLM on harmful prompts (text-only, no audio).
#
# This is the text-only counterpart to 3_evaluate_jailbreaking.sh.
# Evaluates text LLMs finetuned with 2_finetune_text.sh on harmful text prompts.
#
# Output goes to results_text/ (not results/ or results_semantic/)
#
# Usage:
#   ./3_evaluate_jailbreaking_text.sh                                    # Evaluate on advbench (default)
#   ./3_evaluate_jailbreaking_text.sh --dataset safetybench              # Evaluate on safetybench
#   ./3_evaluate_jailbreaking_text.sh --filter_type semantic --percentage 50
#   ./3_evaluate_jailbreaking_text.sh --eval_mode finetuned_only \
#       --finetuned_path checkpoints_text_bft/af3_text_finetuned_percentage_50/best_model

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
FILTER_TYPE="text"  # "acoustic" or "semantic" (for checkpoint path selection) or text
THRESHOLD=""
PERCENTAGE="75"
NUM_SAMPLES=""
NUM_EPOCHS=3

DATASET="safetybench"  # Options: advbench, safetybench
PRETRAINED_PATH=""
FINETUNED_PATH=""  # Will be auto-set based on filtering mode
EVAL_MODE="finetuned_only"
SYSTEM_PROMPT="You are a helpful assistant who tries to help answer the user's question."
MAX_NEW_TOKENS=512
MAX_SAMPLES=""

# These will be set based on DATASET
OUTPUT_DIR=""
BENCHMARK_CSV=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --filter_type)
            FILTER_TYPE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pretrained_path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --finetuned_path)
            FINETUNED_PATH="$2"
            shift 2
            ;;
        --eval_mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --system_prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --benchmark_csv)
            BENCHMARK_CSV="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            THRESHOLD=""
            NUM_SAMPLES=""
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Evaluate text LLM on harmful prompts (text-only, no audio)."
            echo ""
            echo "Options:"
            echo "  --filter_type TYPE         Filter type used for finetuning: 'acoustic' or 'semantic' (default: semantic)"
            echo "  --dataset NAME             Dataset to evaluate: 'advbench' or 'safetybench' (default: safetybench)"
            echo "  --output_dir PATH          Directory to save results (auto-set based on dataset)"
            echo "  --pretrained_path PATH     Path to pretrained model"
            echo "  --finetuned_path PATH      Path to finetuned checkpoint dir (will auto-find best_model inside)"
            echo "  --percentage VALUE         Percentage mode for auto-selecting finetuned path (default: 50)"
            echo "  --num_samples VALUE        Num samples mode for auto-selecting finetuned path"
            echo "  --threshold VALUE          Threshold mode for auto-selecting finetuned path"
            echo "  --num_epochs N             Number of epochs used during training (default: 10)"
            echo "  --eval_mode MODE           Evaluation mode: 'pretrained_only', 'finetuned_only', or 'both' (default: finetuned_only)"
            echo "  --system_prompt TEXT       System prompt to use for inference"
            echo "  --max_new_tokens N         Maximum tokens to generate (default: 512)"
            echo "  --max_samples N            Maximum samples to evaluate (for debugging)"
            echo "  --benchmark_csv PATH       Path to benchmark CSV (auto-set based on dataset)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate semantic text-finetuned model on safetybench"
            echo "  $0 --filter_type semantic --percentage 50"
            echo ""
            echo "  # Evaluate on advbench"
            echo "  $0 --dataset advbench --percentage 50"
            echo ""
            echo "  # Quick test with 10 samples"
            echo "  $0 --max_samples 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Output always goes to results_text/
RESULTS_DIR="${SCRIPT_DIR}/results_text"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_text_bft"

# Set dataset-specific paths if not explicitly provided
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$DATASET" = "safetybench" ]; then
        OUTPUT_DIR="${RESULTS_DIR}/safetybench_eval"
    else
        OUTPUT_DIR="${RESULTS_DIR}/advbench_eval"
    fi
fi

if [ -z "$BENCHMARK_CSV" ]; then
    if [ "$DATASET" = "safetybench" ]; then
        BENCHMARK_CSV="${SCRIPT_DIR}/../harmful_data/safetybench.csv"
    else
        BENCHMARK_CSV="${SCRIPT_DIR}/../harmful_data/advbench.csv"
    fi
fi

# Auto-set finetuned path based on filtering mode if not explicitly provided
if [ -z "$FINETUNED_PATH" ]; then
    if [ -n "$NUM_SAMPLES" ]; then
        FINETUNED_PATH="${CHECKPOINT_DIR}/af3_text_finetuned_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}"
    elif [ -n "$THRESHOLD" ]; then
        FINETUNED_PATH="${CHECKPOINT_DIR}/af3_text_finetuned_thresh_${THRESHOLD}_epoch_${NUM_EPOCHS}"
    elif [ -n "$PERCENTAGE" ]; then
        FINETUNED_PATH="${CHECKPOINT_DIR}/af3_text_finetuned_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}"
    else
        echo "Error: Must specify --finetuned_path, --percentage, --num_samples, or --threshold"
        exit 1
    fi
fi

# Process finetuned path: find best_model inside and extract model name
FINETUNED_MODEL_PATH=""
FINETUNED_MODEL_NAME=""
if [ -n "$FINETUNED_PATH" ]; then
    # Check if best_model exists inside the provided path
    if [ -d "${FINETUNED_PATH}/best_model" ]; then
        FINETUNED_MODEL_PATH="${FINETUNED_PATH}/best_model"
        FINETUNED_MODEL_NAME="$(basename "$FINETUNED_PATH")"
        echo "Found best_model in ${FINETUNED_PATH}"
        echo "Model name for output: ${FINETUNED_MODEL_NAME}"
    elif [ -d "$FINETUNED_PATH" ]; then
        # Use as-is if best_model doesn't exist
        FINETUNED_MODEL_PATH="$FINETUNED_PATH"
        FINETUNED_MODEL_NAME="$(basename "$FINETUNED_PATH")"
        echo "Using finetuned path directly: ${FINETUNED_PATH}"
    else
        echo "Error: Finetuned path not found: ${FINETUNED_PATH}"
        exit 1
    fi
fi

# Build command
CMD="python 3_evaluate_jailbreaking_text.py"
CMD="$CMD --benchmark_csv $BENCHMARK_CSV"
CMD="$CMD --output_dir $OUTPUT_DIR"
if [ -n "$PRETRAINED_PATH" ]; then
    CMD="$CMD --pretrained_path $PRETRAINED_PATH"
fi
CMD="$CMD --eval_mode $EVAL_MODE"
CMD="$CMD --system_prompt \"$SYSTEM_PROMPT\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"

if [ -n "$FINETUNED_MODEL_PATH" ]; then
    CMD="$CMD --finetuned_path $FINETUNED_MODEL_PATH"
    CMD="$CMD --model_name $FINETUNED_MODEL_NAME"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo "=== Text LLM Jailbreaking Evaluation ==="
echo "Modality:         TEXT ONLY (no audio)"
echo "Filter type:      ${FILTER_TYPE}"
echo "Dataset:          ${DATASET}"
echo "Eval mode:        ${EVAL_MODE}"
echo "Benchmark CSV:    ${BENCHMARK_CSV}"
echo "Finetuned path:   ${FINETUNED_MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Running: $CMD"
cd "$SCRIPT_DIR"
eval $CMD
