#!/bin/bash
#
# Full Pipeline (TEXT ONLY): Filter -> Prepare -> Finetune Audio Flamingo 3 -> Evaluate
#
# This script runs the complete TEXT-ONLY benign finetuning pipeline:
#   0. Filter VoiceBench samples by SEMANTIC text similarity (text-to-text comparison)
#   1. Prepare filtered dataset in TEXT-ONLY format (no audio)
#   2. Finetune Audio Flamingo 3 on the text-only dataset
#   3. Evaluate the finetuned model on harmful audio prompts
#
# NOTE: This pipeline uses ONLY semantic (text) filtering.
#       Acoustic filtering is NOT applicable for text-only data.
#       The filtering compares:
#         - VoiceBench question TEXT (extracted from conversations)
#         - Harmful prompt TEXT (from advbench.csv "goal" column)
#       No audio files are processed during filtering.
#
# Output directories:
#   - Checkpoints: checkpoints_text_bft/
#   - Results: results_text/
#
# Usage:
#   ./run_full_pipeline_text.sh                          # Run with defaults (50%)
#   ./run_full_pipeline_text.sh --percentage 65          # 65% of closest samples
#   ./run_full_pipeline_text.sh --num_samples 500        # Exact 500 samples
#   ./run_full_pipeline_text.sh --num_epochs 5           # 5 training epochs

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
# Text-only pipeline ALWAYS uses semantic filtering (text-to-text similarity)
PERCENTAGE="75"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=3
DATASET="advbench"  # advbench, safetybench, or both for evaluation
SKIP_FILTER=""
SKIP_PREPARE=""
SKIP_FINETUNE=""
SKIP_EVALUATE=""
SKIP_ASR=""
RUN_ASR="true"  # Run ASR evaluation by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run the full TEXT-ONLY benign finetuning pipeline for Audio Flamingo 3."
            echo "This finetunes Audio Flamingo 3 on text-only data (no audio during training)."
            echo ""
            echo "NOTE: This pipeline uses SEMANTIC (text) filtering only."
            echo "      Acoustic filtering is not applicable for text-only data."
            echo "      Filtering compares VoiceBench question TEXT vs harmful TEXT prompts."
            echo ""
            echo "Pipeline steps:"
            echo "  0. Filter VoiceBench samples by semantic text similarity"
            echo "  1. Prepare filtered dataset in TEXT-ONLY format"
            echo "  2. Finetune Audio Flamingo 3 (text-only)"
            echo "  3. Evaluate on harmful AUDIO prompts"
            echo ""
            echo "Output directories:"
            echo "  - Checkpoints: checkpoints_text_bft/"
            echo "  - Eval results: results_text/"
            echo ""
            echo "Filtering modes (mutually exclusive):"
            echo "  --percentage VALUE      Keep top percentage of closest samples (default: 50)"
            echo "  --num_samples VALUE     Keep exact number of samples"
            echo "  --threshold VALUE       Semantic distance threshold"
            echo ""
            echo "Training options:"
            echo "  --num_epochs N          Number of training epochs (default: 3)"
            echo ""
            echo "Evaluation options:"
            echo "  --dataset NAME          Evaluation dataset: 'advbench', 'safetybench', or 'both' (default: advbench)"
            echo ""
            echo "Skip options (to resume from a specific step):"
            echo "  --skip_filter           Skip step 0 (filtering)"
            echo "  --skip_prepare          Skip step 1 (dataset preparation)"
            echo "  --skip_finetune         Skip step 2 (finetuning)"
            echo "  --skip_evaluate         Skip step 3 (evaluation)"
            echo "  --skip_asr              Skip step 4 (ASR evaluation)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Default: 50%, 3 epochs"
            echo "  $0 --percentage 65                    # 65% of closest samples"
            echo "  $0 --num_samples 500                  # Exact 500 samples"
            echo "  $0 --skip_filter --skip_prepare       # Resume from finetuning"
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

echo "=============================================="
echo "   AUDIO FLAMINGO 3 TEXT-ONLY BFT PIPELINE"
echo "=============================================="
echo "Model:            Audio Flamingo 3"
echo "Training data:    TEXT ONLY (no audio)"
echo "Filtering:        SEMANTIC (text-to-text similarity)"
echo "Mode:             $MODE_DESC"
echo "Training epochs:  $NUM_EPOCHS"
echo "Eval dataset:     $DATASET (AUDIO prompts)"
echo "Checkpoint dir:   checkpoints_text_bft/"
echo "=============================================="
echo ""

# ============================================
# Step 0: Filter VoiceBench samples by semantic text similarity
# ============================================
if [ -z "$SKIP_FILTER" ]; then
    echo "=============================================="
    echo "STEP 0: Filtering VoiceBench by semantic text similarity"
    echo "=============================================="
    echo "Comparing VoiceBench question TEXT vs harmful TEXT prompts"
    echo "(No audio files are processed - pure text comparison)"
    echo ""

    bash 0_filter_voicebench_semantic.sh $FILTER_ARGS

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
    echo "STEP 1: Preparing filtered dataset (text-only format)"
    echo "=============================================="

    bash 1_prepare_filtered_dataset.sh --filter_type semantic $FILTER_ARGS

    echo ""
    echo "Step 1 completed successfully!"
    echo ""
else
    echo "Skipping Step 1 (dataset preparation)..."
fi

# ============================================
# Step 2: Finetune Audio Flamingo 3 on TEXT-ONLY data
# ============================================
if [ -z "$SKIP_FINETUNE" ]; then
    echo "=============================================="
    echo "STEP 2: Finetuning Audio Flamingo 3 (TEXT-ONLY)"
    echo "=============================================="

    bash 2_finetune_af3_text.sh \
        --num_epochs "$NUM_EPOCHS" \
        $FILTER_ARGS

    echo ""
    echo "Step 2 completed successfully!"
    echo ""
else
    echo "Skipping Step 2 (finetuning)..."
fi

# ============================================
# Step 3: Evaluate on harmful AUDIO prompts
# (AF3 is evaluated on audio, not text, even after text-only finetuning)
# Results saved to results_text/
# ============================================
if [ -z "$SKIP_EVALUATE" ]; then
    echo "=============================================="
    echo "STEP 3: Evaluating AF3 on $DATASET (AUDIO prompts)"
    echo "=============================================="
    echo "Note: Evaluating with harmful AUDIO to test if text-only BFT"
    echo "      affects model safety on audio inputs."
    echo ""

    # Determine the finetuned checkpoint path (without /best_model - the eval script will find it)
    if [ -n "$NUM_SAMPLES" ]; then
        FINETUNED_PATH="checkpoints_text_bft/af3_text_finetuned_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}"
    elif [ -n "$PERCENTAGE" ]; then
        FINETUNED_PATH="checkpoints_text_bft/af3_text_finetuned_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}"
    fi

    # Use the audio-based evaluation script for AF3, but save results to results_text/
    if [ "$DATASET" = "both" ]; then
        echo "Results will be saved to: results_text/advbench_eval and results_text/safetybench_eval"
        echo ""

        echo "--- Evaluating on advbench ---"
        bash 3_evaluate_jailbreaking.sh \
            --filter_type semantic \
            --dataset advbench \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "results_text/advbench_eval" \
            --eval_mode finetuned_only

        echo ""
        echo "--- Evaluating on safetybench ---"
        bash 3_evaluate_jailbreaking.sh \
            --filter_type semantic \
            --dataset safetybench \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "results_text/safetybench_eval" \
            --eval_mode finetuned_only
    else
        echo "Results will be saved to: results_text/${DATASET}_eval"
        echo ""

        bash 3_evaluate_jailbreaking.sh \
            --filter_type semantic \
            --dataset "$DATASET" \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "results_text/${DATASET}_eval" \
            --eval_mode finetuned_only
    fi

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
            --filter-type text_semantic

        echo ""
        echo "Running ASR evaluation for safetybench..."
        bash ../run_asr_eval.sh \
            --model audio-flamingo \
            --dataset safetybench \
            --filter-type text_semantic
    else
        bash ../run_asr_eval.sh \
            --model audio-flamingo \
            --dataset "$DATASET" \
            --filter-type text_semantic
    fi

    echo ""
    echo "Step 4 completed successfully!"
    echo ""
else
    echo "Skipping Step 4 (ASR evaluation)..."
fi

echo "=============================================="
echo "   AF3 TEXT-ONLY BFT COMPLETED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Model: Audio Flamingo 3"
echo "  Training: TEXT ONLY (no audio)"
echo "  Filtering: SEMANTIC (text similarity)"
echo "  Mode: $MODE_DESC"
echo "  Epochs: $NUM_EPOCHS"
echo "  Eval dataset: $DATASET"
echo ""
echo "Checkpoint saved to: checkpoints_text_bft/"
echo "Eval results saved to: results_text/"
echo "ASR results saved to: ../asr_results/audio-flamingo/"
echo "=============================================="
