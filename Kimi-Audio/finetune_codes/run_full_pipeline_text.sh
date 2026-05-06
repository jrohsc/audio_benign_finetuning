#!/bin/bash
#
# Full Pipeline (TEXT ONLY): Filter -> Prepare -> Finetune -> Merge -> Evaluate
#
# This script runs the complete TEXT-ONLY benign finetuning pipeline for Kimi-Audio:
#   0. Filter VoiceBench samples by SEMANTIC text similarity
#   1. Convert filtered data to TEXT-ONLY format (no audio)
#   2. Finetune Kimi-Audio on text-only data (LoRA)
#   3. Merge LoRA weights
#   4. Evaluate on harmful AUDIO prompts
#
# NOTE: This pipeline uses ONLY semantic (text) filtering.
#       The filtering compares:
#         - VoiceBench question TEXT
#         - Harmful prompt TEXT (from advbench.csv "goal" column)
#       Training uses TEXT-ONLY conversations (no audio tokens).
#       Evaluation uses AUDIO prompts to test if text-only BFT affects audio safety.
#
# Output directories:
#   - Text data: data_semantic_text/
#   - Checkpoints: output_text_bft/
#
# Usage:
#   ./run_full_pipeline_text.sh                          # Run with defaults (50%)
#   ./run_full_pipeline_text.sh --percentage 65          # 65% of closest samples
#   ./run_full_pipeline_text.sh --num_samples 500        # Exact 500 samples
#   ./run_full_pipeline_text.sh --num_epochs 5           # 5 training epochs

set -e

# Load CUDA
module load cuda/12.6

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
PERCENTAGE="50"
NUM_SAMPLES=""
THRESHOLD=""
NUM_EPOCHS=3
DATASET="advbench"  # advbench, safetybench, or both for evaluation
SKIP_FILTER=""
SKIP_PREPARE=""
SKIP_FINETUNE=""
SKIP_MERGE=""
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
            RUN_ASR=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run the full TEXT-ONLY benign finetuning pipeline for Kimi-Audio."
            echo "This finetunes Kimi-Audio on text-only conversations (no audio processing)."
            echo ""
            echo "NOTE: This pipeline uses SEMANTIC (text) filtering only."
            echo "      Training is TEXT-ONLY (no audio tokens)."
            echo "      Evaluation uses AUDIO prompts to test cross-modal transfer."
            echo ""
            echo "Pipeline steps:"
            echo "  0. Filter VoiceBench samples by semantic text similarity"
            echo "  1. Convert to TEXT-ONLY format"
            echo "  2. Finetune Kimi-Audio on text-only data"
            echo "  3. Merge LoRA weights"
            echo "  4. Evaluate on harmful AUDIO prompts"
            echo ""
            echo "Output directories:"
            echo "  - Text data: data_semantic_text/"
            echo "  - Checkpoints: output_text_bft/"
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
            echo "  --skip_prepare          Skip step 1 (text-only conversion)"
            echo "  --skip_finetune         Skip step 2 (finetuning)"
            echo "  --skip_merge            Skip step 3 (LoRA merge)"
            echo "  --skip_evaluate         Skip step 4 (evaluation)"
            echo "  --skip_asr              Skip step 5 (ASR evaluation)"
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

# Determine filtering mode description and arguments
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
echo "   KIMI-AUDIO TEXT-ONLY BFT PIPELINE"
echo "=============================================="
echo "Model:            Kimi-Audio-7B"
echo "Training data:    TEXT ONLY (no audio)"
echo "Filtering:        SEMANTIC (text-to-text similarity)"
echo "Mode:             $MODE_DESC"
echo "Training epochs:  $NUM_EPOCHS"
echo "Eval dataset:     $DATASET (AUDIO prompts)"
echo "Checkpoint dir:   output_text_bft/"
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
    echo ""

    bash 0_filter_voicebench_semantic.sh $FILTER_ARGS

    echo ""
    echo "Step 0 completed successfully!"
    echo ""
else
    echo "Skipping Step 0 (filtering)..."
fi

# ============================================
# Step 1: Convert to text-only format
# ============================================
if [ -z "$SKIP_PREPARE" ]; then
    echo "=============================================="
    echo "STEP 1: Converting to text-only format"
    echo "=============================================="
    echo ""

    python 1_convert_to_text_only.py $FILTER_ARGS --harmful_source advbench

    echo ""
    echo "Step 1 completed successfully!"
    echo ""
else
    echo "Skipping Step 1 (text-only conversion)..."
fi

# ============================================
# Step 2: Finetune Kimi-Audio on TEXT-ONLY data
# ============================================
if [ -z "$SKIP_FINETUNE" ]; then
    echo "=============================================="
    echo "STEP 2: Finetuning Kimi-Audio (TEXT-ONLY)"
    echo "=============================================="
    echo ""

    bash 3_finetune_lora_text.sh \
        --epochs "$NUM_EPOCHS" \
        $FILTER_ARGS

    echo ""
    echo "Step 2 completed successfully!"
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
    echo ""

    bash 5_merge_lora_for_inference_text.sh \
        --num_epochs "$NUM_EPOCHS" \
        $FILTER_ARGS

    echo ""
    echo "Step 3 completed successfully!"
    echo ""
else
    echo "Skipping Step 3 (LoRA merge)..."
fi

# ============================================
# Step 4: Evaluate on harmful AUDIO prompts
# ============================================
if [ -z "$SKIP_EVALUATE" ]; then
    echo "=============================================="
    echo "STEP 4: Evaluating on $DATASET (AUDIO prompts)"
    echo "=============================================="
    echo "Note: Evaluating with harmful AUDIO to test if text-only BFT"
    echo "      affects model safety on audio inputs."
    echo ""

    # Determine the merged model path
    if [ -n "$NUM_SAMPLES" ]; then
        MERGED_PATH="output_text_bft/finetuned_lora_text_n${NUM_SAMPLES}_epoch_${NUM_EPOCHS}_merged"
    elif [ -n "$PERCENTAGE" ]; then
        MERGED_PATH="output_text_bft/finetuned_lora_text_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}_merged"
    fi

    # Run evaluation with the text-trained model on audio prompts
    if [ "$DATASET" = "both" ]; then
        echo "--- Evaluating on advbench ---"
        bash 6_evaluate_harmful_audio.sh \
            --filter_type text \
            --dataset advbench \
            --model_path "$MERGED_PATH" \
            $FILTER_ARGS

        echo ""
        echo "--- Evaluating on safetybench ---"
        bash 6_evaluate_harmful_audio.sh \
            --filter_type text \
            --dataset safetybench \
            --model_path "$MERGED_PATH" \
            $FILTER_ARGS
    else
        bash 6_evaluate_harmful_audio.sh \
            --filter_type text \
            --dataset "$DATASET" \
            --model_path "$MERGED_PATH" \
            $FILTER_ARGS
    fi

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

    if [ "$DATASET" = "both" ]; then
        echo "Running ASR evaluation for advbench..."
        bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset advbench \
            --filter-type text_semantic

        echo ""
        echo "Running ASR evaluation for safetybench..."
        bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset safetybench \
            --filter-type text_semantic
    else
        bash ../../run_asr_eval.sh \
            --model Kimi-Audio \
            --dataset "$DATASET" \
            --filter-type text_semantic
    fi

    echo ""
    echo "Step 5 completed successfully!"
    echo ""
else
    echo "Skipping Step 5 (ASR evaluation)..."
fi

echo "=============================================="
echo "   KIMI-AUDIO TEXT-ONLY BFT COMPLETED"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Model: Kimi-Audio-7B"
echo "  Training: TEXT ONLY (no audio)"
echo "  Filtering: SEMANTIC (text similarity)"
echo "  Mode: $MODE_DESC"
echo "  Epochs: $NUM_EPOCHS"
echo "  Eval dataset: $DATASET"
echo ""
echo "Text data saved to: data_semantic_text/"
echo "Checkpoint saved to: output_text_bft/"
echo "ASR results saved to: ../../asr_results/Kimi-Audio/"
echo "=============================================="
