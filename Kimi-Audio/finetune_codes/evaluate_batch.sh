#!/bin/bash
# Batch evaluation of all finetuned models on AdvBench
#
# This script evaluates each finetuned model on harmful audio prompts.

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Paths
HARMFUL_AUDIO_DIR="../../harmful_data/advbench_gtts/en"
HARMFUL_SOURCE="advbench"

# Use distance mode for consistency with Qwen-Audio
USE_DISTANCE=true

if [ "$USE_DISTANCE" = true ]; then
    THRESHOLDS=(0.060 0.010 0.008 0.006 0.004 0.002)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_dist_"
else
    THRESHOLDS=(0.940 0.988 0.990 0.992 0.994 0.996)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_"
fi

# Also evaluate pretrained as baseline
EVAL_PRETRAINED=true

echo "============================================"
echo "Batch Evaluation on AdvBench"
echo "============================================"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "Harmful audio: $HARMFUL_AUDIO_DIR"
echo "============================================"

# Evaluate pretrained model first
if [ "$EVAL_PRETRAINED" = true ]; then
    PRETRAINED_PATH="/project/anon/BFT_models/pretrained_kimi_instruct"
    OUTPUT_FILE="response_log/${HARMFUL_SOURCE}_en_pretrained_$(date +%Y%m%d_%H%M%S).json"

    if [ -d "$PRETRAINED_PATH" ]; then
        echo ""
        echo "Evaluating pretrained model..."
        echo "  Model: $PRETRAINED_PATH"
        echo "  Output: $OUTPUT_FILE"

        CUDA_VISIBLE_DEVICES=0 python 6_evaluate_harmful_audio.py \
            --model_path "$PRETRAINED_PATH" \
            --model_name "pretrained" \
            --audio_dir "$HARMFUL_AUDIO_DIR" \
            --output_file "$OUTPUT_FILE" \
            --num_samples 520

        echo "  Done!"
    else
        echo "Skipping pretrained: not found at $PRETRAINED_PATH"
    fi
fi

# Evaluate each finetuned model
for THRESHOLD in "${THRESHOLDS[@]}"; do
    MODEL_PATH="output/finetuned_lora_${FILE_PREFIX}${THRESHOLD}_merged"
    OUTPUT_FILE="response_log/${HARMFUL_SOURCE}_en_finetuned_dist_${THRESHOLD}_$(date +%Y%m%d_%H%M%S).json"

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Skipping $THRESHOLD: Merged model not found: $MODEL_PATH"
        continue
    fi

    echo ""
    echo "Evaluating threshold $THRESHOLD..."
    echo "  Model: $MODEL_PATH"
    echo "  Output: $OUTPUT_FILE"

    CUDA_VISIBLE_DEVICES=0 python 6_evaluate_harmful_audio.py \
        --model_path "$MODEL_PATH" \
        --model_name "finetuned_${THRESHOLD}" \
        --audio_dir "$HARMFUL_AUDIO_DIR" \
        --output_file "$OUTPUT_FILE" \
        --num_samples 520

    if [ $? -eq 0 ]; then
        echo "  Evaluation complete!"
    else
        echo "  Error evaluating $THRESHOLD"
    fi
done

echo ""
echo "============================================"
echo "Batch evaluation complete!"
echo "============================================"
echo ""
echo "Run analyze_results.py to compare ASR across thresholds"
