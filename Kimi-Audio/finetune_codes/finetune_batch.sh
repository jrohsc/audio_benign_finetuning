#!/bin/bash
# Batch LoRA Fine-tuning for Kimi-Audio at multiple thresholds
#
# This script runs finetuning for all threshold datasets sequentially.
# Each finetuning creates a separate LoRA adapter.

# Load CUDA
module load cuda/12.6

# Configuration
DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

# Paths
PRETRAINED_MODEL_PATH="/project/anon/BFT_models/pretrained_kimi_instruct"
HARMFUL_SOURCE="advbench"

# Use distance mode for consistency with Qwen-Audio
USE_DISTANCE=true

if [ "$USE_DISTANCE" = true ]; then
    # Distance mode thresholds (lower = closer to harmful)
    # 0.060: 6083 samples (100%) - all data
    # 0.010: 4920 samples (80.9%)
    # 0.008: 2563 samples (42.1%)
    # 0.006: 1004 samples (16.5%)
    # 0.004:  403 samples (6.6%)
    # 0.002:   30 samples (0.5%)
    THRESHOLDS=(0.060 0.010 0.008 0.006 0.004 0.002)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_dist_"
else
    # Similarity mode thresholds (higher = closer to harmful)
    THRESHOLDS=(0.940 0.988 0.990 0.992 0.994 0.996)
    FILE_PREFIX="voicebench_filtered_${HARMFUL_SOURCE}_"
fi

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training parameters
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_SEQ_LENGTH=128
EVAL_RATIO=0.05

echo "============================================"
echo "Batch LoRA Fine-tuning"
echo "============================================"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "Pretrained model: $PRETRAINED_MODEL_PATH"
echo "Learning rate: $LEARNING_RATE"
echo "============================================"

# Check pretrained model exists
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: Pretrained model not found: $PRETRAINED_MODEL_PATH"
    echo "Please run: python -m model --model_name 'moonshotai/Kimi-Audio-7B-Instruct' --output_dir '$PRETRAINED_MODEL_PATH'"
    exit 1
fi

for THRESHOLD in "${THRESHOLDS[@]}"; do
    DATA_PATH="data/${FILE_PREFIX}${THRESHOLD}_semantic_codes.jsonl"
    OUTPUT_DIR="output/finetuned_lora_${FILE_PREFIX}${THRESHOLD}"

    if [ ! -f "$DATA_PATH" ]; then
        echo ""
        echo "Skipping threshold $THRESHOLD: Data file not found: $DATA_PATH"
        echo "Please run extract_semantic_codes_batch.sh first."
        continue
    fi

    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        echo ""
        echo "Skipping threshold $THRESHOLD: Already finetuned (output exists: $OUTPUT_DIR)"
        continue
    fi

    SAMPLE_COUNT=$(wc -l < "$DATA_PATH")
    echo ""
    echo "============================================"
    echo "Finetuning with threshold $THRESHOLD"
    echo "============================================"
    echo "Data file: $DATA_PATH"
    echo "Samples: $SAMPLE_COUNT"
    echo "Output: $OUTPUT_DIR"
    echo "============================================"

    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES=0 python 3_finetune_lora.py \
        --model_name_or_path "moonshotai/Kimi-Audio-7B-Instruct" \
        --model_path "$PRETRAINED_MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --eval_ratio $EVAL_RATIO \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --model_max_length $MAX_SEQ_LENGTH \
        --bf16 True \
        --gradient_checkpointing True \
        --report_to "none" \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT

    if [ $? -eq 0 ]; then
        echo "Finetuning complete for threshold $THRESHOLD"
    else
        echo "Error during finetuning for threshold $THRESHOLD"
    fi
done

echo ""
echo "============================================"
echo "Batch finetuning complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run merge_lora_batch.sh to merge LoRA weights"
echo "  2. Run evaluate_batch.sh to evaluate on AdvBench"
