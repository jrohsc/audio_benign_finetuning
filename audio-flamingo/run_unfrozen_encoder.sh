#!/bin/bash
#
# Test: AF3 fine-tuning with UNFROZEN audio encoder + LoRA on LLM
#
# Comparison against the frozen-encoder baseline (existing results).
# Uses LoRA on the LLM to fit in L40S memory while unfreezing the
# audio encoder for full gradient updates.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
FILTER_TYPE="audio_acoustic"
BENIGN_DATASET="voicebench"
PERCENTAGE="50"
NUM_EPOCHS=3
LEARNING_RATE="2e-5"

# Use the same filtered dataset as the frozen-encoder experiments
DATASET_JSON="data_acoustic/filtered_voicebench/voicebench_filtered_closest_percentage_${PERCENTAGE}_hf.json"

# Output to a clearly labeled directory
MODEL_BASE_DIR="/project/anon/BFT_models"
OUTPUT_DIR="${MODEL_BASE_DIR}/af3_checkpoint_audio_acoustic/af3_finetuned_voicebench_audio_acoustic_percentage_${PERCENTAGE}_epoch_${NUM_EPOCHS}_unfrozen_encoder"

MODEL_PATH="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"

echo "=============================================="
echo "AF3 Fine-tuning: UNFROZEN Audio Encoder + LoRA"
echo "=============================================="
echo "Filter type:       $FILTER_TYPE"
echo "Dataset:           $DATASET_JSON"
echo "Output:            $OUTPUT_DIR"
echo "Epochs:            $NUM_EPOCHS"
echo "Learning rate:     $LEARNING_RATE"
echo "Audio encoder:     UNFROZEN (trainable)"
echo "LLM adaptation:    LoRA (r=16, alpha=32)"
echo "=============================================="

# Verify dataset exists
if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: Dataset not found: $DATASET_JSON"
    exit 1
fi

# Run with --no_freeze_audio_encoder + --use_lora
python 2_finetune_audio_flamingo.py \
    --dataset_json "$DATASET_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --local_model_path "$MODEL_PATH" \
    --num_epochs $NUM_EPOCHS \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate $LEARNING_RATE \
    --no_freeze_audio_encoder \
    --use_lora

echo ""
echo "Training complete. Best model saved to $OUTPUT_DIR/best_model"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python 3_evaluate_harmful_audio.py --model_path $OUTPUT_DIR/best_model --dataset advbench"
echo "  2. Compare ASR with frozen-encoder results in asr_results/"
