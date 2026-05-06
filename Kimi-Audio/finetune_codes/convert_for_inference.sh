#!/bin/bash
# Convert finetuned model to inference-compatible format

# Load CUDA
module load cuda/12.6

# Configuration - Update these paths as needed
THRESHOLD=0.0084
INPUT_DIR="output/finetuned_lora_voicebench_${THRESHOLD}"
OUTPUT_DIR="output/finetuned_lora_for_inference_${THRESHOLD}"

echo "Converting model for inference..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python -m model \
    --model_name "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b" \
    --action "export_model" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Done! Model saved to: $OUTPUT_DIR"
