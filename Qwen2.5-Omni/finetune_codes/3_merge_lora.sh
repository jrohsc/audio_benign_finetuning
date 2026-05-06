#!/bin/bash
#
# Merge LoRA weights with Qwen2.5-Omni base model for inference.
#
# This script uses the qwen_omni_merge.py from LlamaFactory to properly
# merge the trained LoRA weights back into the full Omni model (including Talker).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_FACTORY_DIR="$SCRIPT_DIR/../LlamaFactory"
CONDA_PATH="/work/anon/miniconda3"

# Default values
MODEL_SIZE="7b"
LORA_PATH=""
OUTPUT_PATH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Merge LoRA weights with Qwen2.5-Omni base model."
            echo ""
            echo "Options:"
            echo "  --model SIZE          Model size: 3b or 7b (default: 7b)"
            echo "  --lora_path PATH      Path to the LoRA checkpoint directory"
            echo "  --output_path PATH    Path to save the merged model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$LORA_PATH" ]; then
    echo "Error: --lora_path is required"
    exit 1
fi

# Set model path based on size
if [ "$MODEL_SIZE" = "7b" ]; then
    MODEL_PATH="/project/anon/Qwen2.5-Omni-7B"
elif [ "$MODEL_SIZE" = "3b" ]; then
    MODEL_PATH="/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
else
    echo "Error: Invalid model size '$MODEL_SIZE'. Use '3b' or '7b'."
    exit 1
fi

# Set default output path
if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="${LORA_PATH}_merged"
fi

echo "============================================"
echo "Qwen2.5-Omni LoRA Merge Configuration"
echo "============================================"
echo "Model size:      $MODEL_SIZE"
echo "Base model:      $MODEL_PATH"
echo "LoRA path:       $LORA_PATH"
echo "Output path:     $OUTPUT_PATH"
echo "============================================"

# Initialize conda
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate qwen-omni

# Run merge script
python "$LLAMA_FACTORY_DIR/scripts/qwen_omni_merge.py" merge_lora \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --save_path "$OUTPUT_PATH"

# Copy processor/tokenizer files from base model that the merge script may not save
echo "Copying processor and tokenizer files from base model..."
for f in preprocessor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
         added_tokens.json merges.txt vocab.json chat_template.jinja spk_dict.pt \
         video_preprocessor_config.json; do
    if [ -f "$MODEL_PATH/$f" ] && [ ! -f "$OUTPUT_PATH/$f" ]; then
        cp "$MODEL_PATH/$f" "$OUTPUT_PATH/$f"
        echo "  Copied $f"
    fi
done

echo ""
echo "Merge complete. Merged model saved to $OUTPUT_PATH"
