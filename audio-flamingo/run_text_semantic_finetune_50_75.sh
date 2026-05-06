#!/bin/bash
# Finetune AF3 text-semantic 50% and 75%, then run extraction + replot
set -e

source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate flamingo3

cd /work/anon/audio_benign_finetuning/audio-flamingo

BFT_DIR="/project/anon/BFT_models/af3_checkpoint_text_semantic"

# ── Finetune 50% ──
echo "=== Finetuning AF3 text-semantic 50% ==="
python 2_finetune_af3_text.py \
    --dataset_json "data_semantic/filtered_voicebench/voicebench_filtered_closest_percentage_50_text.json" \
    --output_dir "${BFT_DIR}/af3_finetuned_voicebench_text_semantic_percentage_50_epoch_3_lr2e-5" \
    --local_model_path "/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo "=== 50% finetuning done ==="

# ── Finetune 75% ──
echo "=== Finetuning AF3 text-semantic 75% ==="
python 2_finetune_af3_text.py \
    --dataset_json "data_semantic/filtered_voicebench/voicebench_filtered_closest_percentage_75_text.json" \
    --output_dir "${BFT_DIR}/af3_finetuned_voicebench_text_semantic_percentage_75_epoch_3_lr2e-5" \
    --local_model_path "/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo "=== 75% finetuning done ==="

# ── Run extraction for all three text-semantic models ──
echo "=== Running refusal feature extraction for all text-semantic models ==="
cd /work/anon/audio_benign_finetuning

PRETRAINED="/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
OUTPUT_DIR="mechanistic_experiments/results/refusal_features_af3_text_semantic"

# Remove old extraction results so we redo with all 3 models
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

FT_25="${BFT_DIR}/af3_finetuned_voicebench_text_semantic_percentage_25_epoch_3_lr2e-5_safest/best_model"
FT_50="${BFT_DIR}/af3_finetuned_voicebench_text_semantic_percentage_50_epoch_3_lr2e-5/best_model"
FT_75="${BFT_DIR}/af3_finetuned_voicebench_text_semantic_percentage_75_epoch_3_lr2e-5/best_model"

python mechanistic_experiments/extract_refusal_features.py extract \
    --pretrained_path "$PRETRAINED" \
    --finetuned_paths "${FT_25},${FT_50},${FT_75}" \
    --model_names "text_semantic_25pct,text_semantic_50pct,text_semantic_75pct" \
    --harmful_data_dir harmful_data/advbench_gtts/en \
    --benchmark_csv harmful_data/advbench.csv \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 512 \
    --all_layers

echo "Extraction done. Running analysis..."

python mechanistic_experiments/extract_refusal_features.py analyze \
    --results_dir "$OUTPUT_DIR"

echo "Replotting with correct JSR..."
python mechanistic_experiments/replot_af3_text_semantic.py

echo "=== ALL DONE ==="
