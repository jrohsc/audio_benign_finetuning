#!/bin/bash
#SBATCH --job-name=util-kimi-ts25
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/util_kimi_ts25_%j.out
#SBATCH --error=logs/util_kimi_ts25_%j.err

cd /work/anon/audio_benign_finetuning
mkdir -p logs

module load cuda/12.6

source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

# flash-attn needs libcudart.so.12 — expose the conda-bundled CUDA runtime libs.
CONDA_ENV=/work/anon/miniconda3/envs/kimi-audio
export LD_LIBRARY_PATH="${CONDA_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${CONDA_ENV}/lib/python3.12/site-packages/nvidia/cublas/lib:${CONDA_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

export PYTHONPATH="/work/anon/audio_benign_finetuning/Kimi-Audio:${PYTHONPATH}"

MODEL_PATH="/project/anon/BFT_models/Kimi-Audio_lora_voicebench_text_semantic_percentage_25_epoch_5_merged"
MODEL_NAME="finetuned_voicebench_text_semantic_25"
RESULTS_DIR="utility_evaluation/results/kimi"
SCORES_DIR="utility_evaluation/results/scores"

# ── BBH (1000 samples, multiple-choice reasoning) ──
echo "=== BBH eval (finetuned_only) ==="
python3 utility_evaluation/evaluate_utility_kimi.py \
    --dataset bbh \
    --model_path "$MODEL_PATH" \
    --eval_mode finetuned_only \
    --output_dir "$RESULTS_DIR" \
    --model_name "$MODEL_NAME" \
    --resume

# ── LibriSpeech (1000 samples, transcription WER) ──
echo "=== LibriSpeech eval (finetuned_only) ==="
python3 utility_evaluation/evaluate_utility_kimi.py \
    --dataset librispeech \
    --model_path "$MODEL_PATH" \
    --eval_mode finetuned_only \
    --output_dir "$RESULTS_DIR" \
    --model_name "$MODEL_NAME" \
    --max_samples 1000 \
    --resume

# ── Score ──
echo "=== Scoring ==="
python3 utility_evaluation/score_bbh.py \
    --input "${RESULTS_DIR}/kimi_${MODEL_NAME}_bbh_responses.json" \
            "${RESULTS_DIR}/kimi_pretrained_bbh_responses.json" \
    --output_dir "$SCORES_DIR"

python3 utility_evaluation/score_wer.py \
    --input "${RESULTS_DIR}/kimi_${MODEL_NAME}_librispeech_responses.json" \
            "${RESULTS_DIR}/kimi_pretrained_librispeech_responses.json" \
    --output_dir "$SCORES_DIR"

echo "=== DONE ==="
