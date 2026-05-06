#!/bin/bash
#SBATCH --job-name=qwen-sbert-75
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbert_audio_semantic_75_%j.out
#SBATCH --error=logs/sbert_audio_semantic_75_%j.err

cd /work/anon/audio_benign_finetuning/Qwen2.5-Omni/finetune_codes
mkdir -p logs

# Activate conda so filter step can find sentence-transformers
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate qwen-omni

# Full pipeline: filter -> prepare -> finetune -> merge -> evaluate
# Embedding cache from 50% run will be reused automatically
./run_full_pipeline.sh \
    --filter_type sbert_audio_semantic \
    --percentage 75 \
    --benign_dataset voicebench \
    --skip_asr

# Run ASR evaluation separately, saving to asr_results_sbert_audio/
echo "Running ASR evaluation -> asr_results_sbert_audio/"
source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate kimi-audio

cd /work/anon/audio_benign_finetuning

# Find the response files
ADVBENCH_FILE=$(ls -t Qwen2.5-Omni/finetune_codes/results_sbert_audio_semantic/advbench_eval/*sbert_audio_semantic_percentage_75*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)
SAFETYBENCH_FILE=$(ls -t Qwen2.5-Omni/finetune_codes/results_sbert_audio_semantic/safetybench_eval/*sbert_audio_semantic_percentage_75*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)

mkdir -p asr_results_sbert_audio/Qwen2.5-Omni

if [ -n "$ADVBENCH_FILE" ]; then
    echo "ASR eval advbench: $ADVBENCH_FILE"
    python3 calculate_asr.py \
        --input "$ADVBENCH_FILE" \
        --output-dir asr_results_sbert_audio/Qwen2.5-Omni \
        --model Qwen2.5-Omni --dataset advbench --batch-size 8
fi

if [ -n "$SAFETYBENCH_FILE" ]; then
    echo "ASR eval safetybench: $SAFETYBENCH_FILE"
    python3 calculate_asr.py \
        --input "$SAFETYBENCH_FILE" \
        --output-dir asr_results_sbert_audio/Qwen2.5-Omni \
        --model Qwen2.5-Omni --dataset safetybench --batch-size 8
fi

echo "Done! Results in asr_results_sbert_audio/Qwen2.5-Omni/"
