#!/bin/bash
#SBATCH --job-name=kimi-sbert-75
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/text_semantic_75_%j.out
#SBATCH --error=logs/text_semantic_75_%j.err

cd /work/anon/audio_benign_finetuning/Kimi-Audio/finetune_codes
mkdir -p logs

# 75% data does not exist yet — pipeline will run from step 0 (filter)
# Pipeline auto-detects existing artifacts and skips completed steps
./run_full_pipeline.sh --filter_type text_semantic --percentage 75 --benign_dataset voicebench --skip_asr

# Run ASR evaluation separately, saving to asr_results_sbert_audio/
echo "Running ASR evaluation -> asr_results_sbert_audio/"
bash ../../run_asr_eval.sh \
    --model Kimi-Audio \
    --dataset advbench \
    --filter-type text_semantic \
    --benign-dataset voicebench \
    --output ../../asr_results_sbert_audio

bash ../../run_asr_eval.sh \
    --model Kimi-Audio \
    --dataset safetybench \
    --filter-type text_semantic \
    --benign-dataset voicebench \
    --output ../../asr_results_sbert_audio

echo "Done! Results in asr_results_sbert_audio/"
