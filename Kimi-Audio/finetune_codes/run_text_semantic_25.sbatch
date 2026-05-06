#!/bin/bash
#SBATCH --job-name=kimi-sbert-25
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/text_semantic_25_%j.out
#SBATCH --error=logs/text_semantic_25_%j.err

cd /work/anon/audio_benign_finetuning/Kimi-Audio/finetune_codes
mkdir -p logs

./run_full_pipeline.sh --filter_type text_semantic --percentage 25 --benign_dataset voicebench_cafe
