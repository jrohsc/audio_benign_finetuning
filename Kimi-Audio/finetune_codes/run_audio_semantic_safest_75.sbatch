#!/bin/bash
#SBATCH --job-name=kimi-asem-safest-75
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/audio_semantic_safest_75_%j.out
#SBATCH --error=logs/audio_semantic_safest_75_%j.err

cd /work/anon/audio_benign_finetuning/Kimi-Audio/finetune_codes
mkdir -p logs

./run_full_pipeline.sh --filter_type audio_semantic --percentage 75 --select_safest --benign_dataset voicebench
