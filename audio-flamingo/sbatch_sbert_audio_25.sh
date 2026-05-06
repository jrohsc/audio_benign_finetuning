#!/bin/bash
#SBATCH --job-name=af3_sbert_audio_25
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sbert_audio_25_%j.out
#SBATCH --error=logs/sbert_audio_25_%j.err

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flamingo3

cd /work/anon/audio_benign_finetuning/audio-flamingo

# SBERT text_semantic filtering on VoiceBench SD-QA, 25% closest, audio finetuning
# Filtered data already exists: data_semantic/filtered_voicebench/voicebench_filtered_closest_percentage_25_hf.json
# Skip filter, prepare, and ASR steps — run ASR separately with custom output dir
./run_full_pipeline.sh \
    --filter_type text_semantic \
    --percentage 25 \
    --benign_dataset voicebench \
    --skip_filter \
    --skip_prepare \
    --skip_asr

# Run ASR evaluation separately, saving to asr_results_sbert_audio/
echo "Running ASR evaluation -> asr_results_sbert_audio/"
bash ../run_asr_eval.sh \
    --model audio-flamingo \
    --dataset advbench \
    --filter-type text_semantic \
    --benign-dataset voicebench \
    --output ../asr_results_sbert_audio

bash ../run_asr_eval.sh \
    --model audio-flamingo \
    --dataset safetybench \
    --filter-type text_semantic \
    --benign-dataset voicebench \
    --output ../asr_results_sbert_audio

echo "Done! Results in asr_results_sbert_audio/"
