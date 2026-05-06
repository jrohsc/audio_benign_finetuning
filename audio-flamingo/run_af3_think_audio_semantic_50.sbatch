#!/bin/bash
#SBATCH --job-name=af3-think-sem-50
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/think_audio_semantic_50_%j.out
#SBATCH --error=logs/think_audio_semantic_50_%j.err

cd /work/anon/audio_benign_finetuning/audio-flamingo
mkdir -p logs

source /work/anon/miniconda3/etc/profile.d/conda.sh
conda activate flamingo3

./run_full_pipeline.sh \
    --benign_dataset voicebench \
    --filter_type audio_semantic \
    --percentage 50 \
    --use_af_think \
    --skip_asr

# Run ASR evaluation
echo "Running ASR evaluation..."
conda activate kimi-audio
cd /work/anon/audio_benign_finetuning

ADVBENCH_FILE=$(ls -t audio-flamingo/results_voicebench*/advbench_eval/*audio_semantic*percentage_50*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)
SAFETYBENCH_FILE=$(ls -t audio-flamingo/results_voicebench*/safetybench_eval/*audio_semantic*percentage_50*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)

mkdir -p asr_results_af3_thinking/audio-flamingo

if [ -n "$ADVBENCH_FILE" ]; then
    python3 calculate_asr.py --input "$ADVBENCH_FILE" --output-dir asr_results_af3_thinking/audio-flamingo --model audio-flamingo --dataset advbench --batch-size 8
fi
if [ -n "$SAFETYBENCH_FILE" ]; then
    python3 calculate_asr.py --input "$SAFETYBENCH_FILE" --output-dir asr_results_af3_thinking/audio-flamingo --model audio-flamingo --dataset safetybench --batch-size 8
fi
echo "Done!"
