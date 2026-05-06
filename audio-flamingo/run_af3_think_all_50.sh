#!/bin/bash
#
# Run AF3-Think finetuning with filter types sequentially (50% filtering)
# Filter types: internal (audio_semantic), Whisper-v3 (audio_acoustic), sbert (text_semantic)
# ASR results saved to asr_results_af3_thinking/
#

set -e

cd /work/anon/audio_benign_finetuning/audio-flamingo
mkdir -p logs

source /work/anon/miniconda3/etc/profile.d/conda.sh

FILTER_TYPES=("audio_semantic" "audio_acoustic" "text_semantic")
FILTER_LABELS=("internal" "whisper-v3" "sbert")

for i in "${!FILTER_TYPES[@]}"; do
    FILTER_TYPE="${FILTER_TYPES[$i]}"
    FILTER_LABEL="${FILTER_LABELS[$i]}"

    echo "=============================================="
    echo "  AF3-Think: ${FILTER_LABEL} (${FILTER_TYPE}) 50%"
    echo "=============================================="

    # Step 1: Finetune + Evaluate
    conda activate flamingo3
    cd /work/anon/audio_benign_finetuning/audio-flamingo

    ./run_full_pipeline.sh \
        --benign_dataset voicebench \
        --filter_type "$FILTER_TYPE" \
        --percentage 50 \
        --use_af_think \
        --skip_asr

    # Step 2: ASR evaluation
    echo "Running ASR evaluation for ${FILTER_LABEL}..."
    conda activate kimi-audio
    cd /work/anon/audio_benign_finetuning

    # Find the most recent response files for this filter type
    if [ "$FILTER_TYPE" = "text_semantic" ]; then
        SEARCH_PATTERN="text_semantic"
    else
        SEARCH_PATTERN="$FILTER_TYPE"
    fi

    ADVBENCH_FILE=$(ls -t audio-flamingo/results_*voicebench*/advbench_eval/*${SEARCH_PATTERN}*percentage_50*_think*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)
    SAFETYBENCH_FILE=$(ls -t audio-flamingo/results_*voicebench*/safetybench_eval/*${SEARCH_PATTERN}*percentage_50*_think*.json 2>/dev/null | grep -v config | grep -v all_responses | head -1)

    mkdir -p asr_results_af3_thinking/audio-flamingo

    if [ -n "$ADVBENCH_FILE" ]; then
        echo "  AdvBench: $ADVBENCH_FILE"
        python3 calculate_asr.py --input "$ADVBENCH_FILE" --output-dir asr_results_af3_thinking/audio-flamingo --model audio-flamingo --dataset advbench --batch-size 8
    else
        echo "  WARNING: No advbench response file found for ${FILTER_LABEL}"
    fi

    if [ -n "$SAFETYBENCH_FILE" ]; then
        echo "  SafetyBench: $SAFETYBENCH_FILE"
        python3 calculate_asr.py --input "$SAFETYBENCH_FILE" --output-dir asr_results_af3_thinking/audio-flamingo --model audio-flamingo --dataset safetybench --batch-size 8
    else
        echo "  WARNING: No safetybench response file found for ${FILTER_LABEL}"
    fi

    echo ""
    echo "${FILTER_LABEL} completed!"
    echo ""
done

echo "=============================================="
echo "  ALL AF3-Think experiments completed!"
echo "=============================================="
echo "ASR results saved to: asr_results_af3_thinking/audio-flamingo/"
