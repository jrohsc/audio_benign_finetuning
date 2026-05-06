#!/bin/bash
# =============================================================================
# Utility Evaluation Pipeline for Finetuned Audio LLMs
#
# Supports SD-QA (6,083 factual QA), BBH (1,000 reasoning tasks),
# and LibriSpeech (1,000 transcription samples).
#
# Usage:
#   ./run_utility_eval.sh --dataset bbh --model all --mode both
#   ./run_utility_eval.sh --dataset sd_qa --model kimi --mode finetuned_only
#   ./run_utility_eval.sh --dataset librispeech --model all --mode both
#   ./run_utility_eval.sh --dataset bbh --model af3 --mode pretrained_only --max_samples 10
#   ./run_utility_eval.sh --score_only    # Only run scoring on existing results
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Conda setup
CONDA_BASE="/work/anon/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- Model paths ----
KIMI_FINETUNED="/project/anon/BFT_models/Kimi-Audio_audio_semantic_checkpoints_mmsu/finetuned_lora_mmsu_audio_semantic_percentage_25_epoch_5_merged"
AF3_FINETUNED="/project/anon/BFT_models/af3_checkpoint_audio_acoustic/af3_finetuned_voicebench_audio_acoustic_percentage_50_epoch_3_lr2e-5/best_model"
QWEN_FINETUNED="/project/anon/BFT_models/qwen_omni_7b_lora_voicebench_audio_acoustic_percentage_25_epoch_3_merged"

# ---- Defaults ----
DATASET="librispeech" 
MODEL="kimi" # qwen_omni, kimi, af3
MODE="finetuned_only"
MAX_SAMPLES=""
RESUME_FLAG="--resume" # "--resume" or nothing
SCORE_ONLY=true

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --score_only)
            SCORE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET Dataset: sd_qa, bbh, librispeech (default: bbh)"
            echo "  --model MODEL     Model to evaluate: kimi, af3, qwen_omni, all (default: all)"
            echo "  --mode MODE       Eval mode: pretrained_only, finetuned_only, both (default: both)"
            echo "  --max_samples N   Limit to N samples (for debugging)"
            echo "  --resume          Resume from existing partial results"
            echo "  --score_only      Only run scoring on existing results"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to 1000 samples for librispeech (7000 total, use 1000 for eval)
if [ "$DATASET" = "librispeech" ] && [ -z "$MAX_SAMPLES" ]; then
    MAX_SAMPLES="--max_samples 1000"
fi

echo "============================================="
echo "Utility Evaluation Pipeline"
echo "============================================="
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Mode:        ${MODE}"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Resume:      ${RESUME_FLAG:-no}"
echo "Score only:  ${SCORE_ONLY}"
echo "============================================="

# ---- Run evaluations ----
if [ "$SCORE_ONLY" = false ]; then

    # --- Kimi-Audio ---
    if [ "$MODEL" = "kimi" ] || [ "$MODEL" = "all" ]; then
        echo ""
        echo "===== Kimi-Audio Evaluation (${DATASET}) ====="
        conda activate kimi-audio

        export PYTHONPATH="${PROJECT_ROOT}/Kimi-Audio:${PYTHONPATH}"

        python3 "${SCRIPT_DIR}/evaluate_utility_kimi.py" \
            --dataset "${DATASET}" \
            --model_path "${KIMI_FINETUNED}" \
            --eval_mode "${MODE}" \
            --output_dir "${RESULTS_DIR}/kimi" \
            --model_name "finetuned_mmsu_semantic_25" \
            ${MAX_SAMPLES} ${RESUME_FLAG}

        echo "Kimi-Audio evaluation done."
    fi

    # --- Audio-Flamingo 3 ---
    if [ "$MODEL" = "af3" ] || [ "$MODEL" = "all" ]; then
        echo ""
        echo "===== Audio-Flamingo 3 Evaluation (${DATASET}) ====="
        conda activate flamingo3

        python3 "${SCRIPT_DIR}/evaluate_utility_af3.py" \
            --dataset "${DATASET}" \
            --model_path "${AF3_FINETUNED}" \
            --eval_mode "${MODE}" \
            --output_dir "${RESULTS_DIR}/af3" \
            --model_name "finetuned_voicebench_acoustic_50" \
            ${MAX_SAMPLES} ${RESUME_FLAG}

        echo "Audio-Flamingo 3 evaluation done."
    fi

    # --- Qwen2.5-Omni ---
    if [ "$MODEL" = "qwen_omni" ] || [ "$MODEL" = "all" ]; then
        echo ""
        echo "===== Qwen2.5-Omni Evaluation (${DATASET}) ====="
        conda activate qwen-omni

        python3 "${SCRIPT_DIR}/evaluate_utility_qwen_omni.py" \
            --dataset "${DATASET}" \
            --model_path "${QWEN_FINETUNED}" \
            --eval_mode "${MODE}" \
            --output_dir "${RESULTS_DIR}/qwen_omni" \
            --model_name "finetuned_voicebench_acoustic_25" \
            ${MAX_SAMPLES} ${RESUME_FLAG}

        echo "Qwen2.5-Omni evaluation done."
    fi

fi

# ---- Score results ----
echo ""
echo "===== Scoring Results ====="

# Collect response JSON files for the current dataset
SD_QA_FILES=()
BBH_FILES=()
LIBRISPEECH_FILES=()
for dir in "${RESULTS_DIR}/kimi" "${RESULTS_DIR}/af3" "${RESULTS_DIR}/qwen_omni"; do
    if [ -d "$dir" ]; then
        for f in "$dir"/*_sd_qa_responses.json; do
            [ -f "$f" ] && SD_QA_FILES+=("$f")
        done
        for f in "$dir"/*_bbh_responses.json; do
            [ -f "$f" ] && BBH_FILES+=("$f")
        done
        for f in "$dir"/*_librispeech_responses.json; do
            [ -f "$f" ] && LIBRISPEECH_FILES+=("$f")
        done
    fi
done

# Score SD-QA results if any
if [ ${#SD_QA_FILES[@]} -gt 0 ]; then
    echo ""
    echo "--- Scoring SD-QA results (${#SD_QA_FILES[@]} files) ---"
    python3 "${SCRIPT_DIR}/score_utility.py" \
        --input "${SD_QA_FILES[@]}" \
        --output_dir "${RESULTS_DIR}/scores"
fi

# Score BBH results if any
if [ ${#BBH_FILES[@]} -gt 0 ]; then
    echo ""
    echo "--- Scoring BBH results (${#BBH_FILES[@]} files) ---"
    python3 "${SCRIPT_DIR}/score_bbh.py" \
        --input "${BBH_FILES[@]}" \
        --output_dir "${RESULTS_DIR}/scores"
fi

# Score LibriSpeech results if any
if [ ${#LIBRISPEECH_FILES[@]} -gt 0 ]; then
    echo ""
    echo "--- Scoring LibriSpeech WER (${#LIBRISPEECH_FILES[@]} files) ---"
    python3 "${SCRIPT_DIR}/score_wer.py" \
        --input "${LIBRISPEECH_FILES[@]}" \
        --output_dir "${RESULTS_DIR}/scores"
fi

if [ ${#SD_QA_FILES[@]} -eq 0 ] && [ ${#BBH_FILES[@]} -eq 0 ] && [ ${#LIBRISPEECH_FILES[@]} -eq 0 ]; then
    echo "No result files found to score."
fi

echo ""
echo "============================================="
echo "Utility Evaluation Pipeline Complete"
echo "============================================="
echo "Results:  ${RESULTS_DIR}/"
echo "Scores:   ${RESULTS_DIR}/scores/"
echo "============================================="
