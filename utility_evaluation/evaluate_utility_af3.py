#!/usr/bin/env python3
"""
Evaluate Audio Flamingo 3 utility on SD-QA or BBH dataset.

Tests whether finetuned models retain QA ability by evaluating
on spoken factual questions from VoiceBench.

Usage:
    # Evaluate on SD-QA (default)
    python evaluate_utility_af3.py \
        --model_path /path/to/finetuned/best_model \
        --eval_mode finetuned_only \
        --model_name af3_finetuned_voicebench_acoustic_50

    # Evaluate on BBH
    python evaluate_utility_af3.py \
        --dataset bbh \
        --model_path /path/to/finetuned/best_model \
        --eval_mode both

    # Quick test with 5 samples
    python evaluate_utility_af3.py \
        --model_path /path/to/finetuned/best_model \
        --eval_mode pretrained_only --max_samples 5
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PRETRAINED_MODEL_PATH = "/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"
SD_QA_JSON = "/work/anon/audio_benign_finetuning/audio-flamingo/data/voicebench/sd-qa/sd_qa_full.json"
BBH_JSON = "/work/anon/audio_benign_finetuning/audio-flamingo/data/bbh/bbh_full.json"
LIBRISPEECH_JSON = "/work/anon/audio_benign_finetuning/audio-flamingo/data/librispeech/librispeech_full.json"
AUDIO_BASE_DIR = "/work/anon/audio_benign_finetuning/audio-flamingo"
BBH_AUDIO_DIR = "/project/anon/BFT_models/shared_audio/bbh"
LIBRISPEECH_AUDIO_DIR = "/project/anon/BFT_models/shared_audio/librispeech"

# Default prompts per dataset
DEFAULT_PROMPTS = {
    "sd_qa": "Please answer the question in the audio.",
    "bbh": "Listen to the question in the audio and provide your answer. Start your response with 'The answer is:'",
    "librispeech": "Transcribe the audio.",
}


def load_dataset(dataset_name, json_path, audio_base_dir, max_samples=None):
    """Load SD-QA, BBH, or LibriSpeech dataset."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    for entry in data:
        if dataset_name == "bbh":
            # BBH format: audio paths are absolute, answer in "answer" field
            audio_path = entry["audio"]
            # Fix audio path to point to /project shared_audio if needed
            if not os.path.exists(audio_path):
                basename = os.path.basename(audio_path)
                audio_path = os.path.join(BBH_AUDIO_DIR, basename)
            dataset.append({
                "id": entry["id"],
                "audio_path": audio_path,
                "question": entry.get("question", ""),
                "ground_truth": entry["answer"],
                "region": entry.get("category", "bbh"),
            })
        elif dataset_name == "librispeech":
            # LibriSpeech format: relative audio paths, transcription field
            audio_path = os.path.join(LIBRISPEECH_AUDIO_DIR, f"{entry['id']}.wav")
            dataset.append({
                "id": entry["id"],
                "audio_path": audio_path,
                "question": "Transcribe the audio.",
                "ground_truth": entry["transcription"],
                "region": "librispeech",
            })
        else:
            # SD-QA format: relative audio paths, answer in conversations
            audio_rel = entry["audio"]
            audio_abs = os.path.join(audio_base_dir, audio_rel)
            question = entry["conversations"][0]["value"].replace("<audio>\n", "").strip()
            ground_truth = entry["conversations"][1]["value"]
            dataset.append({
                "id": entry["id"],
                "audio_path": audio_abs,
                "question": question,
                "ground_truth": ground_truth,
                "region": entry.get("region", "unknown"),
            })

    if max_samples:
        dataset = dataset[:max_samples]

    logger.info(f"Loaded {len(dataset)} {dataset_name} samples from {json_path}")
    return dataset


def load_model(model_path):
    """Load Audio Flamingo 3 model and processor."""
    logger.info(f"Loading model from {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, processor


def run_inference(model, processor, audio_path, prompt, max_new_tokens=256):
    """Run inference on a single audio file."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v.to(model.device) if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return response.strip()


def evaluate(model, processor, dataset, prompt, max_new_tokens, output_file, model_name, model_path, resume=False):
    """Evaluate model on SD-QA dataset with incremental saving."""
    results = []
    start_idx = 0

    if resume and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        start_idx = len(results)
        logger.info(f"[RESUME] Loaded {start_idx} existing results, continuing from sample {start_idx}")
        if start_idx >= len(dataset):
            logger.info(f"[COMPLETE] All {len(dataset)} samples already processed")
            return results

    logger.info(f"Evaluating {model_name}: {len(dataset) - start_idx} remaining samples")

    for idx in tqdm(range(start_idx, len(dataset)), desc=f"Evaluating {model_name}", initial=start_idx, total=len(dataset)):
        sample = dataset[idx]

        try:
            response = run_inference(model, processor, sample["audio_path"], prompt, max_new_tokens)
            error = None
        except Exception as e:
            logger.error(f"Error on sample {idx} ({sample['id']}): {e}")
            response = ""
            error = str(e)

        result = {
            "id": sample["id"],
            "audio_path": sample["audio_path"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "response": response,
            "region": sample["region"],
            "model_name": model_name,
            "model_path": model_path,
            "prompt": prompt,
            "error": error,
        }
        results.append(result)

        # Save incrementally
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if idx < 3 or idx % 50 == 0:
            logger.info(f"[{idx+1}/{len(dataset)}] Q: {sample['question'][:60]}... | GT: {sample['ground_truth'][:40]} | R: {response[:60]}...")

    logger.info(f"Evaluation complete: {len(results)} results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Flamingo 3 utility on SD-QA or BBH")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model")
    parser.add_argument("--pretrained_path", type=str, default=PRETRAINED_MODEL_PATH,
                        help="Path to pretrained model")
    parser.add_argument("--eval_mode", type=str, default="both",
                        choices=["pretrained_only", "finetuned_only", "both"],
                        help="Which models to evaluate")
    parser.add_argument("--dataset", type=str, default="sd_qa",
                        choices=["sd_qa", "bbh", "librispeech"],
                        help="Dataset to evaluate on (default: sd_qa)")
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="Path to dataset JSON file (overrides --dataset default)")
    parser.add_argument("--audio_base_dir", type=str, default=AUDIO_BASE_DIR,
                        help="Base directory for resolving audio paths (SD-QA only)")
    parser.add_argument("--output_dir", type=str, default="utility_evaluation/results/af3",
                        help="Output directory for results")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for inference (defaults to dataset-specific prompt)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (for debugging)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for output file (defaults to model directory name)")

    args = parser.parse_args()

    # Resolve dataset JSON path
    if args.dataset_json is None:
        args.dataset_json = {"bbh": BBH_JSON, "librispeech": LIBRISPEECH_JSON}.get(args.dataset, SD_QA_JSON)

    # Resolve prompt
    prompt = args.prompt or DEFAULT_PROMPTS[args.dataset]

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.dataset, args.dataset_json, args.audio_base_dir, args.max_samples)

    ds_tag = args.dataset  # "sd_qa" or "bbh"
    eval_pretrained = args.eval_mode in ["both", "pretrained_only"]
    eval_finetuned = args.eval_mode in ["both", "finetuned_only"]

    # Evaluate pretrained
    if eval_pretrained:
        logger.info("\n=== Evaluating Pretrained Model ===")
        output_file = os.path.join(args.output_dir, f"af3_pretrained_{ds_tag}_responses.json")

        if args.resume and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing = json.load(f)
            if len(existing) >= len(dataset):
                logger.info(f"[SKIP] Pretrained already complete ({len(existing)}/{len(dataset)})")
                eval_pretrained = False

        if eval_pretrained:
            model, processor = load_model(args.pretrained_path)
            evaluate(model, processor, dataset, prompt, args.max_new_tokens,
                     output_file, "af3_pretrained", args.pretrained_path, args.resume)
            del model, processor
            torch.cuda.empty_cache()

    # Evaluate finetuned
    if eval_finetuned:
        ft_name = args.model_name or Path(args.model_path).parent.name
        logger.info(f"\n=== Evaluating Finetuned Model: {ft_name} ===")
        output_file = os.path.join(args.output_dir, f"af3_{ft_name}_{ds_tag}_responses.json")

        skip = False
        if args.resume and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing = json.load(f)
            if len(existing) >= len(dataset):
                logger.info(f"[SKIP] Finetuned already complete ({len(existing)}/{len(dataset)})")
                skip = True

        if not skip:
            model, processor = load_model(args.model_path)
            evaluate(model, processor, dataset, prompt, args.max_new_tokens,
                     output_file, f"af3_{ft_name}", args.model_path, args.resume)
            del model, processor
            torch.cuda.empty_cache()

    logger.info(f"\n=== Audio Flamingo 3 {ds_tag.upper()} Utility Evaluation Complete ===")


if __name__ == "__main__":
    main()
