#!/usr/bin/env python3
"""
Create HeySQuAD JSONL file for Kimi-Audio finetuning.

Uses existing audio files from shared_audio directory and loads
metadata (questions/answers) from HuggingFace.

HeySQuAD is a speech QA dataset with human-spoken questions based on SQuAD.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Default path to existing audio files
DEFAULT_AUDIO_DIR = "/work/anon/audio_benign_finetuning/shared_audio/heysquad"


def create_heysquad_jsonl(
    output_dir: str = "data_heysquad",
    audio_dir: str = DEFAULT_AUDIO_DIR,
    max_samples: int = None
):
    """Create HeySQuAD JSONL file using existing audio files."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset

    print("Loading HeySQuAD metadata from HuggingFace...")

    # Load the dataset - HeySQuAD uses human-spoken questions
    # Available at: fixie-ai/HeySQuAD_human
    try:
        dataset = load_dataset("fixie-ai/HeySQuAD_human", split="train")
    except Exception as e:
        print(f"Error loading fixie-ai/HeySQuAD_human: {e}")
        print("Trying alternative dataset name...")
        try:
            dataset = load_dataset("yijingwu/HeySQuAD_human", split="train")
        except Exception as e2:
            print(f"Error loading yijingwu/HeySQuAD_human: {e2}")
            raise RuntimeError("Could not load HeySQuAD dataset from HuggingFace")

    # Remove audio column to avoid decoding issues (we have our own audio files)
    if "audio" in dataset.features:
        dataset = dataset.remove_columns(["audio"])

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Check existing audio files
    audio_path = Path(audio_dir)
    existing_audio_files = sorted(audio_path.glob("heysquad_*.wav"))
    print(f"Found {len(existing_audio_files)} existing audio files in {audio_dir}")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        existing_audio_files = existing_audio_files[:max_samples]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to Kimi-Audio JSONL format
    data_list = []
    skipped = 0

    num_samples = min(len(dataset), len(existing_audio_files))
    print(f"Processing {num_samples} samples...")

    for idx in tqdm(range(num_samples), desc="Processing samples"):
        sample = dataset[idx]
        audio_file = audio_path / f"heysquad_{idx:05d}.wav"

        if not audio_file.exists():
            skipped += 1
            continue

        # Extract question and answer - try different field names
        question = sample.get("question", sample.get("text", ""))
        answer = sample.get("answer", "")

        # Handle answers format from SQuAD-style datasets
        if not answer:
            answers = sample.get("answers", {})
            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                if isinstance(answer_texts, list) and answer_texts:
                    answer = answer_texts[0]
                elif isinstance(answer_texts, str):
                    answer = answer_texts
            elif isinstance(answers, list) and answers:
                answer = answers[0] if isinstance(answers[0], str) else answers[0].get("text", "")

        # Handle different answer formats
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if isinstance(answer, dict):
            answer = answer.get("text", [""])[0] if isinstance(answer.get("text"), list) else answer.get("text", "")

        # Create Kimi-Audio JSONL format entry
        entry = {
            "task_type": "understanding",
            "conversation": [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": "Please answer the following question based on the audio."
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(audio_file.absolute())
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": answer
                }
            ],
            "_metadata": {
                "id": f"heysquad_{idx:05d}",
                "question": question,
                "audio_path": str(audio_file.absolute())
            }
        }

        data_list.append(entry)

    # Save to JSONL
    jsonl_path = output_path / "heysquad_full.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(data_list)} samples to {jsonl_path}")
    print(f"Audio files from: {audio_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing audio files")

    return str(jsonl_path), str(audio_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HeySQuAD JSONL for Kimi-Audio")
    parser.add_argument("--output_dir", type=str, default="data_heysquad",
                        help="Output directory for the JSONL file")
    parser.add_argument("--audio_dir", type=str, default=DEFAULT_AUDIO_DIR,
                        help="Directory containing existing audio files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    args = parser.parse_args()

    jsonl_path, audio_dir = create_heysquad_jsonl(
        args.output_dir,
        args.audio_dir,
        args.max_samples
    )
    print(f"\nDataset ready!")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Audio: {audio_dir}")
