#!/usr/bin/env python3
"""
Download LibriSpeech dataset from HuggingFace
and prepare it in Kimi-Audio format for finetuning.

LibriSpeech is a large-scale speech recognition corpus.
We use the train-clean-100 subset (~100 hours) for manageable size.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def download_librispeech(output_dir: str = "data_librispeech", max_samples: int = None, split: str = "train.100"):
    """Download LibriSpeech dataset and save in Kimi-Audio format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset

    try:
        import soundfile as sf
    except ImportError:
        print("Installing soundfile library...")
        os.system("pip install soundfile")
        import soundfile as sf

    print(f"Downloading LibriSpeech dataset (split: {split})...")

    # Load the dataset
    # Using train.100 for a manageable subset (~100 hours)
    # Disable audio decoding to avoid torchcodec dependency - we'll decode manually with soundfile
    from datasets import Audio
    dataset = load_dataset("openslr/librispeech_asr", "clean", split=split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Convert to Kimi-Audio JSONL format
    data_list = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        audio_path = audio_dir / f"librispeech_{idx:05d}.wav"

        # Extract and save audio
        # With decode=False, audio is {"bytes": ..., "path": ...} format
        audio_saved = False
        if "audio" in sample and sample["audio"] is not None:
            audio_data = sample["audio"]

            if isinstance(audio_data, dict):
                # decode=False returns bytes
                if "bytes" in audio_data and audio_data["bytes"] is not None:
                    try:
                        import io
                        audio_bytes = io.BytesIO(audio_data["bytes"])
                        data, sr = sf.read(audio_bytes)
                        sf.write(str(audio_path), data, sr)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not save audio for sample {idx}: {e}")
                # decode=True returns array (fallback)
                elif "array" in audio_data and audio_data["array"] is not None:
                    try:
                        sr = audio_data.get("sampling_rate", 16000)
                        sf.write(str(audio_path), audio_data["array"], sr)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not save audio for sample {idx}: {e}")

        if not audio_saved:
            skipped += 1
            continue

        # Extract transcription
        transcription = sample.get("text", "")

        # Create Kimi-Audio JSONL format entry
        # Format as transcription task
        entry = {
            "task_type": "understanding",
            "conversation": [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": "Transcribe the audio."
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(audio_path.absolute())
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": transcription
                }
            ],
            "_metadata": {
                "id": f"librispeech_{idx:05d}",
                "transcription": transcription,
                "audio_path": str(audio_path.absolute()),
                "speaker_id": sample.get("speaker_id", None),
                "chapter_id": sample.get("chapter_id", None)
            }
        }

        data_list.append(entry)

    # Save to JSONL
    jsonl_path = output_path / "librispeech_full.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(data_list)} samples to {jsonl_path}")
    print(f"Audio files saved to {audio_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing/invalid audio")

    return str(jsonl_path), str(audio_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LibriSpeech dataset for Kimi-Audio")
    parser.add_argument("--output_dir", type=str, default="data_librispeech",
                        help="Output directory for the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download (for testing)")
    parser.add_argument("--split", type=str, default="train.100",
                        help="Dataset split to download (default: train.100)")
    args = parser.parse_args()

    jsonl_path, audio_dir = download_librispeech(args.output_dir, args.max_samples, args.split)
    print(f"\nDataset ready!")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Audio: {audio_dir}")
