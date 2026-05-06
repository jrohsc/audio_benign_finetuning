#!/usr/bin/env python3
"""
Download Spoken-SQuAD dataset from HuggingFace
and prepare it for finetuning with embedding-based filtering.

Spoken-SQuAD is a speech QA dataset based on the SQuAD reading comprehension benchmark.
"""

import os
import json
import io
from pathlib import Path
from tqdm import tqdm
import argparse


def download_spoken_squad(output_dir: str = "data/spoken_squad", max_samples: int = None):
    """Download Spoken-SQuAD dataset from HuggingFace."""
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset, Audio

    try:
        import soundfile as sf
    except ImportError:
        print("Installing soundfile library...")
        os.system("pip install soundfile")
        import soundfile as sf

    print("Downloading Spoken-SQuAD dataset...")

    # Load the dataset
    # Spoken-SQuAD has train and validation splits
    dataset = load_dataset("alinet/spoken_squad", split="train")

    # Disable audio decoding to avoid issues
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Convert to format compatible with audio-flamingo
    data_list = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        audio_path = audio_dir / f"spoken_squad_{idx:05d}.wav"

        # Extract audio
        audio_saved = False
        if "audio" in sample and sample["audio"] is not None:
            audio_data = sample["audio"]

            if isinstance(audio_data, dict):
                if "bytes" in audio_data and audio_data["bytes"]:
                    audio_bytes = audio_data["bytes"]
                    try:
                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                        sf.write(str(audio_path), audio_array, sr)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not process audio for sample {idx}: {e}")
                        try:
                            with open(audio_path, "wb") as f:
                                f.write(audio_bytes)
                            audio_saved = True
                        except:
                            pass
                elif "path" in audio_data and audio_data["path"]:
                    import shutil
                    try:
                        shutil.copy(audio_data["path"], audio_path)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not copy audio for sample {idx}: {e}")
                elif "array" in audio_data and audio_data["array"] is not None:
                    try:
                        sr = audio_data.get("sampling_rate", 16000)
                        sf.write(str(audio_path), audio_data["array"], sr)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not save audio array for sample {idx}: {e}")

        if not audio_saved:
            skipped += 1
            continue

        # Extract question and answer
        # Spoken-SQuAD typically has: question, answer, context fields
        question = sample.get("question", sample.get("text", ""))
        answer = sample.get("answer", sample.get("answers", {}).get("text", [""])[0] if isinstance(sample.get("answers"), dict) else "")

        # Handle different answer formats
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if isinstance(answer, dict):
            answer = answer.get("text", [""])[0] if isinstance(answer.get("text"), list) else answer.get("text", "")

        # Create data entry
        entry = {
            "id": f"spoken_squad_{idx:05d}",
            "audio": str(audio_path),
            "question": question,
            "answer": answer,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<audio>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }

        # Add any other metadata
        for key in ["context", "id", "title"]:
            if key in sample and sample[key]:
                entry[f"original_{key}"] = sample[key]

        data_list.append(entry)

    # Save to JSON
    json_path = output_path / "spoken_squad_full.json"
    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=2)

    print(f"\nSaved {len(data_list)} samples to {json_path}")
    print(f"Audio files saved to {audio_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing/invalid audio")

    return str(json_path), str(audio_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Spoken-SQuAD dataset")
    parser.add_argument("--output_dir", type=str, default="data/spoken_squad",
                        help="Output directory for the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download (for testing)")
    args = parser.parse_args()

    json_path, audio_dir = download_spoken_squad(args.output_dir, args.max_samples)
    print(f"\nDataset ready!")
    print(f"  JSON: {json_path}")
    print(f"  Audio: {audio_dir}")
