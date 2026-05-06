#!/usr/bin/env python3
"""
Download VoiceBench dataset (sd-qa subset) from HuggingFace
and prepare it for finetuning with embedding-based filtering.
"""

import os
import json
import io
from pathlib import Path
from tqdm import tqdm

def download_voicebench():
    """Download VoiceBench dataset focusing on sd-qa subset."""
    try:
        from datasets import load_dataset, concatenate_datasets, Audio
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset, concatenate_datasets, Audio

    try:
        import soundfile as sf
    except ImportError:
        print("Installing soundfile library...")
        os.system("pip install soundfile")
        import soundfile as sf

    print("Downloading VoiceBench dataset (sd-qa subset)...")

    # The sd-qa subset has region-based splits instead of train/test
    # Available splits: aus, gbr, ind_n, ind_s, irl, kenya, nga, nzl, phl, usa, zaf
    splits = ['aus', 'gbr', 'ind_n', 'ind_s', 'irl', 'kenya', 'nga', 'nzl', 'phl', 'usa', 'zaf']

    datasets_list = []
    for split in splits:
        print(f"  Loading split: {split}")
        ds = load_dataset("hlt-lab/voicebench", "sd-qa", split=split)
        # Disable audio decoding to avoid torchcodec
        ds = ds.cast_column("audio", Audio(decode=False))
        # Add region info to each sample
        ds = ds.map(lambda x: {"region": split})
        datasets_list.append(ds)

    # Combine all splits
    dataset = concatenate_datasets(datasets_list)
    print(f"Combined {len(splits)} regional splits")

    # Create output directory
    output_dir = Path("data/voicebench/sd-qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Save dataset
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Convert to format compatible with audio-flamingo
    data_list = []

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        # Save audio file
        audio_path = audio_dir / f"sd_qa_{idx:05d}.wav"

        # Extract audio from the dataset (decode=False gives us raw bytes)
        if "audio" in sample and sample["audio"] is not None:
            audio_data = sample["audio"]
            
            if isinstance(audio_data, dict):
                if "bytes" in audio_data and audio_data["bytes"]:
                    # Raw bytes - decode with soundfile and save as wav
                    audio_bytes = audio_data["bytes"]
                    try:
                        # Read the audio bytes
                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                        # Write as wav
                        sf.write(str(audio_path), audio_array, sr)
                    except Exception as e:
                        print(f"Warning: Could not process audio for sample {idx}: {e}")
                        # Try saving raw bytes directly (might already be wav)
                        with open(audio_path, "wb") as f:
                            f.write(audio_bytes)
                elif "path" in audio_data and audio_data["path"]:
                    # Copy from path
                    import shutil
                    shutil.copy(audio_data["path"], audio_path)

        # Create data entry
        entry = {
            "id": f"sd_qa_{idx:05d}",
            "audio": str(audio_path),
            "conversations": []
        }

        # Add question-answer pairs if available
        # Note: VoiceBench uses 'prompt' and 'reference' instead of 'question' and 'answer'
        if "prompt" in sample and "reference" in sample:
            entry["conversations"] = [
                {
                    "from": "human",
                    "value": f"<audio>\n{sample['prompt']}"
                },
                {
                    "from": "gpt",
                    "value": sample["reference"]
                }
            ]
        elif "question" in sample and "answer" in sample:
            entry["conversations"] = [
                {
                    "from": "human",
                    "value": f"<audio>\n{sample['question']}"
                },
                {
                    "from": "gpt",
                    "value": sample["answer"]
                }
            ]

        # Add any other metadata
        for key in sample.keys():
            if key not in ["audio", "question", "answer", "prompt", "reference"]:
                entry[key] = sample[key]

        data_list.append(entry)

    # Save to JSON
    json_path = output_dir / "sd_qa_full.json"
    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=2)

    print(f"Saved {len(data_list)} samples to {json_path}")
    print(f"Audio files saved to {audio_dir}")

    return str(json_path), str(audio_dir)


if __name__ == "__main__":
    json_path, audio_dir = download_voicebench()
    print(f"\nDataset ready!")
    print(f"  JSON: {json_path}")
    print(f"  Audio: {audio_dir}")