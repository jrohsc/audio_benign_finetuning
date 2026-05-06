#!/usr/bin/env python3
"""
Download HeySQuAD dataset (human-spoken questions) from HuggingFace
and prepare it for finetuning with embedding-based filtering.

HeySQuAD contains spoken questions (audio) with text answers from SQuAD.
This is similar to VoiceBench SD-QA where the QUESTION is spoken.

Dataset: https://huggingface.co/datasets/yijingwu/HeySQuAD_human
Paper: https://arxiv.org/abs/2304.13689
"""

import os
import json
import io
import argparse
from pathlib import Path
from tqdm import tqdm


def download_heysquad(
    output_dir: str = "data/heysquad",
    split: str = "train",
    max_samples: int = None,
    use_machine: bool = False,
):
    """
    Download HeySQuAD dataset.

    Args:
        output_dir: Output directory for audio files and JSON
        split: Dataset split to download ('train' or 'validation')
        max_samples: Maximum number of samples to download (None for all)
        use_machine: If True, use HeySQuAD_machine (TTS) instead of HeySQuAD_human
    """
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

    # Choose dataset
    dataset_name = "yijingwu/HeySQuAD_machine" if use_machine else "yijingwu/HeySQuAD_human"
    dataset_suffix = "machine" if use_machine else "human"

    print(f"Downloading HeySQuAD dataset ({dataset_suffix})...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Split: {split}")

    # Load dataset
    print(f"  Loading {split} split...")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Convert to format compatible with audio-flamingo (same as VoiceBench)
    data_list = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        sample_id = f"heysquad_{dataset_suffix}_{idx:05d}"
        audio_path = audio_dir / f"{sample_id}.wav"

        # Extract and save audio
        audio_saved = False
        if "audio" in sample and sample["audio"] is not None:
            audio_data = sample["audio"]

            if isinstance(audio_data, dict):
                # HuggingFace audio format
                if "array" in audio_data and audio_data["array"] is not None:
                    # Decoded audio array
                    try:
                        audio_array = audio_data["array"]
                        sampling_rate = audio_data.get("sampling_rate", 16000)
                        sf.write(str(audio_path), audio_array, sampling_rate)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not save audio for sample {idx}: {e}")
                elif "bytes" in audio_data and audio_data["bytes"]:
                    # Raw bytes
                    try:
                        audio_bytes = audio_data["bytes"]
                        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                        sf.write(str(audio_path), audio_array, sr)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not process audio bytes for sample {idx}: {e}")
                elif "path" in audio_data and audio_data["path"]:
                    # Copy from path
                    import shutil
                    try:
                        shutil.copy(audio_data["path"], audio_path)
                        audio_saved = True
                    except Exception as e:
                        print(f"Warning: Could not copy audio for sample {idx}: {e}")

        if not audio_saved:
            skipped += 1
            continue

        # Extract question and answer
        question = sample.get("question", "")

        # HeySQuAD has answers as a dict with 'text' list
        answers = sample.get("answers", {})
        if isinstance(answers, dict) and "text" in answers:
            answer_texts = answers["text"]
            answer = answer_texts[0] if answer_texts else ""
        elif isinstance(answers, list) and len(answers) > 0:
            answer = answers[0].get("text", "") if isinstance(answers[0], dict) else str(answers[0])
        else:
            answer = str(answers) if answers else ""

        if not question or not answer:
            skipped += 1
            continue

        # Create data entry (same format as VoiceBench)
        entry = {
            "id": sample_id,
            "audio": str(audio_path),
            "conversations": [
                {
                    "from": "human",
                    "value": f"<audio>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ],
            # Additional metadata
            "original_id": sample.get("id", ""),
            "transcription": sample.get("transcription", ""),
            "context": sample.get("context", "")[:500] if sample.get("context") else "",  # Truncate context
            "dataset": f"heysquad_{dataset_suffix}",
            "split": split,
        }

        data_list.append(entry)

    # Save to JSON
    json_filename = f"heysquad_{dataset_suffix}_{split}.json"
    json_path = output_path / json_filename
    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"  Saved: {len(data_list)} samples")
    print(f"  Skipped: {skipped} samples (missing audio/question/answer)")
    print(f"  JSON: {json_path}")
    print(f"  Audio: {audio_dir}")

    return str(json_path), str(audio_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download HeySQuAD dataset for audio-flamingo finetuning"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/heysquad",
        help="Output directory for audio files and JSON"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "validation"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to download (None for all)"
    )
    parser.add_argument(
        "--use_machine", action="store_true",
        help="Use HeySQuAD_machine (TTS) instead of HeySQuAD_human"
    )

    args = parser.parse_args()

    json_path, audio_dir = download_heysquad(
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        use_machine=args.use_machine,
    )

    print(f"\nDataset ready!")
    print(f"  JSON: {json_path}")
    print(f"  Audio: {audio_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run semantic filtering: python 0_filter_heysquad_semantic.py --input_json {json_path}")
    print(f"  2. Run acoustic filtering: python 0_filter_heysquad_acoustic.py --input_json {json_path}")


if __name__ == "__main__":
    main()
