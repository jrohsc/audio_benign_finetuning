#!/usr/bin/env python3
"""
Download datasets to shared audio location and generate metadata for both
Kimi-Audio and Audio Flamingo.

This avoids downloading the same audio files twice.

Usage:
    python download_shared_dataset.py --dataset spoken_squad --max_samples 6000
    python download_shared_dataset.py --dataset librispeech --max_samples 6000
    python download_shared_dataset.py --dataset heysquad --max_samples 6000
    python download_shared_dataset.py --dataset mmsu --max_samples 3000
    python download_shared_dataset.py --dataset bbh --max_samples 1000
    python download_shared_dataset.py --dataset librispeech --regenerate_metadata_only
"""

import os
import sys
import json
import argparse
import io
from pathlib import Path
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_AUDIO_DIR = SCRIPT_DIR / "shared_audio"
KIMI_DIR = SCRIPT_DIR / "Kimi-Audio" / "finetune_codes"
AF_DIR = SCRIPT_DIR / "audio-flamingo"


def download_spoken_squad(max_samples: int = None):
    """Download Spoken-SQuAD to shared location and generate metadata for both models."""
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset, Audio
        import soundfile as sf

    print("=" * 60)
    print("Downloading Spoken-SQuAD to shared location")
    print("=" * 60)

    # Create directories
    audio_dir = SHARED_AUDIO_DIR / "spoken_squad"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading Spoken-SQuAD from HuggingFace...")
    dataset = load_dataset("alinet/spoken_squad", split="train")

    # Enable audio decoding to get array data
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    # Data for both formats
    af_data = []  # Audio Flamingo format (JSON)
    kimi_data = []  # Kimi-Audio format (JSONL)

    skipped = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"spoken_squad_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        abs_audio_path = str(audio_path.resolve())

        # Save audio if not exists
        if not audio_path.exists():
            try:
                audio_data = sample.get("audio", {})
                saved = False

                # Try bytes first
                if audio_data and "bytes" in audio_data and audio_data["bytes"]:
                    audio_bytes = io.BytesIO(audio_data["bytes"])
                    data, sr = sf.read(audio_bytes)
                    sf.write(str(audio_path), data, sr)
                    saved = True
                # Try path
                elif audio_data and "path" in audio_data and audio_data["path"]:
                    import shutil as shutil_local
                    shutil_local.copy(audio_data["path"], audio_path)
                    saved = True
                # Try array (decoded audio)
                elif audio_data and "array" in audio_data and audio_data["array"] is not None:
                    arr = audio_data["array"]
                    sr = audio_data.get("sampling_rate", 16000)
                    sf.write(str(audio_path), arr, sr)
                    saved = True

                if not saved:
                    skipped += 1
                    if idx < 3:  # Debug first few
                        print(f"  Debug sample {idx}: audio_data keys = {audio_data.keys() if audio_data else 'None'}")
                    continue
            except Exception as e:
                print(f"Error saving audio {idx}: {e}")
                skipped += 1
                continue

        # Get question and answer
        question = sample.get("question", "")
        answer = sample.get("answers", {}).get("text", [""])[0] if sample.get("answers") else ""
        context = sample.get("context", "")

        # Audio Flamingo format
        af_entry = {
            "id": f"spoken_squad_{idx:05d}",
            "audio": f"data/spoken_squad/audio/{audio_filename}",
            "question": question,
            "answer": answer,
            "conversations": [
                {"from": "human", "value": f"<audio>\n{question}"},
                {"from": "gpt", "value": answer}
            ]
        }
        af_data.append(af_entry)

        # Kimi-Audio format - Use generic instruction so model learns to extract question from audio
        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Please answer the following question based on the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": answer}
            ]
        }
        kimi_data.append(kimi_entry)

    print(f"\nProcessed {len(af_data)} samples, skipped {skipped}")

    # Create symlinks and data directories
    af_spoken_squad_dir = AF_DIR / "data" / "spoken_squad"
    af_spoken_squad_dir.mkdir(parents=True, exist_ok=True)

    kimi_spoken_squad_dir = KIMI_DIR / "data_spoken_squad"
    kimi_spoken_squad_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks to shared audio
    import shutil
    af_audio_link = af_spoken_squad_dir / "audio"
    if af_audio_link.is_symlink():
        af_audio_link.unlink()
    elif af_audio_link.is_dir():
        shutil.rmtree(af_audio_link)
    elif af_audio_link.exists():
        af_audio_link.unlink()
    af_audio_link.symlink_to(audio_dir.resolve())

    kimi_audio_link = kimi_spoken_squad_dir / "audio"
    if kimi_audio_link.is_symlink():
        kimi_audio_link.unlink()
    elif kimi_audio_link.is_dir():
        shutil.rmtree(kimi_audio_link)
    elif kimi_audio_link.exists():
        kimi_audio_link.unlink()
    kimi_audio_link.symlink_to(audio_dir.resolve())

    # Save Audio Flamingo metadata
    af_json_path = af_spoken_squad_dir / "spoken_squad_full.json"
    with open(af_json_path, "w") as f:
        json.dump(af_data, f, indent=2)
    print(f"Saved Audio Flamingo metadata: {af_json_path}")

    # Save Kimi-Audio metadata
    kimi_jsonl_path = kimi_spoken_squad_dir / "spoken_squad_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path}")

    print(f"\nShared audio location: {audio_dir}")
    print(f"Audio Flamingo symlink: {af_audio_link} -> {audio_dir}")
    print(f"Kimi-Audio symlink: {kimi_audio_link} -> {audio_dir}")

    return len(af_data)


def regenerate_librispeech_metadata():
    """Regenerate metadata for LibriSpeech from existing shared audio files."""
    print("=" * 60)
    print("Regenerating LibriSpeech metadata from shared audio")
    print("=" * 60)

    audio_dir = SHARED_AUDIO_DIR / "librispeech"
    if not audio_dir.exists():
        print(f"Error: Shared audio directory not found: {audio_dir}")
        return 0

    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    # Load existing Audio Flamingo JSON to get transcriptions
    af_json_path = AF_DIR / "data" / "librispeech" / "librispeech_full.json"
    transcriptions = {}

    if af_json_path.exists():
        print(f"Loading transcriptions from {af_json_path}")
        with open(af_json_path) as f:
            af_existing = json.load(f)
        for entry in af_existing:
            audio_name = Path(entry["audio"]).name
            transcriptions[audio_name] = entry.get("transcription", "")

    # Generate Kimi-Audio metadata
    kimi_data = []
    for audio_path in tqdm(audio_files, desc="Generating Kimi-Audio metadata"):
        audio_name = audio_path.name
        abs_audio_path = str(audio_path.resolve())
        transcription = transcriptions.get(audio_name, "")

        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Transcribe the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": transcription}
            ]
        }
        kimi_data.append(kimi_entry)

    # Save Kimi-Audio metadata
    kimi_librispeech_dir = KIMI_DIR / "data_librispeech"
    kimi_librispeech_dir.mkdir(parents=True, exist_ok=True)

    kimi_jsonl_path = kimi_librispeech_dir / "librispeech_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path} ({len(kimi_data)} samples)")

    return len(kimi_data)


def download_librispeech(max_samples: int = None):
    """Download LibriSpeech to shared location and generate metadata for both models."""
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset, Audio
        import soundfile as sf

    print("=" * 60)
    print("Downloading LibriSpeech to shared location")
    print("=" * 60)

    # Create directories
    audio_dir = SHARED_AUDIO_DIR / "librispeech"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check existing files
    existing_files = set(f.name for f in audio_dir.glob("*.wav"))
    print(f"Found {len(existing_files)} existing audio files")

    # Load dataset
    print("Loading LibriSpeech train-clean-100 from HuggingFace...")
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")

    # Disable audio decoding
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    # Data for both formats
    af_data = []
    kimi_data = []

    skipped = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"librispeech_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        abs_audio_path = str(audio_path.resolve())

        # Save audio if not exists
        if audio_filename not in existing_files:
            try:
                audio_data = sample.get("audio", {})
                if audio_data and "bytes" in audio_data and audio_data["bytes"]:
                    audio_bytes = io.BytesIO(audio_data["bytes"])
                    data, sr = sf.read(audio_bytes)
                    sf.write(str(audio_path), data, sr)
                else:
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Error saving audio {idx}: {e}")
                skipped += 1
                continue

        transcription = sample.get("text", "")

        # Audio Flamingo format
        af_entry = {
            "id": f"librispeech_{idx:05d}",
            "audio": f"data/librispeech/audio/{audio_filename}",
            "transcription": transcription,
            "conversations": [
                {"from": "human", "value": "<audio>\nTranscribe the audio."},
                {"from": "gpt", "value": transcription}
            ],
            "original_speaker_id": sample.get("speaker_id"),
            "original_chapter_id": sample.get("chapter_id"),
            "original_id": sample.get("id")
        }
        af_data.append(af_entry)

        # Kimi-Audio format
        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Transcribe the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": transcription}
            ]
        }
        kimi_data.append(kimi_entry)

    print(f"\nProcessed {len(af_data)} samples, skipped {skipped}")

    # Create directories and symlinks
    af_librispeech_dir = AF_DIR / "data" / "librispeech"
    af_librispeech_dir.mkdir(parents=True, exist_ok=True)

    kimi_librispeech_dir = KIMI_DIR / "data_librispeech"
    kimi_librispeech_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks
    af_audio_link = af_librispeech_dir / "audio"
    if af_audio_link.exists() or af_audio_link.is_symlink():
        af_audio_link.unlink()
    af_audio_link.symlink_to(audio_dir.resolve())

    kimi_audio_link = kimi_librispeech_dir / "audio"
    if kimi_audio_link.exists() or kimi_audio_link.is_symlink():
        kimi_audio_link.unlink()
    kimi_audio_link.symlink_to(audio_dir.resolve())

    # Save Audio Flamingo metadata
    af_json_path = af_librispeech_dir / "librispeech_full.json"
    with open(af_json_path, "w") as f:
        json.dump(af_data, f, indent=2)
    print(f"Saved Audio Flamingo metadata: {af_json_path}")

    # Save Kimi-Audio metadata
    kimi_jsonl_path = kimi_librispeech_dir / "librispeech_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path}")

    return len(af_data)


def download_heysquad(max_samples: int = None, use_human: bool = True):
    """Download HeySQuAD to shared location and generate metadata for both models.

    Args:
        max_samples: Maximum number of samples to download
        use_human: If True, use human-spoken questions (yijingwu/HeySQuAD_human)
                   If False, use machine-generated questions (yijingwu/HeySQuAD_machine)
    """
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset, Audio
        import soundfile as sf

    dataset_name = "yijingwu/HeySQuAD_human" if use_human else "yijingwu/HeySQuAD_machine"

    print("=" * 60)
    print(f"Downloading HeySQuAD to shared location")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    # Create directories
    audio_dir = SHARED_AUDIO_DIR / "heysquad"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check existing files
    existing_files = set(f.name for f in audio_dir.glob("*.wav"))
    print(f"Found {len(existing_files)} existing audio files")

    # Load dataset
    print(f"Loading HeySQuAD from HuggingFace: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading train split, trying validation: {e}")
        dataset = load_dataset(dataset_name, split="validation")

    # Disable audio decoding to avoid torchcodec dependency
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    # Check available columns
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    # Data for both formats
    af_data = []  # Audio Flamingo format (JSON)
    kimi_data = []  # Kimi-Audio format (JSONL)

    skipped = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"heysquad_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        abs_audio_path = str(audio_path.resolve())

        # Save audio if not exists
        if audio_filename not in existing_files and not audio_path.exists():
            try:
                # HeySQuAD has audio in 'audio' column
                audio_data = sample.get("audio", {})
                saved = False

                # Try different audio formats
                if audio_data:
                    if isinstance(audio_data, dict):
                        # Try bytes first
                        if "bytes" in audio_data and audio_data["bytes"]:
                            audio_bytes = io.BytesIO(audio_data["bytes"])
                            data, sr = sf.read(audio_bytes)
                            sf.write(str(audio_path), data, sr)
                            saved = True
                        # Try path
                        elif "path" in audio_data and audio_data["path"]:
                            import shutil as shutil_local
                            shutil_local.copy(audio_data["path"], audio_path)
                            saved = True
                        # Try array (decoded audio)
                        elif "array" in audio_data and audio_data["array"] is not None:
                            arr = audio_data["array"]
                            sr = audio_data.get("sampling_rate", 16000)
                            sf.write(str(audio_path), arr, sr)
                            saved = True

                if not saved:
                    skipped += 1
                    if idx < 3:  # Debug first few
                        print(f"  Debug sample {idx}: audio_data = {type(audio_data)}, keys = {audio_data.keys() if isinstance(audio_data, dict) else 'N/A'}")
                    continue
            except Exception as e:
                print(f"Error saving audio {idx}: {e}")
                skipped += 1
                continue

        # Get question and answer
        # HeySQuAD uses 'question' for original text question, 'context' for passage, 'answers' for answer
        question = sample.get("question", "")

        # Handle answers - could be dict with 'text' key or list
        answers = sample.get("answers", {})
        if isinstance(answers, dict):
            answer = answers.get("text", [""])[0] if answers.get("text") else ""
        elif isinstance(answers, list) and len(answers) > 0:
            answer = answers[0] if isinstance(answers[0], str) else answers[0].get("text", "")
        else:
            answer = ""

        context = sample.get("context", "")

        # Audio Flamingo format - Use generic instruction so model learns to extract question from audio
        af_entry = {
            "id": f"heysquad_{idx:05d}",
            "audio": f"data/heysquad/audio/{audio_filename}",
            "question": question,
            "answer": answer,
            "conversations": [
                {"from": "human", "value": "<audio>\nPlease answer the following question based on the audio."},
                {"from": "gpt", "value": answer}
            ]
        }
        af_data.append(af_entry)

        # Kimi-Audio format - Use generic instruction so model learns to extract question from audio
        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Please answer the following question based on the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": answer}
            ]
        }
        kimi_data.append(kimi_entry)

    print(f"\nProcessed {len(af_data)} samples, skipped {skipped}")

    # Create symlinks and data directories
    af_heysquad_dir = AF_DIR / "data" / "heysquad"
    af_heysquad_dir.mkdir(parents=True, exist_ok=True)

    kimi_heysquad_dir = KIMI_DIR / "data_heysquad"
    kimi_heysquad_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks to shared audio
    import shutil
    af_audio_link = af_heysquad_dir / "audio"
    if af_audio_link.is_symlink():
        af_audio_link.unlink()
    elif af_audio_link.is_dir():
        shutil.rmtree(af_audio_link)
    elif af_audio_link.exists():
        af_audio_link.unlink()
    af_audio_link.symlink_to(audio_dir.resolve())

    kimi_audio_link = kimi_heysquad_dir / "audio"
    if kimi_audio_link.is_symlink():
        kimi_audio_link.unlink()
    elif kimi_audio_link.is_dir():
        shutil.rmtree(kimi_audio_link)
    elif kimi_audio_link.exists():
        kimi_audio_link.unlink()
    kimi_audio_link.symlink_to(audio_dir.resolve())

    # Save Audio Flamingo metadata
    af_json_path = af_heysquad_dir / "heysquad_full.json"
    with open(af_json_path, "w") as f:
        json.dump(af_data, f, indent=2)
    print(f"Saved Audio Flamingo metadata: {af_json_path}")

    # Save Kimi-Audio metadata
    kimi_jsonl_path = kimi_heysquad_dir / "heysquad_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path}")

    print(f"\nShared audio location: {audio_dir}")
    print(f"Audio Flamingo symlink: {af_audio_link} -> {audio_dir}")
    print(f"Kimi-Audio symlink: {kimi_audio_link} -> {audio_dir}")

    return len(af_data)


def download_mmsu(max_samples: int = None):
    """Download VoiceBench MMSU to shared location and generate metadata for both models.

    MMSU is a multimodal subject understanding dataset from VoiceBench with ~3k samples
    covering various academic subjects (law, engineering, biology, etc.).
    """
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset, Audio
        import soundfile as sf

    print("=" * 60)
    print("Downloading VoiceBench MMSU to shared location")
    print("=" * 60)

    # Create directories
    audio_dir = SHARED_AUDIO_DIR / "mmsu"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check existing files
    existing_files = set(f.name for f in audio_dir.glob("*.wav"))
    print(f"Found {len(existing_files)} existing audio files")

    # Load dataset - MMSU has multiple splits (subjects)
    print("Loading VoiceBench MMSU from HuggingFace: lmms-lab/voicebench (mmsu)...")

    # Load all splits and concatenate
    all_samples = []
    splits = ["law", "engineering", "other", "biology", "business", "economics",
              "health", "philosophy", "psychology", "history", "chemistry", "physics"]

    for split in splits:
        try:
            dataset = load_dataset("lmms-lab/voicebench", "mmsu", split=split)
            # Disable audio decoding to avoid torchcodec dependency
            if "audio" in dataset.features:
                dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
            print(f"  Loaded {split}: {len(dataset)} samples")
            for sample in dataset:
                sample["subject"] = split
                all_samples.append(sample)
        except Exception as e:
            print(f"  Warning: Could not load split '{split}': {e}")

    print(f"Total samples loaded: {len(all_samples)}")

    if max_samples and len(all_samples) > max_samples:
        # Sample evenly from all subjects
        import random
        random.seed(42)
        random.shuffle(all_samples)
        all_samples = all_samples[:max_samples]
        print(f"Sampled down to {len(all_samples)} samples")

    print(f"Processing {len(all_samples)} samples...")

    # Data for both formats
    af_data = []  # Audio Flamingo format (JSON)
    kimi_data = []  # Kimi-Audio format (JSONL)

    skipped = 0
    for idx, sample in enumerate(tqdm(all_samples, desc="Processing")):
        audio_filename = f"mmsu_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        abs_audio_path = str(audio_path.resolve())

        # Save audio if not exists
        if audio_filename not in existing_files and not audio_path.exists():
            try:
                # MMSU has audio in 'audio' column
                audio_data = sample.get("audio", {})
                saved = False

                # Try different audio formats
                if audio_data:
                    if isinstance(audio_data, dict):
                        # Try bytes first
                        if "bytes" in audio_data and audio_data["bytes"]:
                            audio_bytes = io.BytesIO(audio_data["bytes"])
                            data, sr = sf.read(audio_bytes)
                            sf.write(str(audio_path), data, sr)
                            saved = True
                        # Try path
                        elif "path" in audio_data and audio_data["path"]:
                            import shutil as shutil_local
                            shutil_local.copy(audio_data["path"], audio_path)
                            saved = True
                        # Try array (decoded audio)
                        elif "array" in audio_data and audio_data["array"] is not None:
                            arr = audio_data["array"]
                            sr = audio_data.get("sampling_rate", 16000)
                            sf.write(str(audio_path), arr, sr)
                            saved = True

                if not saved:
                    skipped += 1
                    if idx < 3:  # Debug first few
                        print(f"  Debug sample {idx}: audio_data = {type(audio_data)}, keys = {audio_data.keys() if isinstance(audio_data, dict) else 'N/A'}")
                    continue
            except Exception as e:
                print(f"Error saving audio {idx}: {e}")
                skipped += 1
                continue

        # Get question and answer
        # MMSU uses 'prompt' for question and 'reference' for answer
        question = sample.get("prompt", "")
        answer = sample.get("reference", "")
        subject = sample.get("subject", sample.get("category", ""))

        # Audio Flamingo format - Use generic instruction so model learns to extract question from audio
        af_entry = {
            "id": f"mmsu_{idx:05d}",
            "audio": f"data/mmsu/audio/{audio_filename}",
            "question": question,
            "answer": answer,
            "subject": subject,
            "conversations": [
                {"from": "human", "value": "<audio>\nPlease answer the following question based on the audio."},
                {"from": "gpt", "value": answer}
            ]
        }
        af_data.append(af_entry)

        # Kimi-Audio format - Use generic instruction so model learns to extract question from audio
        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Please answer the following question based on the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": answer}
            ]
        }
        kimi_data.append(kimi_entry)

    print(f"\nProcessed {len(af_data)} samples, skipped {skipped}")

    # Create symlinks and data directories
    af_mmsu_dir = AF_DIR / "data" / "mmsu"
    af_mmsu_dir.mkdir(parents=True, exist_ok=True)

    kimi_mmsu_dir = KIMI_DIR / "data_mmsu"
    kimi_mmsu_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks to shared audio
    import shutil
    af_audio_link = af_mmsu_dir / "audio"
    if af_audio_link.is_symlink():
        af_audio_link.unlink()
    elif af_audio_link.is_dir():
        shutil.rmtree(af_audio_link)
    elif af_audio_link.exists():
        af_audio_link.unlink()
    af_audio_link.symlink_to(audio_dir.resolve())

    kimi_audio_link = kimi_mmsu_dir / "audio"
    if kimi_audio_link.is_symlink():
        kimi_audio_link.unlink()
    elif kimi_audio_link.is_dir():
        shutil.rmtree(kimi_audio_link)
    elif kimi_audio_link.exists():
        kimi_audio_link.unlink()
    kimi_audio_link.symlink_to(audio_dir.resolve())

    # Save Audio Flamingo metadata
    af_json_path = af_mmsu_dir / "mmsu_full.json"
    with open(af_json_path, "w") as f:
        json.dump(af_data, f, indent=2)
    print(f"Saved Audio Flamingo metadata: {af_json_path}")

    # Save Kimi-Audio metadata
    kimi_jsonl_path = kimi_mmsu_dir / "mmsu_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path}")

    print(f"\nShared audio location: {audio_dir}")
    print(f"Audio Flamingo symlink: {af_audio_link} -> {audio_dir}")
    print(f"Kimi-Audio symlink: {kimi_audio_link} -> {audio_dir}")

    return len(af_data)


def download_bbh(max_samples: int = None):
    """Download VoiceBench BBH (BIG-Bench Hard) to shared location and generate metadata for both models.

    BBH is a challenging reasoning benchmark from VoiceBench with ~1k samples
    covering various reasoning tasks.
    """
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets soundfile")
        from datasets import load_dataset, Audio
        import soundfile as sf

    print("=" * 60)
    print("Downloading VoiceBench BBH to shared location")
    print("=" * 60)

    # Create directories
    audio_dir = SHARED_AUDIO_DIR / "bbh"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check existing files
    existing_files = set(f.name for f in audio_dir.glob("*.wav"))
    print(f"Found {len(existing_files)} existing audio files")

    # Load dataset
    print("Loading VoiceBench BBH from HuggingFace: lmms-lab/voicebench (bbh)...")

    try:
        # Try loading with train split first
        dataset = load_dataset("lmms-lab/voicebench", "bbh", split="train")
    except Exception as e:
        print(f"Could not load train split: {e}")
        try:
            # Try without specifying split
            dataset = load_dataset("lmms-lab/voicebench", "bbh")
            if hasattr(dataset, 'keys'):
                # It's a DatasetDict, get the first available split
                first_split = list(dataset.keys())[0]
                dataset = dataset[first_split]
                print(f"Using split: {first_split}")
        except Exception as e2:
            print(f"Error loading BBH dataset: {e2}")
            return 0

    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")

    # Disable audio decoding to avoid torchcodec dependency
    if "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    # Data for both formats
    af_data = []  # Audio Flamingo format (JSON)
    kimi_data = []  # Kimi-Audio format (JSONL)

    skipped = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        audio_filename = f"bbh_{idx:05d}.wav"
        audio_path = audio_dir / audio_filename
        abs_audio_path = str(audio_path.resolve())

        # Save audio if not exists
        if audio_filename not in existing_files and not audio_path.exists():
            try:
                # BBH has audio in 'audio' column
                audio_data = sample.get("audio", {})
                saved = False

                # Try different audio formats
                if audio_data:
                    if isinstance(audio_data, dict):
                        # Try bytes first
                        if "bytes" in audio_data and audio_data["bytes"]:
                            audio_bytes = io.BytesIO(audio_data["bytes"])
                            data, sr = sf.read(audio_bytes)
                            sf.write(str(audio_path), data, sr)
                            saved = True
                        # Try path
                        elif "path" in audio_data and audio_data["path"]:
                            import shutil as shutil_local
                            shutil_local.copy(audio_data["path"], audio_path)
                            saved = True
                        # Try array (decoded audio)
                        elif "array" in audio_data and audio_data["array"] is not None:
                            arr = audio_data["array"]
                            sr = audio_data.get("sampling_rate", 16000)
                            sf.write(str(audio_path), arr, sr)
                            saved = True

                if not saved:
                    skipped += 1
                    if idx < 3:  # Debug first few
                        print(f"  Debug sample {idx}: audio_data = {type(audio_data)}, keys = {audio_data.keys() if isinstance(audio_data, dict) else 'N/A'}")
                    continue
            except Exception as e:
                print(f"Error saving audio {idx}: {e}")
                skipped += 1
                continue

        # Get question and answer
        # BBH uses 'prompt' for question and 'reference' for answer (similar to MMSU)
        question = sample.get("prompt", "")
        answer = sample.get("reference", "")
        category = sample.get("category", "")

        # Audio Flamingo format - Use generic instruction so model learns to extract question from audio
        af_entry = {
            "id": f"bbh_{idx:05d}",
            "audio": f"data/bbh/audio/{audio_filename}",
            "question": question,
            "answer": answer,
            "category": category,
            "conversations": [
                {"from": "human", "value": "<audio>\nPlease answer the following question based on the audio."},
                {"from": "gpt", "value": answer}
            ]
        }
        af_data.append(af_entry)

        # Kimi-Audio format - Use generic instruction so model learns to extract question from audio
        kimi_entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": "Please answer the following question based on the audio."},
                {"role": "user", "message_type": "audio", "content": abs_audio_path},
                {"role": "assistant", "message_type": "text", "content": answer}
            ]
        }
        kimi_data.append(kimi_entry)

    print(f"\nProcessed {len(af_data)} samples, skipped {skipped}")

    # Create symlinks and data directories
    af_bbh_dir = AF_DIR / "data" / "bbh"
    af_bbh_dir.mkdir(parents=True, exist_ok=True)

    kimi_bbh_dir = KIMI_DIR / "data_bbh"
    kimi_bbh_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks to shared audio
    import shutil
    af_audio_link = af_bbh_dir / "audio"
    if af_audio_link.is_symlink():
        af_audio_link.unlink()
    elif af_audio_link.is_dir():
        shutil.rmtree(af_audio_link)
    elif af_audio_link.exists():
        af_audio_link.unlink()
    af_audio_link.symlink_to(audio_dir.resolve())

    kimi_audio_link = kimi_bbh_dir / "audio"
    if kimi_audio_link.is_symlink():
        kimi_audio_link.unlink()
    elif kimi_audio_link.is_dir():
        shutil.rmtree(kimi_audio_link)
    elif kimi_audio_link.exists():
        kimi_audio_link.unlink()
    kimi_audio_link.symlink_to(audio_dir.resolve())

    # Save Audio Flamingo metadata
    af_json_path = af_bbh_dir / "bbh_full.json"
    with open(af_json_path, "w") as f:
        json.dump(af_data, f, indent=2)
    print(f"Saved Audio Flamingo metadata: {af_json_path}")

    # Save Kimi-Audio metadata
    kimi_jsonl_path = kimi_bbh_dir / "bbh_full.jsonl"
    with open(kimi_jsonl_path, "w") as f:
        for entry in kimi_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved Kimi-Audio metadata: {kimi_jsonl_path}")

    print(f"\nShared audio location: {audio_dir}")
    print(f"Audio Flamingo symlink: {af_audio_link} -> {audio_dir}")
    print(f"Kimi-Audio symlink: {kimi_audio_link} -> {audio_dir}")

    return len(af_data)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets to shared location for both Kimi-Audio and Audio Flamingo"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["spoken_squad", "librispeech", "heysquad", "mmsu", "bbh"],
                        help="Dataset to download")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download")
    parser.add_argument("--regenerate_metadata_only", action="store_true",
                        help="Only regenerate metadata from existing audio files")

    args = parser.parse_args()

    if args.dataset == "spoken_squad":
        if args.regenerate_metadata_only:
            print("Regenerate metadata only not supported for Spoken-SQuAD yet")
            return
        count = download_spoken_squad(args.max_samples)
    elif args.dataset == "librispeech":
        if args.regenerate_metadata_only:
            count = regenerate_librispeech_metadata()
        else:
            count = download_librispeech(args.max_samples)
    elif args.dataset == "heysquad":
        if args.regenerate_metadata_only:
            print("Regenerate metadata only not supported for HeySQuAD yet")
            return
        count = download_heysquad(args.max_samples)
    elif args.dataset == "mmsu":
        if args.regenerate_metadata_only:
            print("Regenerate metadata only not supported for MMSU yet")
            return
        count = download_mmsu(args.max_samples)
    elif args.dataset == "bbh":
        if args.regenerate_metadata_only:
            print("Regenerate metadata only not supported for BBH yet")
            return
        count = download_bbh(args.max_samples)

    print(f"\nDone! Processed {count} samples.")


if __name__ == "__main__":
    main()
