#!/work/anon/miniconda3/envs/audio-llm/bin/python3
"""
Generate GammaCorpus-Fact-QA dataset with 11 accent variations using Edge-TTS.
Matches VoiceBench SDQA accents: aus, gbr, ind_n, ind_s, irl, kenya, nga, nzl, phl, usa, zaf

Downloads from: https://huggingface.co/datasets/rubenroy/GammaCorpus-Fact-QA-450k

Uses parallel downloads for speed (~10 concurrent requests).

Usage:
    python generate_gammacorpus_accents.py                    # Default: 600 samples
    python generate_gammacorpus_accents.py --num_samples 1000 # Custom number
    python generate_gammacorpus_accents.py --num_samples 500 --seed 123
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from tqdm.asyncio import tqdm
import edge_tts

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

# Configuration defaults
DEFAULT_NUM_SAMPLES = 600  # Number of unique questions to use
DEFAULT_SEED = 42  # For reproducibility
MAX_CONCURRENT = 10  # Number of parallel downloads

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")

# Audio files go to shared_audio (to avoid duplication)
SHARED_AUDIO_DIR = BASE_DIR / "shared_audio/gammacorpus_accents"

# JSON files for each model
AF_DATA_DIR = BASE_DIR / "audio-flamingo/data/gammacorpus_accents"
KIMI_DATA_DIR = BASE_DIR / "Kimi-Audio/finetune_codes/data/gammacorpus_accents"

# VoiceBench accent to Edge-TTS voice mapping
ACCENT_VOICES = {
    "aus": "en-AU-NatashaNeural",      # Australian
    "gbr": "en-GB-SoniaNeural",         # British
    "ind_n": "en-IN-NeerjaNeural",      # Indian (North)
    "ind_s": "en-IN-PrabhatNeural",     # Indian (South) - Using male voice to differentiate
    "irl": "en-IE-EmilyNeural",         # Irish
    "kenya": "en-KE-AsiliaNeural",      # Kenyan
    "nga": "en-NG-EzinneNeural",        # Nigerian
    "nzl": "en-NZ-MollyNeural",         # New Zealand
    "phl": "en-PH-RosaNeural",          # Philippines
    "usa": "en-US-JennyNeural",         # American
    "zaf": "en-ZA-LeahNeural",          # South African
}

REGIONS = list(ACCENT_VOICES.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GammaCorpus-Fact-QA dataset with accent variations"
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of unique questions to sample (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--max_concurrent", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent TTS requests (default: {MAX_CONCURRENT})"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without generating audio"
    )
    return parser.parse_args()


async def generate_audio(text: str, voice: str, output_path: Path, semaphore: asyncio.Semaphore) -> bool:
    """Generate audio using Edge-TTS with concurrency limit."""
    async with semaphore:
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
            return True
        except Exception as e:
            print(f"\nError generating {output_path.name}: {e}")
            return False


async def main():
    args = parse_args()

    num_samples = args.num_samples
    seed = args.seed
    max_concurrent = args.max_concurrent
    dry_run = args.dry_run

    # Create output directories
    SHARED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KIMI_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download/load GammaCorpus from HuggingFace
    print("=" * 60)
    print("GammaCorpus-Fact-QA Accent Generation")
    print("=" * 60)
    print(f"Loading dataset from HuggingFace: rubenroy/GammaCorpus-Fact-QA-450k")

    dataset = load_dataset("rubenroy/GammaCorpus-Fact-QA-450k", split="train")
    total_available = len(dataset)
    print(f"Total available samples: {total_available:,}")

    if num_samples > total_available:
        print(f"Warning: Requested {num_samples} samples but only {total_available} available.")
        num_samples = total_available

    # Randomly select samples
    random.seed(seed)
    indices = random.sample(range(total_available), num_samples)
    selected_samples = [dataset[i] for i in indices]

    print(f"Selected {num_samples} samples (seed={seed})")
    print(f"Will generate {num_samples} × {len(REGIONS)} accents = {num_samples * len(REGIONS):,} audio files")

    if dry_run:
        print("\n[DRY RUN] Would generate the following:")
        print(f"  - Audio files: {SHARED_AUDIO_DIR}")
        print(f"  - AF JSON: {AF_DATA_DIR}")
        print(f"  - Kimi JSONL: {KIMI_DATA_DIR}")
        print("\nSample questions:")
        for i, sample in enumerate(selected_samples[:5]):
            print(f"  {i+1}. Q: {sample['question'][:80]}...")
            print(f"     A: {sample['answer'][:80]}")
        return

    # Build list of all tasks
    tasks_info = []  # (sample_idx, sample, region, voice, audio_path, new_id)
    for sample_idx, sample in enumerate(selected_samples):
        for region in REGIONS:
            voice = ACCENT_VOICES[region]
            new_id = f"gammacorpus_accent_{sample_idx:05d}_{region}"
            audio_path = SHARED_AUDIO_DIR / f"{new_id}.wav"
            tasks_info.append((sample_idx, sample, region, voice, audio_path, new_id))

    total_audios = len(tasks_info)

    # Check how many already exist
    existing = sum(1 for _, _, _, _, path, _ in tasks_info if path.exists())
    to_generate = total_audios - existing

    print(f"\nTotal: {total_audios:,} audio files ({num_samples} questions × {len(REGIONS)} accents)")
    print(f"Already exist: {existing:,}")
    print(f"To generate: {to_generate:,}")
    print(f"Concurrent downloads: {max_concurrent}")
    print(f"Audio output: {SHARED_AUDIO_DIR}")

    # Create semaphore for concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    # Generate missing audio files in parallel
    if to_generate > 0:
        print(f"\nGenerating {to_generate:,} audio files...")

        async def generate_task(info):
            sample_idx, sample, region, voice, audio_path, new_id = info
            if not audio_path.exists():
                return await generate_audio(sample["question"], voice, audio_path, semaphore)
            return True

        # Create all tasks
        tasks = [generate_task(info) for info in tasks_info if not info[4].exists()]

        # Run with progress bar
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Generating audio"):
            result = await coro
            results.append(result)

        success_count = sum(results)
        print(f"\nGenerated {success_count:,}/{to_generate:,} files successfully")

    # Build output data from all files (existing + newly generated)
    print("\nBuilding dataset...")
    output_data = []
    for sample_idx, sample, region, voice, audio_path, new_id in tasks_info:
        if audio_path.exists():
            entry = {
                "id": new_id,
                "audio": str(audio_path),
                "question": sample["question"],
                "answer": sample["answer"],
                "region": region,
                "sample_idx": sample_idx,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<audio>\n{sample['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": sample["answer"]
                    }
                ]
            }
            output_data.append(entry)

    # Save the full dataset to Audio-Flamingo (JSON format)
    af_output_json = AF_DATA_DIR / "gammacorpus_accents_full.json"
    with open(af_output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAudio-Flamingo dataset saved to {af_output_json}")

    # Also create per-region JSON files for Audio-Flamingo
    for region in REGIONS:
        region_data = [d for d in output_data if d["region"] == region]
        region_json = AF_DATA_DIR / f"gammacorpus_accents_{region}.json"
        with open(region_json, 'w') as f:
            json.dump(region_data, f, indent=2)

    # Save the full dataset to Kimi-Audio (JSONL format - one JSON object per line)
    kimi_output_jsonl = KIMI_DATA_DIR / "gammacorpus_accents_full.jsonl"
    with open(kimi_output_jsonl, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Kimi-Audio dataset saved to {kimi_output_jsonl}")

    # Also create per-region JSONL files for Kimi-Audio
    for region in REGIONS:
        region_data = [d for d in output_data if d["region"] == region]
        region_jsonl = KIMI_DATA_DIR / f"gammacorpus_accents_{region}.jsonl"
        with open(region_jsonl, 'w') as f:
            for entry in region_data:
                f.write(json.dumps(entry) + '\n')

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total samples in dataset: {len(output_data):,}")
    print(f"Questions sampled: {num_samples}")
    print(f"Accents generated: {len(REGIONS)}")
    print(f"Audio files in: {SHARED_AUDIO_DIR}")
    print(f"\nPer-region files created:")
    print(f"  - Audio-Flamingo (JSON): {AF_DATA_DIR}")
    print(f"  - Kimi-Audio (JSONL): {KIMI_DATA_DIR}")
    print(f"\nTo use with the pipeline:")
    print(f"  ./run_full_pipeline.sh --benign_dataset gammacorpus_accents --percentage 50")


if __name__ == "__main__":
    asyncio.run(main())
