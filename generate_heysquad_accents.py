#!/work/anon/miniconda3/envs/audio-llm/bin/python3
"""
Generate HeySQuAD dataset with 11 accent variations using Edge-TTS.
Matches VoiceBench SDQA accents: aus, gbr, ind_n, ind_s, irl, kenya, nga, nzl, phl, usa, zaf

Uses parallel downloads for speed (~10 concurrent requests).
"""

import asyncio
import json
import random
from pathlib import Path
from tqdm.asyncio import tqdm
import edge_tts

# Configuration
NUM_SAMPLES = 600  # Number of unique questions to use
SEED = 42  # For reproducibility
MAX_CONCURRENT = 10  # Number of parallel downloads

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")
HEYSQUAD_JSON = BASE_DIR / "audio-flamingo/data/heysquad/heysquad_full.json"

# Audio files go to shared_audio (to avoid duplication)
SHARED_AUDIO_DIR = BASE_DIR / "shared_audio/heysquad_accents"

# JSON files for each model
AF_DATA_DIR = BASE_DIR / "audio-flamingo/data/heysquad_accents"
KIMI_DATA_DIR = BASE_DIR / "Kimi-Audio/finetune_codes/data/heysquad_accents"

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
    # Create output directories
    SHARED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KIMI_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load HeySQuAD data
    print(f"Loading HeySQuAD from {HEYSQUAD_JSON}")
    with open(HEYSQUAD_JSON, 'r') as f:
        heysquad_data = json.load(f)

    print(f"Total HeySQuAD samples: {len(heysquad_data)}")

    # Randomly select NUM_SAMPLES questions
    random.seed(SEED)
    selected_samples = random.sample(heysquad_data, NUM_SAMPLES)
    print(f"Selected {NUM_SAMPLES} samples for accent generation")

    # Build list of all tasks
    tasks_info = []  # (sample_idx, sample, region, voice, audio_path)
    for sample_idx, sample in enumerate(selected_samples):
        for region in REGIONS:
            voice = ACCENT_VOICES[region]
            new_id = f"heysquad_accent_{sample_idx:05d}_{region}"
            audio_path = SHARED_AUDIO_DIR / f"{new_id}.wav"
            tasks_info.append((sample_idx, sample, region, voice, audio_path, new_id))

    total_audios = len(tasks_info)

    # Check how many already exist
    existing = sum(1 for _, _, _, _, path, _ in tasks_info if path.exists())
    to_generate = total_audios - existing

    print(f"\nTotal: {total_audios} audio files ({NUM_SAMPLES} questions × {len(REGIONS)} accents)")
    print(f"Already exist: {existing}")
    print(f"To generate: {to_generate}")
    print(f"Concurrent downloads: {MAX_CONCURRENT}")
    print(f"Audio output: {SHARED_AUDIO_DIR}")

    # Create semaphore for concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Generate missing audio files in parallel
    if to_generate > 0:
        print(f"\nGenerating {to_generate} audio files...")

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
        print(f"\nGenerated {success_count}/{to_generate} files successfully")

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
                "original_id": sample["id"],
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
    af_output_json = AF_DATA_DIR / "heysquad_accents_full.json"
    with open(af_output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAudio-Flamingo dataset saved to {af_output_json}")

    # Also create per-region JSON files for Audio-Flamingo
    for region in REGIONS:
        region_data = [d for d in output_data if d["region"] == region]
        region_json = AF_DATA_DIR / f"heysquad_accents_{region}.json"
        with open(region_json, 'w') as f:
            json.dump(region_data, f, indent=2)

    # Save the full dataset to Kimi-Audio (JSONL format - one JSON object per line)
    kimi_output_jsonl = KIMI_DATA_DIR / "heysquad_accents_full.jsonl"
    with open(kimi_output_jsonl, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Kimi-Audio dataset saved to {kimi_output_jsonl}")

    # Also create per-region JSONL files for Kimi-Audio
    for region in REGIONS:
        region_data = [d for d in output_data if d["region"] == region]
        region_jsonl = KIMI_DATA_DIR / f"heysquad_accents_{region}.jsonl"
        with open(region_jsonl, 'w') as f:
            for entry in region_data:
                f.write(json.dumps(entry) + '\n')

    print(f"\nTotal samples in dataset: {len(output_data)}")
    print(f"Audio files in: {SHARED_AUDIO_DIR}")
    print(f"Per-region files created for both models (JSON for AF, JSONL for Kimi)")


if __name__ == "__main__":
    asyncio.run(main())
