#!/work/anon/miniconda3/envs/audio-llm/bin/python3
"""
Generate VoiceBench dataset with realistic background noise augmentations.

Uses REAL noise audio files (noise_cafe.mp3, noise_traffic.mp3, etc.) to create
realistic noisy versions of the VoiceBench dataset.

Creates SEPARATE datasets for each noise type:
  - voicebench_cafe: Coffee shop / restaurant background noise
  - voicebench_traffic: Street traffic noise
  - (Add more noise files to support more types)

Each noise type gets its own folder in shared_audio/ and metadata files.

Usage:
    python generate_voicebench_noisy.py                          # All available noise types
    python generate_voicebench_noisy.py --noise_types cafe       # Specific type only
    python generate_voicebench_noisy.py --snr 5                  # Lower SNR (more noise)
    python generate_voicebench_noisy.py --snr 15                 # Higher SNR (less noise)
"""

import argparse
import json
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

# Configuration
SEED = 42
MAX_WORKERS = 8  # Number of parallel workers

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")
VOICEBENCH_AUDIO_DIR = BASE_DIR / "shared_audio/voicebench"
VOICEBENCH_JSON = BASE_DIR / "audio-flamingo/data/voicebench/sd-qa/sd_qa_full.json"

# Noise files directory
NOISE_FILES_DIR = BASE_DIR

# Output base directories
SHARED_AUDIO_BASE = BASE_DIR / "shared_audio"
AF_DATA_BASE = BASE_DIR / "audio-flamingo/data"
KIMI_DATA_BASE = BASE_DIR / "Kimi-Audio/finetune_codes/data"

# Sample rate for all audio
SAMPLE_RATE = 16000

# Mapping of noise types to their audio files
NOISE_FILE_MAP = {
    "cafe": "noise_cafe.mp3",
    "traffic": "noise_traffic.mp3",
}


def get_available_noise_types() -> List[str]:
    """Get list of noise types that have corresponding audio files."""
    available = []
    for noise_type, filename in NOISE_FILE_MAP.items():
        if (NOISE_FILES_DIR / filename).exists():
            available.append(noise_type)
    return available


def load_noise_audio(noise_type: str, sr: int = 16000) -> np.ndarray:
    """Load noise audio file.

    Args:
        noise_type: Type of noise (cafe, traffic, etc.)
        sr: Target sample rate

    Returns:
        Noise audio as numpy array
    """
    if noise_type not in NOISE_FILE_MAP:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(NOISE_FILE_MAP.keys())}")

    noise_file = NOISE_FILES_DIR / NOISE_FILE_MAP[noise_type]
    if not noise_file.exists():
        raise FileNotFoundError(f"Noise file not found: {noise_file}")

    # Load noise audio
    noise, _ = librosa.load(noise_file, sr=sr, mono=True)
    return noise


def add_noise_to_audio(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Add noise to audio signal at specified SNR.

    Args:
        audio: Clean audio signal
        noise: Noise signal (will be looped/trimmed to match audio length)
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy audio signal
    """
    length = len(audio)

    # Loop or trim noise to match audio length
    if len(noise) < length:
        # Loop the noise
        repeats = (length // len(noise)) + 1
        noise = np.tile(noise, repeats)

    # Random start position for variety
    max_start = len(noise) - length
    if max_start > 0:
        start = random.randint(0, max_start)
        noise_segment = noise[start:start + length]
    else:
        noise_segment = noise[:length]

    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise_segment ** 2) + 1e-10

    # Calculate scaling factor for desired SNR
    # SNR = 10 * log10(signal_power / scaled_noise_power)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    # Mix audio and noise
    noisy_audio = audio + noise_scale * noise_segment

    # Normalize to prevent clipping
    max_val = np.abs(noisy_audio).max()
    if max_val > 0.99:
        noisy_audio = noisy_audio * 0.99 / max_val

    return noisy_audio.astype(np.float32)


# Global variable to store loaded noise (for multiprocessing efficiency)
_loaded_noise = {}


def init_worker(noise_type: str):
    """Initialize worker process with loaded noise."""
    global _loaded_noise
    _loaded_noise[noise_type] = load_noise_audio(noise_type, SAMPLE_RATE)


def process_audio_file(args: Tuple) -> Optional[Dict]:
    """Process a single audio file (worker function for parallel processing).

    Args:
        args: Tuple of (sample_data, per_region_idx, noise_type, snr_db, input_dir, output_dir, noise_array)

    Returns:
        Output entry dict or None if failed
    """
    sample, per_region_idx, noise_type, snr_db, input_dir, output_dir, noise_array = args

    try:
        sample_id = sample["id"]
        region = sample.get("region", "unknown")

        # Use per-region index (audio files are numbered 0-552 per region, not globally)
        idx_no_zeros = str(per_region_idx)
        input_filename = f"{region}_{idx_no_zeros}.wav"
        input_path = input_dir / input_filename

        # Check if input exists, try alternatives with leading zeros
        if not input_path.exists():
            # Try with leading zeros
            alt_input = input_dir / f"{region}_{per_region_idx:05d}.wav"
            if alt_input.exists():
                input_path = alt_input
            else:
                alt_input2 = input_dir / f"{sample_id}.wav"
                if alt_input2.exists():
                    input_path = alt_input2
                else:
                    return None

        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        # Add noise using the real noise audio
        noisy_audio = add_noise_to_audio(audio, noise_array, snr_db)

        # Create output filename (same structure as original voicebench)
        output_filename = f"{region}_{idx_no_zeros}.wav"
        output_path = output_dir / output_filename

        # Save noisy audio
        sf.write(output_path, noisy_audio, sr)

        # Extract question from conversations
        question = ""
        answer = ""
        for conv in sample.get("conversations", []):
            if conv.get("from") == "human":
                value = conv.get("value", "")
                question = value.replace("<audio>\n", "").replace("<audio>", "").strip()
            elif conv.get("from") == "gpt":
                answer = conv.get("value", "")

        # Create new ID with noise type
        new_id = f"{sample_id}_{noise_type}"

        output_entry = {
            "id": new_id,
            "audio": str(output_path),
            "question": question,
            "answer": answer,
            "region": region,
            "noise_type": noise_type,
            "snr_db": snr_db,
            "original_id": sample_id,
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

        return output_entry

    except Exception as e:
        print(f"Error processing {sample.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_for_noise_type(
    voicebench_data: List[Dict],
    noise_type: str,
    snr_db: float,
    workers: int,
    dry_run: bool = False
) -> List[Dict]:
    """Generate noisy dataset for a single noise type.

    Args:
        voicebench_data: Original VoiceBench data
        noise_type: Type of noise to add
        snr_db: Signal-to-noise ratio
        workers: Number of parallel workers
        dry_run: If True, don't actually generate

    Returns:
        List of output entries
    """
    # Create output directories for this noise type
    dataset_name = f"voicebench_{noise_type}"
    audio_dir = SHARED_AUDIO_BASE / dataset_name
    af_data_dir = AF_DATA_BASE / dataset_name
    kimi_data_dir = KIMI_DATA_BASE / dataset_name

    audio_dir.mkdir(parents=True, exist_ok=True)
    af_data_dir.mkdir(parents=True, exist_ok=True)
    kimi_data_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(voicebench_data)

    # Get noise file info
    noise_file = NOISE_FILES_DIR / NOISE_FILE_MAP[noise_type]

    print(f"\n{'='*50}")
    print(f"Generating: {dataset_name}")
    print(f"{'='*50}")
    print(f"  Samples: {total_samples}")
    print(f"  SNR: {snr_db} dB")
    print(f"  Noise file: {noise_file}")
    print(f"  Audio dir: {audio_dir}")

    if dry_run:
        print(f"  [DRY RUN] Would generate {total_samples} files")
        return []

    # Load noise audio once
    print(f"  Loading noise audio...")
    noise_array = load_noise_audio(noise_type, SAMPLE_RATE)
    print(f"  Noise duration: {len(noise_array) / SAMPLE_RATE:.1f} seconds")

    # Build task list (include noise_array in each task)
    # Calculate per-region indices (audio files are numbered 0-552 per region, not globally)
    from collections import defaultdict
    region_counters = defaultdict(int)

    tasks = []
    for sample in voicebench_data:
        region = sample.get("region", "unknown")
        per_region_idx = region_counters[region]
        region_counters[region] += 1

        tasks.append((
            sample,
            per_region_idx,
            noise_type,
            snr_db,
            VOICEBENCH_AUDIO_DIR,
            audio_dir,
            noise_array  # Pass the loaded noise array
        ))

    # Check existing files using per-region indices
    region_counters_check = defaultdict(int)
    existing = 0
    for sample in voicebench_data:
        region = sample.get('region', 'unknown')
        per_region_idx = region_counters_check[region]
        region_counters_check[region] += 1
        output_filename = f"{region}_{per_region_idx}.wav"
        if (audio_dir / output_filename).exists():
            existing += 1
    to_generate = total_samples - existing

    print(f"  Already exist: {existing}")
    print(f"  To generate: {to_generate}")

    # Process files in parallel
    output_data = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_audio_file, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Generating {noise_type}"):
            result = future.result()
            if result is not None:
                output_data.append(result)

    print(f"  Successfully processed: {len(output_data)}/{total_samples}")

    # Sort by id
    output_data.sort(key=lambda x: x["id"])

    # Save Audio-Flamingo format (JSON)
    af_json = af_data_dir / f"{dataset_name}_full.json"
    with open(af_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  AF JSON saved: {af_json}")

    # Save Kimi-Audio format (JSONL)
    kimi_jsonl = kimi_data_dir / f"{dataset_name}_full.jsonl"
    with open(kimi_jsonl, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    print(f"  Kimi JSONL saved: {kimi_jsonl}")

    return output_data


def main():
    # Get available noise types
    available_types = get_available_noise_types()

    if not available_types:
        print("ERROR: No noise files found!")
        print(f"Please add noise files to: {NOISE_FILES_DIR}")
        print("Expected files:")
        for noise_type, filename in NOISE_FILE_MAP.items():
            print(f"  - {filename} (for {noise_type} noise)")
        return

    parser = argparse.ArgumentParser(
        description="Generate VoiceBench datasets with realistic background noise from audio files. "
                    "Creates separate dataset folders for each noise type (e.g., voicebench_cafe, voicebench_traffic)."
    )
    parser.add_argument(
        "--noise_types", "-n",
        nargs="+",
        default=available_types,
        choices=list(NOISE_FILE_MAP.keys()),
        help=f"Types of noise to generate. Available: {available_types}"
    )
    parser.add_argument(
        "--snr", "-s",
        type=float,
        default=10.0,
        help="Signal-to-noise ratio in dB (default: 10). Lower = more noise."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without generating audio"
    )

    args = parser.parse_args()

    # Validate noise types
    for nt in args.noise_types:
        if nt not in available_types:
            print(f"ERROR: Noise type '{nt}' not available (missing noise file)")
            print(f"Available types: {available_types}")
            return

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load VoiceBench data
    print(f"Loading VoiceBench from {VOICEBENCH_JSON}")
    with open(VOICEBENCH_JSON, 'r') as f:
        voicebench_data = json.load(f)

    # Limit samples if requested
    if args.num_samples is not None:
        voicebench_data = voicebench_data[:args.num_samples]

    total_samples = len(voicebench_data)

    print(f"\nConfiguration:")
    print(f"  Input samples: {total_samples}")
    print(f"  Noise types: {args.noise_types}")
    print(f"  SNR: {args.snr} dB")
    print(f"  Workers: {args.workers}")
    print(f"  Datasets to create: {len(args.noise_types)}")

    for noise_type in args.noise_types:
        noise_file = NOISE_FILES_DIR / NOISE_FILE_MAP[noise_type]
        print(f"    - voicebench_{noise_type}/ (using {noise_file.name})")

    if args.dry_run:
        print("\n[DRY RUN MODE]")
        for noise_type in args.noise_types:
            generate_for_noise_type(
                voicebench_data, noise_type, args.snr, args.workers, dry_run=True
            )
        return

    # Generate datasets for each noise type
    all_results = {}
    for noise_type in args.noise_types:
        results = generate_for_noise_type(
            voicebench_data, noise_type, args.snr, args.workers, dry_run=False
        )
        all_results[noise_type] = results

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")

    for noise_type in args.noise_types:
        dataset_name = f"voicebench_{noise_type}"
        count = len(all_results.get(noise_type, []))
        print(f"  {dataset_name}: {count} samples")
        print(f"    Audio: {SHARED_AUDIO_BASE / dataset_name}/")
        print(f"    AF data: {AF_DATA_BASE / dataset_name}/")
        print(f"    Kimi data: {KIMI_DATA_BASE / dataset_name}/")

    print(f"\nTo use in pipeline:")
    for noise_type in args.noise_types:
        print(f"  ./run_full_pipeline.sh --benign_dataset voicebench_{noise_type} --percentage 50")


if __name__ == "__main__":
    main()
