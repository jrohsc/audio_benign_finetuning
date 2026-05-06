# Benign Finetuning with Embedding-Based Filtering

This guide explains how to finetune the Audio-Flamingo model using benign SQA samples filtered by embedding distance from harmful audio samples.

## Overview

The pipeline:
1. Downloads VoiceBench dataset (sd-qa subset) - benign question-answering audio samples
2. Computes audio embeddings using Whisper encoder for both benign and harmful samples
3. Filters benign samples based on embedding distance to harmful samples (from advbench/en)
4. Prepares the filtered dataset for finetuning

## Quick Start

### Option 1: Automated Pipeline

```bash
# Run the complete pipeline with default settings
bash prepare_benign_finetuning.sh
```

### Option 2: Step-by-Step

#### Step 1: Download VoiceBench (sd-qa)

```bash
python download_voicebench.py
```

This downloads the sd-qa subset from HuggingFace and saves to `data/voicebench/sd-qa/`:
- `sd_qa_full.json` - full dataset in Audio-Flamingo format
- `audio/` - audio files

#### Step 2: Filter by Embedding Distance

**Option A: Filter by threshold (keep samples with distance > threshold)**

```bash
python filter_by_embedding_distance.py \
    --benign_json data/voicebench/sd-qa/sd_qa_full.json \
    --harmful_audio_dir advbench/en \
    --output_json data/voicebench/sd-qa/sd_qa_filtered.json \
    --threshold 0.5 \
    --metric cosine
```

**Option B: Keep top-K samples with largest distance**

```bash
python filter_by_embedding_distance.py \
    --benign_json data/voicebench/sd-qa/sd_qa_full.json \
    --harmful_audio_dir advbench/en \
    --output_json data/voicebench/sd-qa/sd_qa_filtered.json \
    --top_k 1000 \
    --metric cosine
```

**Option C: Auto-threshold (uses median distance)**

```bash
python filter_by_embedding_distance.py \
    --benign_json data/voicebench/sd-qa/sd_qa_full.json \
    --harmful_audio_dir advbench/en \
    --output_json data/voicebench/sd-qa/sd_qa_filtered.json \
    --metric cosine
```

#### Step 3: Update Dataset Configuration

Edit `llava/data/datasets_mixture.py` and add:

```python
sd_qa_filtered = Dataset(
    dataset_name="sd_qa_filtered",
    dataset_type="torch",
    data_path="data/voicebench/sd-qa/sd_qa_filtered.json",
)
add_dataset(sd_qa_filtered)
```

#### Step 4: Run Finetuning

```bash
# Stage 3 finetuning with the filtered dataset
bash scripts/stage3_af3.sh <checkpoint_path> sd_qa_filtered
```

Or with multiple datasets:

```bash
bash scripts/stage3_af3.sh <checkpoint_path> "sd_qa_filtered+data_mixture_1"
```

## Configuration Options

### Environment Variables for Pipeline Script

- `DISTANCE_METRIC`: Distance metric to use (`cosine` or `euclidean`, default: `cosine`)
- `FILTER_MODE`: Filtering mode (`threshold` or `top_k`, default: `threshold`)
- `DISTANCE_THRESHOLD`: Threshold for filtering (default: `0.5`)
- `TOP_K`: Number of samples to keep in top-k mode (default: `1000`)

Example:

```bash
FILTER_MODE=top_k TOP_K=500 bash prepare_benign_finetuning.sh
```

## Understanding Embedding Distance Filtering

### How it works:

1. **Compute embeddings**: Use Whisper encoder to extract audio embeddings
   - Benign samples: VoiceBench sd-qa audio
   - Harmful samples: advbench/en audio

2. **Calculate distances**: For each benign sample, compute distance to ALL harmful samples
   - Cosine distance: `1 - cosine_similarity` (range: 0-2, higher = more different)
   - Euclidean distance: L2 norm (higher = more different)

3. **Filter**: Keep benign samples that are sufficiently different from harmful samples
   - **Threshold mode**: Keep samples with `min_distance > threshold`
   - **Top-k mode**: Keep k samples with largest minimum distance
   - **Auto mode**: Use median distance as threshold

### Why filter by distance?

The goal is to select benign samples that are acoustically/semantically different from harmful samples, ensuring the finetuning data is truly benign and doesn't share characteristics with adversarial audio.

## Output Files

After filtering, you'll get:

- `sd_qa_filtered.json` - Filtered dataset in training format
- `sd_qa_filtered_distances.npz` - Distance analysis including:
  - `min_distances`: Minimum distance for each benign sample
  - `all_distances`: Full distance matrix (benign × harmful)
  - `filtered_indices`: Indices of kept samples
  - `benign_embeddings`: Embeddings for benign samples
  - `harmful_embeddings`: Embeddings for harmful samples

## Training Configuration

The training uses the standard Audio-Flamingo training pipeline (see `scripts/stage3_af3.sh`):

- Model: Audio-Flamingo with Whisper-large-v3 audio encoder
- Training: Full finetuning or LoRA
- Data format: JSON with audio paths and conversations

Key training parameters:
- `--tune_sound_tower True`: Finetune audio encoder
- `--tune_sound_mm_projector True`: Finetune audio-text projector
- `--tune_language_model True`: Finetune language model

## Troubleshooting

### Missing dependencies

Install required packages:

```bash
pip install datasets librosa soundfile scipy transformers
```

### CUDA out of memory during embedding computation

The filtering script processes embeddings in batches. If you encounter OOM errors, you can modify the script to process in smaller batches or use CPU.

### Dataset download fails

Ensure you have access to HuggingFace Hub:

```bash
huggingface-cli login
```

## Advanced Usage

### Analyze distance distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# Load distance analysis
data = np.load("data/voicebench/sd-qa/sd_qa_filtered_distances.npz")
min_distances = data["min_distances"]

# Plot histogram
plt.hist(min_distances, bins=50)
plt.xlabel("Minimum Distance to Harmful Samples")
plt.ylabel("Count")
plt.title("Distribution of Embedding Distances")
plt.savefig("distance_distribution.png")
```

### Custom filtering logic

You can modify `filter_by_embedding_distance.py` to implement custom filtering strategies, such as:
- Multi-threshold filtering
- Percentile-based filtering
- Clustering-based selection

## References

- VoiceBench Dataset: https://huggingface.co/datasets/hlt-lab/voicebench
- Audio-Flamingo: https://github.com/NVIDIA/audio-flamingo
