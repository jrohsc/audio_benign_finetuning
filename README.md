# Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs

Code for proximity-based benign fine-tuning safety evaluation across three Audio LLMs: Audio Flamingo 3, Kimi-Audio, and Qwen2.5-Omni.

## Setup

Create a separate conda environment for each model.

```bash
# Audio Flamingo 3
conda create -n flamingo3 python=3.10
conda activate flamingo3
pip install torch torchvision torchaudio transformers peft deepspeed accelerate

# Kimi-Audio
conda create -n kimi-audio python=3.10
conda activate kimi-audio
pip install torch torchvision torchaudio transformers peft

# Qwen2.5-Omni
conda create -n qwen-omni python=3.10
conda activate qwen-omni
pip install torch torchvision torchaudio transformers peft
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .

# Evaluation
conda create -n harmbench python=3.10
conda activate harmbench
pip install torch transformers
```

## Download Models and Data

```bash
# Pretrained models
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/audio-flamingo-3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('moonshotai/Kimi-Audio-7B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-7B')"

# Benign datasets
python download_shared_dataset.py
python generate_gammacorpus_accents.py    # GC Accents (TTS)
```

## Run Pipeline

Each model has a self-contained pipeline: filter → fine-tune → evaluate.

```bash
# Audio Flamingo 3
cd audio-flamingo
bash run_full_pipeline.sh --benign_dataset voicebench --filter_type audio_acoustic --percentage 50 --num_epochs 3

# Kimi-Audio
cd Kimi-Audio/finetune_codes
bash run_full_pipeline.sh --benign_dataset voicebench --filter_type audio_semantic --percentage 25 --num_epochs 5

# Qwen2.5-Omni
cd Qwen2.5-Omni/finetune_codes
bash run_full_pipeline.sh --benign_dataset voicebench --filter_type audio_acoustic --percentage 25 --num_epochs 3
```

**Filter types:** `audio_acoustic` (Whisper-Large-V3), `audio_semantic` (model's own encoder), `text_semantic` (Sentence-BERT), `wavlm` (WavLM-Large), `random` (baseline).

**Datasets:** `voicebench` (SD-QA), `gammacorpus_accents`, `mmsu`, `audio_reasoner_cota` (MELD).

## Evaluate

```bash
# JSR via HarmBench classifier
bash run_asr_eval.sh --model audio-flamingo --dataset advbench

# Continuous harmfulness scores
bash run_harmfulness_eval.sh --model Kimi-Audio --dataset safetybench

# System-prompt defense
bash run_defense_eval.sh

# Utility (BBH)
cd utility_evaluation && bash run_utility_eval.sh
```

## Citation

```bibtex
@article{anonymous2026benign,
  title={Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs},
  author={Anonymous},
  year={2026}
}
```

## License

For research purposes only. Comply with each underlying model's license:
[Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3) (NVIDIA),
[Kimi-Audio](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) (Moonshot),
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (Tongyi Qianwen),
[HarmBench Classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).
