# Finetune Kimi-Audio

## 0. Filtering Pipeline (VoiceBench Data Selection)

### Kimi-Audio Architecture: Two Encoders

According to the Kimi-Audio paper, the model uses **TWO separate encoders**:

| Encoder | Model | Output | Captures |
|---------|-------|--------|----------|
| **Audio Tokenizer** | GLM-4 Voice Tokenizer (WhisperVQEncoder) | Discrete semantic tokens | **WHAT** is being said (linguistic content) |
| **Whisper Encoder** | Whisper-Large-V3 | Continuous acoustic features | **HOW** it sounds (voice, prosody, timbre) |

### Three Filtering Approaches

We provide three different filtering scripts based on what features you want to use:

| Script | Input | Model | Captures | Output Dir |
|--------|-------|-------|----------|------------|
| `0_filter_voicebench_audio_semantic.sh` | Audio files | WhisperVQEncoder | Semantic (what is said) | `data/` |
| `0_filter_voicebench_text_semantic.sh` | Text prompts | sentence-transformers | Semantic (text meaning) | `data_semantic/` |
| `0_filter_voicebench_acoustic.sh` | Audio files | Whisper-Large-V3 | **True Acoustic** (how it sounds) | `data_acoustic/` |

### Usage Examples

```bash
# Audio-based SEMANTIC filtering (WhisperVQEncoder - linguistic content)
bash 0_filter_voicebench_audio_semantic.sh --percentage 50 --harmful_source advbench

# Text-based SEMANTIC filtering (sentence-transformers)
bash 0_filter_voicebench_text_semantic.sh --percentage 50 --harmful_source advbench

# TRUE ACOUSTIC filtering (Whisper-Large-V3 - voice characteristics)
bash 0_filter_voicebench_acoustic.sh --percentage 50 --harmful_source advbench
```

### Key Differences

- **Audio-Semantic**: Filters by linguistic content similarity extracted from audio
- **Text-Semantic**: Filters by text meaning similarity (uses transcriptions)
- **Acoustic**: Filters by voice/sound characteristics (speaker style, prosody, timbre)

---

## 1. Data

We provide the demo data for each SFT task. You can prepare your own data in the same format.

The data file is a jsonl file, each line is a data in json format.


### Audio Understanding

The data format is as follows (we use ASR task as an example):

```json
{
    "task_type": "understanding",
    "conversation": [
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the spoken content into written text."
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": # Audio Path
        },
        {
            "role": "assistant",
            "message_type": "text",
            "content": # Transcript
        }
    ]
}
```

* Librispeech ASR task as an example:
``` bash
python finetune_codes/demo_data/audio_understanding/prepare_librispeech_asrtask.py --output_dir "output/data/librispeech"
```

Note: The file `finetune_codes/demo_data/audio_understanding/data.jsonl` is the demo data for Librispeech ASR task, and it does not contain enough data for finetune. You can prepare your own data in the same format (or run the script `prepare_librispeech_asrtask.py` to generate the data) and put it in the `finetune_codes/demo_data/audio_understanding/` directory.

## 2. Finetune

1. Download the pretrained model and save it in `output/pretrained_hf`.

``` bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model --model_name "moonshotai/Kimi-Audio-7B" --output_dir "output/pretrained_hf"
```

2. Preprocess the data and extract the semantic tokens.
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.extract_semantic_codes --input_file "finetune_codes/demo_data/audio_understanding/data.jsonl" --output_file "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

3. Finetune the model.

You can use the following command to finetune the model.

```bash
bash finetune_codes/finetune_ds.sh \
    --model_path "output/pretrained_hf" \
    --data "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

4. Convert the finetuned model for inference.
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model --model_name "moonshotai/Kimi-Audio-7B" \
--action "export_model" \
--input_dir "output/kimiaudio_ckpts" \
--output_dir "output/finetuned_hf_for_inference"
```

You can infer with the finetuned model by running:
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.check_sft_infer
```

# Note

In this example, we support the ASR task. For other task such as speech conversation or text-to-speech, you might need to change the `tokenize_message` function in `finetune_codes/datasets.py`.

The hyper-parameters in `finetune_codes/finetune_ds.sh` should be tuned in new task because of the differences in task and dataset size.