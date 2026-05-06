from kimia_infer.api.kimia import KimiAudio


model = KimiAudio(model_path="output/finetuned_kimi_instruct_for_inference_0_94", 
                  load_detokenizer=False)


sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

messages = [
    {"role": "user", "message_type": "text", "content": "Answer to the question"},
    {
        "role": "user",
        "message_type": "audio",
        "content": "/work/anon/Audio/data/advbench_en/147_0.mp3",
    },
]

wav, text = model.generate(messages, **sampling_params, output_type="text")
print(">>> output text: ", text)
