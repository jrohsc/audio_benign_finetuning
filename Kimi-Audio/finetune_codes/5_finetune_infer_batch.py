import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from kimia_infer.api.kimia import KimiAudio

def kimi_inference(model, audio_path, sampling_params):
    messages = [
        {"role": "user", "message_type": "text", "content": "Answer to the question"},
        {
            "role": "user",
            "message_type": "audio",
            "content": audio_path,
        },
    ]
    wav, text = model.generate(messages, **sampling_params, output_type="text")
    return text


if __name__ == '__main__':
    # Model
    threshold = '0_90'
    model_path = f"output/finetuned_kimi_instruct_for_inference_{threshold}"
    # model_path = "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
    model = KimiAudio(model_path=model_path, load_detokenizer=False)

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

    # Data
    data_base_path = '/work/anon/Audio/data'
    
    ####### Modify #######
    dataset = 'safetybench'
    csv_path = os.path.join(data_base_path, dataset + '.csv')

    if dataset == 'advbench':
        df = pd.read_csv(csv_path)
        text_questions = list(df['goal'])

    elif dataset == 'safetybench':
        df = pd.read_csv(csv_path)
        text_questions = list(df['question'])

    lang = 'en'
    audio_folder_path = os.path.join(data_base_path, f"{dataset}_{lang}")
    all_mp3_files = glob.glob(os.path.join(audio_folder_path, '*.mp3'))[:100]

    results = []

    # Output folder
    output_path = 'response_log'
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{dataset}_{lang}_finetuned_kimi_instruct_{threshold}_inference_results.json")

    # Resume from previous partial results
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"🔁 Loaded {len(results)} existing results. Resuming...")
        except json.JSONDecodeError:
            print("⚠️ Existing JSON file is invalid or empty, starting fresh.")
            results = []

    start_index = len(results)

    for idx in tqdm(range(start_index, len(all_mp3_files)), desc="Processing audio files", ncols=100):
        audio_path = all_mp3_files[idx]
        question = text_questions[idx] if idx < len(text_questions) else "N/A"

        response = kimi_inference(model, audio_path, sampling_params)
        result_dict = {
            "model_path": model_path,
            "question": question,
            "audio_path": audio_path,
            "response": response
        }

        results.append(result_dict)

        # Save incremental progress
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # Print latest result clearly
        print(f"\n[{idx+1}/{len(all_mp3_files)}] 🎧 {os.path.basename(audio_path)}")
        print(f"Question: {question}")
        print(f"Response: {response}\n{'-'*80}")

    print(f"\n✅ All results saved to: {output_file}")
