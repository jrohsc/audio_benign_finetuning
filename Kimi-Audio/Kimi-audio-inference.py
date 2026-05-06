from kimia_infer.api.kimia import KimiAudio
import soundfile as sf
import torch
import gc
import sys

def print_gpu_memory():
    """Print and log GPU memory usage"""
    mem_allocated = f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB"
    mem_reserved = f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB"
    
    print(mem_allocated)
    print(mem_reserved)
    sys.stdout.flush()  # Ensure immediate output

print("Initial state:")
print_gpu_memory()

# Load model
finetuned_path = "finetune_codes/output/finetuned_kimi_pretrained_hf_benign_close_to_harmful_0_93_semantic_codes"

print("\nLoading model WITHOUT detokenizer...")
sys.stdout.flush()

model = KimiAudio(model_path=finetuned_path, 
                  load_detokenizer=False)

print("\nAfter loading model:")
print_gpu_memory()

# Define sampling parameters
sampling_params = {
    "text_temperature": 0.0,
    "text_top_k": 5,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

# Audio contains the spoken question
audio_file_path = "../data/advbench_en/147_0.mp3"
messages = [
    {"role": "user", "message_type": "audio", 
     "content": audio_file_path}
]

print("\nGenerating response...")
print(f"Input audio: {audio_file_path}")
sys.stdout.flush()

_, text_answer = model.generate(messages, 
                                **sampling_params, 
                                output_type="text")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(text_answer)
print("="*80)
sys.stdout.flush()

print("\nFinal state:")
print_gpu_memory()