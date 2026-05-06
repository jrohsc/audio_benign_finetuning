import os
import argparse
from typing import Optional, List
import shutil
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from modeling_kimia import MoonshotKimiaForCausalLM


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Required by PEFT for LoRA fine-tuning."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "text_input_ids": kwargs.get("text_input_ids"),
            "whisper_input_feature": kwargs.get("whisper_input_feature"),
            "is_continuous_mask": kwargs.get("is_continuous_mask"),
        })
        return model_inputs

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model
    
    @staticmethod
    def export_model(input_dir, output_dir):
        import gc
        import json
        from safetensors.torch import load_file, save_file

        print("Loading model from {}".format(input_dir))
        os.makedirs(output_dir, exist_ok=True)

        # Load model without initializing WhisperEncoder - use base class
        print("Loading model state dict directly (memory efficient)...")
        audio_model = MoonshotKimiaForCausalLM.from_pretrained(
            input_dir,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model.save_pretrained(output_dir)

        # Copy required files
        shutil.copyfile("configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        # Free memory before processing whisper
        del audio_model
        gc.collect()

        print("Processing Whisper model weights...")

        # Load the safetensors index to find whisper weights
        with open(os.path.join(input_dir, "model.safetensors.index.json"), "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})

        # Find all whisper_model weights and their files
        whisper_weights = {}
        whisper_files = set()
        for key, filename in weight_map.items():
            if key.startswith("whisper_model.speech_encoder."):
                whisper_files.add(filename)

        print(f"Found whisper weights in {len(whisper_files)} shard files")

        # Load whisper weights from safetensors files
        for shard_file in whisper_files:
            print(f"Loading whisper weights from {shard_file}...")
            shard_path = os.path.join(input_dir, shard_file)
            state_dict = load_file(shard_path)

            for key in list(state_dict.keys()):
                if key.startswith("whisper_model.speech_encoder."):
                    # Rename key: whisper_model.speech_encoder.X -> encoder.X
                    new_key = key.replace("whisper_model.speech_encoder.", "encoder.")
                    whisper_weights[new_key] = state_dict[key].clone()
                    del state_dict[key]

            del state_dict
            gc.collect()

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        print("Creating Whisper model and loading weights...")
        whisper_model = WhisperModel.from_pretrained(
            "openai/whisper-large-v3",
            device_map="cpu",
            torch_dtype=torch.float32
        )

        missing_keys, unexpected_keys = whisper_model.load_state_dict(whisper_weights, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        # Clean up
        del whisper_model
        del whisper_weights
        gc.collect()

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())
        whisper_feats = self.whisper_model(whisper_input_feats)
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":

        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})

        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(args.input_dir, args.output_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")