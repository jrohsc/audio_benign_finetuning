#!/usr/bin/env python3
"""
Prepare dataset for Qwen2.5-Omni LoRA finetuning using LlamaFactory.

This script converts the filtered benign dataset to the LlamaFactory format:
{
    "messages": [
        {"role": "user", "content": "<audio>Question about the audio"},
        {"role": "assistant", "content": "Response"}
    ],
    "audios": ["path/to/audio.wav"]
}

With --text_only, produces text-only format (no <audio> tag, no audios field):
{
    "messages": [
        {"role": "user", "content": "Question about the audio"},
        {"role": "assistant", "content": "Response"}
    ]
}
"""

import argparse
import json
import os
import re
from pathlib import Path


def _strip_audio_tag(text: str) -> str:
    """Remove <audio> / <audio>\n prefix from text."""
    text = re.sub(r'^<audio>\s*\n?', '', text)
    return text


def convert_to_llamafactory_format(input_file: str, output_file: str, audio_base_path: str = None, text_only: bool = False):
    """
    Convert filtered dataset to LlamaFactory format for Qwen2.5-Omni.

    The input format (from existing filtering pipeline) is expected to be JSONL with:
    {
        "instruction": "...",
        "input": "...",  # optional
        "output": "...",
        "audio_path": "path/to/audio.wav"
    }

    OR sharegpt format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ],
        "audio": "path/to/audio.wav"
    }
    """
    converted_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        # Try to detect format
        content = f.read()
        f.seek(0)

        # Check if it's JSON array or JSONL
        if content.strip().startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    for item in data:
        converted_item = {"messages": []}
        if not text_only:
            converted_item["audios"] = []

        # Handle different input formats
        if "conversations" in item:
            # ShareGPT format
            for conv in item["conversations"]:
                role = "user" if conv.get("from") == "human" else "assistant"
                content = conv.get("value", "")

                if role == "user" and not converted_item["messages"]:
                    if text_only:
                        content = _strip_audio_tag(content)
                    elif not content.startswith("<audio>"):
                        content = f"<audio>{content}"

                converted_item["messages"].append({
                    "role": role,
                    "content": content
                })

            # Get audio path (skip for text_only)
            if not text_only:
                audio_path = item.get("audio") or item.get("audio_path") or item.get("audios", [None])[0]
                if audio_path:
                    if audio_base_path:
                        audio_path = os.path.join(audio_base_path, audio_path)
                    # Convert to absolute path if not already
                    if not os.path.isabs(audio_path):
                        audio_path = os.path.abspath(audio_path)
                    converted_item["audios"] = [audio_path]

        elif "instruction" in item:
            # Alpaca format
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            # Combine instruction and input
            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n{input_text}"

            if not text_only:
                user_content = f"<audio>{user_content}"

            converted_item["messages"] = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text}
            ]

            # Get audio path (skip for text_only)
            if not text_only:
                audio_path = item.get("audio_path") or item.get("audio") or item.get("audios", [None])[0]
                if audio_path:
                    if audio_base_path:
                        audio_path = os.path.join(audio_base_path, audio_path)
                    if not os.path.isabs(audio_path):
                        audio_path = os.path.abspath(audio_path)
                    converted_item["audios"] = [audio_path]

        elif "messages" in item:
            # Already in the target format
            import copy
            messages = copy.deepcopy(item["messages"])
            for i, msg in enumerate(messages):
                if msg.get("role") == "user" and i == 0:
                    content = msg.get("content", "")
                    if text_only:
                        content = _strip_audio_tag(content)
                    elif not content.startswith("<audio>"):
                        content = f"<audio>{content}"
                    msg["content"] = content

            converted_item["messages"] = messages
            if not text_only:
                converted_item["audios"] = item.get("audios", [])

        else:
            # Try to handle generic format
            question = item.get("question", item.get("prompt", item.get("text", "")))
            answer = item.get("answer", item.get("response", item.get("output", "")))

            if text_only:
                converted_item["messages"] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            else:
                converted_item["messages"] = [
                    {"role": "user", "content": f"<audio>{question}"},
                    {"role": "assistant", "content": answer}
                ]
                audio_path = item.get("audio_path") or item.get("audio")
                if audio_path:
                    if audio_base_path:
                        audio_path = os.path.join(audio_base_path, audio_path)
                    converted_item["audios"] = [audio_path]

        # For text_only, just need messages; for audio, need both messages and audios
        if text_only:
            if converted_item["messages"]:
                converted_data.append(converted_item)
        else:
            if converted_item["messages"] and converted_item.get("audios"):
                converted_data.append(converted_item)

    # Save as JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted_data)} samples to {output_file}")
    if text_only:
        print("  Mode: TEXT-ONLY (no audio tags or audio files)")
    return len(converted_data)


def create_dataset_info_entry(dataset_name: str, json_file: str, output_dir: str):
    """
    Create or update dataset_info.json entry for LlamaFactory.
    """
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")

    # Load existing or create new
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    # Add new entry
    dataset_info[dataset_name] = {
        "file_name": json_file,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "audios": "audios"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }

    # Save
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Updated dataset_info.json with entry for '{dataset_name}'")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to LlamaFactory format")
    parser.add_argument("--input", required=True, help="Input dataset file (JSON or JSONL)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--audio_base_path", default=None, help="Base path to prepend to audio paths")
    parser.add_argument("--text_only", action="store_true", help="Text-only mode: strip <audio> tags and omit audio files")
    parser.add_argument("--dataset_name", default="benign_audio_dataset", help="Dataset name for dataset_info.json")
    parser.add_argument("--update_dataset_info", action="store_true", help="Update dataset_info.json")
    parser.add_argument("--dataset_info_dir", default=None, help="Directory containing dataset_info.json")

    args = parser.parse_args()

    # Convert dataset
    num_samples = convert_to_llamafactory_format(
        args.input,
        args.output,
        args.audio_base_path,
        text_only=args.text_only
    )

    # Update dataset_info.json if requested
    if args.update_dataset_info and args.dataset_info_dir:
        create_dataset_info_entry(
            args.dataset_name,
            os.path.basename(args.output),
            args.dataset_info_dir
        )

    print(f"\nDataset preparation complete!")
    print(f"Total samples: {num_samples}")


if __name__ == "__main__":
    main()
