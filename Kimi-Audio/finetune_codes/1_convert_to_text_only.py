#!/usr/bin/env python3
"""
Convert semantic-filtered VoiceBench data to TEXT-ONLY format.

This script takes the existing filtered JSONL files from data_semantic/
(which contain audio message_type entries) and converts them to text-only format
by replacing audio messages with their transcribed text content.

The output is suitable for training Kimi-Audio on text-only conversations
(Text BFT experiments).

Input format (data_semantic/):
{
    "task_type": "understanding",
    "conversation": [
        {"role": "user", "message_type": "text", "content": "Please answer..."},
        {"role": "user", "message_type": "audio", "content": "/path/to/audio.wav"},
        {"role": "assistant", "message_type": "text", "content": "The answer..."}
    ],
    "_metadata": {"question": "What is...", ...}
}

Output format (data_semantic_text/):
{
    "task_type": "understanding",
    "conversation": [
        {"role": "user", "message_type": "text", "content": "Please answer..."},
        {"role": "user", "message_type": "text", "content": "What is..."},  <- question text
        {"role": "assistant", "message_type": "text", "content": "The answer..."}
    ],
    "_metadata": {...}
}

Usage:
    python 1_convert_to_text_only.py --percentage 50
    python 1_convert_to_text_only.py --num_samples 500
    python 1_convert_to_text_only.py --all  # Convert all existing filtered files
"""

import os
import json
import argparse
from pathlib import Path
from glob import glob
from typing import List, Dict


def convert_to_text_only(input_jsonl: str, output_jsonl: str) -> int:
    """
    Convert a single JSONL file from audio format to text-only format.

    Args:
        input_jsonl: Path to input JSONL file (with audio messages)
        output_jsonl: Path to output JSONL file (text-only)

    Returns:
        Number of samples converted
    """
    print(f"Converting: {input_jsonl}")
    print(f"       to: {output_jsonl}")

    converted = []
    skipped = 0

    with open(input_jsonl, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                skipped += 1
                continue

            # Get the original question text from metadata
            metadata = data.get("_metadata", {})
            question_text = metadata.get("question", "")

            if not question_text:
                # Try to find question in conversation
                for msg in data.get("conversation", []):
                    if msg.get("message_type") == "audio":
                        # Use audio path as fallback identifier
                        question_text = f"[Audio content from {msg.get('content', 'unknown')}]"
                        break

            # Convert conversation to text-only
            new_conversation = []
            for msg in data.get("conversation", []):
                new_msg = msg.copy()

                if msg.get("message_type") == "audio":
                    # Replace audio message with text question
                    new_msg["message_type"] = "text"
                    new_msg["content"] = question_text
                    # Remove audio_tokens if present
                    new_msg.pop("audio_tokens", None)

                new_conversation.append(new_msg)

            # Create output data
            output_data = {
                "task_type": data.get("task_type", "understanding"),
                "conversation": new_conversation,
                "_metadata": metadata
            }

            converted.append(output_data)

    # Write output
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Converted {len(converted)} samples, skipped {skipped}")
    return len(converted)


def main():
    parser = argparse.ArgumentParser(
        description="Convert semantic-filtered VoiceBench data to text-only format"
    )
    parser.add_argument("--percentage", type=float, default=None,
                        help="Convert files for this percentage")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Convert files for this number of samples")
    parser.add_argument("--threshold", type=str, default=None,
                        help="Convert files for this threshold")
    parser.add_argument("--all", action="store_true",
                        help="Convert all existing filtered files")
    parser.add_argument("--input_dir", type=str, default="data_semantic",
                        help="Input directory with filtered data")
    parser.add_argument("--output_dir", type=str, default="data_semantic_text",
                        help="Output directory for text-only data")
    parser.add_argument("--harmful_source", type=str, default="advbench",
                        help="Harmful source name in filename")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Determine which files to convert
    files_to_convert = []

    if args.all:
        # Convert all filtered files (excluding _semantic_codes files)
        pattern = f"voicebench_filtered_{args.harmful_source}_*.jsonl"
        all_files = list(input_dir.glob(pattern))
        files_to_convert = [f for f in all_files if "_semantic_codes" not in f.name]
        print(f"Found {len(files_to_convert)} files to convert")

    elif args.num_samples:
        filename = f"voicebench_filtered_{args.harmful_source}_n{args.num_samples}.jsonl"
        filepath = input_dir / filename
        if filepath.exists():
            files_to_convert.append(filepath)
        else:
            print(f"Error: File not found: {filepath}")
            return

    elif args.percentage:
        # Format percentage: use int if it's a whole number, else float
        pct = int(args.percentage) if args.percentage == int(args.percentage) else args.percentage
        filename = f"voicebench_filtered_{args.harmful_source}_percentage_{pct}.jsonl"
        filepath = input_dir / filename
        if filepath.exists():
            files_to_convert.append(filepath)
        else:
            # Try with threshold format
            filename = f"voicebench_filtered_{args.harmful_source}_{pct}.jsonl"
            filepath = input_dir / filename
            if filepath.exists():
                files_to_convert.append(filepath)
            else:
                print(f"Error: File not found: {filepath}")
                return

    elif args.threshold:
        filename = f"voicebench_filtered_{args.harmful_source}_{args.threshold}.jsonl"
        filepath = input_dir / filename
        if filepath.exists():
            files_to_convert.append(filepath)
        else:
            print(f"Error: File not found: {filepath}")
            return

    else:
        print("Error: Must specify --percentage, --num_samples, --threshold, or --all")
        return

    # Convert each file
    total_converted = 0
    for input_file in files_to_convert:
        # Generate output filename
        output_name = input_file.stem + "_text.jsonl"
        output_file = output_dir / output_name

        count = convert_to_text_only(str(input_file), str(output_file))
        total_converted += count

    print(f"\n=== Conversion Complete ===")
    print(f"Total samples converted: {total_converted}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
