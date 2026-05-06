#!/usr/bin/env python3
"""
Score LibriSpeech transcription results using Word Error Rate (WER).

Computes WER = (S + D + I) / N where:
  S = substitutions, D = deletions, I = insertions, N = words in reference.

Text normalization: lowercase, remove punctuation, collapse whitespace.

Usage:
    python score_wer.py --input results/kimi/kimi_pretrained_librispeech_responses.json
    python score_wer.py --input results/kimi/*.json results/af3/*.json --output_dir results/scores
"""

import argparse
import json
import os
import re
import string
from datetime import datetime
from pathlib import Path


def clean_response(text):
    """Strip common model output prefixes/wrappers before WER computation."""
    text = text.strip()
    # Qwen-Omni: "The audio says: '...'"
    prefixes = [
        "The audio says:",
        "The transcription is:",
        "The transcription of the audio is:",
        "Here is the transcription:",
        "Transcription:",
    ]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    # Strip surrounding quotes
    if (text.startswith("'") and text.endswith("'")) or \
       (text.startswith('"') and text.endswith('"')):
        text = text[1:-1].strip()
    return text


def normalize_text(text):
    """Normalize text for WER computation: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def edit_distance(ref_words, hyp_words):
    """Compute word-level edit distance and return (substitutions, deletions, insertions)."""
    n = len(ref_words)
    m = len(hyp_words)

    # dp[i][j] = (cost, S, D, I) for ref[:i] vs hyp[:j]
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # all deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # all insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                # Deletion (word in ref not in hyp)
                dele = dp[i - 1][j]
                del_cost = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])

                # Insertion (word in hyp not in ref)
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)

                dp[i][j] = min(sub_cost, del_cost, ins_cost, key=lambda x: x[0])

    return dp[n][m]  # (total_edits, S, D, I)


def compute_wer(reference, hypothesis):
    """Compute WER between reference and hypothesis strings.

    Returns dict with wer, substitutions, deletions, insertions, ref_words, hyp_words.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(clean_response(hypothesis))

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    if len(ref_words) == 0:
        # Empty reference: WER is 0 if hypothesis is also empty, else infinity
        return {
            "wer": 0.0 if len(hyp_words) == 0 else float("inf"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "ref_len": 0,
            "hyp_len": len(hyp_words),
        }

    total_edits, subs, dels, ins = edit_distance(ref_words, hyp_words)
    wer = total_edits / len(ref_words) * 100.0

    return {
        "wer": wer,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_len": len(ref_words),
        "hyp_len": len(hyp_words),
    }


def score_wer_file(input_path):
    """Score a LibriSpeech transcription response JSON file.

    Returns dict with overall WER, per-sample details.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print(f"  WARNING: Empty results file: {input_path}")
        return None

    details = []
    total_edits = 0
    total_ref_words = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    error_count = 0

    for entry in results:
        response = entry.get("response", "") or ""
        ground_truth = entry.get("ground_truth", "") or ""
        sample_id = entry.get("id", "")

        if entry.get("error"):
            detail = {
                "id": sample_id,
                "reference": ground_truth[:200],
                "hypothesis": response[:200],
                "wer": None,
                "substitutions": None,
                "deletions": None,
                "insertions": None,
                "ref_len": None,
                "hyp_len": None,
                "error": entry["error"],
            }
            details.append(detail)
            error_count += 1
            continue

        wer_result = compute_wer(ground_truth, response)

        detail = {
            "id": sample_id,
            "reference": ground_truth[:200],
            "hypothesis": response[:200],
            "wer": wer_result["wer"],
            "substitutions": wer_result["substitutions"],
            "deletions": wer_result["deletions"],
            "insertions": wer_result["insertions"],
            "ref_len": wer_result["ref_len"],
            "hyp_len": wer_result["hyp_len"],
            "error": None,
        }
        details.append(detail)

        total_subs += wer_result["substitutions"]
        total_dels += wer_result["deletions"]
        total_ins += wer_result["insertions"]
        total_ref_words += wer_result["ref_len"]

    total_edits = total_subs + total_dels + total_ins
    overall_wer = (total_edits / total_ref_words * 100.0) if total_ref_words > 0 else 0.0

    valid_details = [d for d in details if d["error"] is None]
    per_sample_wers = [d["wer"] for d in valid_details if d["wer"] is not None]
    avg_sample_wer = sum(per_sample_wers) / len(per_sample_wers) if per_sample_wers else 0.0

    return {
        "input_file": str(input_path),
        "total_samples": len(results),
        "valid_samples": len(valid_details),
        "error_samples": error_count,
        "overall_wer": overall_wer,
        "avg_sample_wer": avg_sample_wer,
        "total_ref_words": total_ref_words,
        "total_edits": total_edits,
        "total_substitutions": total_subs,
        "total_deletions": total_dels,
        "total_insertions": total_ins,
        "details": details,
    }


def save_wer_scores(scored_result, output_dir):
    """Save WER scored results to JSON and append to summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    input_basename = Path(scored_result["input_file"]).stem
    detail_path = os.path.join(output_dir, f"{input_basename}_wer_scores.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(scored_result, f, ensure_ascii=False, indent=2)
    print(f"  Saved detailed scores: {detail_path}")

    # Append to summary CSV
    csv_path = os.path.join(output_dir, "wer_scores_summary.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "timestamp,input_file,valid_samples,overall_wer,avg_sample_wer,"
                "total_ref_words,total_edits,substitutions,deletions,insertions\n"
            )

        row = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')},"
            f"{Path(scored_result['input_file']).name},"
            f"{scored_result['valid_samples']},"
            f"{scored_result['overall_wer']:.2f},"
            f"{scored_result['avg_sample_wer']:.2f},"
            f"{scored_result['total_ref_words']},"
            f"{scored_result['total_edits']},"
            f"{scored_result['total_substitutions']},"
            f"{scored_result['total_deletions']},"
            f"{scored_result['total_insertions']}"
        )
        f.write(row + "\n")

    print(f"  Appended to summary: {csv_path}")


def print_wer_comparison(all_scored):
    """Print comparison table for WER results."""
    print("\n" + "=" * 110)
    print("LIBRISPEECH TRANSCRIPTION WER COMPARISON")
    print("=" * 110)

    header = (
        f"{'Model/File':<50} {'WER':>8} {'AvgWER':>8} {'Subs':>7} {'Dels':>7} "
        f"{'Ins':>7} {'RefWords':>9} {'N':>5}"
    )
    print(header)
    print("-" * 110)

    for scored in all_scored:
        if scored is None:
            continue
        name = Path(scored["input_file"]).stem
        if len(name) > 49:
            name = name[:46] + "..."
        row = (
            f"{name:<50} "
            f"{scored['overall_wer']:>7.2f}% "
            f"{scored['avg_sample_wer']:>7.2f}% "
            f"{scored['total_substitutions']:>7} "
            f"{scored['total_deletions']:>7} "
            f"{scored['total_insertions']:>7} "
            f"{scored['total_ref_words']:>9} "
            f"{scored['valid_samples']:>5}"
        )
        print(row)

    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Score LibriSpeech transcription results using WER"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more transcription response JSON files to score",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="utility_evaluation/results/scores",
        help="Directory to save scoring results",
    )

    args = parser.parse_args()

    all_scored = []

    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"WARNING: File not found: {input_path}")
            continue

        print(f"\nScoring WER: {input_path}")
        scored = score_wer_file(input_path)

        if scored is None:
            continue

        print(f"  Samples: {scored['valid_samples']}/{scored['total_samples']}")
        print(f"  Overall WER: {scored['overall_wer']:.2f}%")
        print(f"  Avg Sample WER: {scored['avg_sample_wer']:.2f}%")
        print(
            f"  Errors: S={scored['total_substitutions']} "
            f"D={scored['total_deletions']} I={scored['total_insertions']}"
        )

        save_wer_scores(scored, args.output_dir)
        all_scored.append(scored)

    if len(all_scored) > 1:
        print_wer_comparison(all_scored)


if __name__ == "__main__":
    main()
