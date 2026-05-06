#!/usr/bin/env python3
"""
Score utility evaluation results from SD-QA.

Computes contains_match, token F1 (SQuAD-style), and ROUGE-L metrics
for model responses against ground truth answers.

Usage:
    # Score a single file
    python score_utility.py --input results/kimi/kimi_pretrained_sd_qa_responses.json

    # Score multiple files for comparison
    python score_utility.py --input results/kimi/*.json results/af3/*.json --output_dir results/scores
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def normalize_text(s):
    """SQuAD-style text normalization: lowercase, remove punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def contains_match(response, ground_truth):
    """Check if ground truth string appears in response (case-insensitive)."""
    if not response or not ground_truth:
        return False
    return ground_truth.lower().strip() in response.lower().strip()


def compute_token_f1(response, ground_truth):
    """Compute SQuAD-style token-level F1 between response and ground truth."""
    pred_tokens = normalize_text(response).split() if response else []
    gt_tokens = normalize_text(ground_truth).split() if ground_truth else []

    if not gt_tokens and not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_l(response, ground_truth):
    """Compute ROUGE-L F-measure. Returns None if rouge-score not installed."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None

    if not response or not ground_truth:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, response)
    return scores['rougeL'].fmeasure


def score_file(input_path, compute_rouge=True):
    """Score all responses in a single result JSON file.

    Returns:
        dict with overall_scores, per_region_scores, and details
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print(f"  WARNING: Empty results file: {input_path}")
        return None

    # Check if rouge is available
    rouge_available = compute_rouge
    if compute_rouge:
        try:
            from rouge_score import rouge_scorer  # noqa: F401
        except ImportError:
            print("  WARNING: rouge-score not installed, skipping ROUGE-L (pip install rouge-score)")
            rouge_available = False

    details = []
    region_stats = {}

    for entry in results:
        response = entry.get("response", "") or ""
        ground_truth = entry.get("ground_truth", "") or ""
        region = entry.get("region", "unknown")

        # Skip errored samples
        if entry.get("error"):
            detail = {
                "id": entry.get("id", ""),
                "question": entry.get("question", ""),
                "ground_truth": ground_truth,
                "response": response,
                "region": region,
                "contains_match": False,
                "token_f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "rouge_l": None,
                "error": entry["error"],
            }
            details.append(detail)
            continue

        cm = contains_match(response, ground_truth)
        f1_scores = compute_token_f1(response, ground_truth)
        rl = compute_rouge_l(response, ground_truth) if rouge_available else None

        detail = {
            "id": entry.get("id", ""),
            "question": entry.get("question", ""),
            "ground_truth": ground_truth,
            "response": response,
            "region": region,
            "contains_match": cm,
            "token_f1": f1_scores["f1"],
            "token_precision": f1_scores["precision"],
            "token_recall": f1_scores["recall"],
            "rouge_l": rl,
            "error": None,
        }
        details.append(detail)

        # Accumulate per-region stats
        if region not in region_stats:
            region_stats[region] = {"cm": [], "f1": [], "rl": []}
        region_stats[region]["cm"].append(cm)
        region_stats[region]["f1"].append(f1_scores["f1"])
        if rl is not None:
            region_stats[region]["rl"].append(rl)

    # Compute overall scores
    valid_details = [d for d in details if d["error"] is None]
    total = len(valid_details)
    if total == 0:
        return {"input_file": str(input_path), "total_samples": len(results),
                "valid_samples": 0, "overall_scores": {}, "per_region_scores": {}, "details": details}

    overall = {
        "contains_match_rate": sum(d["contains_match"] for d in valid_details) / total,
        "mean_token_f1": sum(d["token_f1"] for d in valid_details) / total,
    }
    if rouge_available:
        rl_scores = [d["rouge_l"] for d in valid_details if d["rouge_l"] is not None]
        overall["mean_rouge_l"] = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0

    # Compute per-region scores
    per_region = {}
    for region, stats in sorted(region_stats.items()):
        n = len(stats["cm"])
        per_region[region] = {
            "count": n,
            "contains_match_rate": sum(stats["cm"]) / n,
            "mean_token_f1": sum(stats["f1"]) / n,
        }
        if stats["rl"]:
            per_region[region]["mean_rouge_l"] = sum(stats["rl"]) / len(stats["rl"])

    return {
        "input_file": str(input_path),
        "total_samples": len(results),
        "valid_samples": total,
        "overall_scores": overall,
        "per_region_scores": per_region,
        "details": details,
    }


def save_scores(scored_result, output_dir):
    """Save scored results to JSON and append to summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    input_basename = Path(scored_result["input_file"]).stem
    detail_path = os.path.join(output_dir, f"{input_basename}_scores.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(scored_result, f, ensure_ascii=False, indent=2)
    print(f"  Saved detailed scores: {detail_path}")

    # Append to summary CSV
    csv_path = os.path.join(output_dir, "utility_scores_summary.csv")
    write_header = not os.path.exists(csv_path)

    overall = scored_result["overall_scores"]
    has_rouge = "mean_rouge_l" in overall

    with open(csv_path, 'a', encoding='utf-8') as f:
        if write_header:
            header = "timestamp,input_file,total_samples,valid_samples,contains_match_rate,mean_token_f1"
            if has_rouge:
                header += ",mean_rouge_l"
            f.write(header + "\n")

        row = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')},"
               f"{Path(scored_result['input_file']).name},"
               f"{scored_result['total_samples']},"
               f"{scored_result['valid_samples']},"
               f"{overall.get('contains_match_rate', 0):.4f},"
               f"{overall.get('mean_token_f1', 0):.4f}")
        if has_rouge:
            row += f",{overall.get('mean_rouge_l', 0):.4f}"
        f.write(row + "\n")

    print(f"  Appended to summary: {csv_path}")

    # Save per-region CSV
    region_csv_path = os.path.join(output_dir, f"{input_basename}_per_region.csv")
    per_region = scored_result["per_region_scores"]
    with open(region_csv_path, 'w', encoding='utf-8') as f:
        header = "region,count,contains_match_rate,mean_token_f1"
        if has_rouge:
            header += ",mean_rouge_l"
        f.write(header + "\n")
        for region, stats in sorted(per_region.items()):
            row = f"{region},{stats['count']},{stats['contains_match_rate']:.4f},{stats['mean_token_f1']:.4f}"
            if has_rouge and "mean_rouge_l" in stats:
                row += f",{stats['mean_rouge_l']:.4f}"
            f.write(row + "\n")
    print(f"  Saved per-region scores: {region_csv_path}")


def print_comparison_table(all_scored):
    """Print a comparison table across all scored files."""
    print("\n" + "=" * 90)
    print("UTILITY EVALUATION COMPARISON")
    print("=" * 90)

    has_rouge = any("mean_rouge_l" in s["overall_scores"] for s in all_scored if s)

    header = f"{'Model/File':<55} {'CM%':>6} {'F1':>6}"
    if has_rouge:
        header += f" {'RL':>6}"
    header += f" {'N':>6}"
    print(header)
    print("-" * 90)

    for scored in all_scored:
        if scored is None:
            continue
        name = Path(scored["input_file"]).stem
        if len(name) > 54:
            name = name[:51] + "..."
        o = scored["overall_scores"]
        row = f"{name:<55} {o.get('contains_match_rate', 0)*100:>5.1f}% {o.get('mean_token_f1', 0):>6.3f}"
        if has_rouge:
            row += f" {o.get('mean_rouge_l', 0):>6.3f}"
        row += f" {scored['valid_samples']:>6}"
        print(row)

    print("=" * 90)
    print("CM% = Contains Match Rate, F1 = Token F1, RL = ROUGE-L, N = Valid Samples")


def main():
    parser = argparse.ArgumentParser(description="Score utility evaluation results")
    parser.add_argument("--input", nargs='+', required=True,
                        help="One or more response JSON files to score")
    parser.add_argument("--output_dir", type=str, default="utility_evaluation/results/scores",
                        help="Directory to save scoring results")
    parser.add_argument("--no_rouge", action="store_true",
                        help="Skip ROUGE-L computation")

    args = parser.parse_args()

    all_scored = []

    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"WARNING: File not found: {input_path}")
            continue

        print(f"\nScoring: {input_path}")
        scored = score_file(input_path, compute_rouge=not args.no_rouge)

        if scored is None:
            continue

        # Print quick summary
        o = scored["overall_scores"]
        print(f"  Samples: {scored['valid_samples']}/{scored['total_samples']}")
        print(f"  Contains Match: {o.get('contains_match_rate', 0)*100:.1f}%")
        print(f"  Token F1:       {o.get('mean_token_f1', 0):.3f}")
        if "mean_rouge_l" in o:
            print(f"  ROUGE-L:        {o.get('mean_rouge_l', 0):.3f}")

        save_scores(scored, args.output_dir)
        all_scored.append(scored)

    # Print comparison table if multiple files
    if len(all_scored) > 1:
        print_comparison_table(all_scored)


if __name__ == "__main__":
    main()
