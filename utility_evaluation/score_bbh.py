#!/usr/bin/env python3
"""
Score BBH (Big-Bench Hard) evaluation results following VoiceBench methodology.

Uses task-specific answer extraction with pattern matching to extract binary
answers from model responses, then computes exact match accuracy.

4 task types (250 samples each):
  - navigate: "Do you return to the starting point?" -> Yes/No
  - sports_understanding: "Is the sentence plausible?" -> Yes/No
  - hyperbaton: "Which is the correct adjective order?" -> (A)/(B)
  - web_of_lies: "Does X tell the truth?" -> Yes/No

Usage:
    python score_bbh.py --input results/kimi/kimi_pretrained_bbh_responses.json
    python score_bbh.py --input results/kimi/*.json results/af3/*.json --output_dir results/scores
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


# Ground truth mapping (VoiceBench convention)
GROUND_TRUTH_MAPPING = {
    "(a)": 0, "(b)": 1,
    "yes": 1, "no": 0,
}


def normalize_response(response):
    """Normalize model response for answer extraction (VoiceBench-style)."""
    response = response.lower().strip()
    # Remove common formatting tokens
    for token in ["<|user|>", "<|assistant|>", "<|end|>", "<1>", "<2>", "</s>"]:
        response = response.replace(token, "")
    # Replace special chars
    for ch in [":", "*", '"', "'", "-", ",", "."]:
        response = response.replace(ch, " ")
    response = " ".join(response.split())
    return response


def extract_answer_yn(response):
    """Extract yes/no answer from response."""
    response = response.strip()

    # Check for boxed format
    boxed = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        inner = boxed.group(1).lower().strip()
        if "yes" in inner:
            return 1
        if "no" in inner:
            return 0

    # Check for "the answer is" pattern
    match = re.search(r'the answer is\s*(yes|no)', response)
    if match:
        return 1 if match.group(1) == "yes" else 0

    # Direct yes/no at start or end
    if response.startswith("yes") or response.endswith("yes"):
        return 1
    if response.startswith("no") or response.endswith("no"):
        return 0

    # "true"/"false" mapping
    if "true" in response and "false" not in response:
        return 1
    if "false" in response and "true" not in response:
        return 0

    return None


def extract_answer_navigate(response):
    """Extract navigate task answer (do you return to starting point?)."""
    response = normalize_response(response)

    # Try yes/no first
    yn = extract_answer_yn(response)
    if yn is not None:
        return yn

    # Positive patterns (return to start)
    positive_patterns = [
        "you return to the starting point",
        "you do return to the starting point",
        "you will return to the starting point",
        "you would return to the starting point",
        "you end up at the starting point",
        "you are back at the starting point",
        "you will be back at the starting point",
        "back to the starting point",
        "back at the starting point",
        "return to the start",
        "back where you started",
        "end up where you started",
        "return to your original position",
        "back to your original position",
        "returned to the starting point",
        "returns to the starting point",
    ]

    # Negative patterns (don't return)
    negative_patterns = [
        "you do not return to the starting point",
        "you don t return to the starting point",
        "you will not return to the starting point",
        "you would not return to the starting point",
        "you won t return to the starting point",
        "you don t end up at the starting point",
        "not return to the starting point",
        "not back at the starting point",
        "not back to the starting point",
        "not end up at the starting point",
        "not return to the start",
        "not back where you started",
        "do not return to your original",
        "will not return to your original",
    ]

    # Check negative first (longer matches take priority over partial positive matches)
    for pat in negative_patterns:
        if pat in response:
            return 0

    for pat in positive_patterns:
        if pat in response:
            return 1

    # Fallback: random
    return random.randint(0, 1)


def extract_answer_sports(response):
    """Extract sports understanding answer (is the sentence plausible?)."""
    response = normalize_response(response)

    # Try yes/no first
    yn = extract_answer_yn(response)
    if yn is not None:
        return yn

    # Negative patterns (not plausible)
    negative_patterns = [
        "not plausible", "is implausible", "implausible",
        "the sentence is not plausible",
        "is not a plausible", "not a plausible",
        "making it implausible", "making this implausible",
        "this is implausible", "that is implausible",
        "sentence is implausible",
    ]

    # Positive patterns (plausible)
    positive_patterns = [
        "is plausible", "the sentence is plausible",
        "this is plausible", "that is plausible",
        "sentence is plausible", "it is plausible",
        "making it plausible", "making this plausible",
    ]

    # Check negative first
    for pat in negative_patterns:
        if pat in response:
            return 0

    for pat in positive_patterns:
        if pat in response:
            return 1

    # Fallback: random
    return random.randint(0, 1)


def extract_answer_hyperbaton(response):
    """Extract hyperbaton answer (correct adjective order: (A) or (B)?)."""
    response = normalize_response(response)

    # Check for "the answer is (a/b)" pattern
    match = re.search(r'the answer is\s*\(?\s*(a|b)\s*\)?', response)
    if match:
        return 0 if match.group(1) == "a" else 1

    # Check for "option (a/b)" pattern
    match = re.search(r'option\s*\(?\s*(a|b)\s*\)?', response)
    if match:
        return 0 if match.group(1) == "a" else 1

    # Check for "correct.*is (a/b)" or "(a/b) is correct"
    match = re.search(r'correct.*\(?\s*(a|b)\s*\)?', response)
    if match:
        return 0 if match.group(1) == "a" else 1
    match = re.search(r'\(?\s*(a|b)\s*\)?\s*is\s*correct', response)
    if match:
        return 0 if match.group(1) == "a" else 1

    # Direct "(a)" or "(b)" at end
    if response.strip().endswith("(a)") or response.strip().endswith("a)"):
        return 0
    if response.strip().endswith("(b)") or response.strip().endswith("b)"):
        return 1

    # Fallback: random
    return random.randint(0, 1)


def extract_answer_lies(response):
    """Extract web of lies answer (does the person tell the truth?)."""
    response = normalize_response(response)

    # Try yes/no first
    yn = extract_answer_yn(response)
    if yn is not None:
        return yn

    # Positive patterns (tells the truth)
    positive_patterns = [
        "tells the truth", "is telling the truth", "is truthful",
        "is honest", "speaks the truth", "said the truth",
    ]

    # Negative patterns (is lying)
    negative_patterns = [
        "is lying", "tells a lie", "is a liar", "does not tell the truth",
        "doesn t tell the truth", "does lie", "is not telling the truth",
        "is not truthful", "is dishonest", "lied",
    ]

    for pat in negative_patterns:
        if pat in response:
            return 0
    for pat in positive_patterns:
        if pat in response:
            return 1

    # Fallback: random
    return random.randint(0, 1)


def get_task_type(sample_id, question=""):
    """Determine BBH task type from sample ID or question content."""
    sid = sample_id.lower() if sample_id else ""
    q = question.lower() if question else ""

    if "navigate" in sid or "return to the starting point" in q:
        return "navigate"
    elif "sports" in sid or "plausible" in q:
        return "sports_understanding"
    elif "hyperbaton" in sid or "adjective order" in q or "option (a)" in q.lower() or "Option (A)" in question:
        return "hyperbaton"
    elif "lies" in sid or "web_of_lies" in sid:
        return "web_of_lies"

    # Infer from question content
    if "return to the starting point" in q:
        return "navigate"
    if "plausible" in q:
        return "sports_understanding"
    if "(a)" in q and "(b)" in q:
        return "hyperbaton"
    if "tells the truth" in q or "lies" in q.split(".")[0]:
        return "web_of_lies"

    return "unknown"


def extract_answer(response, task_type):
    """Route to task-specific answer extractor."""
    if task_type == "navigate":
        return extract_answer_navigate(response)
    elif task_type == "sports_understanding":
        return extract_answer_sports(response)
    elif task_type == "hyperbaton":
        return extract_answer_hyperbaton(response)
    elif task_type == "web_of_lies":
        return extract_answer_lies(response)
    else:
        # Generic yes/no extraction
        yn = extract_answer_yn(normalize_response(response))
        return yn if yn is not None else random.randint(0, 1)


def score_bbh_file(input_path):
    """Score a BBH response JSON file.

    Returns dict with overall accuracy, per-task accuracy, and details.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print(f"  WARNING: Empty results file: {input_path}")
        return None

    # Set random seed for reproducible fallback
    random.seed(42)

    details = []
    task_stats = {}

    for entry in results:
        response = entry.get("response", "") or ""
        ground_truth = entry.get("ground_truth", "") or ""
        sample_id = entry.get("id", "")
        question = entry.get("question", "")

        # Determine task type
        task_type = get_task_type(sample_id, question)

        # Map ground truth to binary
        gt_mapped = GROUND_TRUTH_MAPPING.get(ground_truth.lower().strip())
        if gt_mapped is None:
            # Try direct yes/no
            gt_lower = ground_truth.lower().strip()
            if gt_lower in ("yes", "true"):
                gt_mapped = 1
            elif gt_lower in ("no", "false"):
                gt_mapped = 0
            else:
                gt_mapped = -1  # unknown

        # Skip errored samples
        if entry.get("error"):
            detail = {
                "id": sample_id,
                "question": question[:100],
                "ground_truth": ground_truth,
                "gt_mapped": gt_mapped,
                "response": response[:200],
                "predicted": -1,
                "correct": False,
                "task_type": task_type,
                "error": entry["error"],
            }
            details.append(detail)
            continue

        # Extract answer
        predicted = extract_answer(response, task_type)
        correct = (predicted == gt_mapped) if gt_mapped != -1 else False

        detail = {
            "id": sample_id,
            "question": question[:100],
            "ground_truth": ground_truth,
            "gt_mapped": gt_mapped,
            "response": response[:200],
            "predicted": predicted,
            "correct": correct,
            "task_type": task_type,
            "error": None,
        }
        details.append(detail)

        # Accumulate per-task stats
        if task_type not in task_stats:
            task_stats[task_type] = {"correct": 0, "total": 0}
        task_stats[task_type]["total"] += 1
        if correct:
            task_stats[task_type]["correct"] += 1

    # Compute overall accuracy
    valid_details = [d for d in details if d["error"] is None and d["gt_mapped"] != -1]
    total = len(valid_details)
    correct_count = sum(d["correct"] for d in valid_details)

    overall = {
        "accuracy": (correct_count / total * 100) if total > 0 else 0.0,
        "correct": correct_count,
        "total": total,
    }

    # Per-task accuracy
    per_task = {}
    for task, stats in sorted(task_stats.items()):
        per_task[task] = {
            "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "input_file": str(input_path),
        "total_samples": len(results),
        "valid_samples": total,
        "overall_scores": overall,
        "per_task_scores": per_task,
        "details": details,
    }


def save_bbh_scores(scored_result, output_dir):
    """Save BBH scored results to JSON and append to summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    input_basename = Path(scored_result["input_file"]).stem
    detail_path = os.path.join(output_dir, f"{input_basename}_bbh_scores.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(scored_result, f, ensure_ascii=False, indent=2)
    print(f"  Saved detailed scores: {detail_path}")

    # Append to summary CSV
    csv_path = os.path.join(output_dir, "bbh_scores_summary.csv")
    write_header = not os.path.exists(csv_path)

    overall = scored_result["overall_scores"]
    per_task = scored_result["per_task_scores"]

    with open(csv_path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write("timestamp,input_file,total,accuracy,navigate_acc,sports_acc,hyperbaton_acc,web_of_lies_acc\n")

        row = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')},"
               f"{Path(scored_result['input_file']).name},"
               f"{overall['total']},"
               f"{overall['accuracy']:.2f}")
        for task in ["navigate", "sports_understanding", "hyperbaton", "web_of_lies"]:
            acc = per_task.get(task, {}).get("accuracy", 0.0)
            row += f",{acc:.2f}"
        f.write(row + "\n")

    print(f"  Appended to summary: {csv_path}")

    # Per-task CSV
    task_csv_path = os.path.join(output_dir, f"{input_basename}_bbh_per_task.csv")
    with open(task_csv_path, 'w', encoding='utf-8') as f:
        f.write("task,correct,total,accuracy\n")
        for task, stats in sorted(per_task.items()):
            f.write(f"{task},{stats['correct']},{stats['total']},{stats['accuracy']:.2f}\n")
    print(f"  Saved per-task scores: {task_csv_path}")


def print_bbh_comparison(all_scored):
    """Print comparison table for BBH results."""
    print("\n" + "=" * 100)
    print("BBH UTILITY EVALUATION COMPARISON")
    print("=" * 100)

    header = f"{'Model/File':<45} {'Overall':>8} {'Navigate':>10} {'Sports':>10} {'Hyperbaton':>11} {'WebOfLies':>10} {'N':>5}"
    print(header)
    print("-" * 100)

    for scored in all_scored:
        if scored is None:
            continue
        name = Path(scored["input_file"]).stem
        if len(name) > 44:
            name = name[:41] + "..."
        o = scored["overall_scores"]
        pt = scored["per_task_scores"]
        row = (f"{name:<45} "
               f"{o['accuracy']:>7.2f}% "
               f"{pt.get('navigate', {}).get('accuracy', 0):>9.2f}% "
               f"{pt.get('sports_understanding', {}).get('accuracy', 0):>9.2f}% "
               f"{pt.get('hyperbaton', {}).get('accuracy', 0):>10.2f}% "
               f"{pt.get('web_of_lies', {}).get('accuracy', 0):>9.2f}% "
               f"{o['total']:>5}")
        print(row)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Score BBH evaluation results (VoiceBench-style)")
    parser.add_argument("--input", nargs='+', required=True,
                        help="One or more BBH response JSON files to score")
    parser.add_argument("--output_dir", type=str, default="utility_evaluation/results/scores",
                        help="Directory to save scoring results")

    args = parser.parse_args()

    all_scored = []

    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"WARNING: File not found: {input_path}")
            continue

        print(f"\nScoring BBH: {input_path}")
        scored = score_bbh_file(input_path)

        if scored is None:
            continue

        o = scored["overall_scores"]
        print(f"  Samples: {scored['valid_samples']}/{scored['total_samples']}")
        print(f"  Overall Accuracy: {o['accuracy']:.2f}% ({o['correct']}/{o['total']})")
        for task, stats in sorted(scored["per_task_scores"].items()):
            print(f"    {task}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

        save_bbh_scores(scored, args.output_dir)
        all_scored.append(scored)

    if len(all_scored) > 1:
        print_bbh_comparison(all_scored)


if __name__ == "__main__":
    main()
