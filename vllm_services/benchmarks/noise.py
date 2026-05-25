"""
Noise robustness benchmark for vishing detection via vLLM.

Evaluates classification accuracy under clean, light noise, and heavy noise conditions.
Reports: TP/FP/TN/FN, Accuracy, Precision, Recall, F1, ROC-AUC, detection times.

Adapted from experiment2_noise.py — uses async HTTP to vLLM.
Usage: python -m vllm_services.benchmarks.noise --data_path <path>
"""

import argparse
import asyncio
import copy
import csv
import math
import random
from typing import Any, Dict, List, Optional

from ..vllm_client import VLLMClient
from ..classifier import VishingClassifier, evaluate_at_threshold, find_best_thresholds, compute_roc_auc
from ..heuristics import min_max_normalize
from ..data_loader import load_raw_dataset
from ..noise_injector import apply_noise

NOISE_LEVELS = ["clean", "light", "heavy"]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def build_dataset_for_noise_level(
    dataset: List[Dict[str, Any]],
    level: str,
) -> List[Dict[str, Any]]:
    """Return a copy of the dataset with texts set for this noise level."""
    out = copy.deepcopy(dataset)
    if level == "clean":
        return out
    texts = [item["text"] for item in out]
    if level == "light":
        random.seed(42)
        noisy_texts = apply_noise(texts, level="light")
    else:
        random.seed(43)
        noisy_texts = apply_noise(texts, level="heavy")
    for item, txt in zip(out, noisy_texts):
        item["text"] = txt
    return out


async def run_one_condition(
    classifier: VishingClassifier,
    dataset: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full pipeline on one noise condition."""
    call_ids = []
    labels = []
    scores = []
    heuristic_scores = []
    detection_times_ms = []

    for idx, item in enumerate(dataset, start=1):
        result = await classifier.classify(item["text"])
        call_ids.append(item["call_id"])
        labels.append(item["label"])
        scores.append(result["score"])
        heuristic_scores.append(result["heuristic_score"])
        detection_times_ms.append(result["latency_ms"])

    scores = min_max_normalize(scores)
    metrics_at_05 = evaluate_at_threshold(labels, scores, 0.5)
    roc_auc = compute_roc_auc(labels, scores)
    threshold_info = find_best_thresholds(labels, scores)
    best_f1 = threshold_info["best_f1"] or {}
    best_threshold = best_f1.get("threshold", 0.5)

    # Cumulative metrics
    cumulative_accuracy = []
    cumulative_recall = []
    cumulative_f1 = []
    cumulative_roc_auc = []
    for i in range(1, len(labels) + 1):
        pfx_info = find_best_thresholds(labels[:i], scores[:i])
        pfx_best = pfx_info["best_f1"] or {}
        cumulative_accuracy.append(pfx_best.get("accuracy", 0.0))
        cumulative_recall.append(pfx_best.get("recall", 0.0))
        cumulative_f1.append(pfx_best.get("f1", 0.0))
        pr = compute_roc_auc(labels[:i], scores[:i])
        cumulative_roc_auc.append(pr if not math.isnan(pr) else float("nan"))

    return {
        "call_ids": call_ids,
        "labels": labels,
        "scores": scores,
        "heuristic_scores": heuristic_scores,
        "detection_times_ms": detection_times_ms,
        "cumulative_accuracy": cumulative_accuracy,
        "cumulative_recall": cumulative_recall,
        "cumulative_f1": cumulative_f1,
        "cumulative_roc_auc": cumulative_roc_auc,
        "metrics_at_05": metrics_at_05,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
    }


def write_condition_csv(output_path: str, data: Dict[str, Any]) -> None:
    """Write per-call results CSV for one condition."""
    fieldnames = [
        "call_id", "label", "score", "heuristic_score",
        "detection_time_ms", "accuracy", "recall", "f1_score", "roc_auc",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(data["call_ids"])):
            roc_val = data["cumulative_roc_auc"][i]
            writer.writerow({
                "call_id": data["call_ids"][i],
                "label": data["labels"][i],
                "score": data["scores"][i],
                "heuristic_score": data["heuristic_scores"][i],
                "detection_time_ms": data["detection_times_ms"][i],
                "accuracy": data["cumulative_accuracy"][i],
                "recall": data["cumulative_recall"][i],
                "f1_score": data["cumulative_f1"][i],
                "roc_auc": roc_val if not math.isnan(roc_val) else "",
            })


async def run_benchmark(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    data_path: str = "",
    max_samples: int = 200,
    output_base: str = "noise_results",
):
    """Run the full noise robustness benchmark."""
    client = VLLMClient(base_url=base_url, model=model)
    classifier = VishingClassifier(client)

    dataset = load_raw_dataset(data_path)
    dataset = dataset[:min(len(dataset), max_samples)]
    print(f"Loaded {len(dataset)} samples. Evaluating clean, light, heavy noise.\n")

    results_for_table = []

    for level in NOISE_LEVELS:
        print(f"--- Noise level: {level} ---")
        data_cond = build_dataset_for_noise_level(dataset, level)

        data = await run_one_condition(classifier, data_cond)

        m = data["metrics_at_05"]
        N = len(data["labels"])
        all_dt = data["detection_times_ms"]
        nv_dt = [all_dt[i] for i in range(N) if data["labels"][i] == 0]
        vs_dt = [all_dt[i] for i in range(N) if data["labels"][i] == 1]

        results_for_table.append({
            "noise_level": level,
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"],
            "roc_auc": data["roc_auc"],
            "best_threshold": data["best_threshold"],
            "detection_mean_all_ms": _mean(all_dt),
            "detection_mean_nv_ms": _mean(nv_dt),
            "detection_mean_vs_ms": _mean(vs_dt),
        })

        print(f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        print(f"  Accuracy={m['accuracy']:.4f} Precision={m['precision']:.4f} "
              f"Recall={m['recall']:.4f} F1={m['f1']:.4f}")
        roc_str = f"{data['roc_auc']:.4f}" if not math.isnan(data["roc_auc"]) else "N/A"
        print(f"  ROC-AUC={roc_str} Best_threshold={data['best_threshold']:.2f}")
        print(f"  Detection time (ms): mean={_mean(all_dt):.2f}\n")

        out_path = f"{output_base}_{level}.csv"
        write_condition_csv(out_path, data)
        print(f"  Wrote {out_path}")

    # Comparison table
    print(f"\n{'=' * 80}")
    print("Noise Robustness Comparison: Clean vs Light vs Heavy")
    print("=" * 80)
    header = (
        f"{'Noise':<8} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
        f"{'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} "
        f"{'ROC-AUC':>8} {'DetTime(ms)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results_for_table:
        roc_str = f"{r['roc_auc']:.4f}" if not math.isnan(r["roc_auc"]) else "N/A"
        print(
            f"{r['noise_level']:<8} {r['tp']:>4} {r['fp']:>4} {r['tn']:>4} {r['fn']:>4} "
            f"{r['accuracy']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
            f"{roc_str:>8} {r['detection_mean_all_ms']:>12.2f}"
        )

    await client.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness benchmark (clean/light/heavy)",
    )
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_base", type=str, default="noise_results")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            model=args.model,
            data_path=args.data_path,
            max_samples=args.max_samples,
            output_base=args.output_base,
        )
    )


if __name__ == "__main__":
    main()
