"""
Experiment #2 — Same metrics as Experiment 1 for clean, light, and heavy noise.

One run evaluates all three conditions (clean, light, heavy) and reports for each:
  TP, FP, TN, FN; Accuracy; Precision; Recall; F1; ROC-AUC; Detection times (all / non-vishing / vishing).

Output: three CSVs (base_clean.csv, base_light.csv, base_heavy.csv) and a comparison table.
"""
import argparse
import copy
import csv
import math
import random
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from experiment1_mistral import (
    load_raw_dataset,
    compute_heuristic_score,
    get_label_token_ids,
    compute_llm_probability,
    combine_scores,
    min_max_normalize,
    find_best_thresholds,
    evaluate_at_threshold,
    compute_roc_auc,
)

from noise_injector import apply_noise


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
    """Return a copy of the dataset with texts set for this noise level (clean / light / heavy)."""
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


def run_one_condition(
    dataset: List[Dict[str, Any]],
    model,
    tokenizer,
    skip_llm: bool,
    yes_token_id: int,
    no_token_id: int,
) -> Tuple[
    List[str],
    List[int],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    Dict[str, Any],
    float,
    float,
]:
    """
    Run the full pipeline on one dataset (one noise condition). Returns per-call data
    and aggregate metrics at threshold 0.5.
    """
    call_ids: List[str] = []
    labels: List[int] = []
    scores: List[float] = []
    heuristic_scores: List[float] = []
    detection_times_ms: List[float] = []
    detection_times_s: List[float] = []

    for idx, item in enumerate(dataset, start=1):
        call_id = item["call_id"]
        text = item["text"]
        label = item["label"]

        if skip_llm:
            llm_prob = 0.0
            detection_time_ms = 0.0
        else:
            llm_prob, detection_time_ms = compute_llm_probability(
            model=model,
            tokenizer=tokenizer,
            text=text,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )

        heuristic = compute_heuristic_score(text)
        combined_score = combine_scores(llm_prob, heuristic)

        call_ids.append(call_id)
        labels.append(label)
        scores.append(combined_score)
        heuristic_scores.append(heuristic)
        detection_times_ms.append(detection_time_ms)
        detection_times_s.append(detection_time_ms / 1000.0)

    scores = min_max_normalize(scores)
    metrics_at_05 = evaluate_at_threshold(labels, scores, 0.5)
    roc_auc = compute_roc_auc(labels, scores)
    threshold_info = find_best_thresholds(labels, scores)
    best_f1 = threshold_info["best_f1"] or {}
    best_threshold = best_f1.get("threshold", 0.5)

    cumulative_accuracy = []
    cumulative_recall = []
    cumulative_f1 = []
    cumulative_roc_auc = []
    for i in range(1, len(labels) + 1):
        prefix_labels = labels[:i]
        prefix_scores = scores[:i]
        prefix_thr_info = find_best_thresholds(prefix_labels, prefix_scores)
        prefix_best = prefix_thr_info["best_f1"] or {}
        cumulative_accuracy.append(prefix_best.get("accuracy", 0.0))
        cumulative_recall.append(prefix_best.get("recall", 0.0))
        cumulative_f1.append(prefix_best.get("f1", 0.0))
        pr = compute_roc_auc(prefix_labels, prefix_scores)
        cumulative_roc_auc.append(pr if not math.isnan(pr) else float("nan"))

    return (
        call_ids,
        labels,
        scores,
        heuristic_scores,
        detection_times_ms,
        detection_times_s,
        cumulative_accuracy,
        cumulative_recall,
        cumulative_f1,
        cumulative_roc_auc,
        metrics_at_05,
        roc_auc,
        best_threshold,
    )


def write_condition_csv(
    output_path: str,
    call_ids: List[str],
    labels: List[int],
    scores: List[float],
    heuristic_scores: List[float],
    detection_times_ms: List[float],
    detection_times_s: List[float],
    cumulative_accuracy: List[float],
    cumulative_recall: List[float],
    cumulative_f1: List[float],
    cumulative_roc_auc: List[float],
) -> None:
    fieldnames = [
        "call_id",
        "label",
        "score",
        "heuristic_score",
        "detection_time_ms",
        "detection_time_s",
        "accuracy",
        "recall",
        "f1_score",
        "roc_auc",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(call_ids)):
            roc_val = cumulative_roc_auc[i]
            writer.writerow(
                {
                    "call_id": call_ids[i],
                    "label": labels[i],
                    "score": scores[i],
                    "heuristic_score": heuristic_scores[i],
                    "detection_time_ms": detection_times_ms[i],
                    "detection_time_s": detection_times_s[i],
                    "accuracy": cumulative_accuracy[i],
                    "recall": cumulative_recall[i],
                    "f1_score": cumulative_f1[i],
                    "roc_auc": (
                        cumulative_roc_auc[i]
                        if not math.isnan(roc_val)
                        else ""
                    ),
                }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment #2 — Clean, light, and heavy noise (same metrics as Experiment 1).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSON dataset file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiment2_results.csv",
        help="Base path for output CSVs; will write _clean.csv, _light.csv, _heavy.csv.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mistral", "llama"],
        required=True,
        help="Which model to run: mistral or llama.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override Hugging Face model name.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="If set, use heuristic-only scoring (no LLM load).",
    )

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = (
            "mistralai/Mistral-7B-Instruct-v0.2"
            if args.model_type == "mistral"
            else "meta-llama/Llama-3.2-3B-Instruct"
        )

    # Base path for CSVs: strip .csv and add _clean, _light, _heavy
    base = args.output_csv.rsplit(".csv", 1)[0] if args.output_csv.endswith(".csv") else args.output_csv

    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_raw_dataset(args.data_path)
    if not dataset:
        raise ValueError("Dataset is empty.")
    dataset = dataset[: min(len(dataset), args.max_samples)]
    print(f"Loaded {len(dataset)} samples. Will evaluate clean, light, and heavy noise.")

    if args.skip_llm:
        print("Skipping LLM loading and using heuristic-only scoring.")
        tokenizer = None
        model = None
        yes_token_id = no_token_id = -1
    else:
        print(f"Loading model {args.model_name} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if device == "cuda" and BitsAndBytesConfig is not None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map=None,
            )
            model.to(device)
        elif device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float32,
            )
            model.to(device)
        model.eval()
        yes_token_id, no_token_id = get_label_token_ids(tokenizer)

    results_for_table: List[Dict[str, Any]] = []

    for level in NOISE_LEVELS:
        print(f"\n--- Noise level: {level} ---")
        data_cond = build_dataset_for_noise_level(dataset, level)
        if level != "clean":
            orig = dataset[0]["text"][:100] + "…" if len(dataset[0]["text"]) > 100 else dataset[0]["text"]
            nois = data_cond[0]["text"][:100] + "…" if len(data_cond[0]["text"]) > 100 else data_cond[0]["text"]
            print(f"  Example first call:\n    original: {orig}\n    noised:   {nois}")

        (
            call_ids,
            labels,
            scores,
            heuristic_scores,
            detection_times_ms,
            detection_times_s,
            cumulative_accuracy,
            cumulative_recall,
            cumulative_f1,
            cumulative_roc_auc,
            metrics_at_05,
            roc_auc,
            best_threshold,
        ) = run_one_condition(
            data_cond,
            model,
            tokenizer,
            args.skip_llm,
            yes_token_id,
            no_token_id,
        )

        N = len(labels)
        tp = metrics_at_05["tp"]
        fp = metrics_at_05["fp"]
        tn = metrics_at_05["tn"]
        fn = metrics_at_05["fn"]
        accuracy = metrics_at_05["accuracy"]
        precision = metrics_at_05["precision"]
        recall = metrics_at_05["recall"]
        f1 = metrics_at_05["f1"]

        all_dt = detection_times_ms
        nv_dt = [all_dt[i] for i in range(N) if labels[i] == 0]
        vs_dt = [all_dt[i] for i in range(N) if labels[i] == 1]

        results_for_table.append({
            "noise_level": level,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "best_threshold": best_threshold,
            "detection_mean_all_ms": _mean(all_dt),
            "detection_median_all_ms": _median(all_dt),
            "detection_mean_nv_ms": _mean(nv_dt),
            "detection_mean_vs_ms": _mean(vs_dt),
        })

        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
        roc_str = f"{roc_auc:.4f}" if not math.isnan(roc_auc) else "N/A"
        print(f"  ROC-AUC={roc_str} Best_threshold={best_threshold:.2f}")
        print(f"  Detection time (ms): all mean={_mean(all_dt):.2f} median={_median(all_dt):.2f}")
        print(f"    Non-vishing mean={_mean(nv_dt):.2f}  Vishing mean={_mean(vs_dt):.2f}")

        out_path = f"{base}_{level}.csv"
        write_condition_csv(
            out_path,
            call_ids,
            labels,
            scores,
            heuristic_scores,
            detection_times_ms,
            detection_times_s,
            cumulative_accuracy,
            cumulative_recall,
            cumulative_f1,
            cumulative_roc_auc,
        )
        print(f"  Wrote {out_path}")

    # --- Comparison table ---
    print("\n" + "=" * 80)
    print("Experiment #2 — Comparison: Clean vs Light vs Heavy noise")
    print("=" * 80)
    print(f"Model: {args.model_type} ({args.model_name})  |  Samples: {len(dataset)}")
    print()
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
    print()
    print("Detection times (ms) — mean by condition:")
    print(f"  {'Noise':<8} {'All calls':>12} {'Non-vishing':>14} {'Vishing':>12}")
    print("  " + "-" * 50)
    for r in results_for_table:
        print(
            f"  {r['noise_level']:<8} {r['detection_mean_all_ms']:>12.2f} "
            f"{r['detection_mean_nv_ms']:>14.2f} {r['detection_mean_vs_ms']:>12.2f}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
