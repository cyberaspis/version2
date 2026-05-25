import argparse
import json
import statistics
import time
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_utils import build_prompt, parse_probability


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load vishing dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects")
    return [item for item in data if isinstance(item, dict) and item.get("text")]


def run_inference(model, tokenizer, text: str, max_new_tokens: int = 40) -> tuple:
    """Run inference, return (probability, inference_time_ms)."""
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    end = time.perf_counter()

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if full_output.startswith(prompt):
        completion = full_output[len(prompt) :].strip()
    else:
        completion = full_output.strip()

    prob = parse_probability(completion)
    return prob, (end - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction stability of LLM for vishing detection.",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_path} ...")
    dataset = load_dataset(args.data_path)
    if not dataset:
        raise ValueError("Dataset is empty.")

    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    n_runs = args.n_runs
    per_conv_probabilities: List[List[float]] = []
    all_times_ms: List[float] = []

    total_start = time.perf_counter()

    for idx, item in enumerate(dataset, start=1):
        text = str(item.get("text", "")).strip()
        probs: List[float] = []
        for run in range(n_runs):
            prob, t_ms = run_inference(model, tokenizer, text)
            probs.append(prob)
            all_times_ms.append(t_ms)
        per_conv_probabilities.append(probs)
        print(f"  [{idx}/{len(dataset)}] runs done")

    total_end = time.perf_counter()
    total_runtime_s = total_end - total_start

    # Per-conversation: mean, std, coefficient of variation
    means: List[float] = []
    stds: List[float] = []
    cvs: List[float] = []
    variances: List[float] = []

    for probs in per_conv_probabilities:
        m = statistics.mean(probs)
        s = statistics.stdev(probs) if len(probs) > 1 else 0.0
        cv = s / m if m != 0 else 0.0
        var = s * s if len(probs) > 1 else 0.0
        means.append(m)
        stds.append(s)
        cvs.append(cv)
        variances.append(var)

    avg_variance = statistics.mean(variances) if variances else 0.0
    mean_cv = statistics.mean(cvs) if cvs else 0.0
    stability_index = 1.0 / (1.0 + mean_cv) if mean_cv >= 0 else 0.0

    avg_inference_ms = statistics.mean(all_times_ms) if all_times_ms else 0.0

    results = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "n_conversations": len(dataset),
        "n_runs_per_conversation": n_runs,
        "average_probability_variance": round(avg_variance, 6),
        "stability_index": round(stability_index, 4),
        "average_inference_time_ms": round(avg_inference_ms, 2),
        "total_runtime_seconds": round(total_runtime_s, 2),
        "per_conversation_mean": [round(x, 4) for x in means],
        "per_conversation_std": [round(x, 4) for x in stds],
        "per_conversation_cv": [round(x, 4) for x in cvs],
    }

    print("\n=== Stability Test Results ===")
    print(f"Conversations: {results['n_conversations']}")
    print(f"Runs per conversation: {results['n_runs_per_conversation']}")
    print(f"Average probability variance: {results['average_probability_variance']}")
    print(f"Stability index: {results['stability_index']}")
    print(f"Average inference time (ms): {results['average_inference_time_ms']}")
    print(f"Total runtime (s): {results['total_runtime_seconds']}")

    out_path = "stability_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
