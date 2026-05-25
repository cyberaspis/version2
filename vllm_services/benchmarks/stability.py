"""
Probability stability benchmark for vLLM-based vishing detection.

Tests how consistently the model returns the same probability for the
same input over multiple runs. Reports variance, stability index,
and per-conversation statistics.

Adapted from probability_stability_test.py — uses async HTTP to vLLM.
Usage: python -m vllm_services.benchmarks.stability --data_path <path>
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import Any, Dict, List, Optional

from ..vllm_client import VLLMClient
from ..prompts import build_vishing_probability_prompt
from ..data_loader import load_raw_dataset


async def run_benchmark(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    data_path: str = "",
    n_runs: int = 5,
    max_samples: int = 200,
    output_json: str = "stability_results.json",
):
    """Run the stability benchmark."""
    client = VLLMClient(base_url=base_url, model=model)

    dataset = load_raw_dataset(data_path)
    dataset = dataset[:min(len(dataset), max_samples)]
    print(f"Loaded {len(dataset)} samples. Running {n_runs} passes per sample.\n")

    per_conv_probabilities: List[List[float]] = []
    all_times_ms: List[float] = []

    total_start = time.perf_counter()

    for idx, item in enumerate(dataset, start=1):
        text = item["text"]
        prompt = build_vishing_probability_prompt(text)
        probs: List[float] = []

        for run in range(n_runs):
            prob, t_ms = await client.get_vishing_probability(prompt)
            probs.append(prob)
            all_times_ms.append(t_ms)

        per_conv_probabilities.append(probs)
        print(f"  [{idx}/{len(dataset)}] runs done | "
              f"mean={statistics.mean(probs):.4f} std={statistics.stdev(probs) if len(probs) > 1 else 0:.4f}")

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
        var = s * s
        means.append(m)
        stds.append(s)
        cvs.append(cv)
        variances.append(var)

    avg_variance = statistics.mean(variances) if variances else 0.0
    mean_cv = statistics.mean(cvs) if cvs else 0.0
    stability_index = 1.0 / (1.0 + mean_cv) if mean_cv >= 0 else 0.0
    avg_inference_ms = statistics.mean(all_times_ms) if all_times_ms else 0.0

    results: Dict[str, Any] = {
        "model": model or "default",
        "data_path": data_path,
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

    print(f"\n{'=' * 44}")
    print("Stability Test Results")
    print("=" * 44)
    print(f"Conversations:              {results['n_conversations']}")
    print(f"Runs per conversation:      {results['n_runs_per_conversation']}")
    print(f"Avg probability variance:   {results['average_probability_variance']}")
    print(f"Stability index:            {results['stability_index']}")
    print(f"Avg inference time (ms):    {results['average_inference_time_ms']}")
    print(f"Total runtime (s):          {results['total_runtime_seconds']}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Probability stability benchmark",
    )
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_json", type=str, default="stability_results.json")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            model=args.model,
            data_path=args.data_path,
            n_runs=args.n_runs,
            max_samples=args.max_samples,
            output_json=args.output_json,
        )
    )


if __name__ == "__main__":
    main()
