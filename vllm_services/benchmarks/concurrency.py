"""
Concurrency benchmark for vLLM-based vishing detection.

Measures per concurrency level:
  - Response time: avg latency (ms), total wall-clock time (ms), throughput (req/s)
  - Memory: RAM (MB) — sampled during run
  - CPU: percent usage (avg, max)

Adapted from experiment3_concurrency.py — uses asyncio for true parallel HTTP requests.
Usage: python -m vllm_services.benchmarks.concurrency --data_path <path> --concurrency 1,5,10,20
"""

import argparse
import asyncio
import csv
import logging
import threading
import time
from typing import List, Optional

import psutil

from ..vllm_client import VLLMClient
from ..prompts import build_vishing_probability_prompt
from ..data_loader import load_raw_dataset

logger = logging.getLogger(__name__)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _sample_cpu_ram_worker(
    process: psutil.Process,
    stop_event: threading.Event,
    interval: float,
    cpu_samples: List[float],
    ram_samples_mb: List[float],
) -> None:
    """Background thread: sample CPU % and RAM (MB) until stop_event is set."""
    while not stop_event.wait(interval):
        try:
            cpu_samples.append(process.cpu_percent(interval=0.1))
            ram_samples_mb.append(float(process.memory_info().rss / (1024**2)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break


async def _single_request(
    client: VLLMClient,
    text: str,
) -> float:
    """Send a single classification request and return latency in ms."""
    prompt = build_vishing_probability_prompt(text)
    _, latency_ms = await client.get_vishing_probability(prompt)
    return latency_ms


async def benchmark_concurrency_level(
    client: VLLMClient,
    texts: List[str],
    concurrency: int,
    sample_interval: float = 0.3,
) -> dict:
    """
    Benchmark a given concurrency level using asyncio.gather for true parallelism.
    """
    if not texts:
        raise ValueError("No texts available for benchmarking.")

    batch_texts = [texts[i % len(texts)] for i in range(concurrency)]
    process = psutil.Process()

    cpu_samples: List[float] = []
    ram_samples_mb: List[float] = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_sample_cpu_ram_worker,
        args=(process, stop_event, sample_interval, cpu_samples, ram_samples_mb),
        daemon=True,
    )
    monitor.start()

    t0 = time.perf_counter()

    # True parallel HTTP requests via asyncio
    tasks = [_single_request(client, text) for text in batch_texts]
    latencies_ms = await asyncio.gather(*tasks)

    t1 = time.perf_counter()
    stop_event.set()
    monitor.join(timeout=sample_interval * 2)
    total_time_ms = (t1 - t0) * 1000.0

    ram_mb_end = float(process.memory_info().rss / (1024**2))
    avg_latency_ms = _mean(list(latencies_ms))
    avg_cpu = _mean(cpu_samples)
    max_cpu = max(cpu_samples) if cpu_samples else 0.0
    max_ram_mb = max(ram_samples_mb) if ram_samples_mb else ram_mb_end
    throughput = concurrency / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0

    return {
        "concurrency": concurrency,
        "avg_latency_ms": avg_latency_ms,
        "total_time_ms": total_time_ms,
        "throughput_rps": throughput,
        "ram_used_mb": ram_mb_end,
        "ram_max_mb": max_ram_mb,
        "cpu_avg_percent": avg_cpu,
        "cpu_max_percent": max_cpu,
    }


async def run_benchmark(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    data_path: Optional[str] = None,
    concurrency_levels: Optional[List[int]] = None,
    output_csv: Optional[str] = None,
):
    """Run the full concurrency benchmark."""
    client = VLLMClient(base_url=base_url, model=model)

    if data_path:
        dataset = load_raw_dataset(data_path)
        texts = [item["text"] for item in dataset]
    else:
        texts = [f"Sample transcript {i}" for i in range(40)]

    if not concurrency_levels:
        concurrency_levels = [1, 3, 5, 10, 15, 20]

    print(f"Loaded {len(texts)} texts. Testing concurrency levels: {concurrency_levels}")
    print(f"vLLM server: {client.base_url} | Model: {client.model}\n")

    all_results = []
    for level in concurrency_levels:
        print(f"Benchmarking concurrency F = {level} ...")
        metrics = await benchmark_concurrency_level(client, texts, level)
        all_results.append(metrics)
        print(
            f"  Avg latency: {metrics['avg_latency_ms']:.0f}ms | "
            f"Total: {metrics['total_time_ms']:.0f}ms | "
            f"Throughput: {metrics['throughput_rps']:.2f} req/s"
        )

    # Results table
    header = (
        f"{'F':>4} {'AvgLatency(ms)':>14} {'TotalTime(ms)':>14} "
        f"{'Throughput':>10} {'RAM_MB':>10} {'RAM_max_MB':>12} "
        f"{'CPU_avg_%':>10} {'CPU_max_%':>10}"
    )
    print(f"\n{'=' * 80}")
    print("Concurrency Benchmark Results")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['concurrency']:>4d} {r['avg_latency_ms']:>14.2f} {r['total_time_ms']:>14.2f} "
            f"{r['throughput_rps']:>10.2f} {r['ram_used_mb']:>10.2f} {r['ram_max_mb']:>12.2f} "
            f"{r['cpu_avg_percent']:>10.2f} {r['cpu_max_percent']:>10.2f}"
        )

    if output_csv:
        fieldnames = list(all_results[0].keys())
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nWrote {output_csv}")

    await client.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Concurrency benchmark (async HTTP to vLLM)",
    )
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--concurrency", type=str, default="1,3,5,10,15,20",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            model=args.model,
            data_path=args.data_path,
            concurrency_levels=levels,
            output_csv=args.output_csv,
        )
    )


if __name__ == "__main__":
    main()
