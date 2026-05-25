"""
Experiment #3 — Concurrency benchmark on vishing data.

Measures per concurrency level F:
  (i)   Response time: avg latency (ms) per request, total wall-clock time (ms).
  (ii)  Memory: GPU memory (MB), RAM (MB) — sampled at end and max during run.
  (iii) CPU: percent usage — sampled during concurrent run (avg and max).
  (iv)  F: number of concurrent requests (concurrency level).
"""
import argparse
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from experiment_utils import build_prompt

# Optional: load vishing dataset (list of {call_id, text, label})
try:
    from experiment1_mistral import load_raw_dataset as load_raw_dataset_mistral
except ImportError:
    load_raw_dataset_mistral = None


def run_single_inference(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 32,
) -> float:
    """Run a single generate() call and return latency in milliseconds."""
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    end = time.perf_counter()
    return (end - start) * 1000.0


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
            ram_samples_mb.append(float(process.memory_info().rss / (1024 ** 2)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break


def benchmark_concurrency_level(
    model,
    tokenizer,
    texts: List[str],
    concurrency: int,
    max_new_tokens: int = 32,
    sample_interval: float = 0.3,
) -> dict:
    """
    Benchmark a given concurrency level F using ThreadPoolExecutor.
    Samples CPU and RAM during the run. Returns dict with F, latency, memory, CPU.
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
    latencies_ms: List[float] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_single_inference,
                model,
                tokenizer,
                text,
                max_new_tokens,
            )
            for text in batch_texts
        ]
        for future in as_completed(futures):
            lat = future.result()
            latencies_ms.append(lat)

    t1 = time.perf_counter()
    stop_event.set()
    monitor.join(timeout=sample_interval * 2)
    total_time_ms = (t1 - t0) * 1000.0

    # GPU memory (current at end of run)
    if torch.cuda.is_available():
        gpu_mem_mb = float(torch.cuda.memory_allocated() / (1024 ** 2))
    else:
        gpu_mem_mb = 0.0

    ram_mb_end = float(process.memory_info().rss / (1024 ** 2))
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    max_cpu = max(cpu_samples) if cpu_samples else 0.0
    max_ram_mb = max(ram_samples_mb) if ram_samples_mb else ram_mb_end

    return {
        "concurrency": concurrency,
        "avg_latency_ms": avg_latency_ms,
        "total_time_ms": total_time_ms,
        "gpu_memory_mb": gpu_mem_mb,
        "ram_used_mb": ram_mb_end,
        "ram_max_mb": max_ram_mb,
        "cpu_avg_percent": avg_cpu,
        "cpu_max_percent": max_cpu,
    }


def load_texts_from_data_path(data_path: str) -> List[str]:
    """Load transcript texts from vishing/non-vishing JSON (list of {call_id, text, label})."""
    if load_raw_dataset_mistral is None:
        raise RuntimeError("experiment1_mistral.load_raw_dataset not available.")
    dataset = load_raw_dataset_mistral(data_path)
    return [item["text"] for item in dataset]


def load_texts(path: Optional[str], fallback_n: int = 40) -> List[str]:
    """
    Load texts from JSON: either vishing-style (--data_path) or list of strings/objects with 'text'.
    """
    if path is None:
        return [f"Sample transcript {i}" for i in range(fallback_n)]

    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
    if not texts:
        texts = [f"Sample transcript {i}" for i in range(fallback_n)]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment #3 — Concurrency benchmark (response time, memory, CPU).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name (e.g. mistralai/Mistral-7B-Instruct-v0.2 or meta-llama/Llama-3.2-3B-Instruct).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to vishing/non-vishing JSON dataset (list of {call_id, text, label}).",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default=None,
        help="Alternative: path to JSON list of strings or objects with 'text'.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens per generate() call.",
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,3,5,10,15,20",
        help="Comma-separated concurrency levels F (e.g. 1,3,5,10,15,20).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional: path to write results CSV.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization on GPU to reduce VRAM (recommended for 7B/8B).",
    )
    args = parser.parse_args()

    if args.data_path:
        print(f"Loading vishing-style dataset from {args.data_path} ...")
        texts = load_texts_from_data_path(args.data_path)
    else:
        print(f"Loading texts from {args.texts_path or 'synthetic'} ...")
        texts = load_texts(args.texts_path)
    print(f"Loaded {len(texts)} texts.")

    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    if not concurrency_levels:
        concurrency_levels = [1, 3, 5, 10, 15, 20]

    print(f"Loading model {args.model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if device == "cuda" and args.use_4bit and BitsAndBytesConfig is not None:
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

    all_results = []
    for level in concurrency_levels:
        print(f"\nBenchmarking concurrency F = {level} ...")
        metrics = benchmark_concurrency_level(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            concurrency=level,
            max_new_tokens=args.max_new_tokens,
        )
        all_results.append(metrics)

    # Table: F, response time, memory, CPU
    header = (
        f"{'F':>4} {'AvgLatency(ms)':>14} {'TotalTime(ms)':>14} "
        f"{'GPU_MB':>10} {'RAM_MB':>10} {'RAM_max_MB':>12} "
        f"{'CPU_avg_%':>10} {'CPU_max_%':>10}"
    )
    print("\n=== Experiment #3 — Concurrency Results ===")
    print("(i) Response time: AvgLatency(ms), TotalTime(ms)  (ii) Memory: GPU_MB, RAM_MB  (iii) CPU: avg %, max %  (iv) F = concurrency)")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['concurrency']:>4d} {r['avg_latency_ms']:>14.2f} {r['total_time_ms']:>14.2f} "
            f"{r['gpu_memory_mb']:>10.2f} {r['ram_used_mb']:>10.2f} {r['ram_max_mb']:>12.2f} "
            f"{r['cpu_avg_percent']:>10.2f} {r['cpu_max_percent']:>10.2f}"
        )

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "concurrency",
                    "avg_latency_ms",
                    "total_time_ms",
                    "gpu_memory_mb",
                    "ram_used_mb",
                    "ram_max_mb",
                    "cpu_avg_percent",
                    "cpu_max_percent",
                ],
            )
            w.writeheader()
            w.writerows(all_results)
        print(f"\nWrote {args.output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
