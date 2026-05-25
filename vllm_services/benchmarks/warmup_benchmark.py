import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--runs", type=int, default=5)
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_RUNS = args.runs

# Enable cuDNN benchmark mode for optimized performance
torch.backends.cudnn.benchmark = True

print(f"\nLoading model: {MODEL_NAME}\n")

start_load = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set evaluation mode
model.eval()

load_time = time.time() - start_load
print(f"Model Load Time: {load_time:.2f} seconds")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

prompt = """Παρακαλώ επιβεβαιώστε τον τραπεζικό σας κωδικό
για να ξεμπλοκάρουμε τον λογαριασμό σας άμεσα."""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# GPU preparation
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

print("\nStarting warm-up runs...\n")

times = []

for i in range(NUM_RUNS):
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True
        )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    times.append(elapsed)

    print(f"Run {i+1}: {elapsed:.3f} seconds")

# GPU memory after
if device == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nPeak GPU Memory Usage: {peak_mem:.2f} GB")

# Exclude first run (cold start)
if len(times) > 1:
    avg_time = sum(times[1:]) / (len(times) - 1)
else:
    avg_time = times[0]

print("\n========== RESULTS ==========")
print(f"Average Warm Inference Time (excluding first run): {avg_time:.3f} sec")
print("================================\n")
