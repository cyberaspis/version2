import asyncio
import json
import sys
import os
from classifier import VishingClassifier
from vllm_client import VLLMClient
from data_loader import load_dataset
import config

async def run_evaluation(dataset_path: str):
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    texts, labels = load_dataset(dataset_path)
    print(f"Loaded {len(texts)} samples.")

    # Initialize client and classifier
    # vLLM must be running (e.g. via docker-compose)
    client = VLLMClient(base_url=config.VLLM_BASE_URL, model_name=config.MODEL_NAME)
    classifier = VishingClassifier(client)

    print("Starting classification...")
    results = await classifier.classify_batch(texts, labels)
    
    metrics = results.get("metrics", {})
    print("\n--- Evaluation Results ---")
    if "best_f1" in metrics:
        f1_m = metrics["best_f1"]
        print(f"Best F1 Score: {f1_m['f1']:.4f} at threshold {f1_m['threshold']}")
        print(f"Precision: {f1_m['precision']:.4f}, Recall: {f1_m['recall']:.4f}")
        print(f"Confusion Matrix: TP={f1_m['tp']}, FP={f1_m['fp']}, TN={f1_m['tn']}, FN={f1_m['fn']}")
    
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Optionally save results
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_classifier.py <path_to_dataset.json>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    asyncio.run(run_evaluation(dataset_path))
