import asyncio
import json
import os
from .classifier import VishingClassifier
from .vllm_client import VLLMClient
from .data_loader import load_dataset
from . import config

async def run_merged_evaluation():
    # Detect environment paths
    data_dir = os.getenv("DATA_DIR", "/data")
    vishing_path = os.path.join(data_dir, "vishing_dataset.json")
    non_vishing_path = os.path.join(data_dir, "non_vishing_dataset.json")

    print(f"Loading vishing dataset from: {vishing_path}")
    vishing_texts, vishing_labels = load_dataset(vishing_path)
    print(f"Loaded {len(vishing_texts)} vishing samples.")

    print(f"Loading non-vishing dataset from: {non_vishing_path}")
    non_vishing_texts, non_vishing_labels = load_dataset(non_vishing_path)
    print(f"Loaded {len(non_vishing_texts)} non-vishing samples.")

    # Merge
    all_texts = vishing_texts + non_vishing_texts
    all_labels = vishing_labels + non_vishing_labels
    print(f"Total samples to evaluate: {len(all_texts)}")

    # Initialize client and classifier
    client = VLLMClient(base_url=config.VLLM_BASE_URL, model=config.MODEL_NAME)
    classifier = VishingClassifier(client)

    print("Starting batch classification...")
    results = await classifier.classify_batch(all_texts, all_labels)
    
    metrics = results.get("metrics", {})
    print("\n" + "="*40)
    print("      CONSOLIDATED EVALUATION RESULTS")
    print("="*40)
    
    if "best_f1" in metrics:
        f1_m = metrics["best_f1"]
        print(f"Best F1 Score: {f1_m['f1']:.4f} at threshold {f1_m['threshold']}")
        print(f"Precision: {f1_m['precision']:.4f}")
        print(f"Recall:    {f1_m['recall']:.4f}")
        print(f"ROC-AUC:   {metrics.get('roc_auc', 0.0):.4f}")
        print("-" * 20)
        print(f"Confusion Matrix (at best threshold):")
        print(f"  TP: {f1_m['tp']:4d}  FP: {f1_m['fp']:4d}")
        print(f"  TN: {f1_m['tn']:4d}  FN: {f1_m['fn']:4d}")
    else:
        print("No metrics calculated (possibly empty dataset or all zero labels).")
    print("=" * 40)

    # Save results to file
    output_path = os.getenv("OUTPUT_FILE", "evaluation_results.json")
    print(f"Saving results to: {output_path}")
    try:
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(run_merged_evaluation())
