import asyncio
import json
import os
import sys
import time
from .classifier import VishingClassifier
from .vllm_client import VLLMClient
from .data_loader import load_dataset
from . import config

async def run_streaming_evaluation():
    # Detect environment paths (same as in batch script)
    data_dir = os.getenv("DATA_DIR", "/data")
    vishing_path = os.path.join(data_dir, "vishing_dataset.json")
    non_vishing_path = os.path.join(data_dir, "non_vishing_dataset.json")

    print(f"Loading datasets...")
    v_texts, v_labels = load_dataset(vishing_path)
    nv_texts, nv_labels = load_dataset(non_vishing_path)
    
    all_samples = []
    for t, l in zip(v_texts, v_labels):
        all_samples.append({"text": t, "label": l})
    for t, l in zip(nv_texts, nv_labels):
        all_samples.append({"text": t, "label": l})

    print(f"Total samples to evaluate: {len(all_samples)}")

    # Initialize components
    client = VLLMClient(base_url=config.VLLM_BASE_URL, model=config.MODEL_NAME)
    classifier = VishingClassifier(client)

    results = []
    # --- Parallel Simulation ---
    semaphore = asyncio.Semaphore(30) # 30 concurrent calls
    
    # Counter for progress
    finished_count = 0
    total_samples = len(all_samples)
    
    async def process_sample(idx, sample):
        nonlocal finished_count
        async with semaphore:
            text = sample["text"]
            label = sample["label"]
            call_id = f"eval-{idx}"
            
            classifier.state_manager.cleanup(call_id)
            
            segments = text.split(". ")
            detected = False
            detection_segment = -1
            detection_words = -1
            accumulated_text = ""
            last_classification_words = 0
            
            for s_idx, seg in enumerate(segments):
                accumulated_text += seg + ". "
                words = accumulated_text.split()
                word_count = len(words)
                
                # Word-count based debounce (realistic 5-second equivalent)
                if (word_count >= config.MIN_WORDS_FOR_CLASSIFICATION and 
                    (word_count - last_classification_words) >= 15):
                    
                    res = await classifier.classify(accumulated_text, call_id=call_id)
                    status = res.get("risk_status", "SAFE")
                    last_classification_words = word_count
                    
                    if not detected and status in ["VISHING", "CRITICAL"]:
                        detected = True
                        detection_segment = s_idx + 1
                        detection_words = word_count
            
            res_item = {
                "call_id": call_id,
                "label": label,
                "detected": detected,
                "ttd_segments": detection_segment,
                "ttd_words": detection_words
            }
            results.append(res_item)
            finished_count += 1
            
            if finished_count % 5 == 0:
                # Current Recall (approximation based on processed files)
                processed_v = [r for r in results if r['label'] == 1]
                rec = sum(1 for r in processed_v if r['detected']) / len(processed_v) if processed_v else 0
                print(f"Progress: {finished_count}/{total_samples} samples... (Current Recall: {rec:.2f})")
                sys.stdout.flush()

    print(f"Starting parallel simulation (concurrency=30)...")
    tasks = [process_sample(i, s) for i, s in enumerate(all_samples)]
    await asyncio.gather(*tasks)

    # Compute Metrics
    ttd_segments = [r["ttd_segments"] for r in results if r["label"] == 1 and r["detected"]]
    ttd_words = [r["ttd_words"] for r in results if r["label"] == 1 and r["detected"]]
    
    tp = sum(1 for r in results if r["label"] == 1 and r["detected"])
    fp = sum(1 for r in results if r["label"] == 0 and r["detected"])
    tn = sum(1 for r in results if r["label"] == 0 and not r["detected"])
    fn = sum(1 for r in results if r["label"] == 1 and not r["detected"])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    avg_ttd_seg = sum(ttd_segments) / len(ttd_segments) if ttd_segments else 0
    avg_ttd_words = sum(ttd_words) / len(ttd_words) if ttd_words else 0

    print("\n" + "="*45)
    print("      STREAMING EVALUATION RESULTS (TTD)")
    print("="*45)
    print(f"Total Samples: {len(results)}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f} (Detection Rate)")
    print(f"FPR:           {fpr:.4f} (False Positive Rate)")
    print("-" * 25)
    print(f"Avg TTD (Segments): {avg_ttd_seg:.2f}")
    print(f"Avg TTD (Words):    {avg_ttd_words:.2f}")
    print("-" * 25)
    print(f"Confusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  TN: {tn:4d}  FN: {fn:4d}")
    print("=" * 45)

    # Save results
    output_path = os.getenv("OUTPUT_FILE", "results/streaming_evaluation_results.json")
    print(f"Saving detailed results to: {output_path}")
    try:
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = {
            "metrics": {
                "f1": f1, "precision": precision, "recall": recall, "fpr": fpr,
                "avg_ttd_segments": avg_ttd_seg, "avg_ttd_words": avg_ttd_words
            },
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "details": results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(run_streaming_evaluation())
