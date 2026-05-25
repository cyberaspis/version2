"""
Dataset loading and result parsing utilities.

Ported from experiment1_mistral.py and experiment_utils.py.
"""

import json
import re
from typing import Any, Dict, List, Tuple


def load_raw_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load the dataset from a JSON file.

    Each item is expected to have:
        {
          "call_id": str,
          "text": str,
          "label": 0 or 1   # 1 = vishing, 0 = non-vishing
        }

    Ported from experiment1_mistral.load_raw_dataset().
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects.")

    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("call_id", "")).strip() or "unknown"
        text = str(item.get("text", "")).strip()
        label_raw = item.get("label", 0)
        try:
            label_int = int(label_raw)
        except (TypeError, ValueError):
            label_int = 0
        label = 1 if label_int == 1 else 0
        cleaned.append({"call_id": call_id, "text": text, "label": label})

    return cleaned


def load_dataset(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a dataset from JSON, returning (texts, labels) tuple.

    Ported from experiment_utils.load_dataset().
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects")

    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        label_raw = item.get("label", 0)
        try:
            label = int(label_raw)
        except (TypeError, ValueError):
            label = 0

        texts.append(text)
        labels.append(1 if label == 1 else 0)

    return texts, labels


def parse_probability(output_text: str) -> float:
    """
    Extract vishing_probability float from LLM output.

    Expected format: {"vishing_probability": float}
    Falls back to regex if JSON parsing fails.

    Ported from experiment_utils.parse_probability().
    """
    if not output_text:
        return 0.0

    # Try JSON parsing first
    try:
        obj = json.loads(output_text)
        if isinstance(obj, dict) and "vishing_probability" in obj:
            value = float(obj["vishing_probability"])
            if 0.0 <= value <= 1.0:
                return value
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: regex
    pattern = r"vishing_probability\"?\s*[:=]\s*([01](?:\.\d+)?)"
    match = re.search(pattern, output_text)
    if match:
        try:
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass

    return 0.0
