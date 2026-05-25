#!/usr/bin/env python3
"""
Load all conversations from vishing_dataset.json and non_vishing_dataset.json,
and push them to the classifier dashboard as separate calls so you can see
the classifier operate on each one.
"""
import asyncio
import json
import os
import time
import argparse
from pathlib import Path

import httpx

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://127.0.0.1:8003")
PUSH_URL = f"{CLASSIFIER_URL.rstrip('/')}/push_segment"

# Paths relative to Project_v2
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VISHING_JSON = DATA_DIR / "vishing_dataset.json"
NON_VISHING_JSON = DATA_DIR / "non_vishing_dataset.json"


def load_conversations(limit: int | None):
    out = []
    for path, label_name in [(VISHING_JSON, "vishing"), (NON_VISHING_JSON, "non_vishing")]:
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        for item in data:
            out.append({
                "call_id": item["call_id"],
                "text": item["text"],
                "label": item.get("label", 1 if label_name == "vishing" else 0),
                "source": label_name,
            })
    return out


def text_to_segments(text: str):
    """Split conversation text into segments; alternate role agent/caller."""
    raw = [s.strip() for s in text.split("\n") if s.strip()]
    segments = []
    for i, seg in enumerate(raw):
        role = "agent" if i % 2 == 0 else "caller"
        segments.append({"role": role, "text": seg})
    return segments


async def push_conversation(client: httpx.AsyncClient, conv: dict, delay_segment: float, delay_call: float):
    call_id = conv["call_id"]
    segments = text_to_segments(conv["text"])
    for seg in segments:
        payload = {
            "call_uuid": call_id,
            "role": seg["role"],
            "text": seg["text"],
            "timestamp": time.time(),
        }
        try:
            r = await client.post(PUSH_URL, json=payload, timeout=30.0)
            if r.status_code != 200:
                print(f"  [{call_id[:12]}...] segment HTTP {r.status_code}")
        except Exception as e:
            print(f"  [{call_id[:12]}...] error: {e}")
        await asyncio.sleep(delay_segment)
    await asyncio.sleep(delay_call)


async def main():
    ap = argparse.ArgumentParser(description="Push all dataset conversations to the classifier dashboard.")
    ap.add_argument("--limit", type=int, default=None, help="Max conversations per dataset (default: all)")
    ap.add_argument("--delay-segment", type=float, default=0.25, help="Seconds between segments (default: 0.25)")
    ap.add_argument("--delay-call", type=float, default=0.5, help="Seconds between calls (default: 0.5)")
    ap.add_argument("--vishing-only", action="store_true", help="Push only vishing_dataset.json")
    ap.add_argument("--non-vishing-only", action="store_true", help="Push only non_vishing_dataset.json")
    args = ap.parse_args()

    conversations = load_conversations(args.limit)
    if args.vishing_only:
        conversations = [c for c in conversations if c["source"] == "vishing"]
    if args.non_vishing_only:
        conversations = [c for c in conversations if c["source"] == "non_vishing"]

    print(f"Push URL: {PUSH_URL}")
    print(f"Total conversations to push: {len(conversations)}")
    print("Open the dashboard in your browser to see calls and classifier results.\n")

    async with httpx.AsyncClient() as client:
        for i, conv in enumerate(conversations):
            n = i + 1
            call_id = conv["call_id"]
            src = conv["source"]
            print(f"[{n}/{len(conversations)}] Pushing {src}: {call_id[:20]}...")
            await push_conversation(client, conv, args.delay_segment, args.delay_call)

    print("\nDone. Refresh the dashboard to see all calls.")


if __name__ == "__main__":
    asyncio.run(main())
