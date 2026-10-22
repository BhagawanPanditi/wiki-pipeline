#!/usr/bin/env python3
"""
Downloads BRIGHT long_documents subsets from HuggingFace and writes them
into a local `data/` folder as .jsonl files.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "xlangai/BRIGHT"
SUBSET = "long_documents"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"

TARGET_DOMAINS = {
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "sustainable_living",
    "robotics",
    "stackoverflow",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset: {DATASET_NAME} ({SUBSET})...")
dataset = load_dataset(DATASET_NAME, SUBSET)

file_handles = {}
for domain in TARGET_DOMAINS:
    file_path = OUTPUT_DIR / f"{domain}.jsonl"
    file_handles[domain] = open(file_path, "w", encoding="utf-8")

counts = {d: 0 for d in TARGET_DOMAINS}

try:
    for split_name, split_data in dataset.items():
        domain = split_name.strip().lower()
        if domain in TARGET_DOMAINS:
            print(f"Processing split: {domain} ({len(split_data)} records)...")
            for row in split_data:
                # Add domain field to row if not present
                if "domain" not in row:
                    row["domain"] = domain
                file_handles[domain].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[domain] += 1
        else:
            print(f"Skipping split: {split_name} (not in target domains)")
finally:
    for f in file_handles.values():
        f.close()

print("\nDownload complete! Files saved to:", OUTPUT_DIR)
print("Records per domain:")
for d, count in sorted(counts.items()):
    print(f"  {d:<22}: {count} records")
