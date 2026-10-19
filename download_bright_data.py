#!/usr/bin/env python3
"""
Download BRIGHT long_documents subsets from Hugging Face: xlangai/BRIGHT
Saves each split as JSONL in the standard data directory.
"""
import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset

DOMAINS = [
    "robotics",
    "stackoverflow",
    "biology",
    "earth_science",
    "sustainable_living",
    "pony",
    "aops",
    "leetcode"
]

def download_bright_long_documents(output_dir: Path, requested_domains=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    domains_to_fetch = requested_domains or DOMAINS

    print(f"Loading xlangai/BRIGHT long_documents subset...")
    print(f"Output directory: {output_dir}")

    for domain in domains_to_fetch:
        target_file = output_dir / f"{domain}.jsonl"
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[EXISTS] {domain}.jsonl already downloaded ({target_file.stat().st_size / 1024 / 1024:.2f} MB). Skipping.")
            continue

        print(f"\n[DOWNLOADING] split: {domain}...")
        try:
            ds = load_dataset("xlangai/BRIGHT", "long_documents", split=domain)
            print(f"Loaded {len(ds)} documents for {domain}. Saving to {target_file}...")
            
            with open(target_file, "w", encoding="utf-8") as f:
                for row in ds:
                    # BRIGHT documents have 'id' and 'content'
                    entry = {
                        "id": row.get("id"),
                        "content": row.get("content") or row.get("contents") or row.get("text") or ""
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[SUCCESS] Saved {domain}.jsonl ({target_file.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {domain}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BRIGHT long_documents dataset from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/data/long_documents")
    parser.add_argument("--domains", nargs="+", default=["robotics", "stackoverflow"], help="Domains to download (default: robotics stackoverflow)")
    args = parser.parse_args()

    download_bright_long_documents(Path(args.output_dir), args.domains)
