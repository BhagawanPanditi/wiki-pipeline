#!/usr/bin/env python3
"""
Pipeline Status Checker
-----------------------
Scans all domains under the wiki base directory and reports chunk-level
progress across:
  1. extract-concepts (extracted.json)
  2. verify-pass1    (verification.json)
  3. refine-concepts (pass2_extracted.json)
  4. verify-pass2    (verification_pass2.json)
"""

import argparse
import json
from pathlib import Path

def inspect_domain(domain_dir: Path):
    chunks_path = domain_dir / "cache" / "chunks_cache.json"
    staging_dir = domain_dir / "staging"
    state_dir = domain_dir / ".pipeline_state"

    if not chunks_path.is_file():
        print(f"[{domain_dir.name}] No chunks cache found. Stage: Uninitialized / Pending Extraction.")
        return

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"[{domain_dir.name}] Error reading chunks cache: {e}")
        return

    total_chunks = len(chunks)
    counts = {
        "extract": 0,
        "verify_pass1": 0,
        "refine": 0,
        "verify_pass2": 0,
    }

    for ch in chunks:
        doc_id = str(ch.get("doc_id", ""))
        chunk_id = str(ch.get("chunk_id", ""))
        chunk_folder = staging_dir / doc_id / chunk_id

        if not chunk_folder.is_dir():
            continue

        if (chunk_folder / "concepts.json").is_file():
            counts["extract"] += 1
        if (chunk_folder / "verification.json").is_file():
            counts["verify_pass1"] += 1
        if (chunk_folder / "pass2_refinement.json").is_file() or (chunk_folder / "concepts_pass1_backup.json").is_file():
            counts["refine"] += 1
        if (chunk_folder / "verification_pass2.json").is_file():
            counts["verify_pass2"] += 1

    print(f"\n==================================================")
    print(f" DOMAIN: {domain_dir.name.upper()}  (Total Chunks: {total_chunks})")
    print(f"==================================================")

    stages = [
        ("1. Extract Concepts", counts["extract"], "extract-concepts.done"),
        ("2. Verify Pass 1   ", counts["verify_pass1"], "verify-pass1.done"),
        ("3. Refine Concepts ", counts["refine"], "refine-concepts.done"),
        ("4. Verify Pass 2   ", counts["verify_pass2"], "verify-pass2.done"),
    ]

    for label, done_count, done_file in stages:
        pending = total_chunks - done_count
        pct = (done_count / total_chunks * 100) if total_chunks else 0
        is_marked = (state_dir / done_file).exists()
        status_tag = "[DONE]" if is_marked else ("[IN PROGRESS]" if done_count > 0 else "[PENDING]")
        print(f"  {label} : {done_count:4d}/{total_chunks:<4d} ({pct:5.1f}%) | Pending: {pending:4d} | State: {status_tag}")

    # Check post-verification milestones
    post_stages = [
        ("consolidate-documents", domain_dir / "consolidated"),
        ("consolidate-global", domain_dir / "global_concepts.json"),
        ("resolve-stubs", domain_dir / "global_concepts_stubs_resolved.json"),
        ("fold-thin-concepts", state_dir / "fold-thin-concepts.done"),
        ("write-briefs", state_dir / "write-briefs.done"),
        ("audit-islands", state_dir / "audit-islands.done"),
        ("add-reverse-links", state_dir / "add-reverse-links.done"),
        ("final-consolidation", state_dir / "final-consolidation.done"),
    ]
    post_done = [name for name, path in post_stages if path.exists()]
    print(f"  Downstream Stages Completed: {', '.join(post_done) if post_done else 'None'}")

def main():
    parser = argparse.ArgumentParser(description="Check pipeline status across domains")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki",
                        help="Base directory containing domain subfolders")
    parser.add_argument("--domain", type=str, default=None, help="Specific domain to check (default: all)")
    args = parser.parse_args()

    base_dir = Path(args.wiki_base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory.")
        return

    if args.domain:
        domains = [base_dir / args.domain]
    else:
        domains = sorted([d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    for domain_dir in domains:
        inspect_domain(domain_dir)

if __name__ == "__main__":
    main()
