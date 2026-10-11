#!/usr/bin/env python3
"""
Inspect exact post-consolidation graph enrichment progress across all domains.
Checks:
  1. global_concepts.json (exact consolidation)
  2. global_concepts_stubs_resolved.json (resolve-stubs)
  3. global_concepts_refined.json (fold-thin-concepts / write-briefs / audit-islands)
  4. Decision logs for write-briefs, audit-islands, final-consolidation
  5. concepts_md/ (final-consolidation export)
"""

import argparse
import json
from pathlib import Path


def inspect_domain_enrichment(domain_dir: Path):
    dname = domain_dir.name
    print(f"\n{'=' * 65}")
    print(f"  DOMAIN: {dname.upper()}")
    print(f"{'=' * 65}")

    # Stage 1: Global Exact Consolidation
    exact_file = domain_dir / "global_concepts.json"
    if exact_file.is_file():
        try:
            with open(exact_file, "r", encoding="utf-8") as f:
                c = json.load(f).get("concepts", [])
            print(f"  [1] Global Exact Consolidation  : [DONE] ({len(c)} concepts)")
        except Exception:
            print(f"  [1] Global Exact Consolidation  : [DONE] (file exists)")
    else:
        print(f"  [1] Global Exact Consolidation  : [MISSING]")

    # Stage 2: Stubs Resolved
    stubs_file = domain_dir / "global_concepts_stubs_resolved.json"
    if stubs_file.is_file():
        try:
            with open(stubs_file, "r", encoding="utf-8") as f:
                c = json.load(f).get("concepts", [])
            print(f"  [2] Stubs Resolved              : [DONE] ({len(c)} concepts)")
        except Exception:
            print(f"  [2] Stubs Resolved              : [DONE]")
    else:
        print(f"  [2] Stubs Resolved              : [PENDING]")

    # Stage 3: Refined Graph (Fold Thin Concepts / Write Briefs / Audit Islands)
    refined_file = domain_dir / "global_concepts_refined.json"
    if refined_file.is_file():
        try:
            with open(refined_file, "r", encoding="utf-8") as f:
                c = json.load(f).get("concepts", [])
            has_briefs = sum(1 for x in c if x.get("brief"))
            print(f"  [3] Refined Graph (global_refined): [DONE] ({len(c)} concepts)")
            print(f"      - Concepts with Briefs      : {has_briefs}/{len(c)} ({has_briefs/len(c)*100:.1f}%)")
        except Exception:
            print(f"  [3] Refined Graph (global_refined): [DONE]")
    else:
        print(f"  [3] Refined Graph (global_refined): [PENDING]")

    # Check logs for stages
    log_dir = domain_dir / "log"
    stages_to_check = [
        ("write-briefs", "write-briefs.decisions.jsonl"),
        ("link-entities", "link-entities.decisions.jsonl"),
        ("audit-islands", "audit-islands.decisions.jsonl"),
        ("final-consolidation", "final-deep-verify.decisions.jsonl"),
    ]
    print("  Stage Decision Logs:")
    for label, log_name in stages_to_check:
        lf = log_dir / log_name
        if lf.is_file():
            size = lf.stat().st_size
            print(f"      - {label:<22}: [FOUND] ({size / 1024:.1f} KB)")
        else:
            print(f"      - {label:<22}: [NOT RUN / NO LOG]")

    # Check final output
    md_dir = domain_dir / "concepts_md"
    if md_dir.is_dir():
        md_count = len(list(md_dir.glob("*.md")))
        print(f"  [FINAL] Final Markdown Wiki (concepts_md): [COMPLETED] ({md_count} articles)")
    else:
        print(f"  [FINAL] Final Markdown Wiki (concepts_md): [PENDING]")


def main():
    parser = argparse.ArgumentParser(description="Check graph enrichment status across domains")
    parser.add_argument("--wiki_base_dir", type=str,
                        default="/home/compiling-ganesh/24m0829/long_document/wiki",
                        help="Base directory containing domain subfolders")
    parser.add_argument("--domain", type=str, default=None,
                        help="Specific domain to check")
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
        inspect_domain_enrichment(domain_dir)


if __name__ == "__main__":
    main()
