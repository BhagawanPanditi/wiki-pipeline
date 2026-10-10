#!/usr/bin/env python3
"""
Source provenance analyzer for knowledge graph concepts.
Reports statistics on single-source vs multi-source concepts and their filenames.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_concept_sources(wiki_base_dir: str, domain: str = "biology"):
    domain_dir = Path(wiki_base_dir) / domain
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        fallback = domain_dir / "global_concepts.json"
        if fallback.exists():
            refined_file = fallback
        else:
            raise FileNotFoundError(f"Cannot find concepts file in {domain_dir}")

    print(f"Reading concepts from: {refined_file}")
    with open(refined_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    concepts = data.get("concepts", [])
    if not concepts:
        print("No concepts found.")
        return

    single_source = []
    multi_source = []
    zero_source = []
    source_distribution = defaultdict(int)

    for c in concepts:
        title = c.get("title", "").strip()
        chunks = c.get("source_chunks", [])
        docs = sorted(list({ch.split("::")[0].strip() for ch in chunks if "::" in ch and ch.strip()}))

        count = len(docs)
        source_distribution[count] += 1
        if count == 0:
            zero_source.append(title)
        elif count == 1:
            single_source.append((title, docs[0]))
        else:
            multi_source.append((title, docs))

    total = len(concepts)
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH SOURCE PROVENANCE STATISTICS")
    print("=" * 60)
    print(f"Total Concepts Evaluated: {total}")
    print(f"Single-Source Concepts (1 doc):     {len(single_source):5d} ({len(single_source)/total*100:5.1f}%)")
    print(f"Multi-Source Concepts (>= 2 docs):   {len(multi_source):5d} ({len(multi_source)/total*100:5.1f}%)")
    if zero_source:
        print(f"Zero-Source (Synthetic/Stubs):       {len(zero_source):5d} ({len(zero_source)/total*100:5.1f}%)")
    print("\nDistribution of Source Documents per Concept:")
    for num_docs in sorted(source_distribution.keys()):
        print(f"  {num_docs} source doc(s): {source_distribution[num_docs]} concepts")

    print("\n" + "-" * 60)
    print("Sample Multi-Source Concepts (first 25):")
    print("-" * 60)
    for title, docs in multi_source[:25]:
        print(f"  * {title} ({len(docs)} documents)")
        print(f"    Sources: {', '.join(docs)}")

    print("\nSample Single-Source Concepts (first 10):")
    print("-" * 60)
    for title, doc in single_source[:10]:
        print(f"  * {title} -> {doc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze concept source documents.")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--domain", type=str, default="biology")
    args = parser.parse_args()
    analyze_concept_sources(args.wiki_base_dir, args.domain)
