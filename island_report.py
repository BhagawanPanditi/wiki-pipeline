#!/usr/bin/env python3
"""
Knowledge Graph Island Analyzer (Connected Components)
------------------------------------------------------
Finds all disconnected subgraphs (islands/components) in the knowledge graph.
"""

import json
from pathlib import Path
from collections import Counter
import networkx as nx

from utils.atomic_io import atomic_write_json
from utils.cli import build_parser

STAGE = "island-report"

def analyze_kg_islands(domain_dir: Path):
    """Measure how the concept graph splits into islands and write the report.

    An island is a connected component of the concept graph. The report gives the
    giant component, every smaller island, and the concepts with no links at all.

    Args:
        domain_dir: Run directory. Reads ``global_concepts_refined.json``.

    Returns:
        None. Writes ``islands_report.json``.

    Raises:
        FileNotFoundError: If the refined graph does not exist."""
    refined_file = domain_dir / "global_concepts_refined.json"
    report_file = domain_dir / "islands_report.json"

    if not refined_file.exists():
        raise FileNotFoundError(f"Missing refined graph file: {refined_file}")

    print(f"Loading knowledge graph from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        concepts = json.load(f).get("concepts", [])

    # 1. Build Directed Graph
    G = nx.DiGraph()

    # Add all concepts as nodes
    title_map = {c["title"].lower(): c["title"] for c in concepts}
    for c in concepts:
        G.add_node(c["title"])

    # Add directed relation edges
    edge_count = 0
    for c in concepts:
        src = c["title"]
        for rel in c.get("outbound_relations", []):
            tgt_raw = rel.get("target_concept", "").strip()
            if not tgt_raw:
                continue
            
            # Map to canonical casing if present
            tgt = title_map.get(tgt_raw.lower(), tgt_raw)
            if tgt in G:
                G.add_edge(src, tgt, relation=rel.get("relation", ""))
                edge_count += 1

    total_nodes = G.number_of_nodes()
    print("\n--- GRAPH OVERVIEW ---")
    print(f"Total Concepts (Nodes):  {total_nodes}")
    print(f"Total Relations (Edges): {edge_count}")

    # 2. Extract Weakly Connected Components (Islands)
    # Weakly connected means direction is ignored (A -> B means A and B are in the same island)
    wccs = [list(comp) for comp in nx.weakly_connected_components(G)]
    wccs.sort(key=len, reverse=True)  # Sort by size descending

    total_islands = len(wccs)
    giant_component = wccs[0] if wccs else []
    giant_size = len(giant_component)
    giant_pct = (giant_size / total_nodes * 100) if total_nodes > 0 else 0

    print("\n--- ISLAND ANALYSIS ---")
    print(f"Total Connected Components (Islands): {total_islands}")
    print(f"Giant Component ('Mainland') Size:  {giant_size} concepts ({giant_pct:.2f}% of graph)")

    # Group islands by size
    size_distribution = Counter(len(comp) for comp in wccs)
    isolated_nodes_count = size_distribution.get(1, 0)

    print(f"Isolated Single Concepts (Size 1):     {isolated_nodes_count} ({isolated_nodes_count/total_nodes*100:.2f}%)")

    print("\n--- ISLAND SIZE DISTRIBUTION ---")
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        label = "Mainland" if size == giant_size else ("Isolated Node" if size == 1 else "Small Island")
        print(f"  Component Size {size:4d}: {count:4d} component(s) [{label}]")

    # Collect details on small islands (size 2 to 10)
    small_islands = []
    print("\n--- SAMPLE SMALL ISLANDS (Size 2 to 5) ---")
    small_island_counter = 0
    for idx, comp in enumerate(wccs[1:], 2):
        if 2 <= len(comp) <= 5:
            small_island_counter += 1
            small_islands.append({"island_id": idx, "size": len(comp), "concepts": comp})
            if small_island_counter <= 10:  # Print top 10 small islands to terminal
                print(f"Island #{idx} (Size {len(comp)}): {comp}")

    # Save detailed JSON report
    report_data = {
        "summary": {
            "total_concepts": total_nodes,
            "total_relations": edge_count,
            "total_islands": total_islands,
            "giant_component_size": giant_size,
            "giant_component_percentage": round(giant_pct, 2),
            "isolated_single_concepts": isolated_nodes_count,
            "size_distribution": dict(sorted(size_distribution.items()))
        },
        "islands": [
            {
                "island_id": i + 1,
                "size": len(comp),
                "is_giant_component": (i == 0),
                "concepts": sorted(comp)
            }
            for i, comp in enumerate(wccs)
        ]
    }

    atomic_write_json(report_file, report_data)

    print(f"\nSaved full island report to: {report_file}")

def main() -> None:
    """Measure how disconnected the finished graph is and write a report.

    Reads ``global_concepts_refined.json`` from the run directory, finds
    connected components and isolated concepts, and writes
    ``islands_report.json``. This is the last stage of the pipeline, so its
    report is the run's final quality signal.

    Returns:
        None.
    """
    # This stage makes no LLM calls. It uses the shared parser anyway so that one
    # uniform command line works for every stage in run_pipeline.sh; the server
    # and retry flags are accepted and ignored.
    parser = build_parser(
        stage=STAGE,
        description="Knowledge graph island analyzer (no LLM calls)",
        include_max_tokens=False,
        include_model=False)

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    analyze_kg_islands(domain_dir)

if __name__ == "__main__":
    main()