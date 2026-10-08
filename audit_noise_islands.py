#!/usr/bin/env python3
"""
Island & Isolated Node Relevance Auditor and Sibling Connector
---------------------------------------------------------------
1. Analyzes connected components (islands).
2. For small islands (size 1 to 5) that are disconnected from the Giant Component:
   - Evaluates them against sibling concepts extracted from the EXACT SAME source document(s).
   - Prompts the LLM to choose:
     a) "PRUNE": Web scraping noise, site boilerplate, disclaimers, or out-of-domain fragments.
     b) "MERGE": Concept represents an identical or heavily overlapping topic as a sibling concept from the same document.
     c) "LINK": Valid distinct concept that naturally connects to a sibling concept from the same document.
     d) "KEEP": Valid specialized concept with no clear sibling link (kept as-is).
3. Connects linked concepts reciprocally, merges duplicates, and drops noise,
   pulling legitimate isolated nodes into the mainland graph.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from tqdm import tqdm

from pipeline_lib.atomic_io import atomic_write_json
from pipeline_lib.cli import build_parser, resolve_settings
from pipeline_lib.debug_log import DebugLogger
from pipeline_lib.errors import StageBudget
from pipeline_lib.llm_pool import ServerPool
from pipeline_lib.prompts import domain_scope, out_of_domain_clause

STAGE = "island-audit"

SYSTEM_PROMPT_ISLAND_AUDIT = """\
You are an expert technical knowledge graph quality auditor.
{scope}.
You are also provided with other concepts extracted from the EXACT SAME source document(s).

ISLAND CONCEPT(S) TO EVALUATE:
{island_concepts}

SIBLING CONCEPTS FROM THE SAME SOURCE DOCUMENT:
{document_context}

TASK:
Determine how to handle the island concept(s):
1. "PRUNE": Web scraping noise, site boilerplate, disclaimers, cookies, unrelated legal/author/hardware text, or {off_topic}.
2. "MERGE": The island concept is an identical or heavily overlapping duplicate of one of the sibling concepts from the same document.
3. "LINK": The island concept is a legitimate domain concept that is contextually related to one of the sibling concepts in the same document. Provide reciprocal link descriptions.
4. "KEEP": A legitimate domain concept, but none of the sibling concepts from the same document are suitable to link with.

Return ONLY valid JSON matching this schema:
{{
  "decision": "PRUNE" | "MERGE" | "LINK" | "KEEP",
  "target_sibling": "Exact Title of Sibling Concept (if MERGE or LINK, else null)",
  "relation_to_sibling": "1 sentence describing relationship from Island Concept to Sibling (if LINK, else null)",
  "relation_from_sibling": "1 sentence describing reverse relationship from Sibling to Island Concept (if LINK, else null)",
  "reason": "1 brief sentence explaining the decision."
}}
"""

async def audit_and_connect_islands(domain_dir: Path, domain_name: Optional[str], args):
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        fallback_file = domain_dir / "global_concepts.json"
        if fallback_file.exists():
            refined_file = fallback_file
        else:
            raise FileNotFoundError(f"Missing concept graph file in {domain_dir}")

    print(f"Loading concepts from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    title_map = {c["title"].lower().strip(): c for c in concepts}
    canonical_lookup = {c["title"].lower().strip(): c["title"] for c in concepts}

    # Build doc_id -> list of concepts
    doc_to_concepts: Dict[str, List[Dict[str, Any]]] = {}
    for c in concepts:
        for sc in c.get("source_chunks", []):
            doc_id = sc.split("::", 1)[0] if "::" in sc else sc.strip()
            if doc_id:
                doc_to_concepts.setdefault(doc_id, []).append(c)

    # Build graph
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["title"].lower().strip())
        for rel in c.get("outbound_relations", []):
            tgt = rel.get("target_concept", "").lower().strip()
            if tgt in title_map:
                G.add_edge(c["title"].lower().strip(), tgt)

    components = [list(comp) for comp in nx.connected_components(G)]
    components.sort(key=len, reverse=True)
    giant_component = set(components[0]) if components else set()

    small_islands_to_review = []
    for comp in components[1:]:  # skip giant component
        if len(comp) <= args.max_island_size:
            small_islands_to_review.append(comp)

    print(f"Total graph nodes: {len(concepts)} | Giant component: {len(giant_component)} ({len(giant_component)/len(concepts)*100:.1f}%)")
    print(f"Reviewing and bridging {len(small_islands_to_review)} small islands (size 1 to {args.max_island_size}) via sibling document context...")

    if not small_islands_to_review:
        return
    if not args.servers:
        raise RuntimeError("Island auditing found work to do, but no LLM servers are configured.")

    settings = resolve_settings(args, STAGE)
    debug = DebugLogger(domain_dir / "island_audit_debug.jsonl", label=STAGE)
    budget = StageBudget(settings.budget_minutes)

    prune_titles_set: Set[str] = set()
    merge_actions: List[Dict[str, Any]] = []
    link_actions: List[Dict[str, Any]] = []
    failed_components: List[List[str]] = []

    print(f"Spreading {len(small_islands_to_review)} island(s) across "
          f"{len(settings.servers)} server(s) at {settings.concurrency_per_server} "
          f"concurrent request(s) each (item i -> server i % N).")
    print(f"Per-item attempts: {settings.attempts} | Stage budget: "
          f"{settings.budget_minutes:g} min | Debug log: {debug.path}")

    audit_log_path = domain_dir / "island_audit_log.jsonl"
    log_f = open(audit_log_path, "w", encoding="utf-8")

    def build_audit_prompt(comp: List[str]) -> str:
        island_payload = []
        source_docs = set()
        for t_lower in comp:
            c = title_map.get(t_lower)
            if c:
                island_payload.append({
                    "title": c["title"],
                    "brief": c.get("brief", ""),
                    "aliases": c.get("aliases", []),
                    "entities": c.get("entities", [])[:5]
                })
                for sc in c.get("source_chunks", []):
                    doc_id = sc.split("::", 1)[0] if "::" in sc else sc.strip()
                    if doc_id:
                        source_docs.add(doc_id)

        # Gather sibling concepts from same documents (prioritize giant component / mainland concepts)
        sibling_concepts = []
        seen_siblings = set(comp)
        for doc_id in source_docs:
            for sib in doc_to_concepts.get(doc_id, []):
                sib_lower = sib["title"].lower().strip()
                if sib_lower not in seen_siblings:
                    seen_siblings.add(sib_lower)
                    sibling_concepts.append({
                        "title": sib["title"],
                        "brief": sib.get("brief", ""),
                        "aliases": sib.get("aliases", []),
                        "is_in_mainland": sib_lower in giant_component
                    })
                if len(sibling_concepts) >= 12:
                    break
            if len(sibling_concepts) >= 12:
                break

        # Sort siblings so mainland concepts come first
        sibling_concepts.sort(key=lambda s: not s.get("is_in_mainland", False))

        doc_context_str = json.dumps(sibling_concepts, indent=2) if sibling_concepts else "No other concepts in same document."

        return SYSTEM_PROMPT_ISLAND_AUDIT.format(
            scope=domain_scope(
                "Your job is to examine an isolated concept or small island from "
                "an automated document extraction pipeline", domain_name),
            off_topic=out_of_domain_clause(domain_name),
            island_concepts=json.dumps(island_payload, indent=2),
            document_context=doc_context_str
        )

    async def audit_island(client, comp: List[str], server_index: int):
        """One island verdict. Raises so the pool retries; this stage has always
        parsed the reply with a plain json.loads, and still does."""
        resp = await client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": build_audit_prompt(comp)}],
            temperature=0.1,
            max_tokens=settings.max_tokens,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)

    async def record_verdict(comp: List[str], res_json: Dict[str, Any]):
        """Append the audit line and queue the graph action. No awaits inside, so
        this is already atomic with respect to the event loop - no lock needed."""
        decision = str(res_json.get("decision", "KEEP")).strip().upper()
        target_sib = res_json.get("target_sibling")
        rel_ab = res_json.get("relation_to_sibling")
        rel_ba = res_json.get("relation_from_sibling")

        log_entry = {"comp": comp, "decision": decision, "response": res_json}
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        if decision == "PRUNE":
            for t_lower in comp:
                prune_titles_set.add(t_lower)
        elif decision == "MERGE" and target_sib and target_sib.lower().strip() in title_map:
            merge_actions.append({
                "island_titles": comp,
                "target_sibling": target_sib
            })
        elif decision == "LINK" and target_sib and target_sib.lower().strip() in title_map:
            link_actions.append({
                "island_titles": comp,
                "target_sibling": target_sib,
                "rel_ab": rel_ab,
                "rel_ba": rel_ba
            })

    pool = ServerPool(
        settings.servers,
        concurrency_per_server=settings.concurrency_per_server,
        request_timeout=settings.profile.request_timeout,
        connect_timeout=settings.profile.connect_timeout,
        label=STAGE,
        failover=not getattr(args, "no_failover", False),
    )
    try:
        async with pool:
            result = await pool.map(
                small_islands_to_review,
                audit_island,
                attempts=settings.attempts,
                desc="Auditing & Connecting Islands",
                key_fn=lambda comp: "|".join(sorted(comp)),
                budget=budget,
                debug=debug,
                attempt_timeout=settings.profile.attempt_timeout,
                on_result=record_verdict,
            )
    finally:
        log_f.close()
        debug.close()

    # An island that never produced a verdict is a failure, exactly as before.
    for failure in result.failures:
        failed_components.append(small_islands_to_review[failure["index"]])

    print("Requests served per server:")
    for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
        print(f"  {url:<40} {count}")

    if failed_components:
        raise RuntimeError(
            f"Island audit failed for {len(failed_components)} component(s) after retries. "
            f"Examples: {failed_components[:5]}"
        )

    print(f"\nAudit & Bridge Results:")
    print(f"  - Confirmed Noise Islands Pruned:   {len(prune_titles_set)}")
    print(f"  - Sibling Document Merges:          {len(merge_actions)}")
    print(f"  - Sibling Document Links Added:     {len(link_actions)}")

    # 1. Apply Merges
    merged_absorbed_set = set()
    for ma in merge_actions:
        target_c = title_map[ma["target_sibling"].lower().strip()]
        for t_lower in ma["island_titles"]:
            if t_lower == target_c["title"].lower().strip():
                continue
            island_c = title_map.get(t_lower)
            if not island_c:
                continue

            # Merge description
            if island_c.get("description"):
                target_c["description"] = f"{target_c.get('description', '')}\n\n{island_c['description']}".strip()

            # Merge aliases & entities
            target_c.setdefault("aliases", []).append(island_c["title"])
            for a in island_c.get("aliases", []):
                if a not in target_c["aliases"]: target_c["aliases"].append(a)
            for e in island_c.get("entities", []):
                if e not in target_c.get("entities", []): target_c.setdefault("entities", []).append(e)

            # Merge source chunks
            sc_set = set(target_c.get("source_chunks", []))
            sc_set.update(island_c.get("source_chunks", []))
            target_c["source_chunks"] = sorted(list(sc_set))

            merged_absorbed_set.add(t_lower)

    # 2. Apply Sibling Links
    for la in link_actions:
        sib_c = title_map[la["target_sibling"].lower().strip()]
        sib_title = sib_c["title"]

        for t_lower in la["island_titles"]:
            if t_lower in prune_titles_set or t_lower in merged_absorbed_set:
                continue
            island_c = title_map.get(t_lower)
            if not island_c:
                continue

            rel_to_sib = la["rel_ab"] or f"Directly related in source document to {sib_title}."
            rel_from_sib = la["rel_ba"] or f"Directly related in source document to {island_c['title']}."

            # Island -> Sibling
            existing_island = {r.get("target_concept", "").lower().strip() for r in island_c.get("outbound_relations", [])}
            if sib_title.lower().strip() not in existing_island:
                island_c.setdefault("outbound_relations", []).append({
                    "target_concept": sib_title,
                    "relation": rel_to_sib,
                    "source_chunks": []
                })

            # Sibling -> Island
            existing_sib = {r.get("target_concept", "").lower().strip() for r in sib_c.get("outbound_relations", [])}
            if island_c["title"].lower().strip() not in existing_sib:
                sib_c.setdefault("outbound_relations", []).append({
                    "target_concept": island_c["title"],
                    "relation": rel_from_sib,
                    "source_chunks": []
                })

    # 3. Filter out pruned and absorbed concepts
    surviving_concepts = [
        c for c in concepts
        if c["title"].lower().strip() not in prune_titles_set and c["title"].lower().strip() not in merged_absorbed_set
    ]

    # Clean relations
    surviving_title_lookup = {c["title"].lower().strip(): c["title"] for c in surviving_concepts}
    for c in surviving_concepts:
        clean_rels = []
        seen = set()
        for r in c.get("outbound_relations", []):
            tgt_raw = r.get("target_concept", "").lower().strip()
            if tgt_raw in surviving_title_lookup and tgt_raw != c["title"].lower().strip():
                canonical_tgt = surviving_title_lookup[tgt_raw]
                if canonical_tgt not in seen:
                    seen.add(canonical_tgt)
                    r["target_concept"] = canonical_tgt
                    clean_rels.append(r)
        c["outbound_relations"] = clean_rels

    # Save updated clean graph
    print(f"\nSaving updated bridged graph ({len(surviving_concepts)} concepts) to {refined_file}...")
    atomic_write_json(refined_file, {"concepts": surviving_concepts})

    # Re-evaluate island metrics
    G_clean = nx.Graph()
    for c in surviving_concepts:
        G_clean.add_node(c["title"].lower().strip())
        for rel in c.get("outbound_relations", []):
            tgt = rel.get("target_concept", "").lower().strip()
            if tgt in G_clean:
                G_clean.add_edge(c["title"].lower().strip(), tgt)

    new_components = [list(comp) for comp in nx.connected_components(G_clean)]
    new_components.sort(key=len, reverse=True)
    new_giant = len(new_components[0]) if new_components else 0
    print(f"Updated Graph: {len(new_components)} islands total. Giant component: {new_giant}/{len(surviving_concepts)} ({new_giant/len(surviving_concepts)*100:.2f}%)")

def main():
    parser = build_parser(
        stage=STAGE,
        description="Audit, connect, and prune small islands using sibling document context")
    parser.add_argument("--max_island_size", type=int, default=5, help="Maximum component size to evaluate")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    # domain_name feeds prompts only, so pass the optional hint (None when
    # --domain was omitted); args.domain stays the output folder name.
    asyncio.run(audit_and_connect_islands(domain_dir, args.domain_label, args))

if __name__ == "__main__":
    main()
