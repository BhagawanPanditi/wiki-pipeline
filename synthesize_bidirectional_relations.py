#!/usr/bin/env python3
"""
Bidirectional Relation Synthesizer
----------------------------------
Makes all outbound relations bidirectional:
- If Concept A -> Concept B exists with relation R_ab:
  Concept B will have an inbound/counterpart relation pointing back to Concept A.
- Generates a concise 1-2 sentence description for the reverse direction:
  (e.g., using A's brief, B's brief, and R_ab to articulate how Concept B is affected/utilized by A).
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from tqdm import tqdm
from openai import AsyncOpenAI
import httpx

from pipeline_config import load_global_config
from atomic_io import atomic_write_json
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

REVERSE_RELATION_PROMPT = """\
You are an expert technical knowledge graph agent.
Given a relationship from Concept A to Concept B, formulate the REVERSE relationship description from Concept B to Concept A in 1 concise sentence.

Concept A: {title_a}
Brief A: {brief_a}

Concept B: {title_b}
Brief B: {brief_b}

Original Relation (A -> B): {rel_ab}

TASK:
Write exactly 1 clear sentence describing how Concept B relates to, influences, is acted upon by, or provides context for Concept A.
Do not include filler or markdown. Just return the single sentence.
"""

async def synthesize_reverse_relations(domain_dir: Path, args):
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
    canonical_titles = {c["title"].lower().strip(): c["title"] for c in concepts}

    # Identify missing back-links AND template relations that need LLM formulation
    missing_reverse: List[Tuple[str, str, str]] = []  # (source_b, target_a, original_rel_ab)
    existing_edges: Set[Tuple[str, str]] = set()

    for c in concepts:
        src = c["title"].lower().strip()
        for rel in c.get("outbound_relations", []):
            tgt = rel.get("target_concept", "").lower().strip()
            if tgt in title_map and tgt != src:
                existing_edges.add((src, tgt))

    for (src, tgt) in list(existing_edges):
        if (tgt, src) not in existing_edges:
            # Need reverse link from tgt -> src
            source_concept = title_map[src]
            rel_text = ""
            for r in source_concept.get("outbound_relations", []):
                if r.get("target_concept", "").lower().strip() == tgt:
                    rel_text = r.get("relation", "")
                    break
            missing_reverse.append((tgt, src, rel_text))

    # Also rewrite template forward relations from embedding or entity linker
    for c in concepts:
        for rel in c.get("outbound_relations", []):
            rel_str = rel.get("relation", "")
            if "semantic affinity" in rel_str.lower() or "shared key entities" in rel_str.lower():
                tgt = rel.get("target_concept", "").lower().strip()
                if tgt in title_map and tgt != c["title"].lower().strip():
                    missing_reverse.append((c["title"].lower().strip(), tgt, rel_str))

    print(f"Found {len(missing_reverse)} directional edges that lack a reverse link.")

    if not missing_reverse:
        print("All relationships are already bidirectional!")
        return

    # If no LLM servers requested, use fast symmetric template fallback
    if not args.servers or args.fast:
        print("Applying template-based reverse relation formulations...")
        for (src_b_lower, tgt_a_lower, rel_ab) in tqdm(missing_reverse, desc="Adding reverse relations"):
            concept_b = title_map[src_b_lower]
            title_a = canonical_titles[tgt_a_lower]
            reverse_desc = f"Interacts with and is functionally linked to {title_a} in this domain."
            if rel_ab:
                reverse_desc = f"Associated counterpart to {title_a}: {rel_ab}"
            concept_b.setdefault("outbound_relations", []).append({
                "target_concept": title_a,
                "relation": reverse_desc,
                "source_chunks": []
            })
    else:
        print(f"Synthesizing reverse relations using LLM pool across {len(args.servers)} servers...")
        queue = asyncio.Queue()
        for item in missing_reverse:
            await queue.put(item)

        effective_concurrency = args.concurrency_per_server
        for _ in range(len(args.servers) * effective_concurrency):
            await queue.put(None)

        pbar = tqdm(total=len(missing_reverse), desc="LLM Reverse Relations")
        failed_relations: List[Tuple[str, str]] = []

        async def worker(client: AsyncOpenAI):
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break

                src_b_lower, tgt_a_lower, rel_ab = item
                concept_b = title_map[src_b_lower]
                concept_a = title_map[tgt_a_lower]

                prompt = REVERSE_RELATION_PROMPT.format(
                    title_a=concept_a["title"],
                    brief_a=concept_a.get("brief", ""),
                    title_b=concept_b["title"],
                    brief_b=concept_b.get("brief", ""),
                    rel_ab=rel_ab
                )

                rev_text = f"Functionally related to {concept_a['title']}."
                generated = False
                for attempt in range(10):
                    try:
                        resp = await client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=256
                        )
                        raw = (resp.choices[0].message.content or "").strip()
                        if raw:
                            rev_text = raw
                            generated = True
                            break
                        if attempt < 9:
                            await asyncio.sleep(min(1.5 * (attempt + 1), 8.0))
                    except Exception:
                        if attempt < 9:
                            await asyncio.sleep(min(1.5 * (attempt + 1), 8.0))

                if not generated:
                    failed_relations.append((src_b_lower, tgt_a_lower))

                # If relation already exists update it with synthesized description otherwise append
                existing_rel = None
                for r in concept_b.get("outbound_relations", []):
                    if r.get("target_concept", "").lower().strip() == concept_a["title"].lower().strip():
                        existing_rel = r
                        break

                if existing_rel:
                    existing_rel["relation"] = rev_text
                else:
                    concept_b.setdefault("outbound_relations", []).append({
                        "target_concept": concept_a["title"],
                        "relation": rev_text,
                        "source_chunks": []
                    })
                pbar.update(1)
                queue.task_done()

        tasks, http_clients = [], []
        for url in args.servers:
            hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=httpx.Timeout(60.0))
            http_clients.append(hc)
            client = AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0)
            for _ in range(effective_concurrency):
                tasks.append(asyncio.create_task(worker(client)))

        await asyncio.gather(*tasks)
        pbar.close()
        for hc in http_clients:
            await hc.aclose()

        if failed_relations:
            raise RuntimeError(
                f"Reverse-relation synthesis failed for {len(failed_relations)} edge(s) after retries. "
                f"Examples: {failed_relations[:10]}"
            )

    # STRICT STUB PURGE & DEDUPLICATION:
    # Ensure every single outbound relation targets an EXISTING canonical concept, no self-loops, and deduplicated
    print("Purging any residual stubs and deduplicating all relations across the graph...")
    total_valid_relations = 0
    stubs_purged = 0

    for c in concepts:
        clean_relations = []
        seen_targets = set()
        c_title_lower = c["title"].lower().strip()

        for rel in c.get("outbound_relations", []):
            raw_tgt = rel.get("target_concept", "").strip()
            if not raw_tgt:
                continue
            tgt_lower = raw_tgt.lower()

            # Target MUST exist in canonical concepts and cannot be a self-loop
            if tgt_lower in canonical_titles and tgt_lower != c_title_lower:
                canonical_tgt_title = canonical_titles[tgt_lower]
                if canonical_tgt_title not in seen_targets:
                    seen_targets.add(canonical_tgt_title)
                    rel["target_concept"] = canonical_tgt_title
                    clean_relations.append(rel)
                    total_valid_relations += 1
            else:
                stubs_purged += 1

        c["outbound_relations"] = clean_relations

    print(f"Residual stubs purged: {stubs_purged}. Total verified canonical relations: {total_valid_relations}.")

    # Save updated file
    print(f"Writing updated concepts with bidirectional links to {refined_file}...")
    atomic_write_json(refined_file, {"concepts": concepts})

    print("Bidirectional relation synthesis complete! Zero stubs guaranteed.")


def main():
    parser = argparse.ArgumentParser(description="Synthesize bidirectional relations")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)
    parser.add_argument("--fast", action="store_true", help="Use template-based synthesis without LLM calls")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    asyncio.run(synthesize_reverse_relations(domain_dir, args))


if __name__ == "__main__":
    main()
