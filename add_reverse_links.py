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

import asyncio
import json
import re
from pathlib import Path
from typing import List, Set, Tuple

from tqdm import tqdm

from utils.atomic_io import atomic_write_json
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.layout import log_path
from utils.errors import StageBudget
from utils.llm_pool import pool_for_stage

from pydantic import BaseModel, Field
from utils.json_repair import clean_json_string

STAGE = "add-reverse-links"

class ReverseRelationSchema(BaseModel):
    relation: str = Field(description="Exactly 1 clear sentence describing how Concept B relates to, influences, is acted upon by, or provides context for Concept A.")

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

OUTPUT FORMAT:
Return valid JSON with the exact key "relation":
{{
  "relation": "One clear sentence describing how Concept B relates to Concept A."
}}
"""

async def synthesize_reverse_relations(domain_dir: Path, args):
    """Add the missing back-link for every one-way relation.

    Finds each relation A to B that has no matching B to A, and adds one. Legacy
    template wording ("semantic affinity") is rewritten as well. With ``--fast`` or no
    servers the sentence comes from a template; otherwise the LLM writes it from both
    briefs. Each addition is recorded in the decision log.

    Args:
        domain_dir: Run directory. Reads and rewrites ``global_concepts_refined.json``.
        args: Parsed CLI namespace. ``args.fast`` forces template synthesis.

    Returns:
        None.

    Raises:
        FileNotFoundError: If no concept graph exists."""
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

    seen_missing: Set[Tuple[str, str]] = set()
    for (src, tgt) in list(existing_edges):
        if (tgt, src) not in existing_edges:
            # Need reverse link from tgt -> src
            source_concept = title_map[src]
            rel_text = ""
            for r in source_concept.get("outbound_relations", []):
                if r.get("target_concept", "").lower().strip() == tgt:
                    rel_text = r.get("relation", "")
                    break
            pair = (tgt, src)
            if pair not in seen_missing:
                seen_missing.add(pair)
                missing_reverse.append((tgt, src, rel_text))

    # Also rewrite template forward relations planted by the link-entities
    # linker. "semantic affinity" is legacy wording from earlier graph files
    # and is kept so re-running over old inputs still upgrades those edges.
    for c in concepts:
        for rel in c.get("outbound_relations", []):
            rel_str = rel.get("relation", "")
            if "semantic affinity" in rel_str.lower() or "shared key entities" in rel_str.lower():
                tgt = rel.get("target_concept", "").lower().strip()
                if tgt in title_map and tgt != c["title"].lower().strip():
                    pair = (c["title"].lower().strip(), tgt)
                    if pair not in seen_missing:
                        seen_missing.add(pair)
                        missing_reverse.append((pair[0], pair[1], rel_str))

    print(f"Found {len(missing_reverse)} directional edges that lack a reverse link.")
    decisions = EventLog(log_path(domain_dir, STAGE, "decisions"), label=STAGE)

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
            decisions.record("reverse_link_added", method="template", source=concept_b["title"],
                             target=title_a, relation=reverse_desc)
    else:
        settings = resolve_settings(args, STAGE)
        debug = EventLog(log_path(domain_dir, STAGE, "debug"), label=STAGE)
        budget = StageBudget(settings.budget_minutes)
        failed_relations: List[Tuple[str, str]] = []

        async def synthesize_one(client, item, server_index):
            """One reverse-relation sentence. Raises so the pool classifies/retries."""
            src_b_lower, tgt_a_lower, rel_ab = item
            concept_a = title_map[tgt_a_lower]
            concept_b = title_map[src_b_lower]

            prompt = REVERSE_RELATION_PROMPT.format(
                title_a=concept_a["title"],
                brief_a=concept_a.get("brief", ""),
                title_b=concept_b["title"],
                brief_b=concept_b.get("brief", ""),
                rel_ab=rel_ab
            )

            resp = await client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"}
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                # Recoverable: the pool retries it on a short backoff, as before.
                raise ValueError("empty completion for reverse relation")

            try:
                parsed = ReverseRelationSchema.model_validate_json(clean_json_string(raw, allow_bare_object=True))
                final_line = parsed.relation.strip()
            except Exception:
                # Fallback: Strip <think> and extract the last non-scratchpad line
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
                valid_lines = [
                    l for l in lines
                    if not re.match(r"^(?:need\b|we need\b|task:|note:|thinking:|- |\* )", l, re.IGNORECASE)
                ]
                final_line = valid_lines[-1] if valid_lines else lines[-1]

            # Strip surrounding markdown quotes or bullet prefixes
            final_line = re.sub(r"^[\"'\-*\s]+|[\"'\s]+$", "", final_line).strip()
            return final_line

        completed_count = 0
        save_lock = asyncio.Lock()

        async def attach_reverse_relation(item, rev_text):
            """Insert or overwrite the back-link on concept B."""
            nonlocal completed_count
            src_b_lower, tgt_a_lower, _rel_ab = item
            concept_a = title_map[tgt_a_lower]
            concept_b = title_map[src_b_lower]

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
            decisions.record("reverse_link_added", method="llm", source=concept_b["title"],
                             target=concept_a["title"], relation=rev_text,
                             updated_existing=existing_rel is not None)

            async with save_lock:
                completed_count += 1
                # Periodically save checkpoints every 200 items so any crash can resume safely
                if completed_count % 200 == 0:
                    atomic_write_json(refined_file, {"concepts": concepts})

        print(f"Synthesizing reverse relations using LLM pool across {len(settings.servers)} servers...")
        print(f"Spreading {len(missing_reverse)} edge(s) at {settings.concurrency_per_server} "
              f"concurrent request(s) per server (item i -> server i % N).")
        print(f"Per-item attempts: {settings.attempts} | Stage budget: "
              f"{settings.budget_minutes:g} min | Debug log: {debug.path}")

        pool = pool_for_stage(args, settings, STAGE)
        try:
            async with pool:
                result = await pool.map(
                    missing_reverse,
                    synthesize_one,
                    attempts=settings.attempts,
                    desc="LLM Reverse Relations",
                    key_fn=lambda it: f"{it[0]} <- {it[1]}",
                    budget=budget,
                    debug=debug,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=attach_reverse_relation,
                )
        finally:
            debug.close()

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        # Unchanged contract: an edge that never produced text still receives the
        # deterministic fallback wording, is recorded, and fails the stage loudly.
        for failure in result.failures:
            item = missing_reverse[failure["index"]]
            concept_a = title_map[item[1]]
            await attach_reverse_relation(item, f"Functionally related to {concept_a['title']}.")
            failed_relations.append((item[0], item[1]))

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
    """Parse the command line and add the reverse relations.

    Returns:
        None."""
    parser = build_parser(stage=STAGE, description="Synthesize bidirectional relations")
    parser.add_argument("--fast", action="store_true", help="Use template-based synthesis without LLM calls")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    asyncio.run(synthesize_reverse_relations(domain_dir, args))

if __name__ == "__main__":
    main()
