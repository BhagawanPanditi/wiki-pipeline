#!/usr/bin/env python3
"""Stage 6: Global Cross-Document Concept Consolidation.
------------------------------------------------------
Unifies duplicate and variant concepts across the entire corpus:
1. Fast indexing finds exact title matches, lemma variants, and title-to-alias pairs.
2. Parallel LLM calls verify whether candidate pairs denote the exact same subject.
3. Disjoint Set (Union-Find) fuses confirmed merges, combining titles, aliases,
   descriptions, and document citations into global_concepts.json.

Usage:
    python3 consolidate_global.py --wiki_path /path/to/wiki
"""


import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel
from tqdm import tqdm

from utils.atomic_io import atomic_write_json
from utils.cli import build_parser, resolve_settings
from utils.concept_ops import fuse_descriptions, generate_stable_slug
from utils.lemma import canonical_title_key
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest
from utils.stats import print_concept_word_distribution
from utils.unionfind import UnionFind

STAGE = "consolidate-global"

def get_context_hint(desc: str, max_words: int = 40) -> str:
    """Extracts first ~40 words from description as rich context for LLM comparison."""
    if not desc:
        return "No description available."
    words = desc.strip().split()
    snippet = " ".join(words[:max_words])
    if len(words) > max_words:
        snippet += "..."
    return snippet

# ── XML System Prompt for LLM Concept Merge Verification ──────────────────────

SYSTEM_PROMPT_MERGE_VERIFY = """\
You are an expert knowledge consolidator evaluating candidate concepts across documents.
You are given candidate pairs of concepts that share identical titles, plural/spelling variants, or title-to-alias alignments.
Each concept is presented with its Canonical Title, Aliases, and a ~40-word Context Snippet from its description.

Your task is to determine whether both concepts refer to the EXACT SAME subject, entity, item, process, condition, or topic.

CRITICAL RULES:
1. MERGE (is_same_entity: true): Only if both concepts denote the identical subject (e.g., direct synonymous naming or alternative formal titles for the same entity).
2. KEEP SEPARATE (is_same_entity: false): Do NOT merge if one is an acronym that is polysemous, or if one is a sub-component/subtype/variant of the other.
3. When in doubt, prefer FALSE. Keeping concepts separate preserves integrity.

OUTPUT FORMAT:
Output ONLY the final <verdicts>...</verdicts> XML block with no introductory or concluding text:
<verdicts>
<verdict>
<pair_index>1</pair_index>
<is_same_entity>true</is_same_entity>
<chosen_canonical_title>Exact Canonical Title</chosen_canonical_title>
<reason>Direct synonymous reference.</reason>
</verdict>
</verdicts>
"""

class MergeVerdict(BaseModel):
    pair_index: int = 1
    is_same_entity: bool = False
    chosen_canonical_title: str = ""
    reason: str = ""

def parse_xml_merge_verdicts(text: str) -> List[MergeVerdict]:
    """Parse <verdicts> blocks into MergeVerdict objects."""
    if not text:
        return []

    verdicts: List[MergeVerdict] = []
    blocks = re.findall(r"<verdict>(.*?)</verdict>", text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        idx_m = re.search(r"<pair_index>(\d+)</pair_index>", block, re.IGNORECASE)
        idx = int(idx_m.group(1)) if idx_m else len(verdicts) + 1

        same_m = re.search(r"<is_same_entity>(.*?)</is_same_entity>", block, re.IGNORECASE)
        is_same = False
        if same_m:
            is_same = same_m.group(1).strip().lower() in ("true", "1", "yes")

        canon_m = re.search(r"<chosen_canonical_title>(.*?)</chosen_canonical_title>", block, re.DOTALL | re.IGNORECASE)
        canon = canon_m.group(1).strip() if canon_m else ""

        reason_m = re.search(r"<reason>(.*?)</reason>", block, re.DOTALL | re.IGNORECASE)
        reason = reason_m.group(1).strip() if reason_m else ""

        verdicts.append(MergeVerdict(
            pair_index=idx,
            is_same_entity=is_same,
            chosen_canonical_title=canon,
            reason=reason
        ))

    # JSON fallback
    if not verdicts and "{" in text:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                for item in payload.get("pair_verdicts", []):
                    verdicts.append(MergeVerdict(
                        pair_index=item.get("pair_index", len(verdicts) + 1),
                        is_same_entity=bool(item.get("is_same_entity", False)),
                        chosen_canonical_title=item.get("chosen_canonical_title", ""),
                        reason=item.get("reason", "")
                    ))
        except Exception:
            pass

    return verdicts

def make_candidate_pairs_prompt(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    blocks = []
    for i, (c1, c2) in enumerate(pairs, 1):
        c1_context = get_context_hint(c1.get("description", ""), max_words=40)
        c2_context = get_context_hint(c2.get("description", ""), max_words=40)
        c1_aliases = ", ".join(c1.get("aliases", [])) or "None"
        c2_aliases = ", ".join(c2.get("aliases", [])) or "None"

        block = (
            f"PAIR #{i}:\n"
            f"  Concept 1: \"{c1.get('title')}\"\n"
            f"    Aliases: {c1_aliases}\n"
            f"    Context (~40 words): {c1_context}\n"
            f"  Concept 2: \"{c2.get('title')}\"\n"
            f"    Aliases: {c2_aliases}\n"
            f"    Context (~40 words): {c2_context}"
        )
        blocks.append(block)
    return "VERIFY THE FOLLOWING CONCEPT PAIRS (Decide if they denote the EXACT SAME entity):\n\n" + "\n\n".join(blocks)

async def verify_merge_batch(client: Any, batch: List[Tuple[Dict[str, Any], Dict[str, Any]]], settings: Any, args: Any) -> List[MergeVerdict]:
    user_prompt = make_candidate_pairs_prompt(batch)
    raw_text = await call_llm(
        client=client,
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MERGE_VERIFY},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=settings.max_tokens or 2048,
        args=args,
        task="judge",
        max_growth=0,
    )
    return parse_xml_merge_verdicts(raw_text)

# ── Main Pipeline Execution ───────────────────────────────────────────────────

def global_consolidate(domain_dir: Path, args: Any):
    consolidated_dir = domain_dir / "consolidated"
    concepts_dir = domain_dir / "concepts"
    log_file = domain_dir / "global_consolidation.json"

    if not consolidated_dir.exists():
        raise FileNotFoundError(f"No consolidated directory found at {consolidated_dir}. Run document consolidation first.")

    verify_stage_manifest(domain_dir, "consolidate-documents")

    # 1. Load All Consolidated Concepts from per-document folders
    all_raw_concepts = []
    doc_files = sorted(list(consolidated_dir.glob("*/concepts.json")))
    if not doc_files:
        raise RuntimeError(f"No consolidated concept files found under {consolidated_dir}")

    print(f"Loading concepts from {len(doc_files)} document folders...")
    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("concepts", []):
                t = c.get("title", "").strip()
                if not t:
                    continue
                all_raw_concepts.append(c)

    raw_concepts_count = len(all_raw_concepts)
    print(f"Total concepts loaded: {raw_concepts_count}")

    # Display initial word-count distribution before global merge
    print_concept_word_distribution(all_raw_concepts, title_label="STAGE 6: BEFORE GLOBAL CONSOLIDATION")

    # 2. Gather Candidate Merge Pairs (Exact Title, Lemma/Plural Variants, and Title-to-Alias alignments)
    # Group by canonical lemma key to find candidate exact and lemma variants across documents
    lemma_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        t = c.get("title", "").strip()
        if not t:
            continue
        key = canonical_title_key(t)
        lemma_groups.setdefault(key, []).append(c)

    candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_pair_keys: Set[Tuple[str, str]] = set()

    # (A) Candidate pairs from exact/lemma match across different documents/instances
    for key, group in lemma_groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, min(i + 5, len(group))):  # pair across distinct concepts in group
                    c1, c2 = group[i], group[j]
                    pk = tuple(sorted([c1.get("title", ""), c2.get("title", "")]))
                    if pk[0].lower() != pk[1].lower() and pk not in seen_pair_keys:
                        seen_pair_keys.add(pk)
                        candidate_pairs.append((c1, c2))

    # (B) Candidate pairs from Title-to-Alias alignments
    alias_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        for alias in c.get("aliases", []):
            clean_al = canonical_title_key(alias)
            if len(clean_al) > 2:
                alias_map.setdefault(clean_al, []).append(c)

    for c in all_raw_concepts:
        norm_title = canonical_title_key(c.get("title", ""))
        if norm_title in alias_map:
            for match in alias_map[norm_title]:
                if canonical_title_key(match.get("title", "")) != norm_title:
                    pk = tuple(sorted([c.get("title", ""), match.get("title", "")]))
                    if pk not in seen_pair_keys:
                        seen_pair_keys.add(pk)
                        candidate_pairs.append((c, match))

    print(f"Identified {len(candidate_pairs)} candidate merge pair(s) (exact/lemma/alias) with contextual descriptions.")

    # 3. LLM Verification of Candidate Pairs
    # F-09: Union-find is keyed strictly by exact normalized title (strip().casefold())
    # Lemma keys are used purely to generate candidate pairs; union happens ONLY on LLM approval
    uf = UnionFind()
    # N-10: Key canonical choices by member title rather than ephemeral union root
    title_to_chosen_canonical: Dict[str, str] = {}
    for c in all_raw_concepts:
        t = c.get("title", "").strip()
        if t:
            uf.add(t.casefold())

    approved_merges = 0
    if candidate_pairs and not getattr(args, "skip_alias_verification", False):
        settings = resolve_settings(args, STAGE)
        batch_size = 10
        batches = [candidate_pairs[i:i + batch_size] for i in range(0, len(candidate_pairs), batch_size)]

        async def run_verification():
            pool = pool_for_stage(args, settings, STAGE)

            async def handler(client, batch, server_index):
                return await verify_merge_batch(client, batch, settings, args)

            async def on_batch_result(batch, verdicts):
                nonlocal approved_merges
                if verdicts:
                    for verdict in verdicts:
                        idx = verdict.pair_index - 1
                        if 0 <= idx < len(batch):
                            c1, c2 = batch[idx]
                            if verdict.is_same_entity:
                                t1, t2 = c1["title"].strip(), c2["title"].strip()
                                uf.union(t1.casefold(), t2.casefold())
                                approved_merges += 1
                                # F-10 & N-10: Record LLM's chosen_canonical_title for both members
                                if verdict.chosen_canonical_title.strip():
                                    title_to_chosen_canonical[t1.casefold()] = verdict.chosen_canonical_title.strip()
                                    title_to_chosen_canonical[t2.casefold()] = verdict.chosen_canonical_title.strip()

            async with pool:
                res = await pool.map(
                    batches,
                    handler,
                    attempts=settings.attempts,
                    desc="LLM Verifying Candidate Merge Pairs",
                    budget=None,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=on_batch_result,
                )
                res.raise_if_failed(STAGE)

        asyncio.run(run_verification())
        print(f"LLM confirmed and approved {approved_merges} concept merges.")
    else:
        print("Skipping LLM candidate pair verification (or no candidates found).")

    # 4. Final Attribute, Provenance & Description Fusion
    final_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title:
            continue
        root = uf.find(title.casefold())
        final_groups.setdefault(root, []).append(c)

    global_concepts = []
    used_slugs: Set[str] = set()

    for root, concepts in tqdm(final_groups.items(), desc="Fusing global concept attributes"):
        # F-10 & N-10: Prefer LLM chosen canonical title from any group member > frequency > shortest
        chosen_title = None
        for c in concepts:
            c_norm = c.get("title", "").strip().casefold()
            if c_norm in title_to_chosen_canonical:
                chosen_title = title_to_chosen_canonical[c_norm]
                break

        if not chosen_title:
            title_counter = Counter(c.get("title", "").strip() for c in concepts)
            # Sort by frequency descending, then length ascending
            sorted_candidates = sorted(title_counter.items(), key=lambda kv: (-kv[1], len(kv[0])))
            chosen_title = sorted_candidates[0][0]

        # F-12: Assign collision-free stable slug once
        slug = generate_stable_slug(chosen_title, used_slugs)

        merged_aliases_set = set()
        merged_subtopics_set = set()
        global_source_chunks = set()

        for c in concepts:
            merged_aliases_set.add(c.get("title", "").strip())
            for a in c.get("aliases", []):
                if a.strip():
                    merged_aliases_set.add(a.strip())
            for s in c.get("subtopics", []):
                if s.strip():
                    merged_subtopics_set.add(s.strip())

            for sc in c.get("source_chunks", []):
                if sc.strip():
                    global_source_chunks.add(sc.strip())

        fused_desc = fuse_descriptions([c.get("description", "") for c in concepts])
        final_aliases = sorted([a for a in merged_aliases_set if a.casefold() != chosen_title.casefold()])
        final_subtopics = sorted([s for s in merged_subtopics_set if s.casefold() != chosen_title.casefold()])

        global_concepts.append({
            "title": chosen_title,
            "slug": slug,
            "aliases": final_aliases,
            "subtopics": final_subtopics,
            "description": fused_desc,
            "source_chunks": sorted(list(global_source_chunks)),
        })

    # Save summary and global concepts
    log_data = {
        "summary": {
            "canonical_docs_evaluated": len(doc_files),
            "raw_input_concepts": raw_concepts_count,
            "candidate_pairs_checked": len(candidate_pairs),
            "merges_approved_by_llm": approved_merges,
            "final_global_concepts": len(global_concepts),
        }
    }
    atomic_write_json(log_file, log_data)

    concepts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving global_concepts.json to {domain_dir}...")
    atomic_write_json(domain_dir / "global_concepts.json", {"concepts": global_concepts})

    # F-11 & F-12: Write individual concept files using deterministic, collision-free slugs
    # Clean previous files to prevent stale _1/_2 duplicates on re-runs
    print("Writing concepts to concepts/ by unique slug...")
    for old_f in concepts_dir.glob("*.json"):
        try:
            old_f.unlink()
        except OSError:
            pass

    for gc in global_concepts:
        slug = gc.get("slug")
        if not slug:
            slug = generate_stable_slug(gc["title"], used_slugs)
            gc["slug"] = slug
        file_path = concepts_dir / f"{slug}.json"
        atomic_write_json(file_path, gc)

    assert len(set(c["slug"] for c in global_concepts)) == len(global_concepts), "Slug collision detected!"

    write_stage_manifest(
        wiki_path=domain_dir,
        stage=STAGE,
        expected=len(global_concepts),
        completed=len(global_concepts),
        failed=0,
        raw_input_concepts=raw_concepts_count,
        approved_merges=approved_merges,
        duplicates_fused=raw_concepts_count - len(global_concepts),
    )

    # Print final word count distribution across global concepts
    print_concept_word_distribution(global_concepts, title_label="STAGE 6: AFTER GLOBAL CONSOLIDATION")

    print("\n============================================================")
    print("GLOBAL CONSOLIDATION COMPLETE")
    print("============================================================")
    print(f"Total Canonical Docs Merged:  {len(doc_files)}")
    print(f"Raw Input Concepts Loaded:    {raw_concepts_count}")
    print(f"Candidate Pairs Audited:      {len(candidate_pairs)}")
    print(f"Approved LLM Merges:          {approved_merges}")
    print(f"Final Global Concepts:        {len(global_concepts)} (fused {raw_concepts_count - len(global_concepts)} duplicates)")
    print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Consolidate concepts across documents using LLM-verified matches", include_resume=True)
    parser.add_argument("--skip_alias_verification", action="store_true",
                        help="Skip LLM verification of candidate pairs.")
    args = parser.parse_args()

    wiki_path = getattr(args, "wiki_path", None)
    if not wiki_path:
        raise ValueError("Missing required --wiki_path argument.")

    domain_dir = Path(wiki_path)
    global_consolidate(domain_dir, args)

if __name__ == "__main__":
    main()
