#!/usr/bin/env python3
"""
Exact & Verified-Alias Global Concept Consolidation
---------------------------------------------------
Fuses concepts across canonical documents before final graph enrichment:

Phase 1: Deterministic Exact Title Consolidation
  - Normalizes titles using idempotent lemma/singularization rules.
  - Merges identical normalized titles across documents immediately via UnionFind.
  - No LLM calls needed (100% deterministic, high precision).

Phase 2: LLM-Verified Title-to-Alias Equivalence
  - Identifies candidate pairs where Concept A's title matches Concept B's alias
    (ignoring short ambiguous acronyms <= 3 chars).
  - Sends pairs with their titles, aliases, and 150-char contextual briefs to the LLM.
  - Only merges pairs verified by the model as referring to the exact same canonical entity.

Phase 3: Attribute & Provenance Fusion
  - Fuses source_chunks [canon_xxx::chunk_yyy], aliases, entities, and outbound relations.
  - Generates cited description paragraphs.
  - Exports domain/global_concepts.json and domain/concepts/<title>.json.
"""

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.json_repair import clean_json_string
from utils.layout import log_path, run_dir_for
from utils.llm_pool import pool_for_stage

STAGE = "consolidate-global"

# ── Stopwords and Scraping Boilerplate ──────────────────────────────────────
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
    'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't",
    'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn',
    "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
    'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most',
    'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
    'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's',
    'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so',
    'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

STRUCTURAL_BOILERPLATE_REGEXES = [
    r'^(table of contents|contents|outline of .*|overview of .*)$',
    r'^(references|see also|further reading|bibliography|notes|external links)$',
    r'^(history of .*|introduction to .*|timeline of .*)$',
    r'^(glossary of .*|index of .*|list of .*)$',
    r'^(main article|primary topic|disambiguation page)$',
]

def is_boilerplate_concept(title: str) -> bool:
    t = title.strip().lower()
    return any(re.match(pattern, t) for pattern in STRUCTURAL_BOILERPLATE_REGEXES)

def canonical_lemma_key(title: str) -> str:
    """Normalizes titles to catch typographical plurals, punctuation differences,
    and spacing variations (e.g. 'Hand Washing' vs 'Handwashing').
    Strictly idempotent: f(f(x)) == f(x).
    """
    t = re.sub(r'[-_]+', ' ', title.strip().lower())
    t = re.sub(r'[^\w\s]', '', t)
    tokens = t.split()
    normalized = []
    for tok in tokens:
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("es") and len(tok) > 4 and tok not in {"species", "series", "genes", "diabetes"}:
            if tok.endswith("sses") or tok.endswith("shes") or tok.endswith("ches") or tok.endswith("xes") or tok.endswith("zes"):
                tok = tok[:-2]
            elif tok.endswith("ses") and not tok.endswith("sses"):
                tok = tok[:-1]
            else:
                tok = tok[:-1]
        elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss") and tok not in {"this", "gas", "bias", "lens", "virus", "status", "mucus", "genus"}:
            tok = tok[:-1]
        normalized.append(tok)
    return " ".join(normalized)

class UnionFind:
    """Disjoint Set with iterative path compression and cycle prevention."""
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        key = canonical_lemma_key(item)
        if not key:
            return ""

        curr = key
        path = []
        visited = set()

        while curr in self.parent and self.parent[curr] != curr:
            if curr in visited:
                self.parent[curr] = curr
                break
            visited.add(curr)
            path.append(curr)
            curr = self.parent[curr]

        root = curr
        for node in path:
            self.parent[node] = root
        if root not in self.parent:
            self.parent[root] = root
        return root

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 and root2 and root1 != root2:
            self.parent[root2] = root1

# ── Schemas for Alias Verification ──────────────────────────────────────────
class AliasPairVerification(BaseModel):
    is_same_entity: bool = Field(description="True if both concepts refer to the exact same canonical entity.")
    chosen_canonical_title: str = Field(description="The preferred canonical title if merged, or empty string if separate.")
    reason: str = Field(description="Brief 1-sentence rationale.")

class BatchAliasVerificationSchema(BaseModel):
    pair_verdicts: List[AliasPairVerification] = Field(
        default_factory=list,
        description="Verdicts for each concept pair in the input batch."
    )

SYSTEM_PROMPT_ALIAS_VERIFY = """\
You are an expert technical knowledge graph consolidator.
You are given pairs of concepts where the Title of Concept 1 matches an Alias of Concept 2 (or vice versa).
Your task is to determine whether both concepts refer to the EXACT SAME real-world entity, process, tool, gene, condition, or technical subject.

CRITICAL RULES:
1. MERGE (is_same_entity: true): Only if both concepts denote the identical subject (e.g., 'Amyotrophic Lateral Sclerosis' and 'Lou Gehrig's Disease').
2. KEEP SEPARATE (is_same_entity: false): Do NOT merge if one is an acronym that is polysemous (e.g. 'AN' = Anorexia Nervosa vs Adult Neurogenesis), or if one is a component/subtype of the other (e.g. 'T Cell' vs 'Cell', 'Cornea' vs 'Retina').
3. When in doubt, prefer FALSE. Keeping concepts separate preserves graph integrity.

OUTPUT FORMAT:
Return valid JSON matching:
{
  "pair_verdicts": [
    {
      "is_same_entity": true,
      "chosen_canonical_title": "Concept Title A",
      "reason": "Direct synonymous reference."
    }
  ]
}
"""

def make_alias_pairs_prompt(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    blocks = []
    for i, (c1, c2) in enumerate(pairs, 1):
        c1_desc = " ".join(c1.get("description", "").split())[:150]
        c2_desc = " ".join(c2.get("description", "").split())[:150]
        c1_aliases = ", ".join(c1.get("aliases", [])) or "None"
        c2_aliases = ", ".join(c2.get("aliases", [])) or "None"
        
        block = (
            f"PAIR #{i}:\n"
            f"  Concept 1: \"{c1.get('title')}\"\n"
            f"    Aliases: {c1_aliases}\n"
            f"    Context: {c1_desc}\n"
            f"  Concept 2: \"{c2.get('title')}\"\n"
            f"    Aliases: {c2_aliases}\n"
            f"    Context: {c2_desc}"
        )
        blocks.append(block)
    return "VERIFY THE FOLLOWING CONCEPT PAIRS:\n\n" + "\n\n".join(blocks)

async def verify_alias_batch(client, batch: List[Tuple[Dict[str, Any], Dict[str, Any]]], settings):
    user_prompt = make_alias_pairs_prompt(batch)
    response = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ALIAS_VERIFY},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=settings.max_tokens,
        response_format={"type": "json_object"}
    )
    raw_text = response.choices[0].message.content
    payload = json.loads(clean_json_string(raw_text))
    return BatchAliasVerificationSchema.model_validate(payload)

# ── Main Pipeline Function ───────────────────────────────────────────────────
def global_consolidate(domain_dir: Path, args):
    consolidated_dir = domain_dir / "consolidated"
    concepts_dir = domain_dir / "concepts"
    log_file = domain_dir / "global_consolidation.json"
    decisions = EventLog(log_path(domain_dir, STAGE, "decisions"), label=STAGE)

    if not consolidated_dir.exists():
        raise FileNotFoundError(f"No consolidated directory found at {consolidated_dir}")

    # 1. Load All Raw Concepts from per-document consolidated folders
    all_raw_concepts = []
    doc_files = sorted(list(consolidated_dir.glob("canon_*/concepts.json")))
    if not doc_files:
        raise RuntimeError(f"No consolidated concept files found under {consolidated_dir}")

    print(f"Loading concepts from {len(doc_files)} canonical document files...")
    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("concepts", []):
                t = c.get("title", "").strip()
                if not t or is_boilerplate_concept(t):
                    continue
                all_raw_concepts.append(c)

    raw_concepts_count = len(all_raw_concepts)
    print(f"Total concepts loaded (after dropping scraping boilerplate): {raw_concepts_count}")

    # 2. Phase 1: Pure Exact Title Normalization (100% deterministic, NO LLM)
    # Strictly union identical normalized canonical titles across documents.
    # NEVER union aliases transitively in Phase 1!
    uf = UnionFind()
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if title:
            uf.find(title)

    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title:
            continue
        # Only identical normalized titles merge here
        uf.union(title, title)

    # Group raw concepts by their normalized title root
    title_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title:
            continue
        root = uf.find(title)
        title_groups.setdefault(root, []).append(c)

    # Build intermediate exact concepts
    exact_concepts: List[Dict[str, Any]] = []
    for root, members in title_groups.items():
        title_counter = Counter(c.get("title", "").strip() for c in members)
        chosen_title = title_counter.most_common(1)[0][0]
        
        merged_aliases = set()
        for c in members:
            merged_aliases.add(c.get("title", "").strip())
            for a in c.get("aliases", []):
                if a.strip():
                    merged_aliases.add(a.strip())
        merged_aliases.discard(chosen_title)

        all_desc_sentences = []
        for c in members:
            d = c.get("description", "").strip()
            if d:
                all_desc_sentences.append(d)

        exact_concepts.append({
            "title": chosen_title,
            "root": root,
            "aliases": sorted(list(merged_aliases)),
            "description": " ".join(all_desc_sentences[:3]),
            "members": members,
        })

    print(f"Concepts after Deterministic Exact Title Matching: {len(exact_concepts)} "
          f"(merged {raw_concepts_count - len(exact_concepts)} identical titles)")

    # 3. Phase 2: LLM-Verified Title-to-Alias Matching
    # Find candidate pairs where Concept A's normalized title matches Concept B's alias
    # Filter out short ambiguous acronyms (length <= 3) to prevent transitive black holes!
    alias_to_concept: Dict[str, List[Dict[str, Any]]] = {}
    for ec in exact_concepts:
        for alias in ec["aliases"]:
            clean_al = canonical_lemma_key(alias)
            if len(clean_al) > 3 and clean_al not in STOP_WORDS:
                alias_to_concept.setdefault(clean_al, []).append(ec)

    candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_pair_keys: Set[Tuple[str, str]] = set()

    for ec in exact_concepts:
        norm_title = ec["root"]
        if norm_title in alias_to_concept:
            for match in alias_to_concept[norm_title]:
                if match["root"] != ec["root"]:
                    pair_key = tuple(sorted([ec["root"], match["root"]]))
                    if pair_key not in seen_pair_keys:
                        seen_pair_keys.add(pair_key)
                        candidate_pairs.append((ec, match))

    print(f"Identified {len(candidate_pairs)} candidate title-to-alias pair(s) for verification.")

    approved_alias_merges = 0
    if candidate_pairs and not getattr(args, "skip_alias_verification", False):
        settings = resolve_settings(args, STAGE)
        batch_size = 10
        batches = [candidate_pairs[i:i + batch_size] for i in range(0, len(candidate_pairs), batch_size)]
        
        async def run_alias_verification():
            nonlocal approved_alias_merges
            pool = pool_for_stage(args, settings, STAGE)
            
            async def handler(client, batch, server_index):
                return await verify_alias_batch(client, batch, settings)

            async def on_batch_result(batch, res):
                nonlocal approved_alias_merges
                if res and hasattr(res, "pair_verdicts"):
                    for pair, verdict in zip(batch, res.pair_verdicts):
                        c1, c2 = pair
                        if verdict.is_same_entity:
                            decisions.record("alias_merge_approved",
                                             concept1=c1["title"], concept2=c2["title"],
                                             canonical=verdict.chosen_canonical_title,
                                             reason=verdict.reason)
                            uf.union(c1["title"], c2["title"])
                            approved_alias_merges += 1
                        else:
                            decisions.record("alias_merge_rejected",
                                             concept1=c1["title"], concept2=c2["title"],
                                             reason=verdict.reason)

            async with pool:
                await pool.map(
                    batches,
                    handler,
                    attempts=settings.attempts,
                    desc="Verifying Alias Candidate Pairs",
                    budget=None,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=on_batch_result,
                )

        asyncio.run(run_alias_verification())
        print(f"LLM approved {approved_alias_merges} title-to-alias merges.")
    else:
        print("Skipping alias LLM verification (or no candidates found).")

    decisions.close()

    # 4. Phase 3: Final Attribute, Citations & Relation Fusion
    final_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title:
            continue
        final_groups.setdefault(uf.find(title), []).append(c)

    global_concepts = []
    canonical_title_map: Dict[str, str] = {}

    for root, concepts in tqdm(final_groups.items(), desc="Fusing concept attributes"):
        title_counter = Counter(c.get("title", "").strip() for c in concepts)
        chosen_title = title_counter.most_common(1)[0][0]
        canonical_title_map[root] = chosen_title

        merged_aliases_set = set()
        merged_entities = set()
        global_source_chunks = set()
        seen_sentences = set()
        description_sentences = []
        raw_relations = []

        for c in concepts:
            merged_aliases_set.add(c.get("title", "").strip())
            for a in c.get("aliases", []):
                if a.strip():
                    merged_aliases_set.add(a.strip())

            for ent in c.get("entities", []):
                if ent.strip():
                    merged_entities.add(ent.strip())

            for sc in c.get("source_chunks", []):
                if sc.strip():
                    global_source_chunks.add(sc.strip())

        desc_by_chunks: Dict[Tuple[str, ...], List[str]] = {}
        for c in concepts:
            desc = c.get("description", "").strip()
            chunks_key = tuple(sorted(c.get("source_chunks", [])))
            if desc:
                for sent in re.split(r'(?<=[.!?])\s+', desc):
                    clean_sent = sent.strip()
                    if clean_sent and clean_sent.lower() not in seen_sentences:
                        seen_sentences.add(clean_sent.lower())
                        desc_by_chunks.setdefault(chunks_key, []).append(clean_sent)
                        description_sentences.append(clean_sent)

            concept_chunks = c.get("source_chunks", [])
            for rel in c.get("outbound_relations", []):
                if not rel.get("source_chunks"):
                    rel["source_chunks"] = concept_chunks
            raw_relations.extend(c.get("outbound_relations", []))

        cited_paragraphs = []
        for chunks_tuple, sents in desc_by_chunks.items():
            if not sents:
                continue
            para_text = " ".join(sents)
            if chunks_tuple:
                cites = " ".join(f"[{ch}]" for ch in chunks_tuple)
                para_text = f"{para_text} {cites}"
            cited_paragraphs.append(para_text)

        final_aliases = sorted([a for a in merged_aliases_set if a.lower() != chosen_title.lower()])

        global_concepts.append({
            "title": chosen_title,
            "aliases": final_aliases,
            "description": "\n\n".join(cited_paragraphs) if cited_paragraphs else " ".join(description_sentences),
            "entities": sorted(list(merged_entities)),
            "source_chunks": sorted(list(global_source_chunks)),
            "_raw_relations": raw_relations
        })

    # Repoint and fuse outbound relations
    stub_title_counters: Dict[str, Counter] = {}
    for gc in global_concepts:
        for rel in gc["_raw_relations"]:
            tgt = rel.get("target_concept", "").strip()
            if tgt:
                root_tgt = uf.find(tgt)
                if root_tgt not in canonical_title_map:
                    stub_title_counters.setdefault(root_tgt, Counter())[tgt] += 1

    for root, counter in stub_title_counters.items():
        if root not in canonical_title_map:
            canonical_title_map[root] = counter.most_common(1)[0][0]

    canonical_titles_set = {gc["title"] for gc in global_concepts}
    total_global_relations = 0
    unresolved_stubs_map: Dict[str, Set[str]] = {}

    for gc in tqdm(global_concepts, desc="Repointing outbound relations"):
        fused_relations: Dict[str, Dict[str, Set[str]]] = {}

        for rel in gc["_raw_relations"]:
            tgt = rel.get("target_concept", "").strip()
            rel_desc = rel.get("relation", "") or rel.get("description", "")
            rel_desc = rel_desc.strip()
            rel_chunks = rel.get("source_chunks", [])

            canonical_target = canonical_title_map.get(uf.find(tgt), tgt)

            if canonical_target.lower() == gc["title"].lower():
                continue

            if canonical_target not in canonical_titles_set:
                unresolved_stubs_map.setdefault(canonical_target, set()).update(rel_chunks)

            if canonical_target not in fused_relations:
                fused_relations[canonical_target] = {
                    "sentences": set(),
                    "source_chunks": set()
                }

            if rel_desc:
                for sent in re.split(r'(?<=[.!?])\s+', rel_desc):
                    if sent.strip():
                        fused_relations[canonical_target]["sentences"].add(sent.strip())
            for sc in rel_chunks:
                fused_relations[canonical_target]["source_chunks"].add(sc)

        final_relations = []
        for target, payload in fused_relations.items():
            final_relations.append({
                "target_concept": target,
                "relation": " ".join(sorted(payload["sentences"])),
                "source_chunks": sorted(list(payload["source_chunks"]))
            })

        gc["outbound_relations"] = final_relations
        total_global_relations += len(final_relations)
        del gc["_raw_relations"]

    sorted_unresolved_stubs = [
        {"stub_name": tgt, "frequency": len(sources), "source_chunks": sorted(list(sources))}
        for tgt, sources in unresolved_stubs_map.items()
    ]
    sorted_unresolved_stubs.sort(key=lambda x: x["frequency"], reverse=True)

    # Save summary and global concepts
    log_data = {
        "summary": {
            "canonical_docs_evaluated": len(doc_files),
            "raw_input_concepts": raw_concepts_count,
            "exact_matched_concepts": len(exact_concepts),
            "alias_merges_approved": approved_alias_merges,
            "final_global_concepts": len(global_concepts),
            "total_global_relations": total_global_relations,
            "total_unresolved_stubs": len(sorted_unresolved_stubs)
        },
        "unresolved_stubs": sorted_unresolved_stubs
    }
    atomic_write_json(log_file, log_data)

    concepts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving global_concepts.json to {domain_dir}...")
    atomic_write_json(domain_dir / "global_concepts.json", {"concepts": global_concepts})

    print("Saving individual concept files to concepts/...")
    for gc in global_concepts:
        safe_name = sanitize_filename(gc["title"]) or "unnamed_concept"
        file_path = concepts_dir / f"{safe_name}.json"
        counter = 1
        while file_path.exists():
            file_path = concepts_dir / f"{safe_name}_{counter}.json"
            counter += 1
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(gc, f, indent=2)

    print("\n==================================================")
    print("GLOBAL CONSOLIDATION (EXACT & VERIFIED) COMPLETE")
    print("==================================================")
    print(f"Total Canonical Docs Merged:  {len(doc_files)}")
    print(f"Raw Input Concepts Loaded:    {raw_concepts_count}")
    print(f"Deterministic Exact Merged:   {len(exact_concepts)}")
    print(f"Approved Alias Merges:        {approved_alias_merges}")
    print(f"Final Global Concepts:        {len(global_concepts)}")
    print(f"Total Outbound Relations:     {total_global_relations}")
    print("==================================================")

def main():
    parser = build_parser(stage=STAGE, description="Consolidate concepts across documents using exact and verified matches")
    parser.add_argument("--skip_alias_verification", action="store_true",
                        help="Skip LLM verification of title-to-alias candidate pairs.")
    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    global_consolidate(domain_dir, args)

if __name__ == "__main__":
    main()
