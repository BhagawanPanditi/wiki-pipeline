#!/usr/bin/env python3
"""
Global Knowledge Graph Consolidation Pipeline
---------------------------------------------------
1. Loads all concepts from `consolidated/canon_*/concepts.json`.
2. Exact-match UnionFind pass on lowercased titles and aliases.
3. Fuzzy Candidate Blocking (Word Containment + Char 3-Gram Jaccard) + Louvain Community Detection.
4. Async LLM Verification of candidate merge clusters.
5. Complete Attribute Fusion: merges aliases, entities, descriptions, and preserves `source_chunks` provenances.
6. Target Stub Case-Normalization & Relation Repointing.
7. Generates detailed audit summary in `global_consolidation.json`.
8. Saves `global_concepts.json` and individual concept JSON files in `concepts/`.
"""

import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
# nltk only supplies a stopword list for the fuzzy-similarity heuristics. It is
# optional: without it the module falls back to a small built-in list rather than
# refusing to import, so the pipeline runs on a bare environment.
try:
    import nltk
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover - depends on the environment
    nltk = None
    stopwords = None
from pydantic import BaseModel, Field
from tqdm import tqdm

# Silent setup for NLTK Stopwords
try:
    if nltk is None:
        raise ImportError("nltk is not installed")
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {"a", "an", "the", "and", "or", "in", "of", "on", "at", "to", "for", "with", "by", "is", "it", "as", "that", "this", "are", "was"}

# ── Server Configuration ─────────────────────────────────────────────────────
from pipeline_lib.atomic_io import atomic_write_json, sanitize_filename
from pipeline_lib.cli import build_parser, resolve_settings
from pipeline_lib.event_log import EventLog
from pipeline_lib.text import snippet_description
from pipeline_lib.layout import log_path, run_dir_for
from pipeline_lib.errors import StageBudget
from pipeline_lib.json_repair import clean_json_string
from pipeline_lib.llm_log import LLMCallLog
from pipeline_lib.llm_pool import ServerPool

STAGE = "global-consolidation"

# ── Data Schemas ─────────────────────────────────────────────────────────────
class MergeCluster(BaseModel):
    primary_title: str = Field(description="The chosen canonical primary title for this concept.")
    merged_titles: List[str] = Field(
        default_factory=list,
        description="Exact titles of other candidate concepts that represent the exact same entity and should be merged."
    )

class GlobalFuzzyMergeSchema(BaseModel):
    merge_clusters: List[MergeCluster] = Field(default_factory=list)

# ── Union-Find Disjoint Set ──────────────────────────────────────────────────
# Domain-agnostic web scraping artifacts and structural document boilerplate
BOILERPLATE_PATTERNS = [
    r"^table\s+of\s+contents",
    r"^references?\b",
    r"^bibliography\b",
    r"^cookie\s+policy",
    r"^privacy\s+policy",
    r"^terms\s+of\s+(service|use)",
    r"^navigation\s+menu",
    r"^site\s+map",
]

def is_boilerplate_concept(title: str) -> bool:
    """Returns True if the title matches universal web scraping artifacts or structural navigation boilerplate."""
    t = title.strip().lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def canonical_lemma_key(title: str) -> str:
    """
    Normalizes titles to catch typographical plurals, punctuation differences,
    and spacing variations (e.g. 'Hand Washing' vs 'Handwashing', 'Hallucinations' vs 'Hallucination').
    Guaranteed to be strictly idempotent: f(f(x)) == f(x).
    """
    # 1. Lowercase and replace hyphens/underscores with spaces
    t = re.sub(r'[-_]+', ' ', title.strip().lower())
    # 2. Remove punctuation
    t = re.sub(r'[^\w\s]', '', t)
    # 3. Collapse multiple whitespaces
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
        """Create an empty union-find structure."""
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the root key of the set that contains ``item``.

        Args:
            item: A concept title or alias.

        Returns:
            The root key for the item's canonical form, or ``""`` when the title is empty
            after normalisation. An unseen key becomes its own root. Paths are compressed
            on the way, using an iterative walk so long chains cannot exhaust the stack."""
        key = canonical_lemma_key(item)
        if not key:
            return ""

        # Iterative path-traversal (completely immune to RecursionError / cycles)
        curr = key
        path = []
        visited = set()

        while curr in self.parent and self.parent[curr] != curr:
            if curr in visited:
                # Cycle break guard
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
        """Join the sets that contain two titles or aliases.

        Args:
            item1: One title or alias.
            item2: Another title or alias.

        Returns:
            None. Does nothing when either item is empty or both already share a root."""
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 and root2 and root1 != root2:
            self.parent[root2] = root1

# ── Helper Utilities ─────────────────────────────────────────────────────────

def get_clean_words(text: str) -> List[str]:
    """Return the content words of a text, with stopwords removed.

    Args:
        text: Any text, typically a concept title.

    Returns:
        Lowercase alphanumeric tokens, minus the stopwords in ``STOP_WORDS``."""
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def get_char_trigrams(text: str) -> Set[str]:
    """Return the set of character trigrams of a text, padded with a space at each end.

    Args:
        text: Any text.

    Returns:
        The set of every three-character window of the lowercased, stripped text.
        Used for fuzzy similarity."""
    clean = f" {text.lower().strip()} "
    return {clean[i:i+3] for i in range(len(clean) - 2)}

def word_containment_similarity(words_a: List[str], words_b: List[str]) -> float:
    """Score how much the smaller of two word lists is contained in the other.

    Args:
        words_a: Content words of the first title.
        words_b: Content words of the second title.

    Returns:
        The overlap divided by the size of the smaller set. Returns ``0.0`` when either
        list is empty, when the smaller set has a single word, or when fewer than two
        words overlap. That guard stops a generic one-word match, such as "Cell"
        against "T Cell", from counting as a fuzzy duplicate."""
    set_a, set_b = set(words_a), set(words_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    min_len = min(len(set_a), len(set_b))
    
    # GUARD RAIL: Do not allow 1-word generic containment overlap (e.g. "Cell" vs "T Cell")
    if min_len <= 1 or intersection < 2:
        return 0.0
    
    return intersection / min_len

def char_trigram_jaccard(trigrams_a: Set[str], trigrams_b: Set[str]) -> float:
    """Compare two trigram sets by Jaccard similarity.

    Args:
        trigrams_a: Trigram set of the first title, from :func:`get_char_trigrams`.
        trigrams_b: Trigram set of the second title.

    Returns:
        Shared trigrams divided by all trigrams, or ``0.0`` if either set is empty."""
    if not trigrams_a or not trigrams_b:
        return 0.0
    intersection = len(trigrams_a.intersection(trigrams_b))
    union = len(trigrams_a.union(trigrams_b))
    return intersection / union

# ── System Prompt for LLM Fuzzy Verification ────────────────────────────────
SYSTEM_PROMPT_FUZZY_MERGE = """\
You are an expert technical knowledge graph consolidator.
You are given a small candidate cluster of concepts that share similar titles or character patterns.

YOUR MISSION:
Identify groups of concepts that refer to the EXACT SAME real-world entity, process, software tool, gene, or technical subject and should be merged.

CRITICAL MERGE RULES:
1. MERGE EXACT EQUIVALENTS & HIGH-OVERLAP TOPICS: 
   - Merge concepts if they refer to the exact same entity, algorithm, economic model, dataset, or technical subject under slightly different wording.
   - Merge identical methodological or analytical topics that differ only by phrasing (e.g. "Study Population and Demographics" vs "Participant Demographics", "Gradient Descent Optimization" vs "Gradient Descent Algorithm").
2. DO NOT MERGE DISTINCT SUB-TYPES OR OPPOSITES: 
   - DO NOT merge distinct sub-types (e.g. "Type 1 Diabetes" vs "Type 2 Diabetes", "Quicksort" vs "Mergesort", "Microeconomics" vs "Macroeconomics").
   - DO NOT merge a general parent concept with a specific, independent sub-component.
3. PRIMARY TITLE: Choose the cleanest, most canonical and comprehensive title from the cluster.

OUTPUT FORMAT:
Return valid JSON matching this structure:
{
  "merge_clusters": [
    {
      "primary_title": "Exact Chosen Canonical Title",
      "merged_titles": ["Alternative Title 1", "Alternative Title 2"]
    }
  ]
}
If no concepts should be merged, return: {"merge_clusters": []}.
"""

# ── Fuzzy Candidate Graph Construction ───────────────────────────────────────
def build_fuzzy_candidate_clusters(concepts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Builds a similarity graph via Word Containment & Char Trigrams, then extracts clusters."""
    G = nx.Graph()
    prepared = []

    for idx, c in enumerate(concepts):
        G.add_node(idx)
        all_names = [c["title"]] + c.get("aliases", [])
        words_list = [get_clean_words(name) for name in all_names]
        trigrams_list = [get_char_trigrams(name) for name in all_names]
        
        prepared.append({
            "idx": idx,
            "title": c["title"],
            "words": words_list,
            "trigrams": trigrams_list,
            "concept_ref": c
        })

    n = len(prepared)
    print(f"Performing pairwise fuzzy candidate blocking across {n} unique concepts...")

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = prepared[i], prepared[j]
            match_found = False

            # Quick token match for compound variations (e.g. 'handwashing' containing 'hand' and 'washing')
            w1_joined = "".join(c1["words"][0]) if c1["words"] else ""
            w2_joined = "".join(c2["words"][0]) if c2["words"] else ""
            if w1_joined and w2_joined and (w1_joined == w2_joined or (len(w1_joined) > 8 and w1_joined in w2_joined) or (len(w2_joined) > 8 and w2_joined in w1_joined)):
                match_found = True

            if not match_found:
                for w1, t1 in zip(c1["words"], c1["trigrams"]):
                    for w2, t2 in zip(c2["words"], c2["trigrams"]):
                        # Rule 1: Word Containment >= 0.60
                        if word_containment_similarity(w1, w2) >= 0.60:
                            match_found = True
                            break
                        # Rule 2: Character Trigram Jaccard >= 0.50 (catches spelling variations, hyphens)
                        if char_trigram_jaccard(t1, t2) >= 0.50:
                            match_found = True
                            break
                    if match_found:
                        break

            if match_found:
                G.add_edge(i, j)

    candidate_clusters = []
    for component in nx.connected_components(G):
        if len(component) <= 1:
            continue  # No merge review needed for isolated concepts

        subgraph = G.subgraph(component)
        if len(component) <= 15:
            cluster_concepts = [prepared[i]["concept_ref"] for i in component]
            candidate_clusters.append(cluster_concepts)
        else:
            # Hairball detected! Break down using Louvain or Greedy Modularity Community Detection
            try:
                communities = nx.community.louvain_communities(subgraph)
            except Exception:
                communities = nx.community.greedy_modularity_communities(subgraph)

            for comm in communities:
                if len(comm) > 1:
                    cluster_concepts = [prepared[i]["concept_ref"] for i in comm]
                    candidate_clusters.append(cluster_concepts)

    print(f"Identified {len(candidate_clusters)} fuzzy candidate clusters for LLM review.")
    return candidate_clusters

# ── Async LLM Verification Runner ────────────────────────────────────────────
async def verify_cluster_llm(client, cluster: List[Dict[str, Any]], settings) -> GlobalFuzzyMergeSchema:
    """One cluster verdict.

    Raises on any unusable response so ServerPool can classify and retry it.
    Returning None after 10 silent attempts used to mean "this cluster failed";
    an exhausted retry means the same thing now, except it is also recorded in
    the debug log and counted per server.
    """
    lightweight_concepts = []
    for idx, c in enumerate(cluster, 1):
        lightweight_concepts.append({
            "id": idx,
            "title": c.get("title"),
            "aliases": c.get("aliases", []),
            "description_snippet": snippet_description(c.get("description", ""), max_words=30)
        })

    user_prompt = f"CANDIDATE CONCEPT CLUSTER FOR REVIEW:\n{json.dumps(lightweight_concepts, indent=2)}\n"

    response = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FUZZY_MERGE},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=settings.max_tokens,
        response_format={"type": "json_object"}
    )
    raw_text = response.choices[0].message.content
    cleaned = clean_json_string(raw_text)
    return GlobalFuzzyMergeSchema.model_validate_json(cleaned)

async def run_llm_fuzzy_pass(candidate_clusters: List[List[Dict[str, Any]]], args) -> Tuple[List[MergeCluster], List[Dict[str, Any]]]:
    """Ask the LLM which fuzzy candidate clusters really are the same concept.

    Args:
        candidate_clusters: Groups of concept dicts that look alike.
        args: Parsed CLI namespace.

    Returns:
        A 2-tuple of the approved :class:`MergeCluster` list and one evaluation record
        per cluster, holding what was sent and what came back. Both are empty when
        there are no clusters.

    Raises:
        RuntimeError: If clusters exist but no LLM server is configured, or if any
            cluster fails after retries. Failed clusters are recorded before the error."""
    if not candidate_clusters:
        return [], []
    if not args.servers:
        raise RuntimeError("Fuzzy concept clusters require at least one configured LLM server.")

    settings = resolve_settings(args, STAGE)
    print(f"Evaluating {len(candidate_clusters)} fuzzy candidate clusters using LLM pool...")
    print(f"Spreading clusters across {len(settings.servers)} server(s) at "
          f"{settings.concurrency_per_server} concurrent request(s) each "
          f"(item i -> server i % N).")

    debug_path = log_path(run_dir_for(args), STAGE, "debug")
    debug = EventLog(debug_path, label=STAGE)
    budget = StageBudget(settings.budget_minutes)
    print(f"Per-item attempts: {settings.attempts} | Stage budget: "
          f"{settings.budget_minutes:g} min | Debug log: {debug.path}")

    # Accumulators shared by the result callback and the failure sweep below.
    approved_merges: List[MergeCluster] = []
    llm_evaluations: List[Dict[str, Any]] = []
    failed_clusters: List[List[str]] = []

    async def handler(client, cluster, server_index):
        """Verify one fuzzy candidate cluster through the pool.

        Args:
            client: The ``AsyncOpenAI`` client bound to one server.
            cluster: The candidate concepts proposed for merging.
            server_index: Index of the server handling this item, used only for
                per-server accounting.

        Returns:
            The parsed ``GlobalFuzzyMergeSchema`` verdict.

        Raises:
            Whatever :func:`verify_cluster_llm` raises, so the pool can classify
            and retry it; failures are collected after the map completes.
        """
        return await verify_cluster_llm(client, cluster, settings)

    async def record(cluster, result):
        """Append one successful verdict to the shared accumulators.

        Args:
            cluster: The candidate concepts that were sent to the model.
            result: The parsed ``GlobalFuzzyMergeSchema`` verdict.

        Returns:
            None. Mutates ``llm_evaluations`` (always, so the audit trail shows
            which candidates were sent) and ``approved_merges`` (only when the
            model actually approved a merge).
        """
        # Record the candidate titles sent, and the LLM's returned merge decisions
        llm_evaluations.append({
            "candidates_sent": [c.get("title", "Unknown") for c in cluster],
            "decisions": [m.model_dump() for m in result.merge_clusters] if result and result.merge_clusters else []
        })
        if result and result.merge_clusters:
            approved_merges.extend(result.merge_clusters)

    pool = ServerPool(
        settings.servers,
        concurrency_per_server=settings.concurrency_per_server,
        request_timeout=settings.profile.request_timeout,
        connect_timeout=settings.profile.connect_timeout,
        max_connections_slack=20,
        label=STAGE,
        llm_log=LLMCallLog.from_args(args, STAGE),
        failover=not getattr(args, "no_failover", False),
    )
    try:
        async with pool:
            result = await pool.map(
                candidate_clusters,
                handler,
                attempts=settings.attempts,
                desc="LLM Fuzzy Cluster Verification",
                key_fn=lambda cl: "|".join(sorted(c.get("title", "") for c in cl)),
                budget=budget,
                debug=debug,
                attempt_timeout=settings.profile.attempt_timeout,
                on_result=record,
            )
    finally:
        debug.close()

    # A cluster that never produced a verdict is still logged with empty
    # decisions, exactly as the old worker did, and still fails the stage.
    for failure in result.failures:
        cluster = candidate_clusters[failure["index"]]
        failed_clusters.append([c.get("title", "Unknown") for c in cluster])
        llm_evaluations.append({
            "candidates_sent": [c.get("title", "Unknown") for c in cluster],
            "decisions": []
        })

    print("Requests served per server:")
    for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
        print(f"  {url:<40} {count}")

    if failed_clusters:
        raise RuntimeError(
            f"LLM fuzzy review failed for {len(failed_clusters)} cluster(s) after retries. "
            f"Examples: {failed_clusters[:5]}"
        )

    print(f"LLM approved {len(approved_merges)} fuzzy merge clusters.")
    return approved_merges, llm_evaluations

# ── Core Pipeline Logic ──────────────────────────────────────────────────────
def global_consolidate(domain_dir: Path, args):
    """Merge the per-document concepts into one global graph.

    Steps: load every ``consolidated/canon_*/concepts.json``, drop boilerplate titles,
    merge exact title matches with union-find, find fuzzy candidates and have the LLM
    confirm them, then fuse each group's aliases, entities, descriptions and citations
    into one concept. Every exact merge and every approved fuzzy merge is recorded in
    the decision log.

    Args:
        domain_dir: Run directory.
        args: Parsed CLI namespace.

    Returns:
        None. Writes ``global_concepts.json`` and the audit summary
        ``global_consolidation.json``.

    Raises:
        FileNotFoundError: If the ``consolidated/`` folder does not exist.
        RuntimeError: If no consolidated concepts exist, or if LLM review fails."""
    consolidated_dir = domain_dir / "consolidated"
    concepts_dir = domain_dir / "concepts"
    log_file = domain_dir / "global_consolidation.json"
    decisions = EventLog(log_path(domain_dir, STAGE, "decisions"), label=STAGE)

    if not consolidated_dir.exists():
        raise FileNotFoundError(f"No consolidated directory found at {consolidated_dir}")

    # 1. Load All Raw Concepts
    all_raw_concepts = []
    doc_files = list(consolidated_dir.glob("canon_*/concepts.json"))
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
    print(f"Total concepts loaded (after dropping structural scraping boilerplate): {raw_concepts_count}")

    # 2. Phase 1: Exact Match Union-Find
    uf = UnionFind()
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title: continue
        uf.find(title)
        for alias in c.get("aliases", []):
            if alias.strip():
                uf.union(title, alias)

    # Temporary Grouping for Intermediate Exact Concepts
    exact_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title: continue
        exact_groups.setdefault(uf.find(title), []).append(c)

    exact_concepts = []
    for root, concepts in exact_groups.items():
        title_counter = Counter(c.get("title", "").strip() for c in concepts)
        chosen_title = title_counter.most_common(1)[0][0]
        merged_titles = sorted({c.get("title", "").strip() for c in concepts} - {chosen_title})
        if merged_titles:
            decisions.record("exact_merge", canonical=chosen_title, merged=merged_titles,
                             members=len(concepts))
        
        all_aliases = set()
        all_descs = []
        for c in concepts:
            all_aliases.add(c.get("title", "").strip())
            for a in c.get("aliases", []): all_aliases.add(a.strip())
            if c.get("description", "").strip(): all_descs.append(c["description"].strip())

        exact_concepts.append({
            "title": chosen_title,
            "aliases": sorted([a for a in all_aliases if a.lower() != chosen_title.lower()]),
            "description": " ".join(all_descs)
        })

    exact_count = len(exact_concepts)
    print(f"Concepts after Exact-Match UnionFind: {exact_count}")

    # 3. Phase 2 & 3: Fuzzy Blocking & LLM Verification
    candidate_clusters = build_fuzzy_candidate_clusters(exact_concepts)
    approved_fuzzy_merges, llm_evaluations = asyncio.run(run_llm_fuzzy_pass(candidate_clusters, args))

    # Apply LLM Approved Fuzzy Merges to UnionFind
    for cluster in approved_fuzzy_merges:
        primary = cluster.primary_title.strip()
        decisions.record("fuzzy_merge", canonical=primary,
                         merged=[m.strip() for m in cluster.merged_titles])
        uf.find(primary)
        for merged in cluster.merged_titles:
            uf.union(primary, merged.strip())

    # 4. Phase 4: Final Comprehensive Attribute & Citation Fusion
    final_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title: continue
        final_groups.setdefault(uf.find(title), []).append(c)

    global_concepts = []
    canonical_title_map: Dict[str, str] = {}  # Map UF Root -> Canonical Title

    for root, concepts in tqdm(final_groups.items(), desc="Merging concept attributes & provenances"):
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
            # Merge Aliases
            merged_aliases_set.add(c.get("title", "").strip())
            for a in c.get("aliases", []):
                if a.strip(): merged_aliases_set.add(a.strip())

            # Merge Entities
            for ent in c.get("entities", []):
                if ent.strip(): merged_entities.add(ent.strip())

            # Merge Source Chunks (Citations)
            for sc in c.get("source_chunks", []):
                if sc.strip(): global_source_chunks.add(sc.strip())

        # Group description sentences by source chunks to form cited paragraphs
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

        # Format paragraphs with inline citations
        cited_paragraphs = []
        for chunks_tuple, sents in desc_by_chunks.items():
            if not sents: continue
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

    # 5. Phase 5: Stub Normalization, Relation Repointing & Fusion
    # Register all stub target titles in UnionFind so casing mismatches map together
    stub_title_counters: Dict[str, Counter] = {}
    for gc in global_concepts:
        for rel in gc["_raw_relations"]:
            tgt = rel.get("target_concept", "").strip()
            if tgt:
                root_tgt = uf.find(tgt)
                if root_tgt not in canonical_title_map:
                    stub_title_counters.setdefault(root_tgt, Counter())[tgt] += 1

    # Assign canonical display name for stub roots
    for root, counter in stub_title_counters.items():
        if root not in canonical_title_map:
            canonical_title_map[root] = counter.most_common(1)[0][0]

    total_global_relations = 0
    canonical_titles_set = {gc["title"] for gc in global_concepts}
    unresolved_stubs_map: Dict[str, Set[str]] = {}

    for gc in tqdm(global_concepts, desc="Repointing & fusing relations"):
        fused_relations: Dict[str, Dict[str, Set[str]]] = {}

        for rel in gc["_raw_relations"]:
            tgt = rel.get("target_concept", "").strip()
            if not tgt: continue

            root_tgt = uf.find(tgt)
            resolved_target = canonical_title_map[root_tgt]

            # Prevent self-loops
            if resolved_target.lower() == gc["title"].lower():
                continue

            fused = fused_relations.setdefault(resolved_target, {"sentences": set(), "source_chunks": set()})

            rel_desc = rel.get("relation", "").strip()
            if rel_desc:
                for sent in re.split(r'(?<=[.!?])\s+', rel_desc):
                    if sent.strip(): fused["sentences"].add(sent.strip())

            for sc in rel.get("source_chunks", []):
                if sc.strip(): fused["source_chunks"].add(sc.strip())

            if resolved_target not in canonical_titles_set:
                unresolved_stubs_map.setdefault(resolved_target, set()).update(fused["source_chunks"])

        gc["outbound_relations"] = []
        for target, data in fused_relations.items():
            gc["outbound_relations"].append({
                "target_concept": target,
                "relation": " ".join(sorted(list(data["sentences"]))),
                "source_chunks": sorted(list(data["source_chunks"]))
            })
            total_global_relations += 1

        del gc["_raw_relations"]

    # 6. Global Audit Logging (New requirements: concepts before/after, stubs)
    print(f"Writing global audit summary to {log_file}...")
    sorted_unresolved_stubs = [
        {"stub_name": tgt, "frequency": len(sources), "source_chunks": sorted(list(sources))}
        for tgt, sources in unresolved_stubs_map.items()
    ]
    sorted_unresolved_stubs.sort(key=lambda x: x["frequency"], reverse=True)

    log_data = {
        "global_audit_summary": {
            "raw_concepts_loaded": raw_concepts_count,
            "exact_merged_concepts": exact_count,
            "fuzzy_candidate_clusters_evaluated": len(candidate_clusters),
            "llm_fuzzy_merges_applied": len(approved_fuzzy_merges),
            "final_global_concepts": len(global_concepts),
            "total_global_relations": total_global_relations,
            "total_unresolved_stubs": len(sorted_unresolved_stubs)
        },
        "llm_fuzzy_evaluations": llm_evaluations,  # <--- Added the new tracking data here
        "concepts_before": [c.get("title", "") for c in all_raw_concepts],
        "concepts_after": [gc["title"] for gc in global_concepts],
        "unresolved_stubs": sorted_unresolved_stubs
    }

    atomic_write_json(log_file, log_data)

    # 7. Output Results directly to domain root and `concepts/`
    concepts_dir.mkdir(parents=True, exist_ok=True)

    print("Saving global_concepts.json...")
    atomic_write_json(domain_dir / "global_concepts.json", {"concepts": global_concepts})

    print("Saving individual concept files...")
    for gc in global_concepts:
        safe_name = sanitize_filename(gc["title"])
        if not safe_name: safe_name = "unnamed_concept"

        file_path = concepts_dir / f"{safe_name}.json"
        counter = 1
        while file_path.exists():
            file_path = concepts_dir / f"{safe_name}_{counter}.json"
            counter += 1

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(gc, f, indent=2)

    print(f"\n==================================================")
    print(f"GLOBAL CONSOLIDATION COMPLETE")
    print(f"==================================================")
    print(f"Total Canonical Docs Merged:  {len(doc_files)}")
    print(f"Raw Input Concepts Loaded:    {raw_concepts_count}")
    print(f"After Exact-Match UnionFind:  {exact_count}")
    print(f"LLM Approved Fuzzy Merges:    {len(approved_fuzzy_merges)}")
    print(f"Final Global Concepts:        {len(global_concepts)}")
    print(f"Final Global Relations:       {total_global_relations}")
    print(f"Unresolved Stub Targets:      {len(sorted_unresolved_stubs)}")
    print(f"==================================================")
    print(f"Outputs written to:\n  - {domain_dir / 'global_concepts.json'}\n  - {concepts_dir}/\n  - {log_file}")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def main():
    """Parse the command line and run global consolidation.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Global Knowledge Graph Consolidation (Exact + Fuzzy Blocking + LLM Verification)")
    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain

    global_consolidate(domain_dir, args)

if __name__ == "__main__":
    main()
