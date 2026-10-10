#!/usr/bin/env python3
"""
Global Knowledge Graph Consolidation Pipeline
---------------------------------------------------
1. Loads all concepts from `consolidated/canon_*/concepts.json`.
2. Exact-match UnionFind pass on lowercased titles and aliases.
3. Word-containment graph, balanced cuts of large components, LLM-reviewed merges.
4. Rebuild a character-trigram graph after word merges, cut and LLM-review again.
5. Complete Attribute Fusion: merges aliases, entities, descriptions, and preserves `source_chunks` provenances.
6. Target Stub Case-Normalization & Relation Repointing.
7. Generates detailed audit summary in `global_consolidation.json`.
8. Saves `global_concepts.json` and individual concept JSON files in `concepts/`.
"""

import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from openai import BadRequestError
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
from utils.atomic_io import atomic_write_json, read_json, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.text import snippet_description
from utils.layout import log_path, run_dir_for
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage

STAGE = "consolidate-global"
FUZZY_REVIEW_STRATEGY = "word-then-character-balanced-cut-ids-v2"
MAX_FUZZY_CLUSTER_CONCEPTS = 29  # strictly fewer than 30
MAX_FUZZY_BATCH_CONCEPTS = MAX_FUZZY_CLUSTER_CONCEPTS
MAX_FUZZY_BATCH_CHARS = 20000

# ── Data Schemas ─────────────────────────────────────────────────────────────
class MergeCluster(BaseModel):
    primary_title: str = Field(description="The chosen canonical primary title for this concept.")
    merged_titles: List[str] = Field(
        default_factory=list,
        description="Exact titles of other candidate concepts that represent the exact same entity and should be merged."
    )

class GlobalFuzzyMergeSchema(BaseModel):
    merge_clusters: List[MergeCluster] = Field(default_factory=list)

class FuzzyIdMerge(BaseModel):
    primary_id: int
    merged_ids: List[int]

class FuzzyIdResponse(BaseModel):
    merge_clusters: List[FuzzyIdMerge]

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
   - DO NOT merge a topic with studies, terminology, statistical artifacts, or
     evidence *about* that topic. Do not merge two different named products just
     because they are in the same category.
   - A two-candidate batch is NOT a request to merge those two candidates.
     If equivalence is uncertain, return {"merge_clusters": []}.
3. PRIMARY CONCEPT: Choose the best existing numbered candidate as the primary.
   Only return numbers shown in this request. Do not return names, aliases,
   new titles, or concepts from other requests. Each group needs two or more
   different candidate IDs; do not repeat an ID across groups.

OUTPUT FORMAT:
Return valid JSON matching this structure:
{
  "merge_clusters": [
    {"primary_id": 1, "merged_ids": [2, 3]}
  ]
}
If no concepts should be merged, return: {"merge_clusters": []}.
"""

# ── Fuzzy Candidate Graph Construction ───────────────────────────────────────
def balanced_cut_clusters(graph: nx.Graph, nodes: Set[int],
                          limit: int = MAX_FUZZY_CLUSTER_CONCEPTS,
                          fits_prompt=None) -> List[Set[int]]:
    """Recursively minimize cut-edge weight with balanced bisections.

    An unconstrained global min-cut usually peels off a weak leaf (1 vs N-1),
    which is not useful for bounding prompts. Kernighan–Lin is a deterministic
    heuristic for the *balanced* min-cut; exact balanced min-cut is NP-hard.
    Edge weights are within one metric, not a blend of incomparable scores.
    Edges cut by partitioning are intentionally not offered for review here.
    """
    if len(nodes) <= limit and (fits_prompt is None or fits_prompt(nodes)):
        return [nodes]
    if len(nodes) == 1:
        raise ValueError("One fuzzy concept cannot fit within the review prompt limit")
    subgraph = graph.subgraph(nodes)
    left, right = nx.community.kernighan_lin_bisection(subgraph, weight="weight", seed=0)
    return (balanced_cut_clusters(graph, set(left), limit, fits_prompt) +
            balanced_cut_clusters(graph, set(right), limit, fits_prompt))


def build_fuzzy_candidate_clusters(concepts: List[Dict[str, Any]], method: str) -> List[List[Dict[str, Any]]]:
    """Build ONE weighted similarity graph and cut its oversized components.

    Word containment and character trigrams are deliberately separate passes;
    each pass weights edges with only its own similarity score.
    """
    if method not in {"word", "character"}:
        raise ValueError(f"Unknown fuzzy candidate method: {method}")
    graph = nx.Graph()
    prepared = []
    for idx, concept in enumerate(concepts):
        graph.add_node(idx)
        names = [concept["title"]] + concept.get("aliases", [])
        features = ([get_clean_words(name) for name in names] if method == "word" else
                    [get_char_trigrams(name) for name in names])
        prepared.append(features)

    print(f"Building {method} similarity graph across {len(concepts)} concepts...")
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            best = 0.0
            for feature_a in prepared[i]:
                for feature_b in prepared[j]:
                    score = (word_containment_similarity(feature_a, feature_b) if method == "word"
                             else char_trigram_jaccard(feature_a, feature_b))
                    if score > best:
                        best = score
                    if best == 1.0:
                        break
                if best == 1.0:
                    break
            threshold = 0.60 if method == "word" else 0.50
            if best >= threshold:
                graph.add_edge(i, j, weight=best)

    clusters = []
    for component in nx.connected_components(graph):
        if len(component) < 2:
            continue
        def fits_prompt(nodes: Set[int]) -> bool:
            # The length limit is a second constraint on the same graph cut,
            # rather than an unrelated alphabetical partition afterwards.
            return len(make_fuzzy_review_batches([concepts[i] for i in sorted(nodes)])) == 1

        for partition in balanced_cut_clusters(graph, component, fits_prompt=fits_prompt):
            # A balanced cut can leave a partition with multiple disconnected
            # groups; never ask the model to compare unrelated nodes in it.
            for connected in nx.connected_components(graph.subgraph(partition)):
                if len(connected) > 1:
                    clusters.append([concepts[i] for i in sorted(connected)])

    print(f"Identified {len(clusters)} {method} candidate clusters (each < 30 concepts).")
    return clusters


def merged_fuzzy_views(concepts: List[Dict[str, Any]], uf: UnionFind) -> List[Dict[str, Any]]:
    """Rebuild stage-two nodes from stage-one LLM-approved unions."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for concept in concepts:
        groups.setdefault(uf.find(concept["title"]), []).append(concept)
    merged = []
    for root, members in groups.items():
        # UnionFind keeps the approved primary as the root when possible.
        title = next((c["title"] for c in members
                      if canonical_lemma_key(c["title"]) == root), members[0]["title"])
        aliases = sorted({name for c in members
                          for name in [c["title"]] + c.get("aliases", [])
                          if name.strip() and name.lower() != title.lower()})
        descriptions = [c["description"] for c in members if c.get("description")]
        merged.append({"title": title, "aliases": aliases,
                       "description": " ".join(descriptions)})
    return merged


# ── Async LLM Verification Runner ────────────────────────────────────────────
def format_fuzzy_review_batch(batch: List[Dict[str, Any]]) -> str:
    """Render a compact two-line numbered list, not verbose JSON input.

    Numbers are local to each batch; the model returns those IDs as JSON. The
    caller maps valid IDs back to exact stored titles, so display whitespace or
    a rewritten title can never create a new concept.
    """
    lines = []
    for number, concept in enumerate(batch, 1):
        title = " ".join(str(concept["title"]).split())
        aliases = [" ".join(str(a).split()) for a in concept["aliases"] if str(a).strip()]
        hint = " ".join(str(concept["description_hint"]).split())
        heading = f"{number}. {title}"
        if aliases:
            heading += f" (aliases: {'; '.join(aliases)})"
        lines.append(heading + (f"\n   {hint}" if hint else ""))
    return ("CANDIDATE CONCEPTS FOR REVIEW. Merge only equivalent concepts. "
            "Return only the numbered IDs below in your JSON answer:\n\n"
            + "\n\n".join(lines) + "\n")


def make_fuzzy_review_batches(cluster: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Bound both the concept count and the *rendered* prompt length.

    Exact-match fusion can accumulate many aliases, and graph components have
    no inherent size limit. Count characters on the actual text we send, rather than an
    approximate serialized representation. This is a character budget, not a
    substitute for the server's token budget (context errors are also split).
    """
    views = sorted(({
        "title": c.get("title"),
        "aliases": c.get("aliases", [])[:8],
        "description_hint": snippet_description(c.get("description", ""), max_words=30)[:400],
    } for c in cluster), key=lambda c: str(c["title"] or "").casefold())
    batches: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    for view in views:
        if len(format_fuzzy_review_batch([view])) > MAX_FUZZY_BATCH_CHARS:
            raise ValueError(f"Fuzzy candidate {view['title']!r} is too large for a review batch")
        if batch and (len(batch) >= MAX_FUZZY_BATCH_CONCEPTS or
                      len(format_fuzzy_review_batch(batch + [view])) > MAX_FUZZY_BATCH_CHARS):
            batches.append(batch)
            batch = []
        batch.append(view)
    if batch:
        batches.append(batch)
    return batches


def map_fuzzy_ids(reply: FuzzyIdResponse, batch: List[Dict[str, Any]]) -> List[MergeCluster]:
    """Translate model IDs to stored titles; never let fabricated IDs enter UF."""
    converted = []
    used_ids: Set[int] = set()
    for group in reply.merge_clusters:
        primary = group.primary_id
        if not 1 <= primary <= len(batch):
            print(f"WARNING: ignoring fuzzy merge with unknown primary ID {primary}; "
                  f"valid IDs are 1..{len(batch)}", flush=True)
            continue
        valid = []
        for candidate in group.merged_ids:
            if not 1 <= candidate <= len(batch):
                print(f"WARNING: ignoring unknown fuzzy merge ID {candidate}; "
                      f"valid IDs are 1..{len(batch)}", flush=True)
            elif candidate != primary and candidate not in valid:
                valid.append(candidate)
        if not valid:
            continue
        if used_ids.intersection([primary, *valid]):
            print(f"WARNING: ignoring overlapping fuzzy merge IDs "
                  f"{[primary, *valid]}; a candidate was already merged", flush=True)
            continue
        used_ids.update([primary, *valid])
        converted.append(MergeCluster(
            primary_title=batch[primary - 1]["title"],
            merged_titles=[batch[i - 1]["title"] for i in valid],
        ))
    return converted


def parse_fuzzy_id_response(text: str) -> FuzzyIdResponse:
    """Accept the requested wrapper or one unwrapped merge object.

    Small two-concept requests often produce {"primary_id":1,"merged_ids":[2]}
    despite the wrapper in the prompt. Both shapes carry the same decision;
    anything else still fails rather than masquerading as an empty verdict.
    """
    payload = json.loads(clean_json_string(text))
    if not isinstance(payload, dict):
        raise ValueError("Fuzzy review answer must be a JSON object")
    if "merge_clusters" not in payload and "primary_id" in payload:
        payload = {"merge_clusters": [payload]}
    elif isinstance(payload.get("merge_clusters"), dict):
        payload = {"merge_clusters": [payload["merge_clusters"]]}
    return FuzzyIdResponse.model_validate(payload)


async def verify_cluster_llm(client, cluster: List[Dict[str, Any]], settings) -> GlobalFuzzyMergeSchema:
    """Review a candidate cluster using bounded requests; all sub-batches must pass."""
    batches = make_fuzzy_review_batches(cluster)
    semaphore = asyncio.Semaphore(2)

    async def review_batch(batch: List[Dict[str, Any]]) -> List[MergeCluster]:
        async with semaphore:
            prompt = format_fuzzy_review_batch(batch)
            finish_reason = None
            try:
                response = await client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_FUZZY_MERGE},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=settings.max_tokens,
                    response_format={"type": "json_object"}
                )
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                if finish_reason == "length":
                    raise ValueError("Fuzzy review truncated at max_tokens")
                result = parse_fuzzy_id_response(choice.message.content)
                return map_fuzzy_ids(result, batch)
            except Exception as exc:
                too_long = (finish_reason == "length" or
                            isinstance(exc, BadRequestError) and
                            ("maximum context length" in str(exc).lower() or
                             "input_tokens" in str(exc).lower()))
                if not too_long or len(batch) == 1:
                    raise
        # Release the semaphore before child requests so recursive splitting
        # cannot deadlock. Never accept a partially reviewed cluster.
        mid = len(batch) // 2
        left, right = await asyncio.gather(review_batch(batch[:mid]),
                                           review_batch(batch[mid:]))
        return left + right

    results = await asyncio.gather(*(review_batch(batch) for batch in batches))
    return GlobalFuzzyMergeSchema(
        merge_clusters=[merge for part in results for merge in part]
    )

def fuzzy_review_key(cluster: List[Dict[str, Any]], method: str, settings) -> str:
    """Hash everything that affects the LLM verdict, including rendered inputs."""
    prompts = [format_fuzzy_review_batch(batch)
               for batch in make_fuzzy_review_batches(cluster)]
    payload = {
        "cache_format": 1, "strategy": FUZZY_REVIEW_STRATEGY,
        "method": method, "model": settings.model_name,
        "max_tokens": settings.max_tokens, "system": SYSTEM_PROMPT_FUZZY_MERGE,
        "prompts": prompts,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def valid_fuzzy_verdict(cluster: List[Dict[str, Any]], verdict: GlobalFuzzyMergeSchema) -> bool:
    """Never replay a checkpoint referring to titles outside its candidate group."""
    titles = {concept["title"] for concept in cluster}
    return all(merge.primary_title in titles and
               all(title in titles for title in merge.merged_titles)
               for merge in verdict.merge_clusters)


async def run_llm_fuzzy_pass(candidate_clusters: List[List[Dict[str, Any]]], args,
                             method: str = "word") -> Tuple[List[MergeCluster], List[Dict[str, Any]]]:
    """Review one metric's candidate groups, checkpointing each successful group.

    A failed group is never cached. A later attempt reuses successful decisions,
    including empty merge lists, and asks the model only about pending groups.
    Cache keys include the exact prompts, model, strategy, and pass name so a
    changed input cannot silently reuse an unrelated decision.
    """
    if not candidate_clusters:
        return [], []
    settings = resolve_settings(args, STAGE)
    cache_dir = run_dir_for(args) / "cache" / "consolidate-global" / method
    approved_merges: List[MergeCluster] = []
    llm_evaluations: List[Dict[str, Any]] = []
    pending: List[List[Dict[str, Any]]] = []
    pending_keys: Dict[int, str] = {}
    reused = 0

    def collect(cluster: List[Dict[str, Any]], verdict: GlobalFuzzyMergeSchema) -> None:
        llm_evaluations.append({
            "candidates_sent": [c.get("title", "Unknown") for c in cluster],
            "decisions": [m.model_dump() for m in verdict.merge_clusters],
        })
        approved_merges.extend(verdict.merge_clusters)

    for cluster in candidate_clusters:
        key = fuzzy_review_key(cluster, method, settings)
        checkpoint = read_json(cache_dir / f"{key}.json")
        if (isinstance(checkpoint, dict) and checkpoint.get("key") == key and
                isinstance(checkpoint.get("verdict"), dict) and
                isinstance(checkpoint["verdict"].get("merge_clusters"), list)):
            try:
                verdict = GlobalFuzzyMergeSchema.model_validate(checkpoint["verdict"])
                if valid_fuzzy_verdict(cluster, verdict):
                    collect(cluster, verdict)
                    reused += 1
                    continue
            except (KeyError, ValueError, TypeError):
                pass  # Corrupt or incompatible checkpoints are reviewed again.
        pending_keys[id(cluster)] = key
        pending.append(cluster)

    print(f"{method.capitalize()} fuzzy review: {len(candidate_clusters)} clusters, "
          f"{reused} cached verdicts, {len(pending)} pending requests.")
    if not pending:
        return approved_merges, llm_evaluations
    if not args.servers:
        raise RuntimeError("Fuzzy concept clusters require at least one configured LLM server.")

    print(f"Spreading clusters across {len(settings.servers)} server(s) at "
          f"{settings.concurrency_per_server} concurrent request(s) each "
          f"(item i -> server i % N).")
    debug = EventLog(log_path(run_dir_for(args), STAGE, "debug"), label=STAGE)
    budget = StageBudget(settings.budget_minutes)
    print(f"Per-item attempts: {settings.attempts} | Stage budget: "
          f"{settings.budget_minutes:g} min | Debug log: {debug.path}")
    failed_clusters: List[List[str]] = []

    async def handler(client, cluster, server_index):
        return await verify_cluster_llm(client, cluster, settings)

    async def record(cluster, verdict):
        # Commit a verdict before marking this task complete. An interruption or
        # later failure cannot lose results that were already accepted.
        key = pending_keys[id(cluster)]
        await asyncio.to_thread(atomic_write_json, cache_dir / f"{key}.json",
                                {"key": key, "verdict": verdict.model_dump()})
        collect(cluster, verdict)

    pool = pool_for_stage(args, settings, STAGE)
    try:
        async with pool:
            result = await pool.map(
                pending,
                handler,
                attempts=settings.attempts,
                desc=f"LLM {method.capitalize()} Fuzzy Verification",
                key_fn=lambda cl: "|".join(sorted(c.get("title", "") for c in cl)),
                budget=budget,
                debug=debug,
                attempt_timeout=settings.profile.attempt_timeout,
                on_result=record,
            )
    finally:
        debug.close()

    for failure in result.failures:
        cluster = pending[failure["index"]]
        failed_clusters.append([c.get("title", "Unknown") for c in cluster])
        llm_evaluations.append({"candidates_sent": failed_clusters[-1], "decisions": []})

    print("Requests served per server:")
    for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
        print(f"  {url:<40} {count}")
    if failed_clusters:
        raise RuntimeError(
            f"LLM {method} fuzzy review failed for {len(failed_clusters)} cluster(s) "
            f"after retries. Examples: {failed_clusters[:5]}"
        )
    print(f"LLM approved {len(approved_merges)} {method} fuzzy merges.")
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

    # 3. Optional fuzzy blocking & LLM review. Exact-match merging and all
    # downstream citation/relation fusion still run when fuzzy review is skipped.
    skip_fuzzy = getattr(args, "skip_fuzzy_review", False)
    candidate_clusters = []
    approved_fuzzy_merges = []
    llm_evaluations = []
    fuzzy_counts = {}
    if skip_fuzzy:
        print("Skipping global fuzzy LLM review; continuing with exact/alias matches only.")
    else:
        current_concepts = exact_concepts
        for method in ("word", "character"):
            clusters = build_fuzzy_candidate_clusters(current_concepts, method)
            approved, evaluations = asyncio.run(run_llm_fuzzy_pass(clusters, args, method))
            fuzzy_counts[method] = {"clusters": len(clusters), "merges": len(approved)}
            candidate_clusters.extend(clusters)
            approved_fuzzy_merges.extend(approved)
            llm_evaluations.extend({"method": method, **record} for record in evaluations)

            # Only LLM-approved unions from the word pass feed the character
            # pass. Neither similarity metric merges anything automatically.
            for merge in approved:
                primary = merge.primary_title.strip()
                decisions.record("fuzzy_merge", method=method, canonical=primary,
                                 merged=[title.strip() for title in merge.merged_titles])
                for title in merge.merged_titles:
                    uf.union(primary, title.strip())
            if method == "word":
                current_concepts = merged_fuzzy_views(current_concepts, uf)
    decisions.close()

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
            "fuzzy_passes": fuzzy_counts,
            "fuzzy_review_strategy": FUZZY_REVIEW_STRATEGY if not skip_fuzzy else None,
            "fuzzy_review_skipped": skip_fuzzy,
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

    print("\n==================================================")
    print("GLOBAL CONSOLIDATION COMPLETE")
    print("==================================================")
    print(f"Total Canonical Docs Merged:  {len(doc_files)}")
    print(f"Raw Input Concepts Loaded:    {raw_concepts_count}")
    print(f"After Exact-Match UnionFind:  {exact_count}")
    print(f"LLM Approved Fuzzy Merges:    {len(approved_fuzzy_merges)}")
    print(f"Final Global Concepts:        {len(global_concepts)}")
    print(f"Final Global Relations:       {total_global_relations}")
    print(f"Unresolved Stub Targets:      {len(sorted_unresolved_stubs)}")
    print("==================================================")
    print(f"Outputs written to:\n  - {domain_dir / 'global_concepts.json'}\n  - {concepts_dir}/\n  - {log_file}")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def main():
    """Parse the command line and run global consolidation.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Global Knowledge Graph Consolidation (Exact + Fuzzy Blocking + LLM Verification)")
    parser.add_argument("--skip_fuzzy_review", action="store_true",
                        help="Skip fuzzy candidate generation/LLM review; keep only exact-title and alias merges.")
    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain

    global_consolidate(domain_dir, args)

if __name__ == "__main__":
    main()
