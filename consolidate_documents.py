#!/usr/bin/env python3
"""Stage 5: Document-Level Concept Consolidation.
----------------------------------------------
Deduplicates and merges concept mentions within each document across its multiple
chunks, consolidating descriptions and tracking chunk citations.

Usage:
    python3 consolidate_documents.py --wiki_path /path/to/wiki
"""


import asyncio
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import BadRequestError
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.atomic_io import atomic_write_text
from utils.checkpoint import load_bool_log, save_log_async
from utils.cli import build_parser, resolve_settings
from utils.concept_ops import fuse_descriptions
from utils.corpus import chunking_enabled
from utils.errors import StageBudget
from utils.lemma import canonical_title_key
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest
from utils.unionfind import UnionFind


def union_aliases(alias_lists: Iterable[List[str]]) -> List[str]:
    """Union lists of aliases, deduplicating case-insensitively while preserving clean display."""
    seen = set()
    result = []
    for alist in alias_lists:
        for a in alist:
            norm = a.strip().casefold()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(a.strip())
    return result

STAGE = "consolidate-documents"
MAX_MERGE_BATCH_CHARS = 12000
MAX_MERGE_BATCH_CONCEPTS = 40
CALL_TIMEOUT_S = 120.0
CALL_RETRIES = 2

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class MergeCluster(BaseModel):
    primary_title: str = Field(description="The canonical primary title for this concept.")
    merged_titles: List[str] = Field(
        default_factory=list,
        description="Exact titles of other concepts in the input list that represent the same entity and should be merged into primary_title."
    )

class DocumentConsolidationSchema(BaseModel):
    merge_clusters: List[MergeCluster] = Field(default_factory=list)

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_CONSOLIDATE = """\
You are an expert technical knowledge graph consolidator.
You are given a list of candidate concepts extracted across multiple chunks of the SAME document.
Because these concepts were extracted independently per chunk, many represent the EXACT SAME subject, entity, item, process, condition, or topic under slightly different titles or acronyms.

YOUR MISSION:
Identify groups of concepts that refer to the exact same core concept/subject within this document and should be merged together.

CRITICAL RULES:
1. MERGE CRITERIA: Merge concepts ONLY if they refer to the exact same canonical subject, entity, item, process, acronym, or topic.
2. DO NOT MERGE DISTINCT CONCEPTS: Do NOT merge distinct sub-components, related but separate concepts, or parent/child hierarchy terms. If two entries differ only by a qualifier that changes meaning (Type I / Type II, Part 1 / Part 2, version numbers, before/after), do NOT merge them.
3. PRIMARY TITLE SELECTION: Pick the most standard, canonical, explicit title as `primary_title`.
4. SCOPE: Return only clusters of two or more titles. Titles in `<merged_titles>` must be copied EXACTLY as written in the candidate list.

OUTPUT FORMAT:
Output ONLY the <merge_clusters>...</merge_clusters> XML block:
<merge_clusters>
<cluster>
<primary_title>Exact Chosen Canonical Title</primary_title>
<merged_titles>
- Alternative Title 1
- Alternative Title 2
</merged_titles>
</cluster>
</merge_clusters>

If no concepts need to be merged, output:
<merge_clusters>
</merge_clusters>
"""

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_merge_clusters(text: str) -> List[MergeCluster]:
    """Parse <merge_clusters> blocks into MergeCluster objects directly."""
    if not text:
        return []

    clusters: List[MergeCluster] = []
    # Match each <cluster> block
    cluster_blocks = re.findall(r"<cluster>(.*?)</cluster>", text, re.DOTALL | re.IGNORECASE)
    for block in cluster_blocks:
        title_m = re.search(r"<primary_title>(.*?)</primary_title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m or not title_m.group(1).strip():
            continue
        primary_title = title_m.group(1).strip()

        merged_titles = []
        merged_m = re.search(r"<merged_titles>(.*?)</merged_titles>", block, re.DOTALL | re.IGNORECASE)
        if merged_m:
            lines = merged_m.group(1).strip().splitlines()
            for line in lines:
                cleaned = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
                if cleaned and cleaned.lower() != primary_title.lower():
                    merged_titles.append(cleaned)

        clusters.append(MergeCluster(primary_title=primary_title, merged_titles=merged_titles))

    # JSON fallback in case model emitted JSON
    if not clusters and "{" in text:
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                if "merge_clusters" in payload:
                    schema = DocumentConsolidationSchema.model_validate(payload)
                    return schema.merge_clusters
        except Exception:
            pass

    return clusters

# ── Word Count Thin Concept Statistics Helper ─────────────────────────────────

def print_concept_word_distribution(concepts: List[Dict[str, Any]], title_label: str = "CONCEPTS"):
    """Compute and print word count distribution statistics for concept descriptions."""
    if not concepts:
        print(f"\n[{title_label}] No concepts found.")
        return

    word_counts = []
    for c in concepts:
        desc = c.get("description", "")
        # Count words
        words = len(re.findall(r"\b\w+\b", desc))
        word_counts.append(words)

    bins = [
        ("0 - 10 words", lambda w: 0 <= w <= 10),
        ("11 - 20 words", lambda w: 11 <= w <= 20),
        ("21 - 30 words", lambda w: 21 <= w <= 30),
        ("31 - 50 words", lambda w: 31 <= w <= 50),
        ("51 - 100 words", lambda w: 51 <= w <= 100),
        ("101 - 200 words", lambda w: 101 <= w <= 200),
        ("201 - 350 words", lambda w: 201 <= w <= 350),
        ("351 - 500 words", lambda w: 351 <= w <= 500),
        ("501 - 750 words", lambda w: 501 <= w <= 750),
        ("751 - 1000 words", lambda w: 751 <= w <= 1000),
        ("1001+ words", lambda w: w > 1000),
    ]

    bin_counts = {label: 0 for label, _ in bins}
    for count in word_counts:
        for label, fn in bins:
            if fn(count):
                bin_counts[label] += 1
                break

    total = len(word_counts)
    avg_words = sum(word_counts) / total if total else 0
    thin_30 = sum(1 for w in word_counts if w <= 30)

    print("\n============================================================")
    print(f"WORD-COUNT DISTRIBUTION [{title_label}] (Total Concepts: {total})")
    print(f"Average Words per Concept: {avg_words:.1f} | Thin Concepts (<= 30 words): {thin_30} ({thin_30 / total * 100:.1f}%)")
    print("------------------------------------------------------------")
    print(f"{'Word Count Range':<20} {'Count':<10} {'Percentage':<12}")
    print("------------------------------------------------------------")
    for label, _ in bins:
        count = bin_counts[label]
        pct = (count / total * 100) if total else 0.0
        bar = "█" * int(pct / 4)
        print(f"{label:<20} {count:<10} {pct:>5.1f}%   {bar}")
    print("============================================================\n")

# ── Batching Utilities ────────────────────────────────────────────────────────

def make_merge_batches(concepts: List[Dict[str, Any]], overlap: int = 10) -> List[List[Dict[str, Any]]]:
    """Group concept title-stubs into token-bounded batches with overlap, sorted by lemma key."""
    # F-13: Sort by lemma key then title so near-duplicates land in the same or adjacent batches
    ordered = sorted(concepts, key=lambda c: (canonical_title_key(str(c.get("title") or "")), str(c.get("title") or "").casefold()))
    batches: List[List[Dict[str, Any]]] = []
    
    i = 0
    while i < len(ordered):
        batch: List[Dict[str, Any]] = []
        size = 2
        j = i
        while j < len(ordered):
            concept = ordered[j]
            item_size = len(json.dumps(concept, ensure_ascii=False, separators=(",", ":"))) + 1
            if batch and (len(batch) >= MAX_MERGE_BATCH_CONCEPTS or size + item_size > MAX_MERGE_BATCH_CHARS):
                break
            batch.append(concept)
            size += item_size
            j += 1
        
        batches.append(batch)
        if j >= len(ordered):
            break
        # Advance with overlap so boundary items are compared together
        step = max(1, len(batch) - overlap)
        i += step

    return batches

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiDocumentConsolidatorPipeline:
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")

        self.base_wiki_dir = Path(wiki_path)
        verify_stage_manifest(self.base_wiki_dir, "refine-concepts")
        self.staging_dir = self.base_wiki_dir / "staging"
        self.consolidated_dir = self.base_wiki_dir / "consolidated"
        self.log_file = self.base_wiki_dir / "doc_consolidation_log.json"

        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory missing at {self.staging_dir}. Run extraction first.")

        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.log_data = load_bool_log(self.log_file, corrupt_message="Log corrupted. Initializing fresh.")
        self.settings = resolve_settings(args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.flush_count = 0
        self.log_lock = asyncio.Lock()

    def prepare_document_items(self) -> List[Dict[str, Any]]:
        """Collect all canonical documents and their raw chunk concepts."""
        print(f"Scanning staging directory: {self.staging_dir}...")
        doc_dirs = sorted([d for d in self.staging_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        doc_items = []

        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            chunk_dirs = sorted([c for c in doc_dir.iterdir() if c.is_dir() and c.name.startswith("chunk_")])
            raw_concepts = []

            for chunk_dir in chunk_dirs:
                concepts_file = chunk_dir / "concepts.json"
                chunk_id = chunk_dir.name

                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for concept in data.get("concepts", []):
                                if isinstance(concept, dict) and concept.get("title"):
                                    c_copy = dict(concept)
                                    c_copy["source_chunk"] = f"{doc_id}::{chunk_id}"
                                    raw_concepts.append(c_copy)
                    except Exception:
                        continue

            if raw_concepts:
                doc_items.append({
                    "doc_id": doc_id,
                    "raw_concepts": raw_concepts,
                    "output_dir": self.consolidated_dir / doc_id,
                })

        return doc_items

    async def get_merge_clusters(self, client: Any, item: Dict[str, Any], server_index: int) -> DocumentConsolidationSchema:
        doc_id = item["doc_id"]
        raw_concepts = item["raw_concepts"]

        # Deduplicate exact case-insensitive titles before constructing LLM prompts
        by_key: Dict[str, Dict[str, Any]] = {}
        for c in raw_concepts:
            title = (c.get("title") or "").strip()
            if not title:
                continue
            k = title.casefold()
            entry = by_key.setdefault(k, {
                "title": title,
                "aliases": [],
                "description_hint": " ".join((c.get("description") or "").split()[:12])[:200],
            })
            for a in c.get("aliases", []):
                if a and a not in entry["aliases"] and len(entry["aliases"]) < 6:
                    entry["aliases"].append(a)

        lightweight_concepts = list(by_key.values())

        batches = make_merge_batches(lightweight_concepts)
        # Allow concurrent processing of batches within a document to saturate server capacity
        batch_concurrency = max(1, getattr(self.args, "batch_concurrency", 4))
        semaphore = asyncio.Semaphore(batch_concurrency)
        batch_pbar = None
        if len(batches) > 1:
            batch_pbar = tqdm(total=len(batches), desc=f"  [doc {doc_id[:12]}] {len(raw_concepts)} concepts", leave=False)

        def _batch_cache_path(batch: List[Dict[str, Any]]) -> Path:
            h = hashlib.sha1("\n".join(str(c.get("title", "")) for c in batch).encode("utf-8")).hexdigest()[:16]
            return self.consolidated_dir / ".cache" / f"{doc_id}_{h}.json"

        def format_batch_for_prompt(batch_concepts: List[Dict[str, Any]]) -> str:
            lines = []
            for i, c in enumerate(batch_concepts, 1):
                title = c.get("title", "")
                aliases = ", ".join(c.get("aliases", []))
                hint = c.get("description_hint", "").strip()
                item_lines = [f"{i}. Title: \"{title}\""]
                if aliases:
                    item_lines.append(f"   Aliases: {aliases}")
                if hint:
                    item_lines.append(f"   Context: {hint}")
                lines.append("\n".join(item_lines))
            return "\n\n".join(lines)

        async def merge_batch(batch: List[Dict[str, Any]]) -> List[MergeCluster]:
            cache_path = _batch_cache_path(batch)
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                    clusters = [MergeCluster(**c) for c in data]
                    if batch_pbar is not None:
                        batch_pbar.update(1)
                    return clusters
                except Exception:
                    pass

            last_exc: Exception | None = None
            for attempt in range(CALL_RETRIES):
                async with semaphore:
                    batch_text = format_batch_for_prompt(batch)
                    user_prompt = (
                        f"DOCUMENT ID: {doc_id}\nTOTAL UNIQUE EXTRACTED CONCEPTS: "
                        f"{len(lightweight_concepts)}\nCONCEPTS IN THIS BATCH: {len(batch)}\n\n"
                        "CANDIDATE CONCEPTS LIST (only merge titles present in this list):\n\n"
                        f"{batch_text}\n"
                    )

                    try:
                        raw_text = await asyncio.wait_for(
                            call_llm(
                                client=client,
                                model=self.settings.model_name,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT_CONSOLIDATE},
                                    {"role": "user", "content": user_prompt}
                                ],
                                max_tokens=min(self.settings.max_tokens or 2048, 2048),
                                args=self.args,
                                cfg=getattr(self, "cfg", None),
                                task="judge",
                                max_growth=1,
                            ),
                            timeout=CALL_TIMEOUT_S,
                        )
                        clusters = parse_xml_merge_clusters(raw_text)
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            atomic_write_text(cache_path, json.dumps([c.model_dump() for c in clusters]))
                        except Exception:
                            pass
                        if batch_pbar is not None:
                            batch_pbar.update(1)
                        return clusters
                    except Exception as exc:
                        msg = str(exc).lower()
                        is_timeout = isinstance(exc, asyncio.TimeoutError)
                        too_long = (
                            "hit max_tokens" in msg
                            or (isinstance(exc, BadRequestError) and
                                ("maximum context length" in msg or "input_tokens" in msg))
                        )
                        if not (is_timeout or too_long):
                            raise
                        last_exc = exc
                        if too_long:
                            break  # retrying same input won't help; split immediately

            if len(batch) <= 1:
                if last_exc:
                    raise last_exc
                return []

            if batch_pbar is not None:
                batch_pbar.total += 1
                batch_pbar.refresh()

            mid = len(batch) // 2
            left, right = await asyncio.gather(merge_batch(batch[:mid]), merge_batch(batch[mid:]))
            return left + right

        try:
            batch_results = await asyncio.gather(*(merge_batch(batch) for batch in batches))
        finally:
            if batch_pbar is not None:
                batch_pbar.close()

        return DocumentConsolidationSchema(
            merge_clusters=[cluster for clusters in batch_results for cluster in clusters]
        )

    def execute_programmatic_merge(self, raw_concepts: List[Dict[str, Any]], delta: DocumentConsolidationSchema, doc_id: str = "unknown") -> List[Dict[str, Any]]:
        """Group identical concepts and fuse their aliases, descriptions, and chunks using UnionFind."""
        # F-13: Resolve conflicting or overlapping clusters via UnionFind instead of dict overwrites
        uf = UnionFind()
        primary_titles: Dict[str, str] = {}

        for cluster in delta.merge_clusters:
            primary = cluster.primary_title.strip()
            if not primary:
                continue
            norm_primary = primary.casefold()
            uf.add(norm_primary)
            primary_titles.setdefault(norm_primary, primary)

            for alt in cluster.merged_titles:
                alt_clean = alt.strip()
                if alt_clean:
                    norm_alt = alt_clean.casefold()
                    uf.add(norm_alt)
                    uf.union(norm_primary, norm_alt)

        # Map each raw concept to its cluster root
        grouped_concepts: Dict[str, List[Dict[str, Any]]] = {}
        for c in raw_concepts:
            raw_title = c.get("title", "").strip()
            if not raw_title:
                continue
            norm_title = raw_title.casefold()
            root = uf.find(norm_title)
            grouped_concepts.setdefault(root, []).append(c)

        consolidated_concepts = []
        groups = uf.get_groups()  # Compute groups once before the loop to avoid O(N^2) blocking
        for root, concept_group in grouped_concepts.items():
            # Pick canonical display title: LLM primary title > most frequent title
            candidate_primaries = [primary_titles[item] for item in groups.get(root, []) if item in primary_titles]
            if candidate_primaries:
                chosen_title = candidate_primaries[0]
            else:
                chosen_title = concept_group[0].get("title", "").strip()

            aliases_list = []
            subtopics_list = []
            source_chunks_set = set()

            for c in concept_group:
                for alias in c.get("aliases", []):
                    if alias.strip().casefold() != chosen_title.casefold():
                        aliases_list.append([alias.strip()])
                for sub in c.get("subtopics", []):
                    if sub.strip().casefold() != chosen_title.casefold():
                        subtopics_list.append([sub.strip()])
                raw_title = c.get("title", "").strip()
                if raw_title.casefold() != chosen_title.casefold():
                    aliases_list.append([raw_title])

                src_chunk = c.get("source_chunk")
                if src_chunk:
                    source_chunks_set.add(src_chunk)

            # Preserve per-chunk paragraphs with chunk-level provenance (e.g. doc_XXXXX_chunk_YYY)
            chunk_to_descs: Dict[str, List[str]] = defaultdict(list)
            chunk_order: List[str] = []
            for c in concept_group:
                src = (c.get("source_chunk") or doc_id).replace("::", "_")
                if src not in chunk_to_descs:
                    chunk_order.append(src)
                desc_text = (c.get("description") or "").strip()
                if desc_text:
                    chunk_to_descs[src].append(desc_text)

            chunk_paragraphs = []
            for src in chunk_order:
                fused_chunk_desc = fuse_descriptions(chunk_to_descs[src])
                if fused_chunk_desc:
                    chunk_paragraphs.append({
                        "source_chunk": src,
                        "doc_id": doc_id,
                        "text": fused_chunk_desc,
                    })

            fused_desc = "\n\n".join(p["text"] for p in chunk_paragraphs) if chunk_paragraphs else fuse_descriptions([c.get("description", "") for c in concept_group])

            consolidated_concepts.append({
                "title": chosen_title,
                "aliases": union_aliases(aliases_list),
                "subtopics": union_aliases(subtopics_list),
                "description": fused_desc,
                "paragraphs": chunk_paragraphs or [{"source_chunk": doc_id, "doc_id": doc_id, "text": fused_desc}],
                "source_chunks": sorted(list(source_chunks_set)),
            })

        return consolidated_concepts

    async def persist_consolidation(self, item: Dict[str, Any], delta: DocumentConsolidationSchema) -> None:
        doc_id = item["doc_id"]
        output_dir: Path = item["output_dir"]
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

        consolidated = self.execute_programmatic_merge(item["raw_concepts"], delta, doc_id=doc_id)
        out_payload = json.dumps({"doc_id": doc_id, "concepts": consolidated}, indent=2)
        await asyncio.to_thread(atomic_write_text, output_dir / "concepts.json", out_payload)

        async with self.log_lock:
            self.log_data[doc_id] = True
            await save_log_async(self.log_file, self.log_data)

    async def pass_through(self, pending: List[Dict[str, Any]]) -> None:
        no_merges = DocumentConsolidationSchema()
        for item in pending:
            await self.persist_consolidation(item, no_merges)
        if pending:
            await save_log_async(self.log_file, self.log_data)
            self.flush_count = 0

    async def run(self):
        doc_items = self.prepare_document_items()
        all_raw_concepts = [c for item in doc_items for c in item["raw_concepts"]]
        total_concepts_before = len(all_raw_concepts)

        pending = [d for d in doc_items if not self.log_data.get(d["doc_id"], False)]

        print(f"Total documents: {len(doc_items)} | Pending consolidation: {len(pending)}")
        print(f"Total raw concepts before consolidation: {total_concepts_before}")

        # Display Initial Word Distribution before consolidation
        print_concept_word_distribution(all_raw_concepts, title_label="STAGE 5: BEFORE DOCUMENT CONSOLIDATION")

        settings = self.settings
        pbar = None

        def on_attempt_err(attempt: int, exc: BaseException, classification: str, doc_id: str, server_url: str):
            tqdm.write(f"[{doc_id}] Attempt {attempt} failed on {server_url}: {type(exc).__name__}: {exc}")

        async def wrap_get_merge_clusters(client: Any, item: Dict[str, Any], server_index: int) -> DocumentConsolidationSchema:
            doc_id = item["doc_id"]
            num_concepts = len(item["raw_concepts"])
            server_url = settings.servers[server_index] if server_index < len(settings.servers) else f"server-{server_index}"
            try:
                t0 = asyncio.get_event_loop().time()
                res = await self.get_merge_clusters(client, item, server_index)
                elapsed = asyncio.get_event_loop().time() - t0
                tqdm.write(f"✓ [{doc_id}] Finished ({num_concepts} raw concepts -> {len(res.merge_clusters)} merge clusters) in {elapsed:.1f}s on {server_url}")
                return res
            except Exception as e:
                tqdm.write(f"✗ [{doc_id}] Error on {server_url} ({type(e).__name__}: {e})")
                raise

        if not chunking_enabled(self.base_wiki_dir):
            print("Chunking disabled: passing documents through without LLM calls.")
            await self.pass_through(pending)
        elif pending:
            print(f"Distributing {len(pending)} document(s) across {len(settings.servers)} server(s).")
            print(f"Per-item attempts: {settings.attempts} | Stage budget: {settings.budget_minutes:g} min")

            for item in pending:
                self.log_data.setdefault(item["doc_id"], False)

            pool = pool_for_stage(self.args, settings, STAGE)
            async with pool:
                res = await pool.map(
                    pending,
                    wrap_get_merge_clusters,
                    attempts=settings.attempts,
                    desc="Consolidating Documents",
                    key_fn=lambda it: it["doc_id"],
                    budget=self.budget,
                    attempt_timeout=min(settings.profile.attempt_timeout, 300.0),
                    on_result=self.persist_consolidation,
                )
                res.raise_if_failed(STAGE)
        else:
            print("All documents are already consolidated!")

        await save_log_async(self.log_file, self.log_data)

        # Collect Consolidated Concepts and Display Final Word Distribution
        all_consolidated_concepts = []
        for doc_dir in self.consolidated_dir.iterdir():
            if doc_dir.is_dir():
                concepts_file = doc_dir / "concepts.json"
                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_consolidated_concepts.extend(data.get("concepts", []))
                    except Exception:
                        pass

        total_concepts_after = len(all_consolidated_concepts)
        print_concept_word_distribution(all_consolidated_concepts, title_label="STAGE 5: AFTER DOCUMENT CONSOLIDATION")

        write_stage_manifest(
            wiki_path=self.base_wiki_dir,
            stage=STAGE,
            expected=len(doc_items),
            completed=len(doc_items),
            failed=0,
            concepts_before=total_concepts_before,
            concepts_after=total_concepts_after,
        )

        print("============================================================")
        print("DOCUMENT CONSOLIDATION COMPLETE")
        print(f"Documents Processed:         {len(doc_items)}")
        print(f"Concepts Before Merge:       {total_concepts_before}")
        print(f"Concepts After Merge:        {total_concepts_after} (fused {total_concepts_before - total_concepts_after} duplicates)")
        print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Consolidate concepts across chunks within each document", include_resume=True)
    args = parser.parse_args()

    pipeline = WikiDocumentConsolidatorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
