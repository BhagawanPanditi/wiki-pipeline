#!/usr/bin/env python3
"""
Document-Level Knowledge Graph Consolidation Pass
"""

import re
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from openai import BadRequestError
from pydantic import BaseModel, Field

# ── Server & Concurrency Configuration ───────────────────────────────────────
from utils.checkpoint import load_json_log, save_log_async
from utils.cli import build_parser, resolve_settings
from utils.corpus import chunking_enabled
from utils.attempts import AttemptHistory
from utils.event_log import EventLog
from utils.text import snippet_description
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage

STAGE = "consolidate-documents"
# A whole document can contain hundreds of extracted concepts. Keep each
# request far below the model context window even with a large output reserve.
MAX_MERGE_BATCH_CHARS = 30000
MAX_MERGE_BATCH_CONCEPTS = 80

# ── Pydantic Data Schemas for LLM Consolidation ──────────────────────────────
class MergeCluster(BaseModel):
    primary_title: str = Field(description="The chosen canonical primary title for this concept.")
    merged_titles: List[str] = Field(
        default_factory=list,
        description="Exact titles of other concepts in the input list that represent the same entity and should be merged into primary_title."
    )

class DocumentConsolidationSchema(BaseModel):
    merge_clusters: List[MergeCluster] = Field(
        default_factory=list,
        description="List of concept clusters that represent identical concepts within this document."
    )

# ── Helper Utilities ─────────────────────────────────────────────────────────

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_CONSOLIDATE = """\
You are an expert technical knowledge graph consolidator.
You are given a list of concepts extracted across multiple chunks of the SAME document.
Because these concepts were extracted independently per chunk, many represent the EXACT SAME real-world entity, process, software tool, gene, condition, or technical subject under slightly different titles or acronyms.

YOUR MISSION:
Identify groups of concepts that refer to the exact same core concept/entity within this document and should be merged together.

CRITICAL MERGE RULES:
1. MERGE MERGING CRITERIA: Merge concepts if they refer to the exact same canonical entity, tool, process, acronym, or subject.
2. DO NOT MERGE DISTINCT CONCEPTS: Do NOT merge distinct sub-components, related but separate concepts, or parent/child hierarchy terms.
3. PRIMARY TITLE SELECTION: Pick the most standard, canonical, explicit title as `primary_title`.
4. SCOPE: ONLY include titles that exist in the provided concept list.

OUTPUT FORMAT:
Return ONLY valid JSON matching this structure:
{
  "merge_clusters": [
    {
      "primary_title": "Exact Chosen Canonical Title",
      "merged_titles": [
        "Alternative Title 1",
        "Alternative Title 2"
      ]
    }
  ]
}
If no concepts need to be merged, return: {"merge_clusters": []}.
"""

# ── Pipeline Implementation ──────────────────────────────────────────────────
def make_merge_batches(concepts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Pack similar titles together while bounding each request's JSON payload.

    Exact-title duplicates still merge programmatically across batch boundaries;
    the LLM considers near-duplicates within each batch. Extremely long single
    entries are rejected rather than silently omitted.
    """
    ordered = sorted(concepts, key=lambda c: str(c.get("title") or "").casefold())
    batches: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    size = 2  # opening and closing JSON brackets
    for concept in ordered:
        item_size = len(json.dumps(concept, ensure_ascii=False, separators=(",", ":"))) + 1
        if item_size + 2 > MAX_MERGE_BATCH_CHARS:
            raise ValueError(f"Concept {concept.get('title')!r} is too large for a merge batch")
        if batch and (len(batch) >= MAX_MERGE_BATCH_CONCEPTS or
                      size + item_size > MAX_MERGE_BATCH_CHARS):
            batches.append(batch)
            batch = []
            size = 2
        batch.append(concept)
        size += item_size
    if batch:
        batches.append(batch)
    return batches


class DocumentConsolidatorPipeline:
    def __init__(self, args):
        """Prepare document consolidation, creating the ``consolidated/`` folder.

        Args:
            args: Parsed CLI namespace.

        Sets:
            Paths, the stage settings, the debug and decision logs, the time budget and
            the per-document checkpoint."""
        self.args = args
        self.domain = args.domain
        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.consolidated_dir = self.base_wiki_dir / "consolidated"
        self.log_file = self.base_wiki_dir / "doc_consolidation_log.json"

        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.settings = resolve_settings(args, STAGE)
        self.debug = EventLog(log_path(self.base_wiki_dir, STAGE, "debug"),
                                 label=STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.decisions = EventLog(log_path(self.base_wiki_dir, STAGE, "decisions"), label=STAGE)
        self.attempts = AttemptHistory()
        self.flush_count = 0
        self.log_lock = asyncio.Lock()
        self.log_data = load_json_log(
            self.log_file,
            corrupt_message="Consolidation log corrupted. Re-initializing.")

    def prepare_document_items(self) -> List[Dict[str, Any]]:
        """Collect each document's concepts from the staging folder.

        A document's concepts are gathered across all its chunks, and each concept is
        stamped with the chunk it came from (``source_chunk``). Documents with no
        concepts are skipped. A concepts file that cannot be read is recorded in the
        debug log, not dropped silently.

        Returns:
            A list with one item per document: ``doc_id``, ``raw_concepts`` and
            ``output_dir``."""
        print(f"Scanning staging directory: {self.staging_dir}...")
        doc_dirs = [d for d in self.staging_dir.iterdir() if d.is_dir() and d.name.startswith("canon_")]
        
        doc_items = []
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            chunk_dirs = [c for c in doc_dir.iterdir() if c.is_dir() and c.name.startswith("chunk_")]
            raw_concepts = []
            
            for chunk_dir in chunk_dirs:
                concepts_file = chunk_dir / "concepts.json"
                chunk_id = chunk_dir.name
                
                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Inject source_chunk into each raw concept
                            for concept in data.get("concepts", []):
                                concept["source_chunk"] = f"{doc_id}::{chunk_id}"
                                raw_concepts.append(concept)
                    except Exception as exc:
                        self.debug.record("unreadable_concepts_file", doc_id=doc_id,
                                          chunk_id=chunk_id, path=str(concepts_file),
                                          error=str(exc))
                        continue
                        
            if raw_concepts:
                doc_items.append({
                    "doc_id": doc_id,
                    "raw_concepts": raw_concepts,
                    "output_dir": self.consolidated_dir / doc_id
                })
        return doc_items

    async def get_merge_clusters(self, client, item: Dict[str, Any],
                                 server_index: int) -> DocumentConsolidationSchema:
        """Build a document's merge plan from context-bounded title batches.

        Raises on any unusable response so ServerPool classifies and retries it.
        Context overflow or truncated output splits the affected batch further.
        """
        doc_id = item["doc_id"]
        raw_concepts = item["raw_concepts"]

        # Merge identity is decided from names and a short disambiguation hint.
        # Relations and full descriptions belong in the saved concepts, not in
        # this prompt: repeating them for every chunk made large documents exceed
        # the server's input-token allowance.
        lightweight_concepts = []
        for c in raw_concepts:
            lightweight_concepts.append({
                "title": c.get("title"),
                "aliases": c.get("aliases", [])[:8],
                "description_hint": snippet_description(
                    c.get("description", ""), max_words=30)[:400],
            })

        batches = make_merge_batches(lightweight_concepts)
        if len(batches) > 1:
            self.debug.record("document_batched", doc_id=doc_id,
                              concepts=len(lightweight_concepts), batches=len(batches))
        # A large document is one checkpointed unit. Persist it only after EVERY
        # batch succeeds; partial merge plans never mark the document complete.
        semaphore = asyncio.Semaphore(16)
        batch_pbar = None
        if len(batches) > 1:
            batch_pbar = tqdm(total=len(batches), desc=f"  [doc {doc_id[:12]}] {len(raw_concepts)} concepts", leave=False)

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
            async with semaphore:
                batch_text = format_batch_for_prompt(batch)
                user_prompt = (
                    f"DOCUMENT ID: {doc_id}\nTOTAL EXTRACTED CONCEPTS ACROSS CHUNKS: "
                    f"{len(lightweight_concepts)}\nCONCEPTS IN THIS BATCH: {len(batch)}\n\n"
                    "CANDIDATE CONCEPTS LIST (only merge titles present in this list):\n\n"
                    f"{batch_text}\n"
                )
                raw_text = None
                finish_reason = None
                try:
                    response = await client.chat.completions.create(
                        model=self.settings.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_CONSOLIDATE},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1, max_tokens=self.settings.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    choice = response.choices[0]
                    finish_reason = choice.finish_reason
                    raw_text = choice.message.content
                    if finish_reason == "length":
                        raise ValueError(f"Response truncated: Hit max_tokens ({self.settings.max_tokens}).")
                    delta = DocumentConsolidationSchema.model_validate_json(clean_json_string(raw_text))
                    if batch_pbar is not None:
                        batch_pbar.update(1)
                    return delta.merge_clusters
                except Exception as exc:
                    self.attempts.record(doc_id, finish_reason=finish_reason, raw_text=raw_text,
                                         error_type=type(exc).__name__, error_message=str(exc),
                                         batch_size=len(batch))
                    # Server-side tokenization is authoritative. If a batch
                    # unexpectedly exceeds its context/output allowance, halve
                    # it and try again; a single unfit concept fails explicitly.
                    too_long = (finish_reason == "length" or
                                isinstance(exc, BadRequestError) and
                                ("maximum context length" in str(exc).lower() or
                                 "input_tokens" in str(exc).lower()))
                    if not too_long or len(batch) == 1:
                        raise
                    self.debug.record("split_merge_batch", doc_id=doc_id,
                                      batch_size=len(batch), error=str(exc))
            # Split outside the semaphore to avoid deadlock when child calls
            # need the same limited pool of request slots.
            mid = len(batch) // 2
            left, right = await asyncio.gather(
                merge_batch(batch[:mid]), merge_batch(batch[mid:]))
            return left + right

        batch_results = await asyncio.gather(*(merge_batch(batch) for batch in batches))
        if batch_pbar is not None:
            batch_pbar.close()
        return DocumentConsolidationSchema(
            merge_clusters=[cluster for clusters in batch_results for cluster in clusters]
        )

    async def persist_consolidation(self, item: Dict[str, Any],
                                    delta: DocumentConsolidationSchema) -> None:
        """SUCCESS: stream the merged document and advance the checkpoint."""
        doc_id = item["doc_id"]
        output_dir: Path = item["output_dir"]
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

        consolidated = self.execute_programmatic_merge(item["raw_concepts"], delta)
        out_payload = json.dumps({"doc_id": doc_id, "concepts": consolidated}, indent=2)
        await asyncio.to_thread((output_dir / "concepts.json").write_text,
                                out_payload, encoding="utf-8")
        await asyncio.to_thread((output_dir / "merge_clusters.json").write_text,
                                delta.model_dump_json(indent=2), encoding="utf-8")

        self.decisions.record(
            "document_consolidated", doc_id=doc_id,
            concepts_before=len(item["raw_concepts"]), concepts_after=len(consolidated),
            merges=[{"canonical": c.primary_title, "merged": list(c.merged_titles)}
                    for c in delta.merge_clusters])
        async with self.log_lock:
            self.log_data[doc_id] = True
            self.flush_count += 1
            if self.flush_count >= 10:
                await save_log_async(self.log_file, self.log_data)
                self.flush_count = 0

    async def write_failure_debug(self, item: Dict[str, Any],
                                  failure: Dict[str, Any]) -> None:
        """FAILURE: save every attempt for this document, untruncated."""
        doc_id = item["doc_id"]
        output_dir: Path = item["output_dir"]
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        debug_data = {
            "doc_id": doc_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempts": self.attempts.get(doc_id),
            "final_error": failure.get("error"),
            "final_classification": failure.get("classification"),
            "last_server": failure.get("server"),
        }
        payload = json.dumps(debug_data, indent=2)
        await asyncio.to_thread((output_dir / "consolidation_debug.json").write_text,
                                payload, encoding="utf-8")

    def execute_programmatic_merge(self, raw_concepts: List[Dict[str, Any]], consolidation_delta: DocumentConsolidationSchema) -> List[Dict[str, Any]]:
        """Apply the LLM's merge clusters to one document's concepts, deterministically.

        Each cluster maps its member titles onto its primary title. Concepts that then
        share a title are combined: aliases, entities and source chunks are unioned, and
        description sentences are joined without repeats.

        Args:
            raw_concepts: The document's concepts, as produced by
                :meth:`prepare_document_items`.
            consolidation_delta: The LLM's merge clusters for this document.

        Returns:
            The merged concept list, one entry per distinct title."""
        title_map: Dict[str, str] = {}
        for cluster in consolidation_delta.merge_clusters:
            primary = cluster.primary_title.strip()
            title_map[primary.lower()] = primary
            for alt in cluster.merged_titles:
                title_map[alt.strip().lower()] = primary

        grouped_concepts: Dict[str, List[Dict[str, Any]]] = {}
        for c in raw_concepts:
            raw_title = c.get("title", "").strip()
            if not raw_title: continue
            primary_title = title_map.get(raw_title.lower(), raw_title)
            grouped_concepts.setdefault(primary_title, []).append(c)

        consolidated_concepts = []
        for primary_title, concept_group in grouped_concepts.items():
            aliases_set = set()
            entities_set = set()
            source_chunks_set = set()
            
            for c in concept_group:
                for alias in c.get("aliases", []):
                    if alias.strip().lower() != primary_title.lower(): aliases_set.add(alias.strip())
                raw_title = c.get("title", "").strip()
                if raw_title.lower() != primary_title.lower(): aliases_set.add(raw_title)

                for ent in c.get("entities", []):
                    if ent.strip(): entities_set.add(ent.strip())
                    
                src_chunk = c.get("source_chunk")
                if src_chunk:
                    source_chunks_set.add(src_chunk)

            description_sentences = []
            seen_sentences = set()
            for c in concept_group:
                desc = c.get("description", "").strip()
                if desc:
                    for s in re.split(r'(?<=[.!?])\s+', desc):
                        s_clean = s.strip()
                        s_key = s_clean.lower()
                        if s_clean and s_key not in seen_sentences:
                            seen_sentences.add(s_key)
                            description_sentences.append(s_clean)
            merged_description = " ".join(description_sentences)

            merged_relations_map: Dict[str, Dict[str, Any]] = {}
            for c in concept_group:
                src_chunk = c.get("source_chunk")
                for rel in c.get("outbound_relations", []):
                    raw_target = rel.get("target_concept", "").strip()
                    if not raw_target: continue
                    mapped_target = title_map.get(raw_target.lower(), raw_target)
                    if mapped_target.lower() == primary_title.lower(): continue
                    
                    rel_text = rel.get("relation", "").strip()
                    merged_relations_map.setdefault(mapped_target, {"texts": [], "chunks": set()})
                    
                    if rel_text and rel_text not in merged_relations_map[mapped_target]["texts"]:
                        merged_relations_map[mapped_target]["texts"].append(rel_text)
                    
                    # Track provenance: Use relation's chunks if they exist, otherwise fallback to the concept's chunk
                    if rel.get("source_chunks"):
                        merged_relations_map[mapped_target]["chunks"].update(rel["source_chunks"])
                    elif src_chunk:
                        merged_relations_map[mapped_target]["chunks"].add(src_chunk)

            final_outbound_relations = [
                {
                    "target_concept": tgt, 
                    "relation": " ".join(data["texts"]),
                    "source_chunks": sorted(list(data["chunks"]))
                }
                for tgt, data in merged_relations_map.items()
            ]

            consolidated_concepts.append({
                "title": primary_title,
                "aliases": sorted(list(aliases_set)),
                "description": merged_description,
                "entities": sorted(list(entities_set)),
                "source_chunks": sorted(list(source_chunks_set)),
                "outbound_relations": final_outbound_relations
            })

        return consolidated_concepts

    async def pass_through(self, pending: List[Dict[str, Any]]) -> None:
        """Copy each document's concepts to ``consolidated/`` without any LLM call.

        This stage exists to merge duplicate concepts found in *different chunks
        of the same document*. When the corpus was ingested with
        ``--no-chunking`` every document is a single unit, so there are no
        cross-chunk duplicates to find and no merge to decide.

        Writing through the normal :meth:`persist_consolidation` path keeps the
        output files, their schema and the checkpoint identical to a chunked
        run, so every later stage is unaffected.

        Args:
            pending: Document items not yet recorded as consolidated. Each has
                ``doc_id``, ``raw_concepts`` and ``output_dir``.

        Returns:
            None. Mutates ``self.log_data`` to mark each document complete.
        """
        # An empty merge decision is exactly "change nothing": running it through
        # execute_programmatic_merge reuses the same normalisation as a real run
        # instead of hand-rolling a second writer.
        no_merges = DocumentConsolidationSchema()
        for item in pending:
            await self.persist_consolidation(item, no_merges)
        if pending:
            await save_log_async(self.log_file, self.log_data)
            self.flush_count = 0

    async def run(self):
        """Consolidate concepts within each document, then report the totals.

        Chooses between the LLM merge pass and a no-op pass-through based on how
        the corpus was ingested (see :func:`utils.corpus.chunking_enabled`
        and ``run_config.json``). Resumes from ``doc_consolidation_log.json``,
        processing only documents not already marked complete.

        Returns:
            None. Writes ``consolidated/<doc_id>/concepts.json`` and
            ``merge_clusters.json`` per document, plus the before/after concept
            counts in the checkpoint log.

        Raises:
            RuntimeError: If any document is still unconsolidated after the pass,
                which keeps the pipeline from advancing on partial output.
        """
        doc_items = self.prepare_document_items()
        
        # Calculate total concepts before consolidation
        total_concepts_before = sum(len(item["raw_concepts"]) for item in doc_items)

        pending = [d for d in doc_items if not self.log_data.get(d["doc_id"], False)]

        print(f"Total documents: {len(doc_items)} | Pending consolidation: {len(pending)}")
        print(f"Total raw concepts before consolidation: {total_concepts_before}")
        
        settings = self.settings
        if not chunking_enabled(self.base_wiki_dir):
            print("Chunking was disabled for this corpus, so every document is a single")
            print("unit and there are no cross-chunk duplicates to merge. Passing")
            print(f"{len(pending)} document(s) through without any LLM call.")
            await self.pass_through(pending)
        elif pending:
            print(f"Spreading {len(pending)} document(s) across {len(settings.servers)} "
                  f"server(s) at {settings.concurrency_per_server} concurrent request(s) "
                  f"each (item i -> server i % N).")
            print(f"Per-item attempts: {settings.attempts} | Stage budget: "
                  f"{settings.budget_minutes:g} min | Debug log: {self.debug.path}")

            # Mark every pending document in-progress, as the old worker did under
            # a lock, so a later resume can see what was attempted.
            for item in pending:
                self.log_data.setdefault(item["doc_id"], False)

            pool = pool_for_stage(self.args, settings, STAGE)
            try:
                async with pool:
                    result = await pool.map(
                        pending,
                        self.get_merge_clusters,
                        attempts=settings.attempts,
                        desc=f"Consolidating Documents [{self.domain}]",
                        key_fn=lambda it: it["doc_id"],
                        budget=self.budget,
                        debug=self.debug,
                        attempt_timeout=settings.profile.attempt_timeout,
                        on_result=self.persist_consolidation,
                    )
            finally:
                self.debug.close()

            for failure in result.failures:
                await self.write_failure_debug(pending[failure["index"]], failure)

            print("Documents consolidated per server:")
            for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
                print(f"  {url:<40} {count}")
            print("Document attempts per server (one attempt may contain several batch requests):")
            for url, count in sorted(pool.per_server_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {url:<40} {count}")
        else:
            print("All documents are already consolidated!")

        incomplete_docs = [
            item["doc_id"] for item in doc_items
            if not self.log_data.get(item["doc_id"], False)
        ]
        if incomplete_docs:
            await save_log_async(self.log_file, self.log_data)
            raise RuntimeError(
                f"Document consolidation left {len(incomplete_docs)} document(s) incomplete. "
                f"Examples: {incomplete_docs[:10]}. Rerun to retry pending documents."
            )

        # Always calculate 'after' statistics directly from disk to ensure accuracy
        total_concepts_after = 0
        for doc_dir in self.consolidated_dir.iterdir():
            if doc_dir.is_dir():
                concepts_file = doc_dir / "concepts.json"
                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            total_concepts_after += len(data.get("concepts", []))
                    except Exception:
                        pass
        
        # Save stats to log
        self.log_data["total_concepts_before"] = total_concepts_before
        self.log_data["total_concepts_after"] = total_concepts_after
        
        await save_log_async(self.log_file, self.log_data)
        
        print(f"\nDocument consolidation successfully finished for domain: {self.domain}!")
        print("Stat Summary:")
        print(f"  - Concepts Before: {total_concepts_before}")
        print(f"  - Concepts After:  {total_concepts_after}")

def main():
    """Parse the command line and run document consolidation.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Cross-Chunk Concept Consolidation Pipeline")
    args = parser.parse_args()
    pipeline = DocumentConsolidatorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()