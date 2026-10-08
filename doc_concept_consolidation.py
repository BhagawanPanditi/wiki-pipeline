#!/usr/bin/env python3
"""
Document-Level Knowledge Graph Consolidation Pass
"""

import os
import re
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

# ── Server & Concurrency Configuration ───────────────────────────────────────
from pipeline_lib.checkpoint import load_json_log, save_log_async, save_log_sync
from pipeline_lib.cli import build_parser, resolve_settings
from pipeline_lib.corpus import chunking_enabled
from pipeline_lib.event_log import EventLog
from pipeline_lib.layout import log_path
from pipeline_lib.errors import StageBudget
from pipeline_lib.json_repair import clean_json_string
from pipeline_lib.llm_log import LLMCallLog
from pipeline_lib.llm_pool import ServerPool

STAGE = "document-consolidation"

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

def snippet_description(text: str, max_words: int = 30) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."

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
class DocumentConsolidatorPipeline:
    def __init__(self, args):
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
        self.attempt_errors: Dict[str, List[Dict[str, Any]]] = {}
        self.flush_count = 0
        self.log_data = load_json_log(
            self.log_file,
            corrupt_message="Consolidation log corrupted. Re-initializing.")

    def prepare_document_items(self) -> List[Dict[str, Any]]:
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
                    except Exception:
                        continue
                        
            if raw_concepts:
                doc_items.append({
                    "doc_id": doc_id,
                    "raw_concepts": raw_concepts,
                    "output_dir": self.consolidated_dir / doc_id
                })
        return doc_items

    def _record_attempt(self, doc_id: str, finish_reason, raw_text,
                        error_type: str, error_message: str) -> None:
        """Keep every attempt's full, untruncated evidence for the debug file."""
        logs = self.attempt_errors.setdefault(doc_id, [])
        logs.append({
            "attempt": len(logs) + 1, "finish_reason": finish_reason,
            "error_type": error_type, "error_message": error_message,
            "raw_response": raw_text
        })

    async def get_merge_clusters(self, client, item: Dict[str, Any],
                                 server_index: int) -> DocumentConsolidationSchema:
        """One document's merge plan.

        Raises on any unusable response so ServerPool classifies and retries it.
        Token-limit truncation is raised as a plain ValueError, so - as before -
        it stays retryable rather than failing fast.
        """
        doc_id = item["doc_id"]
        raw_concepts = item["raw_concepts"]

        lightweight_concepts = []
        for idx, c in enumerate(raw_concepts, 1):
            relations_summary = [
                {"target": r.get("target_concept"), "relation_snippet": snippet_description(r.get("relation", ""), max_words=15)}
                for r in c.get("outbound_relations", [])
            ]
            lightweight_concepts.append({
                "id": idx, "title": c.get("title"), "aliases": c.get("aliases", []),
                "description_snippet": snippet_description(c.get("description", ""), max_words=30),
                "outbound_relations": relations_summary
            })

        user_prompt = f"DOCUMENT ID: {doc_id}\nTOTAL EXTRACTED CONCEPTS ACROSS CHUNKS: {len(lightweight_concepts)}\n\nCONCEPT LIST:\n{json.dumps(lightweight_concepts, indent=2)}\n"

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

            cleaned_json = clean_json_string(raw_text)
            return DocumentConsolidationSchema.model_validate_json(cleaned_json)
        except Exception as exc:
            self._record_attempt(doc_id, finish_reason, raw_text,
                                 type(exc).__name__, str(exc))
            raise

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
            "attempts": self.attempt_errors.get(doc_id, []),
            "final_error": failure.get("error"),
            "final_classification": failure.get("classification"),
            "last_server": failure.get("server"),
        }
        payload = json.dumps(debug_data, indent=2)
        await asyncio.to_thread((output_dir / "consolidation_debug.json").write_text,
                                payload, encoding="utf-8")

    def execute_programmatic_merge(self, raw_concepts: List[Dict[str, Any]], consolidation_delta: DocumentConsolidationSchema) -> List[Dict[str, Any]]:
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
        the corpus was ingested (see :func:`pipeline_lib.corpus.chunking_enabled`
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

            pool = ServerPool(
                settings.servers,
                concurrency_per_server=settings.concurrency_per_server,
                request_timeout=settings.profile.request_timeout,
                connect_timeout=settings.profile.connect_timeout,
                max_connections_slack=20,
                label=STAGE,
                llm_log=LLMCallLog.from_args(self.args, STAGE),
                failover=not getattr(self.args, "no_failover", False),
            )
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

            print("Requests served per server:")
            for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
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
        print(f"Stat Summary:")
        print(f"  - Concepts Before: {total_concepts_before}")
        print(f"  - Concepts After:  {total_concepts_after}")

def main():
    parser = build_parser(
        stage=STAGE,
        description="Cross-Chunk Concept Consolidation Pipeline")
    args = parser.parse_args()
    pipeline = DocumentConsolidatorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()