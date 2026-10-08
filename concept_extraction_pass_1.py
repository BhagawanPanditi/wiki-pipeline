#!/usr/bin/env python3
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from tqdm.asyncio import tqdm

from pipeline_lib.atomic_io import atomic_write_json
from pipeline_lib.checkpoint import load_bool_log, save_log_async, save_log_sync
from pipeline_lib.cli import build_parser, resolve_settings
from pipeline_lib.corpus import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
                                 build_units, cache_manifest, deduplicate,
                                 load_corpus, resolve_corpus_path,
                                 save_run_config)
from pipeline_lib.attempts import AttemptHistory
from pipeline_lib.event_log import EventLog
from pipeline_lib.layout import log_path
from pipeline_lib.errors import PermanentFailure, StageBudget
from pipeline_lib.json_repair import clean_json_string
from pipeline_lib.llm_log import LLMCallLog
from pipeline_lib.llm_pool import ServerPool, pool_for_stage

STAGE = "pass1-extraction"

# ── Pydantic Data Schemas for Extraction ──────────────────────────────────────

class OutboundRelation(BaseModel):
    target_concept: str = Field(description="Exact title of the target canonical concept.")
    relation: str = Field(description="1 sentence describing how this concept affects/relates outward to Target Concept.")

class Concept(BaseModel):
    title: str = Field(description="Exact canonical title (1-6 words, singular).")
    aliases: List[str] = Field(default_factory=list, description="Explicit acronyms or 'also known as' names.")
    description: str = Field(description="Single dense factual paragraph with all metrics, parameters, and details.")
    entities: List[str] = Field(default_factory=list, description="Flat list of minor named entities, tools, genes, datasets, proteins, dates, or specific terms mentioned.")
    outbound_relations: List[OutboundRelation] = Field(default_factory=list)

class ChunkExtractionSchema(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)

# ── Helper Utilities ─────────────────────────────────────────────────────────
# short_hash, chunk_document and the rest of the corpus contract live in
# pipeline_lib/corpus.py so that every stage agrees on document ids, chunk ids
# and citation keys.

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert technical knowledge extractor building an LLM Wiki Knowledge Graph. 
Analyze the CURRENT CHUNK and extract its core thematic concepts, dense factual descriptions, related named entities, and outbound relationships.

INPUT & NOISE CLEANING:
- The text is raw and uncleaned. SILENTLY IGNORE all HTML tags, scraping artifacts, navigation links, headers, footers, cookie notices, and promotional clutter. Focus ONLY on substantive informational text.
- If the CURRENT CHUNK contains no substantive core concepts (e.g., solely bibliographies, navigational boilerplate, or non-informative text), return an empty list: {"concepts": []}.

CONTEXT BOUNDARIES:
- CURRENT CHUNK is your ONLY source for facts.
- Use CONTEXT BEFORE / CONTEXT AFTER ONLY to resolve pronouns (e.g., "this architecture", "they") or acronyms to find full canonical names.
- DO NOT extract facts, numbers, or statements that appear exclusively in CONTEXT BEFORE or CONTEXT AFTER.

EXHAUSTIVE CONCEPT COVERAGE (UNCAPPED):
- Extract ALL distinct concepts, core thematic subjects, major sub-topics, conditions, pathways, systems, or technical components discussed in this chunk.
- Do NOT artificially restrict concept counts. Extract EVERY substantive sub-topic that has meaningful explanatory detail in the text.

FACT vs. ENTITY STRICT SEPARATION:
1. DESCRIPTIONS HOLD ALL FACTS: Write a single, dense, factual paragraph for each concept. ALL quantitative metrics, ranges, parameters, statistical findings, experimental conditions, and procedural details MUST be written in full complete sentences inside `description`.
2. ENTITIES LIST: `entities` is STRICTLY a flat list of short named string identifiers (e.g., software tools, biological terms, gene symbols, dataset identifiers, model names, author names, dates). DO NOT place factual claims or parameter values in `entities`.

EXHAUSTIVE FACTUAL RETENTION MANDATE:
Descriptions MUST retain exact detail rather than abstracting away technical facts. You MUST preserve in sentence form:
1. QUANTITATIVE METRICS & BOUNDS: Exact numbers, statistical findings, percentages, measurements, ranges, and parameter limits.
2. FORMAL ENTITIES & NAMED TERMS: Specific algorithms, software/tool versions, taxonomies, codes, benchmarks, locations, or formal classifications.
3. CONDITIONS & SPECIFICATIONS: Specific operational bounds, environmental constraints, experimental conditions, procedural steps, and dates/years.
4. CREDITED ATTRIBUTIONS: Specific expert, study, or author attributions when linked to explicit technical claims.

OUTBOUND RELATIONSHIPS:
- List OUTBOUND relationships to other target concepts mentioned in the chunk.
- The concept being extracted is ALWAYS the source/subject (Concept A). Describe how Concept A affects, utilizes, replaces, leads to, or relates to Target Concept B in 1 concise sentence.
- Ensure `target_concept` matches the canonical title of the target concept as defined in the chunk or context.
- If a concept has no outbound relations to other concepts in this chunk, set "outbound_relations": [].

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact structure:
{
  "concepts": [
    {
      "title": "Exact Canonical Title",
      "aliases": ["Acronym or Alt Name"],
      "description": "A single comprehensive paragraph detailing what the concept is along with all raw quantitative metrics, parameter bounds, specific findings, and detailed statements from this chunk.",
      "entities": ["Tool/Software Name", "Dataset Code", "Gene Symbol", "Formal Benchmark"],
      "outbound_relations": [
        {
          "target_concept": "Target Concept Title",
          "relation": "1 sentence: How THIS concept affects or relates outward to Target Concept."
        }
      ]
    }
  ]
}
"""

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiExtractorPipeline:
    def __init__(self, args):
        """Prepare Pass 1 extraction, creating the staging and cache folders.

        Args:
            args: Parsed CLI namespace.

        Sets:
            Paths for staging, cache and ``extraction_log.json``, and the extraction
            checkpoint loaded from that file."""
        self.args = args
        self.domain = args.domain
        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.log_file = self.base_wiki_dir / "extraction_log.json"

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_log = load_bool_log(
            self.log_file,
            corrupt_message="Log file corrupted. Initializing fresh extraction log.")

    def cleanup_incomplete_states(self):
        """Remove partial concept output for failed chunks but preserve debug evidence."""
        print("Sanitizing incomplete extraction states...")
        purged = 0
        for key, completed in list(self.extraction_log.items()):
            if not completed:
                doc_id, chunk_id = key.split("::", 1)
                target_subfolder = self.staging_dir / doc_id / chunk_id
                partial_concepts = target_subfolder / "concepts.json"
                if partial_concepts.exists():
                    partial_concepts.unlink()
                    purged += 1
                del self.extraction_log[key]
        save_log_sync(self.log_file, self.extraction_log)
        print(f"Cleared {purged} partial concept files; preserved per-chunk debug logs.")

    def chunk_is_complete(self, chunk: Dict[str, Any]) -> bool:
        """Trust a completion-log entry only when its extracted JSON is readable."""
        key = f"{chunk['doc_id']}::{chunk['chunk_id']}"
        if not self.extraction_log.get(key, False):
            return False

        concepts_file = self.staging_dir / chunk["doc_id"] / chunk["chunk_id"] / "concepts.json"
        try:
            with concepts_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data.get("concepts"), list)
        except Exception:
            self.extraction_log[key] = False
            return False

    def prepare_chunks(self) -> List[Dict[str, Any]]:
        """Load the corpus, deduplicate it, and build this stage's work units.

        The corpus may be a single JSON array or JSON Lines, and may be given
        with ``--corpus`` or resolved from the legacy ``--data_dir``/``--domain``
        layout. Documents with identical content are collapsed into one
        canonical document; the original ids are written to
        ``doc_mapping.json`` in the run directory so citations can be traced back to
        the source.

        Chunking is optional. With ``--no-chunking`` each document becomes a
        single unit and no tokenizer is loaded, so ``transformers`` and an
        offline HuggingFace model are not required.

        A previously built ``cache/chunks_cache.json`` is reused only when its
        manifest matches today's corpus digest, tokenizer and chunking
        parameters exactly; otherwise it is rebuilt.

        Returns:
            A list of unit dicts with keys ``doc_id``, ``chunk_id``,
            ``current_chunk``, ``context_before`` and ``context_after``.

        Raises:
            CorpusError: If no corpus was specified, the file is missing, or a
                record has no usable content.
            ValueError: If ``--chunk_overlap`` is not smaller than
                ``--chunk_size``.
        """
        chunks_cache_file = self.cache_dir / "chunks_cache.json"
        cache_manifest_file = self.cache_dir / "chunks_cache_manifest.json"
        mapping_file = self.base_wiki_dir / "doc_mapping.json"

        corpus_path = resolve_corpus_path(self.args)
        chunking = bool(getattr(self.args, "chunking", True))
        chunk_size = int(getattr(self.args, "chunk_size", DEFAULT_CHUNK_SIZE))
        overlap = int(getattr(self.args, "chunk_overlap", DEFAULT_CHUNK_OVERLAP))

        # A pre-existing chunks_cache.json is not sufficient evidence that its
        # chunks describe today's input. Hash the source and bind the cache to
        # the tokenizer name and chunking parameters before reusing it.
        expected_manifest = cache_manifest(
            corpus_path,
            tokenizer_name=self.args.model_name if chunking else None,
            chunking=chunking,
            chunk_size=chunk_size,
            overlap=overlap)

        if chunks_cache_file.exists() and cache_manifest_file.exists() and mapping_file.exists():
            try:
                with cache_manifest_file.open("r", encoding="utf-8") as f:
                    cached_manifest = json.load(f)
                with chunks_cache_file.open("r", encoding="utf-8") as f:
                    cached_chunks = json.load(f)
                required_fields = {
                    "doc_id", "chunk_id", "current_chunk", "context_before", "context_after"
                }
                chunks_are_valid = isinstance(cached_chunks, list) and all(
                    isinstance(chunk, dict) and required_fields.issubset(chunk)
                    for chunk in cached_chunks
                )
                if cached_manifest == expected_manifest and chunks_are_valid:
                    print(f"Loading validated cached chunks from {chunks_cache_file}...")
                    return cached_chunks
            except (OSError, json.JSONDecodeError, TypeError):
                pass
            print("Cached chunks are stale or invalid; rebuilding them from the current input.")
        elif chunks_cache_file.exists():
            print("Cached chunks are missing required metadata; rebuilding them for safety.")

        print(f"Loading corpus from {corpus_path}...")
        docs = load_corpus(corpus_path)

        print("Deduplicating documents based on content hash...")
        canon_docs, doc_mapping = deduplicate(docs)
        # Refresh the mapping whenever the source signature changes, rather than
        # retaining mappings that may belong to an older version of the corpus.
        atomic_write_json(mapping_file, doc_mapping)
        print(f"Deduplicated {len(docs)} docs down to {len(canon_docs)} unique canonical docs.")

        # Record how this run was prepared so later stages (notably
        # per-document consolidation) can tell whether chunking was on without
        # being told again on their command line.
        save_run_config(self.base_wiki_dir, {
            "corpus": str(corpus_path),
            "corpus_sha256": expected_manifest["source_sha256"],
            "domain": self.args.domain_label,
            "run_name": self.domain,
            "model_name": self.args.model_name,
            "chunking": chunking,
            "chunk_size": chunk_size if chunking else None,
            "chunk_overlap": overlap if chunking else None,
            "documents": len(docs),
            "unique_documents": len(canon_docs),
        })

        tokenizer = None
        if chunking:
            # Imported lazily: an unchunked corpus needs no tokenizer, so the
            # pipeline runs without transformers installed.
            from transformers import AutoTokenizer

            print(f"Tokenizing and creating chunks "
                  f"({chunk_size} token window, {overlap} overlap)...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name, trust_remote_code=True, local_files_only=True)
        else:
            print("Chunking disabled: treating each document as a single unit.")

        all_chunks = build_units(canon_docs, tokenizer, chunking=chunking,
                                 chunk_size=chunk_size, overlap=overlap)

        atomic_write_json(chunks_cache_file, all_chunks)
        atomic_write_json(cache_manifest_file, expected_manifest)

        print(f"Cached {len(all_chunks)} unit(s) to disk.")
        return all_chunks

    async def extract_chunk(self, client, chunk_data: Dict[str, Any], server_index: int):
        """Performs one LLM call and validates the response via Pydantic.

        Raises on any unusable response so ServerPool can classify it and apply
        the retry ladder. Token-limit truncation is still fail-fast: it is raised
        as a permanent error and never retried.
        """
        key = f"{chunk_data['doc_id']}::{chunk_data['chunk_id']}"
        user_prompt = f"""\
    CONTEXT BEFORE:
    {chunk_data['context_before']}

    CURRENT CHUNK:
    {chunk_data['current_chunk']}

    CONTEXT AFTER:
    {chunk_data['context_after']}
    """
        raw_text = None
        finish_reason = None
        try:
            response = await client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=self.settings.max_tokens,
                response_format={"type": "json_object"}
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            raw_text = choice.message.content

            if finish_reason == "length":
                # FAIL FAST: Do not retry if we hit the token limit
                self.attempts.record(key, finish_reason=finish_reason, raw_text=raw_text,
                                     error_type="TokenLimitExceeded",
                                     error_message="Response truncated: Hit max_tokens limit.")
                raise PermanentFailure(
                    f"{key}: response truncated, hit max_tokens limit",
                    cause=ValueError("Response truncated: Hit max_tokens limit."))

            cleaned_json = clean_json_string(raw_text)
            return ChunkExtractionSchema.model_validate_json(cleaned_json)

        except PermanentFailure:
            raise
        except Exception as exc:
            self.attempts.record(key, finish_reason=finish_reason, raw_text=raw_text,
                                 error_type=type(exc).__name__, error_message=str(exc))
            raise

    async def persist_extraction(self, chunk_data: Dict[str, Any],
                                 extracted: ChunkExtractionSchema) -> None:
        """SUCCESS: stream concepts.json to disk and advance the checkpoint."""
        doc_id = chunk_data["doc_id"]
        chunk_id = chunk_data["chunk_id"]
        key = f"{doc_id}::{chunk_id}"

        subfolder = self.staging_dir / doc_id / chunk_id
        await asyncio.to_thread(subfolder.mkdir, parents=True, exist_ok=True)

        concepts_file = subfolder / "concepts.json"
        json_payload = extracted.model_dump_json(indent=2)
        await asyncio.to_thread(concepts_file.write_text, json_payload, encoding="utf-8")

        # Clean up any stale debug.json if it existed
        debug_file = subfolder / "debug.json"
        if debug_file.exists():
            try:
                debug_file.unlink()
            except Exception:
                pass

        self.extraction_log[key] = True
        self.flush_count += 1
        if self.flush_count >= 50:
            await save_log_async(self.log_file, self.extraction_log)
            self.flush_count = 0

    async def write_failure_debug(self, chunk_data: Dict[str, Any],
                                  failure: Dict[str, Any]) -> None:
        """FAILURE: save debug.json with every attempt, untruncated."""
        doc_id = chunk_data["doc_id"]
        chunk_id = chunk_data["chunk_id"]
        key = f"{doc_id}::{chunk_id}"

        subfolder = self.staging_dir / doc_id / chunk_id
        await asyncio.to_thread(subfolder.mkdir, parents=True, exist_ok=True)

        error_logs = self.attempts.get(key)
        debug_data = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_attempts": len(error_logs),
            "chunk_metadata": {
                "context_before_chars": len(chunk_data.get("context_before", "")),
                "current_chunk_chars": len(chunk_data.get("current_chunk", "")),
                "context_after_chars": len(chunk_data.get("context_after", "")),
            },
            "attempts": error_logs,
            "final_error": failure.get("error"),
            "final_classification": failure.get("classification"),
            "last_server": failure.get("server"),
        }
        debug_payload = json.dumps(debug_data, indent=2)
        await asyncio.to_thread((subfolder / "debug.json").write_text,
                                debug_payload, encoding="utf-8")

    async def run(self):
        """Extract concepts from every unit of the corpus and persist each result.

        Units come from :meth:`prepare_chunks`, which reuses a validated chunk cache when
        one matches. Units already recorded as complete are skipped. The rest go through
        the server pool, and each unit's ``concepts.json`` is written as it finishes. The
        checkpoint is flushed periodically, so an interrupted run resumes where it
        stopped.

        Returns:
            None. Results are in ``staging/<doc>/<chunk>/concepts.json``.

        Raises:
            RuntimeError: If any unit is still incomplete after retries.
            StageBudgetExceeded: If the stage's wall-clock budget runs out. The
                checkpoint is kept, so the run is resumable."""
        self.settings = resolve_settings(self.args, STAGE)
        self.debug = EventLog(log_path(self.base_wiki_dir, STAGE, "debug"),
                                 label=STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.attempts = AttemptHistory()
        self.flush_count = 0

        if self.args.resume:
            self.cleanup_incomplete_states()

        all_chunks = self.prepare_chunks()
        pending_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]

        total_pending = len(pending_chunks)
        settings = self.settings
        print(f"Total chunks: {len(all_chunks)} | Pending chunks: {total_pending}")
        print(f"Spreading work across {len(settings.servers)} server(s) at "
              f"{settings.concurrency_per_server} concurrent request(s) each "
              f"(item i -> server i % N).")
        print(f"Total Active Concurrency Limit: "
              f"{len(settings.servers) * settings.concurrency_per_server} in flight.")
        print(f"Per-item attempts: {settings.attempts} | Stage budget: "
              f"{settings.budget_minutes:g} min | Debug log: {self.debug.path}")

        if not pending_chunks:
            print("All chunks are already processed!")
            return

        # Mark every pending chunk in-progress, as the old worker did under a lock,
        # so --resume can purge anything left half-written.
        for chunk in pending_chunks:
            key = f"{chunk['doc_id']}::{chunk['chunk_id']}"
            self.extraction_log.setdefault(key, False)

        pool = pool_for_stage(self.args, settings, STAGE)
        try:
            async with pool:
                result = await pool.map(
                    pending_chunks,
                    self.extract_chunk,
                    attempts=settings.attempts,
                    desc=f"Extracting [{self.domain}]",
                    key_fn=lambda c: f"{c['doc_id']}::{c['chunk_id']}",
                    budget=self.budget,
                    debug=self.debug,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=self.persist_extraction,
                )
        finally:
            await save_log_async(self.log_file, self.extraction_log)
            self.debug.close()

        for failure in result.failures:
            await self.write_failure_debug(pending_chunks[failure["index"]], failure)

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        failed_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]
        if failed_chunks:
            failed_keys = [f"{c['doc_id']}::{c['chunk_id']}" for c in failed_chunks]
            print(f"[ERROR] {len(failed_chunks)} chunk(s) are incomplete after retries. Examples: {failed_keys[:10]}")
            raise RuntimeError(
                f"Pass 1 left {len(failed_chunks)} chunk(s) incomplete; rerun with --resume after checking server health."
            )

        print(f"\nSuccessfully finished extraction loop for domain: {self.domain}! All {len(all_chunks)} chunks are valid.")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run Pass 1 concept extraction to completion.

    Reads the corpus given by ``--corpus`` (or the legacy
    ``--data_dir``/``--domain`` pair), extracts concepts from every unit through
    the server pool, and writes per-chunk results under ``staging/`` plus the
    citation mapping under ``doc_mapping.json``. Resumes from
    ``extraction_log.json`` unless ``--resume`` is omitted.

    Returns:
        None. Exits non-zero if a stage budget expires or the pool reports
        unresolved failures.
    """
    parser = build_parser(
        stage=STAGE,
        description="High-throughput concept extractor: corpus -> per-chunk concepts",
        include_data_dir=True,
        include_corpus=True,
        include_resume=True)
    args = parser.parse_args()
    pipeline = WikiExtractorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()