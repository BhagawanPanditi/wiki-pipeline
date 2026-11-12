#!/usr/bin/env python3
"""Stage 1: Concept Extraction (Pass 1).
-------------------------------------
Extracts substantive, canonical concepts from raw document chunks using direct,
domain-agnostic XML extraction without thinking-mode overhead. Retains concrete
names, metrics, facts, and attributes while capturing narrow sub-topics in aliases.

Usage:
    python3 extract_concepts.py --input_path /path/to/domain.jsonl --wiki_path /path/to/wiki
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json
from utils.checkpoint import load_bool_log, save_log_async, save_log_sync
from utils.cli import build_parser, resolve_settings
from config import _read_config
from utils.corpus import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
                          build_units, cache_manifest, deduplicate,
                          load_corpus, resolve_corpus_path,
                          save_run_config)
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import write_stage_manifest

STAGE = "extract-concepts"
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_CHUNK_OVERLAP = 500

# ── Data Schemas ─────────────────────────────────────────────────────────────

class Concept(BaseModel):
    title: str = Field(description="Exact canonical title (1-6 words, singular).")
    aliases: List[str] = Field(default_factory=list, description="Explicit acronyms or alternate names.")
    subtopics: List[str] = Field(default_factory=list, description="Narrower absorbed subtopics or parameters.")
    description: str = Field(description="Single dense factual paragraph retaining all metrics and details.")

class ChunkExtractionSchema(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)

# ── System Prompt (XML Format) ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert knowledge extractor building a Knowledge Wiki from raw documents.
Your goal is to transform the CURRENT CHUNK into solid, fact-dense concepts that serve as a complete, superior replacement for the raw text. Once extracted, a reader should never need to look at the original chunk.

INPUT & NOISE CLEANING:
- Silently ignore HTML tags, scraping artifacts, navigation menus, headers/footers, cookie notices, and boilerplate.
- If the chunk contains no substantive informational content, output nothing.

CONTEXT BOUNDARIES:
- CURRENT CHUNK is your source for facts. Use CONTEXT BEFORE / CONTEXT AFTER ONLY to resolve pronouns (e.g. "this system", "they") or acronyms to standard canonical titles.
- Never invent facts or add outside knowledge.

TITLES AND ALIASES:
- Title: 1-6 words, Title Case, the full standard name (expand acronyms), singular unless the standard name is plural. Use the canonical name any other document would use for the same subject, ensuring concepts merge cleanly across sources.
- Aliases: Semicolon-separated list of acronyms, alternate designations, or specific minor sub-topics/parameters absorbed into this concept. If none, leave empty.

SIZE GUIDELINES (words):
- A well-formed concept description is typically 150-450 words.
- If a topic has fewer than ~45 words of substance in this chunk, do not create a concept; absorb its facts into the parent concept's description and record its name in <subtopics>.
- If one topic spans well over 700 words in this chunk, split at natural thematic boundaries into separate concepts.
- Never pad. Do not restate the title in every sentence or add commentary ("This concept covers...").

ALIASES vs SUBTOPICS:
- <aliases>: only true synonyms, acronyms, and alternate names for THIS concept.
- <subtopics>: names of narrower parameters, variants, or sub-steps whose facts you absorbed into this description.

EMPTY CASE:
If the CURRENT CHUNK has no substantive informational content, output exactly: <no_concepts/>

OUTPUT FORMAT:
Output ONLY completed <concept>...</concept> XML blocks with no introductory or concluding text:
<concept>
<title>Exact Canonical Title</title>
<aliases>Acronym or Alt Name 1; Alt Name 2</aliases>
<subtopics>Absorbed Subtopic 1; Absorbed Subtopic 2</subtopics>
<description>A single comprehensive, fact-dense paragraph detailing what the concept is along with all substantive facts, properties, and specific details from this chunk.</description>
</concept>

If no substantive concepts exist in the chunk, output exactly: <no_concepts/>
"""

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_concepts(text: str) -> List[Concept]:
    """Parse all completed <concept>...</concept> blocks directly from raw LLM output."""
    if not text:
        return []

    concepts = []
    blocks = re.findall(r"<concept>((?:(?!<concept>).)*?)</concept>", text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m:
            continue
        title = title_m.group(1).strip()
        if not title:
            continue

        desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE)
        description = desc_m.group(1).strip() if desc_m else ""

        # 3. Clean up in case description contains residual nested tags
        if "<concept>" in description or "<title>" in description:
            inner_desc_m = re.findall(r"<description>(.*?)(?:</description>|$)", description, re.DOTALL | re.IGNORECASE)
            if inner_desc_m:
                description = inner_desc_m[-1].strip()
            description = re.sub(r"<.*?>", "", description).strip()

        aliases = []
        aliases_m = re.search(r"<aliases>(.*?)</aliases>", block, re.DOTALL | re.IGNORECASE)
        if aliases_m:
            raw_aliases = aliases_m.group(1).strip()
            aliases = [a.strip() for a in re.split(r"[;,]", raw_aliases) if a.strip()]

        subtopics = []
        subtopics_m = re.search(r"<subtopics>(.*?)</subtopics>", block, re.DOTALL | re.IGNORECASE)
        if subtopics_m:
            raw_subs = subtopics_m.group(1).strip()
            subtopics = [s.strip() for s in re.split(r"[;,]", raw_subs) if s.strip()]

        concepts.append(Concept(
            title=title,
            aliases=aliases,
            subtopics=subtopics,
            description=description,
        ))

    return concepts

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiExtractorPipeline:
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument. Specify where to save wiki outputs.")

        self.base_wiki_dir = Path(wiki_path)
        self.base_wiki_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.log_file = self.base_wiki_dir / "extraction_log.json"

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_log = load_bool_log(
            self.log_file,
            corrupt_message="Log file corrupted. Initializing fresh extraction log.")
        self.cfg = _read_config()

    def cleanup_incomplete_states(self):
        """Remove partial concept files for incomplete chunks."""
        purged = 0
        for key, completed in list(self.extraction_log.items()):
            if not completed:
                doc_id, chunk_id = key.split("::", 1)
                partial_concepts = self.staging_dir / doc_id / chunk_id / "concepts.json"
                if partial_concepts.exists():
                    partial_concepts.unlink()
                    purged += 1
                del self.extraction_log[key]
        save_log_sync(self.log_file, self.extraction_log)
        if purged > 0:
            print(f"Cleared {purged} partial concept files.")

    def chunk_is_complete(self, chunk: Dict[str, Any]) -> bool:
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
        chunks_cache_file = self.cache_dir / "chunks_cache.json"
        cache_manifest_file = self.cache_dir / "chunks_cache_manifest.json"
        mapping_file = self.base_wiki_dir / "doc_mapping.json"

        corpus_path = resolve_corpus_path(self.args)
        chunking = bool(getattr(self.args, "chunking", True))
        chunk_size = int(getattr(self.args, "chunk_size", DEFAULT_CHUNK_SIZE))
        overlap = int(getattr(self.args, "chunk_overlap", DEFAULT_CHUNK_OVERLAP))

        expected_manifest = cache_manifest(
            corpus_path,
            tokenizer_name=self.settings.model_name if chunking else None,
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
            print("Cached chunks are stale or invalid; rebuilding them from current input.")

        print(f"Loading corpus from {corpus_path}...")
        docs = load_corpus(corpus_path)

        print("Deduplicating documents based on content hash...")
        canon_docs, doc_mapping = deduplicate(docs)
        atomic_write_json(mapping_file, doc_mapping)
        print(f"Deduplicated {len(docs)} docs down to {len(canon_docs)} unique canonical docs.")

        save_run_config(self.base_wiki_dir, {
            "corpus": str(corpus_path),
            "corpus_sha256": expected_manifest["source_sha256"],
            "wiki_path": str(self.base_wiki_dir),
            "model_name": self.settings.model_name,
            "chunking": chunking,
            "chunk_size": chunk_size if chunking else None,
            "chunk_overlap": overlap if chunking else None,
            "documents": len(docs),
            "unique_documents": len(canon_docs),
        })

        tokenizer = None
        if chunking:
            print(f"Loading instant offline tiktoken (cl100k_base) for chunking ({chunk_size} tokens, {overlap} overlap)...")
            import tiktoken
            tok_arg = getattr(self.args, "tokenizer", None)
            enc_name = "o200k_base" if tok_arg == "o200k_base" else "cl100k_base"
            enc = tiktoken.get_encoding(enc_name)

            class TiktokenWrapper:
                def encode(self, text: str) -> list:
                    return enc.encode(text, disallowed_special=())

                def decode(self, tokens: list, skip_special_tokens: bool = True) -> str:
                    return enc.decode(tokens)

            tokenizer = TiktokenWrapper()
        else:
            print("Chunking disabled: treating each document as a single unit.")

        all_chunks = build_units(canon_docs, tokenizer, chunking=chunking,
                                 chunk_size=chunk_size, overlap=overlap)

        atomic_write_json(chunks_cache_file, all_chunks)
        atomic_write_json(cache_manifest_file, expected_manifest)
        print(f"Cached {len(all_chunks)} unit(s) to disk.")
        return all_chunks

    async def extract_chunk(self, client, chunk_data: Dict[str, Any], server_index: int) -> ChunkExtractionSchema:
        user_prompt = f"""\
CONTEXT BEFORE:
{chunk_data['context_before']}

CURRENT CHUNK:
{chunk_data['current_chunk']}

CONTEXT AFTER:
{chunk_data['context_after']}

Extract the substantive concepts from CURRENT CHUNK now according to the system instructions.
"""
        max_model_len = getattr(self.args, "max_model_len", None) or (self.cfg.get("max_model_len") if hasattr(self, "cfg") else None)
        raw_text = await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.settings.max_tokens,
            args=self.args,
            cfg=getattr(self, "cfg", None),
            task="generate",
            max_growth=1,
            max_model_len=max_model_len,
        )

        completed_concepts = parse_xml_concepts(raw_text)

        # F-04: Disambiguate empty output from failure
        # If output has neither <concept> nor sentinel <no_concepts/>, and chunk has substance, raise recoverable error to retry
        if not completed_concepts and "<no_concepts" not in raw_text.lower():
            chunk_text = chunk_data.get("current_chunk", "")
            chunk_word_count = len(chunk_text.split())
            if chunk_word_count > 150:
                raise ValueError(
                    f"LLM produced no concepts and missing <no_concepts/> sentinel for non-empty chunk "
                    f"({chunk_word_count} words). Retrying."
                )

        return ChunkExtractionSchema(concepts=completed_concepts)

    async def persist_extraction(self, chunk_data: Dict[str, Any], extracted: ChunkExtractionSchema) -> None:
        doc_id = chunk_data["doc_id"]
        chunk_id = chunk_data["chunk_id"]
        key = f"{doc_id}::{chunk_id}"

        subfolder = self.staging_dir / doc_id / chunk_id
        await asyncio.to_thread(subfolder.mkdir, parents=True, exist_ok=True)

        concepts_file = subfolder / "concepts.json"
        json_payload = extracted.model_dump_json(indent=2)
        await asyncio.to_thread(concepts_file.write_text, json_payload, encoding="utf-8")

        self.extraction_log[key] = True
        self.flush_count += 1
        if self.flush_count >= 50:
            await save_log_async(self.log_file, self.extraction_log)
            self.flush_count = 0

    async def run(self):
        self.settings = resolve_settings(self.args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.flush_count = 0

        if self.args.resume:
            self.cleanup_incomplete_states()

        all_chunks = self.prepare_chunks()
        pending_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]

        settings = self.settings
        pool = pool_for_stage(self.args, settings, STAGE)

        total_in_flight = sum(pool.server_concurrencies)
        print(f"Total chunks: {len(all_chunks)} | Pending chunks: {len(pending_chunks)}")
        print(f"Active Server Pool ({len(pool.servers)} servers, {total_in_flight} total concurrency):")
        for url, conc in zip(pool.servers, pool.server_concurrencies):
            print(f"  {url:<38} -> {conc} concurrent workers")

        if not pending_chunks:
            print("All chunks are already processed!")
            return

        for chunk in pending_chunks:
            key = f"{chunk['doc_id']}::{chunk['chunk_id']}"
            self.extraction_log.setdefault(key, False)
        async with pool:
            result = await pool.map(
                pending_chunks,
                self.extract_chunk,
                attempts=settings.attempts,
                desc=f"Extracting [{self.base_wiki_dir.name}]",
                key_fn=lambda c: f"{c['doc_id']}::{c['chunk_id']}",
                budget=self.budget,
                debug=None,
                attempt_timeout=settings.profile.attempt_timeout,
                on_result=self.persist_extraction,
            )

        await save_log_async(self.log_file, self.extraction_log)
        result.raise_if_failed(STAGE)

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        failed_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]
        if failed_chunks:
            failed_keys = [f"{c['doc_id']}::{c['chunk_id']}" for c in failed_chunks]
            print(f"[ERROR] {len(failed_chunks)} chunk(s) incomplete after retries. Examples: {failed_keys[:10]}")
            raise RuntimeError(f"Pass 1 left {len(failed_chunks)} chunk(s) incomplete; rerun with --resume.")

        write_stage_manifest(
            wiki_path=self.base_wiki_dir,
            stage=STAGE,
            expected=len(all_chunks),
            completed=len(all_chunks) - len(failed_chunks),
            failed=len(failed_chunks),
        )
        print(f"\nSuccessfully finished extraction for: {self.base_wiki_dir.name}! All {len(all_chunks)} chunks are valid.")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser(
        stage=STAGE,
        description="High-throughput concept extractor: corpus -> per-chunk concepts",
        include_input_path=True,
        include_resume=True)
    args = parser.parse_args()
    pipeline = WikiExtractorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
