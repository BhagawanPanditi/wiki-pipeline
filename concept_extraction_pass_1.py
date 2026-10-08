#!/usr/bin/env python3
import os
import re
import json
import time
import hashlib
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from atomic_io import atomic_write_json

# ── Server & Concurrency Configuration ───────────────────────────────────────

from pipeline_config import load_global_config
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

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

def short_hash(text: str) -> str:
    """Computes MD5 hash for string deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def clean_json_string(raw_response: str) -> str:
    """Extracts clean JSON string from raw model outputs."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response.strip()

def _save_log_sync(log_file: Path, data: dict):
    """Synchronous atomic write using a temporary file to avoid corrupt states."""
    tmp_file = log_file.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, log_file)

async def save_log_async(log_file: Path, data: dict):
    """Offloads atomic log write to a thread pool to avoid blocking the event loop."""
    await asyncio.to_thread(_save_log_sync, log_file, data)

# ── Pre-Deduplication & Chunking ─────────────────────────────────────────────

def chunk_document(
    doc_id: str,
    content: str,
    tokenizer,
    chunk_size: int = 4500,
    overlap: int = 400
) -> List[Dict[str, Any]]:
    """Splits document into token chunks with context before/after."""
    tokens = tokenizer.encode(content)
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return []

    chunks = []
    step = chunk_size - overlap
    chunk_idx = 0

    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        
        chunk_tokens = tokens[start:end]
        before_tokens = tokens[max(0, start - overlap):start]
        after_tokens = tokens[end:min(total_tokens, end + overlap)]

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"chunk_{chunk_idx:03d}",
            "current_chunk": tokenizer.decode(chunk_tokens, skip_special_tokens=True),
            "context_before": tokenizer.decode(before_tokens, skip_special_tokens=True),
            "context_after": tokenizer.decode(after_tokens, skip_special_tokens=True),
        })
        chunk_idx += 1
        if end >= total_tokens:
            break

    return chunks

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
        self.args = args
        self.domain = args.domain
        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.log_file = self.base_wiki_dir / "extraction_log.json"

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.servers = args.servers
        self.concurrency = args.concurrency_per_server
        self.extraction_log = self._load_log()

    def _load_log(self) -> Dict[str, bool]:
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                print("Warning: Log file corrupted. Initializing fresh extraction log.")
                return {}
        return {}

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
        _save_log_sync(self.log_file, self.extraction_log)
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
        """Deduplicate documents and reuse chunks only when their inputs match."""
        chunks_cache_file = self.cache_dir / "chunks_cache.json"
        cache_manifest_file = self.cache_dir / "chunks_cache_manifest.json"
        mapping_file = self.cache_dir / "doc_mapping.json"
        doc_path = Path(self.args.data_dir) / f"{self.domain}.jsonl"
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing domain file: {doc_path}")

        # A pre-existing chunks_cache.json is not sufficient evidence that its
        # chunks describe today's input. Hash the source and bind the cache to
        # the tokenizer name and chunking parameters before reusing it.
        source_digest = hashlib.sha256()
        with doc_path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                source_digest.update(block)
        expected_manifest = {
            "format_version": 1,
            "source_sha256": source_digest.hexdigest(),
            "tokenizer": self.args.model_name,
            "chunk_size": 4500,
            "overlap": 400,
        }

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

        print(f"Loading documents from {doc_path}...")
        raw_docs = []
        with open(doc_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_docs.append(json.loads(line))

        print("Deduplicating documents based on content hash...")
        by_hash = {}
        for d in raw_docs:
            h = short_hash(d["content"])
            if h not in by_hash:
                by_hash[h] = f"canon_{h}"

        canon_docs = {}
        for d in raw_docs:
            h = short_hash(d["content"])
            cid = by_hash[h]
            if cid not in canon_docs:
                canon_docs[cid] = d["content"]

        # Refresh the mapping whenever the source signature changes, rather than
        # retaining mappings that may belong to an older version of the corpus.
        def fix_mojibake(text: str) -> str:
            try:
                return text.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
                return text

        doc_mapping = {}
        for d in raw_docs:
            raw_id = d.get("id", d.get("doc_id", "UNKNOWN_ID"))
            clean_id = fix_mojibake(raw_id)
            doc_mapping[clean_id] = f"canon_{short_hash(d['content'])}"
        atomic_write_json(mapping_file, doc_mapping)

        print(f"Deduplicated {len(raw_docs)} docs down to {len(canon_docs)} unique canonical docs.")
        print("Tokenizing and creating chunks (4500 tokens window + overlap)...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True, local_files_only=True)
        
        all_chunks = []
        for canon_id, content in canon_docs.items():
            doc_chunks = chunk_document(canon_id, content, tokenizer)
            all_chunks.extend(doc_chunks)

        atomic_write_json(chunks_cache_file, all_chunks)
        atomic_write_json(cache_manifest_file, expected_manifest)

        print(f"Cached {len(all_chunks)} chunks to disk.")
        return all_chunks

    async def extract_chunk(
            self, client: AsyncOpenAI, chunk_data: Dict[str, Any]
        ) -> Tuple[Optional[ChunkExtractionSchema], List[Dict[str, Any]]]:
            """Performs LLM call and validates response via Pydantic."""
            user_prompt = f"""\
    CONTEXT BEFORE:
    {chunk_data['context_before']}

    CURRENT CHUNK:
    {chunk_data['current_chunk']}

    CONTEXT AFTER:
    {chunk_data['context_after']}
    """
            max_attempts = 4  # Reduced from 10
            error_logs = []

            for attempt in range(max_attempts):
                raw_text = None
                finish_reason = None
                try:
                    response = await client.chat.completions.create(
                        model=self.args.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=self.args.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    
                    choice = response.choices[0]
                    finish_reason = choice.finish_reason
                    raw_text = choice.message.content

                    if finish_reason == "length":
                        # FAIL FAST: Do not retry if we hit the token limit
                        error_logs.append({
                            "attempt": attempt + 1,
                            "finish_reason": finish_reason,
                            "truncated": True,
                            "error_type": "TokenLimitExceeded",
                            "error_message": "Response truncated: Hit max_tokens limit.",
                            "raw_response": raw_text
                        })
                        return None, error_logs

                    cleaned_json = clean_json_string(raw_text)
                    parsed = ChunkExtractionSchema.model_validate_json(cleaned_json)
                    return parsed, error_logs

                except Exception as exc:
                    error_info = {
                        "attempt": attempt + 1,
                        "finish_reason": finish_reason,
                        "truncated": finish_reason == "length",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_response": raw_text
                    }
                    error_logs.append(error_info)

                    if attempt == max_attempts - 1:
                        return None, error_logs
                    
                    # Cap the backoff sleep at 15 seconds max (1, 2, 4, 8)
                    await asyncio.sleep(min(15, 2 ** attempt))

            return None, error_logs

    async def worker_task(
        self,
        client: AsyncOpenAI,
        queue: asyncio.Queue,
        lock: asyncio.Lock,
        pbar: tqdm,
        flush_counter: dict
    ):
        """Worker task executing in parallel with batched atomic log updates."""
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            doc_id = item["doc_id"]
            chunk_id = item["chunk_id"]
            key = f"{doc_id}::{chunk_id}"

            # Check execution state atomically
            async with lock:
                if self.extraction_log.get(key, False):
                    queue.task_done()
                    pbar.update(1)
                    continue
                self.extraction_log[key] = False

            extracted, error_logs = await self.extract_chunk(client, item)

            subfolder = self.staging_dir / doc_id / chunk_id
            subfolder.mkdir(parents=True, exist_ok=True)

            if extracted is not None:
                # SUCCESS: Save concepts.json
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

                async with lock:
                    self.extraction_log[key] = True
                    flush_counter["count"] += 1

                    if flush_counter["count"] >= 50:
                        await save_log_async(self.log_file, self.extraction_log)
                        flush_counter["count"] = 0
            else:
                # FAILURE: Save debug.json detailing truncation / errors
                debug_file = subfolder / "debug.json"
                debug_data = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "total_attempts": len(error_logs),
                    "chunk_metadata": {
                        "context_before_chars": len(item.get("context_before", "")),
                        "current_chunk_chars": len(item.get("current_chunk", "")),
                        "context_after_chars": len(item.get("context_after", "")),
                    },
                    "attempts": error_logs
                }
                debug_payload = json.dumps(debug_data, indent=2)
                await asyncio.to_thread(debug_file.write_text, debug_payload, encoding="utf-8")

            pbar.update(1)
            queue.task_done()

    async def run(self):
        if self.args.resume:
            self.cleanup_incomplete_states()

        all_chunks = self.prepare_chunks()
        pending_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]

        total_pending = len(pending_chunks)
        print(f"Total chunks: {len(all_chunks)} | Pending chunks: {total_pending}")
        print(f"Targeting {len(self.servers)} servers with {self.concurrency} concurrent workers each.")
        print(f"Total Active Concurrency Limit: {len(self.servers) * self.concurrency} in flight.")

        if not pending_chunks:
            print("All chunks are already processed!")
            return

        queue = asyncio.Queue()
        for chunk in pending_chunks:
            await queue.put(chunk)

        total_workers = len(self.servers) * self.concurrency
        for _ in range(total_workers):
            await queue.put(None)

        lock = asyncio.Lock()
        flush_counter = {"count": 0}
        pbar = tqdm(total=total_pending, desc=f"Extracting [{self.domain}]")

        tasks = []
        http_clients = []
        openai_clients = []

        for server_url in self.servers:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.concurrency + 20,
                    max_keepalive_connections=self.concurrency
                ),
                timeout=httpx.Timeout(300.0, connect=30.0)
            )
            http_clients.append(http_client)

            client = AsyncOpenAI(
                base_url=server_url,
                api_key="EMPTY",
                http_client=http_client,
                max_retries=0
            )
            openai_clients.append(client)

        # Interleave worker creation round-robin so requests are uniformly distributed across all servers
        for worker_idx in range(self.concurrency):
            for client in openai_clients:
                task = asyncio.create_task(
                    self.worker_task(client, queue, lock, pbar, flush_counter)
                )
                tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        finally:
            await save_log_async(self.log_file, self.extraction_log)
            pbar.close()

            for http_client in http_clients:
                await http_client.aclose()

        failed_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]
        if failed_chunks:
            failed_keys = [f"{c['doc_id']}::{c['chunk_id']}" for c in failed_chunks]
            print(f"[ERROR] {len(failed_chunks)} chunk(s) are incomplete after retries. Examples: {failed_keys[:10]}")
            raise RuntimeError(
                f"Pass 1 left {len(failed_chunks)} chunk(s) incomplete; rerun with --resume after checking server health."
            )

        print(f"\nSuccessfully finished extraction loop for domain: {self.domain}! All {len(all_chunks)} chunks are valid.")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="High-Throughput Concept Extractor for BRIGHT Wiki Benchmark")
    parser.add_argument("--domain", type=str, required=True, help="Target domain (e.g. biology, earth_science)")
    parser.add_argument("--data_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/data/long_documents")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model_name", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max generation length for LLM responses.")
    parser.add_argument("--resume", action="store_true", help="Purge incomplete extractions and resume processing.")

    args = parser.parse_args()
    pipeline = WikiExtractorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()