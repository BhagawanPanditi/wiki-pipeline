#!/usr/bin/env python3
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils.atomic_io import atomic_write_json
from utils.checkpoint import load_bool_log, save_log_async, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.corpus import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
                          build_units, cache_manifest, deduplicate,
                          load_corpus, resolve_corpus_path,
                          save_run_config)
from utils.errors import PermanentFailure, StageBudget
from utils.llm_pool import ServerPool

STAGE = "extract-concepts"

# ── Data Schemas ─────────────────────────────────────────────────────────────

class OutboundRelation(BaseModel):
    target_concept: str = Field(description="Exact canonical title of target concept.")
    relation: str = Field(description="1 concise sentence describing the relationship.")

class Concept(BaseModel):
    title: str = Field(description="Exact canonical title (1-6 words, singular).")
    aliases: List[str] = Field(default_factory=list, description="Acronyms or alternate names.")
    description: str = Field(description="Single dense factual paragraph.")
    outbound_relations: List[OutboundRelation] = Field(default_factory=list)

class ChunkExtractionSchema(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)

# ── System Prompt (XML Format) ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are building a knowledge graph from documents in any domain. Read the CURRENT CHUNK and extract its concepts, each with a fact-dense description and outbound relations.

SOURCE RULES:
- Facts come ONLY from the CURRENT CHUNK. Use CONTEXT BEFORE/AFTER solely to resolve pronouns, acronyms, and full names. Never extract facts that appear only in the context, and never add outside knowledge.
- Silently ignore HTML, scraping artifacts, navigation, headers/footers, ads, cookie notices, and bibliographies. If nothing substantive remains, return an empty response.

WHAT IS A CONCEPT:
- A distinct subject the chunk explains in meaningful detail: an idea, method, algorithm, mechanism, entity, system, event, rule, condition, parameter, etc.
- Extract every such concept; there is no cap. Skip passing mentions that have no detail.
- Do not create concepts from structural labels ("Introduction", "Results", "Example") or a summary of the whole chunk.

TITLES AND ALIASES:
- Title: 1-6 words, Title Case, standard canonical name (expand acronyms), singular unless plural by definition.
- Aliases: Semicolon-separated list of acronyms or alternate names appearing in text. If none, leave empty.

DESCRIPTIONS:
- One self-contained paragraph: name the subject explicitly, no "this" or "the above". Start with what it is, then the details.
- Preserve all specifics in full sentences: numbers, units, ranges, thresholds, formulas, parameters, complexity, constraints, conditions, steps, dates, locations, and attributions.

OUTBOUND RELATIONS:
- The concept being described is always the subject. In one sentence, state how it affects, uses, causes, is part of, replaces, or contrasts with the target.
- Target must be named in the chunk or context. No self-relations.
- Format each relation on a new line: Target Concept :: 1 sentence relation.

OUTPUT FORMAT:
Output each concept using XML tags matching this exact structure:
<concept>
<title>Exact Canonical Title</title>
<aliases>Acronym; Alternative Name</aliases>
<description>A single comprehensive paragraph detailing what the concept is along with all raw quantitative metrics, parameter bounds, specific findings, and detailed statements from this chunk.</description>
<relations>
Target Concept A :: How this concept relates to Target Concept A.
Target Concept B :: How this concept relates to Target Concept B.
</relations>
</concept>

If no concepts exist in the chunk, output nothing.
"""

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_concepts(text: str) -> List[Concept]:
    """Parse all completed <concept>...</concept> blocks from raw LLM output."""
    if not text:
        return []

    concepts = []
    blocks = re.findall(r"<concept>(.*?)</concept>", text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m:
            continue
        title = title_m.group(1).strip()
        if not title:
            continue

        desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE)
        description = desc_m.group(1).strip() if desc_m else ""

        aliases = []
        aliases_m = re.search(r"<aliases>(.*?)</aliases>", block, re.DOTALL | re.IGNORECASE)
        if aliases_m:
            raw_aliases = aliases_m.group(1).strip()
            aliases = [a.strip() for a in re.split(r"[;,]", raw_aliases) if a.strip()]

        relations = []
        rel_m = re.search(r"<relations>(.*?)</relations>", block, re.DOTALL | re.IGNORECASE)
        if rel_m:
            for line in rel_m.group(1).strip().splitlines():
                line = line.strip()
                if "::" in line:
                    parts = line.split("::", 1)
                    tgt = parts[0].strip()
                    desc = parts[1].strip()
                    if tgt and desc:
                        relations.append(OutboundRelation(target_concept=tgt, relation=desc))

        concepts.append(Concept(
            title=title,
            aliases=aliases,
            description=description,
            outbound_relations=relations
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
            tok_arg = getattr(self.args, "tokenizer", None)
            if tok_arg in ("tiktoken", "cl100k_base", "o200k_base"):
                enc_name = "o200k_base" if tok_arg == "o200k_base" else "cl100k_base"
                print(f"Using instant offline tiktoken ({enc_name})...")
                import tiktoken
                enc = tiktoken.get_encoding(enc_name)
                class TiktokenWrapper:
                    def encode(self, text: str) -> list:
                        return enc.encode(text, disallowed_special=())
                    def decode(self, tokens: list, skip_special_tokens: bool = True) -> str:
                        return enc.decode(tokens)
                tokenizer = TiktokenWrapper()
            else:
                from transformers import AutoTokenizer
                print(f"Loading tokenizer for chunking ({chunk_size} tokens, {overlap} overlap)...")
                tok_candidates = []
                if tok_arg:
                    tok_candidates.append(tok_arg)
                tok_candidates.extend([
                    self.settings.model_name,
                    "Qwen/Qwen2.5-32B-Instruct",
                    "Qwen/Qwen2.5-14B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2-7B-Instruct",
                ])

                # Try local cache first for all candidates to avoid network hangs
                for name in tok_candidates:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            name, trust_remote_code=True, local_files_only=True)
                        print(f"Successfully loaded local tokenizer: {name}")
                        break
                    except Exception:
                        continue

                # Fallback to tiktoken if no local transformers model found
                if tokenizer is None:
                    try:
                        print("Local HuggingFace tokenizer not found; falling back to instant tiktoken (cl100k_base)...")
                        import tiktoken
                        enc = tiktoken.get_encoding("cl100k_base")
                        class TiktokenWrapper:
                            def encode(self, text: str) -> list:
                                return enc.encode(text, disallowed_special=())
                            def decode(self, tokens: list, skip_special_tokens: bool = True) -> str:
                                return enc.decode(tokens)
                        tokenizer = TiktokenWrapper()
                    except ImportError:
                        primary = tok_candidates[0]
                        print(f"Downloading/loading tokenizer: {primary}...")
                        tokenizer = AutoTokenizer.from_pretrained(primary, trust_remote_code=True)
        else:
            print("Chunking disabled: treating each document as a single unit.")

        all_chunks = build_units(canon_docs, tokenizer, chunking=chunking,
                                 chunk_size=chunk_size, overlap=overlap)

        atomic_write_json(chunks_cache_file, all_chunks)
        atomic_write_json(cache_manifest_file, expected_manifest)
        print(f"Cached {len(all_chunks)} unit(s) to disk.")
        return all_chunks

    async def extract_chunk(self, client, chunk_data: Dict[str, Any], server_index: int) -> ChunkExtractionSchema:
        key = f"{chunk_data['doc_id']}::{chunk_data['chunk_id']}"
        user_prompt = f"""\
CONTEXT BEFORE:
{chunk_data['context_before']}

CURRENT CHUNK:
{chunk_data['current_chunk']}

CONTEXT AFTER:
{chunk_data['context_after']}
"""
        response = await client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=self.settings.max_tokens,
        )

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason

        completed_concepts = parse_xml_concepts(raw_text)

        # ── Truncation Handling via Continuation ──────────────────────────────
        if finish_reason == "length":
            # Detect unclosed concept block
            last_open_idx = raw_text.rfind("<concept>")
            last_close_idx = raw_text.rfind("</concept>")
            incomplete_title = None

            if last_open_idx > last_close_idx:
                unclosed_block = raw_text[last_open_idx:]
                title_m = re.search(r"<title>(.*?)(?:</title>|\n|$)", unclosed_block, re.DOTALL | re.IGNORECASE)
                if title_m and title_m.group(1).strip():
                    incomplete_title = title_m.group(1).strip()

            already_titles = [c.title for c in completed_concepts]
            cut_off_desc = f"'{incomplete_title}'" if incomplete_title else "the last incomplete concept"

            continuation_prompt = f"""\
You were extracting concepts from the CURRENT CHUNK, but hit the token limit.

ALREADY COMPLETED CONCEPTS:
{', '.join(already_titles) if already_titles else 'None'}

TASK:
Continue extracting for the rest of the CURRENT CHUNK.
Start by providing the complete <concept> block for {cut_off_desc}, then continue extracting any remaining concepts from the CURRENT CHUNK.
Output in the exact same <concept>...</concept> XML format.
"""
            # Valid prefix of prior response up to last completed concept
            assistant_content = raw_text[:last_close_idx + len("</concept>")] if last_close_idx != -1 else raw_text

            try:
                cont_response = await client.chat.completions.create(
                    model=self.settings.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": continuation_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=self.settings.max_tokens,
                )
                cont_text = cont_response.choices[0].message.content or ""
                continued_concepts = parse_xml_concepts(cont_text)

                # Merge and deduplicate by canonical title
                merged_concepts = list(completed_concepts)
                seen_titles = {c.title.lower() for c in completed_concepts}
                for cc in continued_concepts:
                    if cc.title.lower() not in seen_titles:
                        seen_titles.add(cc.title.lower())
                        merged_concepts.append(cc)

                return ChunkExtractionSchema(concepts=merged_concepts)
            except Exception:
                # If continuation fails, return the completed concepts salvaged so far
                if completed_concepts:
                    return ChunkExtractionSchema(concepts=completed_concepts)
                raise PermanentFailure(f"{key}: response truncated and continuation failed.")

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
        pool = ServerPool(
            settings.servers,
            concurrency_per_server=settings.concurrency_per_server,
            request_timeout=settings.profile.request_timeout,
            connect_timeout=settings.profile.connect_timeout,
            label=STAGE,
            llm_log=None,
            failover=not getattr(self.args, "no_failover", False),
        )

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

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        failed_chunks = [c for c in all_chunks if not self.chunk_is_complete(c)]
        if failed_chunks:
            failed_keys = [f"{c['doc_id']}::{c['chunk_id']}" for c in failed_chunks]
            print(f"[ERROR] {len(failed_chunks)} chunk(s) incomplete after retries. Examples: {failed_keys[:10]}")
            raise RuntimeError(f"Pass 1 left {len(failed_chunks)} chunk(s) incomplete; rerun with --resume.")

        print(f"\nSuccessfully finished extraction for: {self.base_wiki_dir.name}! All {len(all_chunks)} chunks are valid.")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser(
        stage=STAGE,
        description="High-throughput concept extractor: corpus -> per-chunk concepts",
        include_corpus=True,
        include_resume=True)
    args = parser.parse_args()
    pipeline = WikiExtractorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
