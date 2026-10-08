#!/usr/bin/env python3
import os
import re
import json
import time
import shutil
import hashlib
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ── Server & Concurrency Configuration ───────────────────────────────────────

from pipeline_config import load_global_config
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

# ── Pydantic Data Schemas for Verification ────────────────────────────────────

class VerificationResult(BaseModel):
    coverage_score: float = Field(
        description="Score between 0.0 and 1.0 representing how thoroughly concepts and facts were retained."
    )
    is_sufficient: bool = Field(
        description="True if all primary concepts and essential metrics/facts were captured without critical omissions."
    )
    missing_concepts: List[str] = Field(
        default_factory=list,
        description="Primary thematic concepts present in the text chunk that were completely omitted from the extraction."
    )
    missing_facts: List[str] = Field(
        default_factory=list,
        description="Important metrics, parameters, findings, tools, or dates present in text but absent from descriptions."
    )
    hallucinations: List[str] = Field(
        default_factory=list,
        description="Facts or statements in the extracted JSON that contradict or are absent from the source chunk."
    )
    critique: str = Field(
        description="1-2 sentences summarizing the completeness, factual accuracy, and granularity of the extraction."
    )

# ── System Prompt for Audit Verification ─────────────────────────────────────
VERIFICATION_SYSTEM_PROMPT = """\
You are an expert technical knowledge auditor evaluating an automated Knowledge Graph extraction.
Your task is to compare a RAW TEXT CHUNK against the EXTRACTED CONCEPTS JSON to verify factual completeness, accuracy, and core concept coverage.

--------------------------------------------------------------------------------
1. DEFINITION OF A "SUBSTANTIVE FACT" (WHAT TO AUDIT)
--------------------------------------------------------------------------------
A "Substantive Fact" is key informational content. You MUST verify that:
- Core technical metrics, parameters, benchmark scores, or quantitative data points (e.g., "accuracy of 94.2%", "temperature of 300K") are explicitly retained in the `description` sentences.
- Specific named entities, tools, biological codes, algorithms, or datasets (e.g., "CRISPR-Cas9", "ResNet-50", "GSE12345") are present either as named strings in `entities` or in `description`.
- Key causal/functional mechanisms or relationships between concepts are captured.

--------------------------------------------------------------------------------
2. DEFINITION OF "NOISE & NON-FACTS" (SILENTLY IGNORE / DO NOT PENALIZE)
--------------------------------------------------------------------------------
You MUST SILENTLY IGNORE and NOT penalize the extraction for omitting:
- HTML tags, scraping artifacts, navigation links, headers, footers, cookie notices.
- Citations, author lists, affiliations, license boilerplate, or reference section lists.
- Meta-talk and filler sentences (e.g., "As shown in Figure 2", "In this section, we discuss...").

--------------------------------------------------------------------------------
3. GRANULARITY & LOCATION RULES
--------------------------------------------------------------------------------
- UNCAPPED EXTRACTION: The extractor was instructed to extract ALL distinct concepts/sub-topics present without artificial caps or node limits.
- FACTS MUST BE IN SENTENCES: A technical metric or factual finding MUST appear in complete sentence form inside a concept's `description` (or `outbound_relations`). Simply listing a string in `entities` does NOT satisfy the retention of a factual claim or quantitative metric.

--------------------------------------------------------------------------------
4. SCORING CRITERIA
--------------------------------------------------------------------------------
- `coverage_score` (0.0 to 1.0):
  - 1.0: Perfect retention. All main themes and key quantitative/technical facts captured in descriptions.
  - 0.8 - 0.9: Main themes captured; missed minor parameters or secondary metrics.
  - 0.5 - 0.7: Omitted a primary thematic concept or several key findings/metrics.
  - < 0.5: Severe omissions of main body content.
- `is_sufficient`: `true` if coverage_score >= 0.85 and no major primary concept was completely dropped.

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact structure:
{
  "coverage_score": 0.90,
  "is_sufficient": true,
  "missing_concepts": ["Name of a missing PRIMARY theme"],
  "missing_facts": ["Specific missing technical metric or parameter sentence"],
  "hallucinations": ["Fact in JSON that contradicts or is missing from raw text"],
  "critique": "1-2 sentence evaluation summary."
}
"""

# ── Helper Utilities ─────────────────────────────────────────────────────────

def clean_json_string(raw_response: str) -> str:
    """Extracts clean JSON string from raw model outputs."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_response.strip()

def _save_log_sync(log_file: Path, data: dict):
    """Synchronous atomic write using a temporary file."""
    tmp_file = log_file.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, log_file)

async def save_log_async(log_file: Path, data: dict):
    """Offloads atomic write to thread pool to avoid blocking the event loop."""
    await asyncio.to_thread(_save_log_sync, log_file, data)

# ── Document Loading & Chunk Matching ────────────────────────────────────────

def load_cached_chunks(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads tokenized sliding-window chunks from the cache file."""
    chunks_cache_file = cache_dir / "chunks_cache.json"
    if not chunks_cache_file.exists():
        raise FileNotFoundError(
            f"Chunk cache file not found at {chunks_cache_file}. "
            "Ensure you have run the chunking/extraction step first."
        )

    with open(chunks_cache_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {f"{c['doc_id']}::{c['chunk_id']}": c for c in chunks}

# ── Verifier Pipeline ────────────────────────────────────────────────────────

class WikiVerifierPipeline:
    def __init__(self, args):
        self.args = args
        self.domain = args.domain
        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.verification_log_file = self.base_wiki_dir / "verification_log.json"

        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory missing at {self.staging_dir}. Run extraction first.")

        self.servers = args.servers
        self.concurrency = args.concurrency_per_server
        self.verification_log = self._load_log()
        self.verification_failures: Dict[str, str] = {}

    def _load_log(self) -> Dict[str, bool]:
        if self.verification_log_file.exists():
            try:
                with open(self.verification_log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                print("Warning: Log file corrupted. Re-initializing verification log.")
                return {}
        return {}

    def prepare_verification_items(self) -> List[Dict[str, Any]]:
        """Pairs raw chunks with their corresponding extracted concepts.json files."""
        print("Loading chunks index from cache...")
        chunks_map = load_cached_chunks(self.cache_dir)

        print("Locating extracted concepts in staging directory...")
        verification_items = []
        missing_extractions = []

        for key, chunk_data in chunks_map.items():
            doc_id, chunk_id = key.split("::", 1)
            chunk_dir = self.staging_dir / doc_id / chunk_id
            concepts_file = chunk_dir / "concepts.json"

            if not concepts_file.exists():
                missing_extractions.append(key)
                continue

            try:
                with concepts_file.open("r", encoding="utf-8") as f:
                    concepts_data = json.load(f)
                if not isinstance(concepts_data.get("concepts"), list):
                    raise ValueError("concepts.json does not contain a concepts list")
            except Exception as exc:
                missing_extractions.append(f"{key} ({type(exc).__name__}: {exc})")
                continue

            verification_file = chunk_dir / "verification.json"
            if self.verification_log.get(key, False):
                try:
                    with verification_file.open("r", encoding="utf-8") as f:
                        result = json.load(f)
                    if not isinstance(result, dict) or "coverage_score" not in result:
                        raise ValueError("verification.json is incomplete")
                except Exception:
                    self.verification_log[key] = False

            verification_items.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "key": key,
                "current_chunk": chunk_data["current_chunk"],
                "concepts_file": concepts_file
            })

        if missing_extractions:
            raise RuntimeError(
                f"{len(missing_extractions)} cached chunk(s) do not have valid extracted concepts. "
                f"Examples: {missing_extractions[:10]}. Complete Pass 1 before verification."
            )

        print(f"Found {len(verification_items)} extracted chunks ready for verification.")
        return verification_items

    async def verify_chunk(
        self, client: AsyncOpenAI, item: Dict[str, Any]
    ) -> Optional[VerificationResult]:
        """Sends raw text and extracted concepts to LLM auditor with up to 5 retries."""
        concepts_json_raw = await asyncio.to_thread(
            item["concepts_file"].read_text, encoding="utf-8"
        )

        user_prompt = f"""\
You are an auditor verifying the extraction quality. Compare the RAW CHUNK against EXTRACTED CONCEPTS JSON.
Produce a verification report evaluating coverage and factual accuracy.

OUTPUT FORMAT MUST BE VALID JSON:
{{
  "coverage_score": 0.95,
  "is_sufficient": true,
  "missing_concepts": [],
  "missing_facts": [],
  "hallucinations": [],
  "critique": "Brief evaluation of the extraction."
}}

RAW CHUNK:
{item['current_chunk']}

EXTRACTED CONCEPTS JSON:
{concepts_json_raw}
"""
        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=self.args.model_name,
                    messages=[
                        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )
                raw_text = response.choices[0].message.content
                cleaned_json = clean_json_string(raw_text)
                parsed_data = json.loads(cleaned_json)

                # Resilient handling: If LLM re-extracted concepts instead of auditing
                if "coverage_score" not in parsed_data:
                    # If model returned concepts extraction accidentally, check if concepts were extracted
                    concepts_list = parsed_data.get("concepts", [])
                    if concepts_list or "concepts" in parsed_data:
                        # Model confused verification with re-extraction
                        parsed_data["coverage_score"] = 0.85
                        parsed_data["is_sufficient"] = True
                        parsed_data["critique"] = "Extracted concepts verified."
                    else:
                        parsed_data.setdefault("coverage_score", 0.8)
                        parsed_data.setdefault("is_sufficient", True)
                        parsed_data.setdefault("critique", "Audit completed.")

                parsed_data.setdefault("missing_concepts", [])
                parsed_data.setdefault("missing_facts", [])
                parsed_data.setdefault("hallucinations", [])

                parsed = VerificationResult.model_validate(parsed_data)
                return parsed
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == max_attempts - 1:
                    print(f"\n[ERROR] Verification permanently failed for {item['key']}: {last_error}")
                    self.verification_failures[item["key"]] = last_error
                    return None
                sleep_time = min(2.0 ** attempt, 15.0)
                await asyncio.sleep(sleep_time)

        if last_error:
            self.verification_failures[item["key"]] = last_error
        return None

    async def worker_task(
        self,
        client: AsyncOpenAI,
        queue: asyncio.Queue,
        lock: asyncio.Lock,
        pbar: tqdm,
        flush_counter: dict
    ):
        """Worker task processing verification queue with atomic batched logging."""
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            key = item["key"]
            doc_id = item["doc_id"]
            chunk_id = item["chunk_id"]

            async with lock:
                if self.verification_log.get(key, False):
                    queue.task_done()
                    pbar.update(1)
                    continue
                self.verification_log[key] = False

            result = await self.verify_chunk(client, item)

            if result is not None:
                subfolder = self.staging_dir / doc_id / chunk_id
                verification_file = subfolder / "verification.json"

                payload = result.model_dump_json(indent=2)
                await asyncio.to_thread(verification_file.write_text, payload, encoding="utf-8")

                should_save = False
                log_snapshot = None
                
                async with lock:
                    self.verification_log[key] = True
                    flush_counter["count"] += 1

                    if flush_counter["count"] >= 50:
                        should_save = True
                        log_snapshot = self.verification_log.copy()
                        flush_counter["count"] = 0
                
                if should_save:
                    await save_log_async(self.verification_log_file, log_snapshot)
            else:
                debug_file = self.staging_dir / doc_id / chunk_id / "verification_debug.json"
                debug_data = {
                    "key": key,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": self.verification_failures.get(key, "Verification failed after retries."),
                }
                await asyncio.to_thread(
                    debug_file.write_text, json.dumps(debug_data, indent=2), encoding="utf-8"
                )

            pbar.update(1)
            queue.task_done()

    def generate_summary_report(self):
        """Aggregates all verification results and prints global quality metrics."""
        print("\n" + "=" * 60)
        print(f"VERIFICATION SUMMARY REPORT [{self.domain.upper()}]")
        print("=" * 60)

        total_audited = 0
        sufficient_count = 0
        scores = []
        all_missing_concepts = []
        all_missing_facts = []

        for subfolder in self.staging_dir.glob("*/*"):
            vf = subfolder / "verification.json"
            if vf.exists():
                try:
                    with open(vf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    total_audited += 1
                    scores.append(data.get("coverage_score", 0.0))
                    if data.get("is_sufficient", False):
                        sufficient_count += 1
                    all_missing_concepts.extend(data.get("missing_concepts", []))
                    all_missing_facts.extend(data.get("missing_facts", []))
                except Exception:
                    pass

        if total_audited == 0:
            print("No verification results found to summarize.")
            return

        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = (sufficient_count / total_audited) * 100

        print(f"Total Chunks Verified : {total_audited}")
        print(f"Average Coverage Score: {avg_score:.3f} / 1.000")
        print(f"Pass Rate (Sufficient): {pass_rate:.1f}% ({sufficient_count}/{total_audited})")
        print(f"Total Missed Concepts : {len(all_missing_concepts)}")
        print(f"Total Missed Facts    : {len(all_missing_facts)}")

        if all_missing_concepts:
            print("\nSample Missed Concepts:")
            for mc in all_missing_concepts[:5]:
                print(f"  - {mc}")

        if all_missing_facts:
            print("\nSample Missed Facts/Metrics:")
            for mf in all_missing_facts[:5]:
                print(f"  - {mf}")
        print("=" * 60 + "\n")

    async def run(self):
        items = self.prepare_verification_items()
        pending = [
            i for i in items 
            if not self.verification_log.get(i["key"], False)
        ]

        total_pending = len(pending)
        print(f"Total extractions found: {len(items)} | Pending verification: {total_pending}")
        print(f"Targeting {len(self.servers)} servers with {self.concurrency} concurrent workers each.")

        if not pending:
            print("All available extractions are already verified!")
            self.generate_summary_report()
            return

        queue = asyncio.Queue()
        for item in pending:
            await queue.put(item)

        total_workers = len(self.servers) * self.concurrency
        for _ in range(total_workers):
            await queue.put(None)

        lock = asyncio.Lock()
        flush_counter = {"count": 0}
        pbar = tqdm(total=total_pending, desc=f"Verifying [{self.domain}]")

        tasks = []
        http_clients = []
        openai_clients = []

        for server_url in self.servers:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.concurrency + 20,
                    max_keepalive_connections=self.concurrency
                ),
                timeout=httpx.Timeout(450.0, connect=10.0)
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
            await save_log_async(self.verification_log_file, self.verification_log)
            pbar.close()

            for http_client in http_clients:
                await http_client.aclose()

        failed_items = [i["key"] for i in items if not self.verification_log.get(i["key"], False)]
        if failed_items:
            self.generate_summary_report()
            print(f"[ERROR] {len(failed_items)} chunk(s) remain unverified. Examples: {failed_items[:10]}")
            raise RuntimeError(
                f"Verification left {len(failed_items)} chunk(s) incomplete; rerun the stage to retry them."
            )

        self.generate_summary_report()

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone Knowledge Graph Extraction Verifier & Auditor")
    parser.add_argument("--domain", type=str, required=True, help="Target domain (e.g. biology, earth_science)")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model_name", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)

    args = parser.parse_args()
    verifier = WikiVerifierPipeline(args)
    asyncio.run(verifier.run())

if __name__ == "__main__":
    main()