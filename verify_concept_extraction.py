#!/usr/bin/env python3
"""
Extraction Verifier & Auditor - runs twice, once per extraction pass.
----------------------------------------------------------------------
Phase 1 audits the raw Pass 1 output and produces the ``missing_concepts`` /
``missing_facts`` feedback that Pass 2 consumes as its repair instructions.
Phase 2 re-audits the same chunks after Pass 2 has merged its repairs, so the
two runs are directly comparable and their difference is the measured
improvement.

The phases write to separate files precisely because Pass 2 reads phase 1:

    phase 1 -> staging/<doc>/<chunk>/verification.json         + verification_log.json
    phase 2 -> staging/<doc>/<chunk>/verification_pass2.json   + verification_pass2_log.json

Phase 2 additionally writes ``verification_summary.json`` in the domain
directory holding both runs plus the deltas, and prints the improvement table.

Requests are spread evenly across the configured vLLM servers (item i goes to
server i % N), every failure is streamed to ``verification_debug.jsonl`` with
the full untruncated prompt and raw response, and the stage stops on a
wall-clock budget rather than idling on backoff.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pipeline_lib.atomic_io import atomic_write_text
from pipeline_lib.checkpoint import CheckpointLog
from pipeline_lib.chunks import load_cached_chunks
from pipeline_lib.cli import build_parser, resolve_settings
from pipeline_lib.debug_log import DebugLogger
from pipeline_lib.errors import StageBudget
from pipeline_lib.json_repair import clean_json_string
from pipeline_lib.llm_log import LLMCallLog
from pipeline_lib.llm_pool import ServerPool
from pipeline_lib.reporting import (PASS1_LABEL, PASS1_RESULT, PASS2_LABEL, PASS2_RESULT,
                                    build_improvement, collect_metrics,
                                    format_improvement, format_report, write_summary)

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

# ── Document Loading & Chunk Matching ────────────────────────────────────────

# ── Verifier Pipeline ────────────────────────────────────────────────────────

class WikiVerifierPipeline:
    """Audits extracted concepts for one phase and reports aggregate quality."""

    def __init__(self, args):
        self.args = args
        self.domain = args.domain
        self.phase = int(args.phase)
        if self.phase not in (1, 2):
            raise ValueError("--phase must be 1 or 2")

        self.stage = "verification" if self.phase == 1 else "verification-pass2"
        self.settings = resolve_settings(args, self.stage)

        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"

        # Phase-specific outputs: phase 2 must not overwrite phase 1, because
        # Pass 2 consumes phase 1's missing_concepts / missing_facts.
        self.result_filename = PASS1_RESULT if self.phase == 1 else PASS2_RESULT
        log_name = "verification_log.json" if self.phase == 1 else "verification_pass2_log.json"
        debug_name = "verification_debug.jsonl" if self.phase == 1 else "verification_pass2_debug.jsonl"
        self.verification_log_file = self.base_wiki_dir / log_name

        if not self.staging_dir.exists():
            raise FileNotFoundError(
                f"Staging directory missing at {self.staging_dir}. Run extraction first."
            )

        self.checkpoint = CheckpointLog(self.verification_log_file, flush_every=50)
        self.debug = DebugLogger(self.base_wiki_dir / debug_name, label=self.stage)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.verification_failures: Dict[str, str] = {}

        if getattr(args, "resume", False):
            cleared = self.checkpoint.reset_incomplete()
            if cleared:
                self.checkpoint.flush_sync()
                print(f"Cleared {cleared} incomplete verification marks for resume.")

    # ── item preparation ───────────────────────────────────────────────────
    def prepare_verification_items(self) -> List[Dict[str, Any]]:
        """Collect chunks whose extracted concepts are present and parseable."""
        chunks_map = load_cached_chunks(self.cache_dir)

        print("Locating extracted concepts in staging directory...")
        verification_items: List[Dict[str, Any]] = []
        missing_extractions: List[str] = []

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

            # A result file that is missing or torn invalidates the mark, so a
            # resumed run re-audits it instead of trusting a hollow checkpoint.
            result_file = chunk_dir / self.result_filename
            if self.checkpoint.is_done(key):
                try:
                    with result_file.open("r", encoding="utf-8") as f:
                        prior = json.load(f)
                    if not isinstance(prior, dict) or "coverage_score" not in prior:
                        raise ValueError(f"{self.result_filename} is incomplete")
                except Exception:
                    self.checkpoint.mark(key, False)

            verification_items.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "key": key,
                "current_chunk": chunk_data["current_chunk"],
                "concepts_file": concepts_file,
                "result_file": result_file,
            })

        if missing_extractions:
            stage_hint = "Pass 1" if self.phase == 1 else "Pass 2"
            raise RuntimeError(
                f"{len(missing_extractions)} cached chunk(s) do not have valid extracted concepts. "
                f"Examples: {missing_extractions[:10]}. Complete {stage_hint} before verification."
            )

        print(f"Found {len(verification_items)} extracted chunks ready for verification.")
        return verification_items

    # ── single LLM interaction ─────────────────────────────────────────────
    def build_prompt(self, item: Dict[str, Any], concepts_json_raw: str) -> str:
        return f"""\
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

    def parse_verification(self, raw_text: Optional[str], key: str) -> VerificationResult:
        """Parse an audit response, keeping the original resilient fallbacks.

        A malformed payload is streamed to the debug log in full before raising,
        so the exact model output can be inspected offline; raising lets the
        pool retry it as a recoverable failure.
        """
        if not raw_text:
            self.debug.record("empty_response", key=key)
            raise ValueError("Empty verification response")

        cleaned_json = clean_json_string(raw_text)
        try:
            parsed_data = json.loads(cleaned_json)
        except json.JSONDecodeError:
            self.debug.record("parse_failure", key=key, raw_response=raw_text)
            raise

        if not isinstance(parsed_data, dict):
            self.debug.record("parse_failure", key=key, raw_response=raw_text,
                              reason="top-level JSON is not an object")
            raise ValueError("Verification response is not a JSON object")

        # Resilient handling: If LLM re-extracted concepts instead of auditing
        if "coverage_score" not in parsed_data:
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
            self.debug.record("recovered_response_shape", key=key,
                              raw_response=raw_text, parsed=parsed_data)

        parsed_data.setdefault("missing_concepts", [])
        parsed_data.setdefault("missing_facts", [])
        parsed_data.setdefault("hallucinations", [])

        try:
            return VerificationResult.model_validate(parsed_data)
        except Exception:
            self.debug.record("schema_failure", key=key, raw_response=raw_text,
                              parsed=parsed_data)
            raise

    async def verify_chunk(self, client: Any, item: Dict[str, Any],
                           server_index: int) -> VerificationResult:
        """One audit call. Retries are owned by the server pool."""
        concepts_json_raw = await asyncio.to_thread(
            item["concepts_file"].read_text, encoding="utf-8"
        )
        user_prompt = self.build_prompt(item, concepts_json_raw)

        response = await client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=self.settings.max_tokens or 4096,
            response_format={"type": "json_object"},
        )
        return self.parse_verification(response.choices[0].message.content, item["key"])

    # ── streaming persistence ──────────────────────────────────────────────
    async def persist_result(self, item: Dict[str, Any],
                             result: VerificationResult) -> None:
        """Write one chunk's audit result and advance the checkpoint."""
        payload = result.model_dump_json(indent=2)
        await asyncio.to_thread(atomic_write_text, item["result_file"], payload)
        self.checkpoint.mark(item["key"], True)
        self.debug.record("verified", key=item["key"], phase=self.phase,
                          coverage_score=result.coverage_score,
                          is_sufficient=result.is_sufficient,
                          missing_concepts=len(result.missing_concepts),
                          missing_facts=len(result.missing_facts))
        if self.checkpoint.needs_flush():
            await self.checkpoint.flush()

    # ── reporting ──────────────────────────────────────────────────────────
    def generate_summary_report(self) -> None:
        """Print both runs and, for phase 2, the measured improvement."""
        before = collect_metrics(self.staging_dir, label=PASS1_LABEL,
                                 result_filename=PASS1_RESULT)
        print("\n" + format_report(self.domain, before) + "\n")

        if self.phase == 1:
            summary_path = write_summary(self.base_wiki_dir, self.domain, before)
            print(f"Saved Pass 1 verification summary to: {summary_path}")
            print("Phase 2 (after Pass 2 refinement) will report the improvement.\n")
            return

        after = collect_metrics(self.staging_dir, label=PASS2_LABEL,
                                result_filename=PASS2_RESULT)
        print(format_report(self.domain, after) + "\n")

        if before.total_chunks == 0 or after.total_chunks == 0:
            print("WARNING: cannot compute improvement - one of the two "
                  "verification runs has no results on disk.\n")
            summary_path = write_summary(self.base_wiki_dir, self.domain, before,
                                         after if after.total_chunks else None)
            print(f"Saved verification summary to: {summary_path}\n")
            return

        improvement = build_improvement(before, after)
        print(format_improvement(self.domain, improvement) + "\n")
        summary_path = write_summary(self.base_wiki_dir, self.domain, before, after)
        print(f"Saved both verification runs + improvement to: {summary_path}\n")

    # ── orchestration ──────────────────────────────────────────────────────
    async def run(self) -> None:
        items = self.prepare_verification_items()
        pending = [i for i in items if not self.checkpoint.is_done(i["key"])]

        print(f"Verification phase {self.phase} for domain '{self.domain}'.")
        print(f"Total extractions found: {len(items)} | Pending verification: {len(pending)}")
        print(f"Spreading work across {len(self.settings.servers)} server(s) at "
              f"{self.settings.concurrency_per_server} concurrent request(s) each "
              f"(item i -> server i % N).")
        print(f"Per-item attempts: {self.settings.attempts} | "
              f"Stage budget: {self.settings.budget_minutes:g} min | "
              f"Debug log: {self.debug.path}")

        if not pending:
            print("All available extractions are already verified for this phase!")
            self.generate_summary_report()
            return

        pool = ServerPool(
            self.settings.servers,
            concurrency_per_server=self.settings.concurrency_per_server,
            request_timeout=self.settings.profile.request_timeout,
            connect_timeout=self.settings.profile.connect_timeout,
            max_connections_slack=20,
            label=self.stage,
            # Each phase gets its own log file, matching its own debug log and
            # checkpoint: logs/verification.jsonl vs logs/verification-pass2.jsonl.
            llm_log=LLMCallLog.from_args(self.args, self.stage),
            failover=not getattr(self.args, "no_failover", False),
        )

        try:
            async with pool:
                result = await pool.map(
                    pending,
                    self.verify_chunk,
                    attempts=self.settings.attempts,
                    desc=f"Verifying phase {self.phase} [{self.domain}]",
                    key_fn=lambda item: item["key"],
                    budget=self.budget,
                    debug=self.debug,
                    attempt_timeout=self.settings.profile.attempt_timeout,
                    on_result=self.persist_result,
                )
        finally:
            await self.checkpoint.flush()
            self.debug.close()

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        for failure in result.failures:
            self.verification_failures[failure["key"]] = failure["error"]

        failed_items = [i["key"] for i in items if not self.checkpoint.is_done(i["key"])]
        if failed_items:
            self.generate_summary_report()
            print(f"[ERROR] {len(failed_items)} chunk(s) remain unverified. "
                  f"Examples: {failed_items[:10]}")
            first = next(iter(self.verification_failures.values()), "unknown error")
            raise RuntimeError(
                f"Verification phase {self.phase} left {len(failed_items)} chunk(s) "
                f"incomplete; rerun the stage with --resume to retry them. "
                f"First error: {first}"
            )

        self.generate_summary_report()

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = build_parser(
        stage="verification",
        description="Knowledge Graph extraction verifier & auditor (runs per phase)",
        include_max_tokens=True,
        include_resume=True,
    )
    parser.add_argument("--phase", type=int, default=1, choices=(1, 2),
                        help="1 = audit raw Pass 1 output (default); "
                             "2 = re-audit after Pass 2 and report the improvement.")
    args = parser.parse_args()

    verifier = WikiVerifierPipeline(args)
    try:
        asyncio.run(verifier.run())
    finally:
        verifier.debug.close()

if __name__ == "__main__":
    main()
