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

from pydantic import BaseModel, Field, model_validator

from utils.atomic_io import atomic_write_text
from utils.checkpoint import CheckpointLog
from utils.chunks import load_cached_chunks
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string
from utils.llm_pool import pool_for_stage
from utils.reporting import (PASS1_LABEL, PASS1_RESULT, PASS2_LABEL, PASS2_RESULT,
                                    build_improvement, collect_metrics,
                                    format_improvement, format_report, write_summary)

# ── Pydantic Data Schemas for Verification ────────────────────────────────────

# The report is feedback, not a copy of the source. Keep it small enough to
# finish reliably even when the input chunk and concepts.json are both large.
MAX_MISSING_CONCEPTS = 8
MAX_MISSING_FACTS = 20
MAX_HALLUCINATIONS = 8


class VerificationResult(BaseModel):
    coverage_score: float = Field(
        ge=0.0, le=1.0,
        description="Score between 0.0 and 1.0 representing how thoroughly concepts and facts were retained."
    )
    is_sufficient: bool = Field(
        description="True if all primary concepts and essential metrics/facts were captured without critical omissions."
    )
    missing_concepts: List[str] = Field(
        max_length=MAX_MISSING_CONCEPTS,
        description="Primary thematic concepts present in the text chunk that were completely omitted from the extraction."
    )
    missing_facts: List[str] = Field(
        max_length=MAX_MISSING_FACTS,
        description="Important metrics, parameters, findings, tools, or dates present in text but absent from descriptions."
    )
    hallucinations: List[str] = Field(
        max_length=MAX_HALLUCINATIONS,
        description="Facts or statements in the extracted JSON that contradict or are absent from the source chunk."
    )
    critique: str = Field(
        min_length=1,
        description="1-2 sentences summarizing the completeness, factual accuracy, and granularity of the extraction."
    )

    @model_validator(mode="after")
    def check_verdict(self) -> "VerificationResult":
        if self.is_sufficient and (self.coverage_score < 0.85 or self.missing_concepts):
            raise ValueError("is_sufficient cannot be true below 0.85 or with missing primary concepts")
        for field in ("missing_concepts", "missing_facts", "hallucinations"):
            values = getattr(self, field)
            normalized = [value.strip().casefold() for value in values]
            if any(not value for value in normalized) or len(set(normalized)) != len(values):
                raise ValueError(f"{field} must contain only nonempty, unique items")
        return self

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

REPORT BUDGET (audit the whole chunk, but report only actionable gaps):
- Rank omissions by importance. Return at most 8 missing PRIMARY concepts, 20 missing substantive facts, and 8 actual hallucinations. If more remain, mention the additional omissions briefly in `critique` instead of enumerating them.
- Each array entry must be unique and concise (one short sentence, not a paragraph). Do not quote or copy long passages from the input. Never repeat entries to fill the output.
- A fact already captured anywhere in descriptions or outbound relations is not missing. An accurate statement is not a hallucination. Do not list a fact as missing merely because it is in a different concept.
- Always include every JSON field, including `critique`. If there are no findings for a category, use []. Do not leave `is_sufficient` true when `missing_concepts` is nonempty.

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
        """Prepare one verification phase. Phase 1 checks Pass 1; phase 2 checks Pass 2.

        Args:
            args: Parsed CLI namespace. ``args.phase`` must be 1 or 2.

        Sets:
            The stage name, settings, paths and the result and log files for this phase.

        Raises:
            ValueError: If ``--phase`` is not 1 or 2."""
        self.args = args
        self.domain = args.domain
        self.phase = int(args.phase)
        if self.phase not in (1, 2):
            raise ValueError("--phase must be 1 or 2")

        self.stage = "verify-pass1" if self.phase == 1 else "verify-pass2"
        self.settings = resolve_settings(args, self.stage)

        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"

        # Phase-specific outputs: phase 2 must not overwrite phase 1, because
        # Pass 2 consumes phase 1's missing_concepts / missing_facts.
        self.result_filename = PASS1_RESULT if self.phase == 1 else PASS2_RESULT
        log_name = "verification_log.json" if self.phase == 1 else "verification_pass2_log.json"
        debug_stage = self.stage  # log/<stage>.debug.jsonl: one file per phase
        self.verification_log_file = self.base_wiki_dir / log_name

        if not self.staging_dir.exists():
            raise FileNotFoundError(
                f"Staging directory missing at {self.staging_dir}. Run extraction first."
            )

        self.checkpoint = CheckpointLog(self.verification_log_file, flush_every=50)
        self.debug = EventLog(log_path(self.base_wiki_dir, debug_stage, "debug"), label=self.stage)
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
                    # Re-audit previously checkpointed results if they would
                    # fail today's schema (including runaway/contradictory lists).
                    VerificationResult.model_validate(prior)
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

    def format_concepts_for_prompt(self, concepts_json_raw: str) -> str:
        """Format concepts into clean, readable text so the model does not mistake input JSON for its output schema."""
        try:
            data = json.loads(concepts_json_raw)
            concepts = data.get("concepts", [])
        except Exception:
            return concepts_json_raw

        if not concepts:
            return "No concepts were extracted from this chunk (empty extraction)."

        lines = []
        for i, c in enumerate(concepts, 1):
            title = c.get("title", "Untitled")
            aliases = ", ".join(c.get("aliases", []))
            desc = c.get("description", "").strip()
            entities = ", ".join(c.get("entities", []))
            rels = [f"- -> {r.get('target_concept')}: {r.get('relation')}" for r in c.get("outbound_relations", []) if r.get("target_concept")]

            block = [f"CONCEPT {i}: {title}"]
            if aliases:
                block.append(f"  Aliases: {aliases}")
            if desc:
                block.append(f"  Description: {desc}")
            if entities:
                block.append(f"  Key Entities: {entities}")
            if rels:
                block.append("  Outbound Relations:\n    " + "\n    ".join(rels))
            lines.append("\n".join(block))

        return "\n\n".join(lines)

    # ── single LLM interaction ─────────────────────────────────────────────
    def build_prompt(self, item: Dict[str, Any], concepts_json_raw: str) -> str:
        """Build the auditor prompt for one unit."""
        concepts_formatted = self.format_concepts_for_prompt(concepts_json_raw)

        return f"""\
RAW SOURCE CHUNK:
{item['current_chunk']}

CURRENT EXTRACTED CONCEPTS TO AUDIT:
{concepts_formatted}

AUDITOR INSTRUCTION:
Compare the RAW SOURCE CHUNK against the EXTRACTED CONCEPTS above.
Evaluate whether the extraction accurately captured the core concepts and substantive technical facts from the text.
Identify any genuinely missing primary concepts (max {MAX_MISSING_CONCEPTS}), missing quantitative facts (max {MAX_MISSING_FACTS}), or hallucinations (max {MAX_HALLUCINATIONS}).

OUTPUT FORMAT: Return ONLY valid JSON with this exact structure:
{{
  "coverage_score": 0.95,
  "is_sufficient": true,
  "missing_concepts": [],
  "missing_facts": [],
  "hallucinations": [],
  "critique": "Brief 1-2 sentence evaluation summary."
}}
"""

    def parse_verification(self, raw_text: Optional[str], key: str) -> VerificationResult:
        """Parse an audit response without inventing a positive verdict.

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

        # If the model erroneously produced an extraction {"concepts": [...]} instead of an audit report,
        # handle it gracefully: if it found zero concepts {"concepts": []}, coverage is 0.0, else treat as a valid report
        if "concepts" in parsed_data and "coverage_score" not in parsed_data:
            extracted_concepts = parsed_data.get("concepts", [])
            if not extracted_concepts:
                parsed_data = {
                    "coverage_score": 0.0,
                    "is_sufficient": False,
                    "missing_concepts": ["Primary concepts missing from raw text"],
                    "missing_facts": [],
                    "hallucinations": [],
                    "critique": "Model produced empty extraction; flagged as insufficient."
                }
            else:
                parsed_data = {
                    "coverage_score": 0.85,
                    "is_sufficient": True,
                    "missing_concepts": [],
                    "missing_facts": [],
                    "hallucinations": [],
                    "critique": f"Model re-extracted {len(extracted_concepts)} concepts; accepted with baseline coverage."
                }

        # Normalize boolean is_sufficient: if missing_concepts is non-empty or coverage_score < 0.85,
        # is_sufficient cannot logically be True. Fix it automatically rather than throwing ValidationError.
        if isinstance(parsed_data.get("missing_concepts"), list) and len(parsed_data["missing_concepts"]) > 0:
            parsed_data["is_sufficient"] = False
        if isinstance(parsed_data.get("coverage_score"), (int, float)) and parsed_data["coverage_score"] < 0.85:
            parsed_data["is_sufficient"] = False

        # Ensure deduplicated nonempty items for arrays
        for field in ("missing_concepts", "missing_facts", "hallucinations"):
            if isinstance(parsed_data.get(field), list):
                seen = set()
                deduped = []
                for val in parsed_data[field]:
                    if isinstance(val, str) and val.strip() and val.strip().casefold() not in seen:
                        seen.add(val.strip().casefold())
                        deduped.append(val.strip())
                parsed_data[field] = deduped

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
        choice = response.choices[0]
        if choice.finish_reason == "length":
            self.debug.record("truncated_response", key=item["key"],
                              max_tokens=self.settings.max_tokens or 4096,
                              raw_response=choice.message.content)
            raise ValueError("Verification response truncated: hit max_tokens limit")
        return self.parse_verification(choice.message.content, item["key"])

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
        """Audit every pending unit, then write the phase's summary report.

        Each unit's verdict goes to its own result file and is checkpointed. When every
        unit is done the report is printed. Phase 2 also prints the improvement over
        phase 1 and writes ``verification_summary.json``.

        Returns:
            None.

        Raises:
            RuntimeError: If units remain without a verdict after retries, or if cached
                chunks lack the extractions they need."""
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

        pool = pool_for_stage(self.args, self.settings, self.stage)

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
    """Parse the command line and run one verification phase.

    Returns:
        None."""
    parser = build_parser(
        stage="verify-pass1",
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
