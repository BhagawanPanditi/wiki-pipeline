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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from config import _read_config
from utils.atomic_io import atomic_write_text
from utils.checkpoint import CheckpointLog
from utils.chunks import load_cached_chunks
from utils.cli import build_parser, resolve_settings
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
Your task is to compare a RAW TEXT CHUNK against the EXTRACTED CONCEPTS to verify factual completeness, accuracy, and core concept coverage.

AUDIT DEFINITIONS & EXPECTATIONS:
1. SOLID CONCEPTS ONLY (NO FRAGMENTATION):
   - Concepts are expected to be major, solid, well-developed subjects (a core idea, system, entity, method, mechanism, theory, phenomenon, or process).
   - Minor parameters, sub-steps, properties, and variants should NOT be standalone concepts; they are expected to be absorbed directly into the comprehensive description of their primary parent concept.
   - Do NOT penalize the extraction for omitting meta-labels ("Introduction", "Summary") or for not creating thin stub concepts for minor properties.

2. SUBSTANTIVE FACTS TO AUDIT (ZERO FACT LOSS):
   - Verify that all core technical metrics, parameters, benchmark scores, formulas, conditions, procedural steps, and quantitative bounds from the raw text are explicitly retained inside concept descriptions.
   - Verify that specific named terms, tools, algorithms, or classifications are retained.

3. NOISE & NON-FACTS (SILENTLY IGNORE):
   - Silently ignore HTML tags, web scraping artifacts, navigation menus, headers/footers, cookie notices, citations, and boilerplate. Do NOT penalize the extraction for omitting them, and do NOT flag them as missing facts or missing concepts.

4. SCORING CRITERIA:
- `coverage_score` (0.0 to 1.0):
  - 1.0: Perfect retention. All main themes and key quantitative/technical facts captured.
  - 0.8 - 0.9: Main themes captured; missed minor secondary metrics.
  - 0.5 - 0.7: Omitted a primary thematic concept or several key findings.
  - < 0.5: Severe omissions of main body content.
- `is_sufficient`: `true` if coverage_score >= 0.85 and NO major primary concept was dropped. Otherwise `false`.

THINKING PROCESS (SANDBOXED):
- If you need to reason or analyze omissions, you MUST do so strictly inside `<think>...</think>` tags at the very start of your response.
- Once `</think>` is closed, output ONLY the final `<verification>...</verification>` XML block.
- Absolutely NO commentary or notes outside `<think>...</think>` or outside `<verification>` tags.

OUTPUT FORMAT:
<think>
[Auditor analysis and comparison notes]
</think>
<verification>
<coverage_score>0.95</coverage_score>
<is_sufficient>true</is_sufficient>
<missing_concepts>
- Name of missing PRIMARY theme (if any, max 8)
</missing_concepts>
<missing_facts>
- Specific missing technical metric or finding (if any, max 20)
</missing_facts>
<hallucinations>
- Fact in extraction that contradicts or is absent from raw text (if any, max 8)
</hallucinations>
<critique>1-2 sentence evaluation summary.</critique>
</verification>
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
        self.phase = int(args.phase)
        if self.phase not in (1, 2):
            raise ValueError("--phase must be 1 or 2")

        wiki_path = getattr(args, "wiki_path", None)
        if wiki_path:
            self.base_wiki_dir = Path(wiki_path)
            self.domain = self.base_wiki_dir.name
        else:
            self.domain = getattr(args, "domain", "default")
            wiki_base = getattr(args, "wiki_base_dir", None) or "wiki"
            self.base_wiki_dir = Path(wiki_base) / self.domain

        self.cfg = _read_config()
        self.stage = "verify-pass1" if self.phase == 1 else "verify-pass2"
        self.settings = resolve_settings(args, self.stage)

        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"

        # Phase-specific outputs: phase 2 must not overwrite phase 1
        self.result_filename = PASS1_RESULT if self.phase == 1 else PASS2_RESULT
        log_name = "verification_pass1_log.json" if self.phase == 1 else "verification_pass2_log.json"
        self.verification_log_file = self.base_wiki_dir / log_name

        if not self.staging_dir.exists():
            raise FileNotFoundError(
                f"Staging directory missing at {self.staging_dir}. Run extraction first."
            )

        self.checkpoint = CheckpointLog(self.verification_log_file, flush_every=50)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.verification_failures: Dict[str, str] = {}

        if getattr(args, "resume", False):
            cleared = self.checkpoint.reset_incomplete()
            if cleared:
                self.checkpoint.flush_sync()
                print(f"Cleared {cleared} incomplete verification marks for resume.")

    # ── item preparation ───────────────────────────────────────────────────
    def prepare_verification_items(self) -> List[Dict[str, Any]]:
        """Collect chunks whose extracted concepts are present and parseable, with deterministic sampling."""
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

        # Deterministic sorting so sampling is identical between Phase 1 and Phase 2
        verification_items.sort(key=lambda x: x["key"])

        sample_size = getattr(self.args, "sample_size", 50)
        seed = getattr(self.args, "seed", 42)
        total_found = len(verification_items)
        if sample_size and sample_size > 0 and total_found > sample_size:
            import random
            rng = random.Random(seed)
            sampled = rng.sample(verification_items, sample_size)
            sampled_keys = {item["key"] for item in sampled}
            verification_items = [item for item in verification_items if item["key"] in sampled_keys]
            print(f"Sampled {len(verification_items)} random chunks for audit (seed={seed}, total available: {total_found}).")
        else:
            print(f"Found {len(verification_items)} extracted chunks ready for verification.")

        return verification_items

    def format_concepts_for_prompt(self, concepts_json_raw: str) -> str:
        """Format concepts into clean, readable text."""
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
            aliases = ", ".join(c.get("aliases", [])) if isinstance(c.get("aliases"), list) else ""
            desc = c.get("description", "").strip()

            block = [f"CONCEPT {i}: {title}"]
            if aliases:
                block.append(f"  Aliases: {aliases}")
            if desc:
                block.append(f"  Description: {desc}")
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
If you need to think, put all thinking strictly inside <think>...</think>, followed immediately by the <verification>...</verification> block.
"""

    def parse_verification(self, raw_text: Optional[str], key: str) -> VerificationResult:
        """Parse an audit response from XML (with JSON fallback) into VerificationResult."""
        if not raw_text:
            raise ValueError("Empty verification response")

        # 1. Strip thinking tags
        clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

        # 2. Check for XML <verification> block
        if "<verification>" in clean_text:
            block_m = re.search(r"<verification>((?:(?!<verification>).)*?)</verification>", clean_text, re.DOTALL | re.IGNORECASE)
            block = block_m.group(1) if block_m else clean_text

            score_m = re.search(r"<coverage_score>(.*?)</coverage_score>", block, re.DOTALL | re.IGNORECASE)
            coverage_score = 0.90
            if score_m:
                val_str = re.search(r"([0-9]*\.?[0-9]+)", score_m.group(1).strip())
                if val_str:
                    try:
                        coverage_score = float(val_str.group(1))
                        if coverage_score > 1.0 and coverage_score <= 100.0:
                            coverage_score /= 100.0
                        coverage_score = max(0.0, min(1.0, coverage_score))
                    except ValueError:
                        pass

            def extract_list(tag: str) -> List[str]:
                m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL | re.IGNORECASE)
                if not m:
                    return []
                items = []
                for line in m.group(1).strip().splitlines():
                    line = line.strip().lstrip("-*•0123456789.) ").strip()
                    if line and line.lower() not in ("none", "n/a", "[]", "none."):
                        items.append(line)
                return items

            missing_concepts = extract_list("missing_concepts")[:MAX_MISSING_CONCEPTS]
            missing_facts = extract_list("missing_facts")[:MAX_MISSING_FACTS]
            hallucinations = extract_list("hallucinations")[:MAX_HALLUCINATIONS]

            suff_m = re.search(r"<is_sufficient>(.*?)</is_sufficient>", block, re.DOTALL | re.IGNORECASE)
            if suff_m:
                raw_suff = suff_m.group(1).strip().lower()
                is_sufficient = "true" in raw_suff or "yes" in raw_suff
            else:
                is_sufficient = (coverage_score >= 0.85 and len(missing_concepts) == 0)

            if missing_concepts or coverage_score < 0.85:
                is_sufficient = False

            crit_m = re.search(r"<critique>(.*?)</critique>", block, re.DOTALL | re.IGNORECASE)
            critique = crit_m.group(1).strip() if crit_m else "Verification audit complete."
            if not critique:
                critique = "Verification audit complete."

            def dedupe(arr: List[str]) -> List[str]:
                seen = set()
                res = []
                for a in arr:
                    norm = a.strip().casefold()
                    if norm and norm not in seen:
                        seen.add(norm)
                        res.append(a.strip())
                return res

            parsed_data = {
                "coverage_score": coverage_score,
                "is_sufficient": is_sufficient,
                "missing_concepts": dedupe(missing_concepts),
                "missing_facts": dedupe(missing_facts),
                "hallucinations": dedupe(hallucinations),
                "critique": critique,
            }
            try:
                return VerificationResult.model_validate(parsed_data)
            except Exception:
                pass

        # 3. Fallback to JSON repair if XML parse didn't match
        try:
            cleaned_json = clean_json_string(clean_text)
            parsed_data = json.loads(cleaned_json)
            if isinstance(parsed_data, dict):
                if isinstance(parsed_data.get("missing_concepts"), list) and len(parsed_data["missing_concepts"]) > 0:
                    parsed_data["is_sufficient"] = False
                if isinstance(parsed_data.get("coverage_score"), (int, float)) and parsed_data["coverage_score"] < 0.85:
                    parsed_data["is_sufficient"] = False
                for field in ("missing_concepts", "missing_facts", "hallucinations"):
                    if isinstance(parsed_data.get(field), list):
                        seen = set()
                        deduped = []
                        for val in parsed_data[field]:
                            if isinstance(val, str) and val.strip() and val.strip().casefold() not in seen:
                                seen.add(val.strip().casefold())
                                deduped.append(val.strip())
                        parsed_data[field] = deduped
                return VerificationResult.model_validate(parsed_data)
        except Exception:
            pass

        raise ValueError(f"Could not parse verification output for {key}")

    async def verify_chunk(self, client: Any, item: Dict[str, Any],
                           server_index: int) -> VerificationResult:
        """One audit call. Retries are owned by the server pool."""
        concepts_json_raw = await asyncio.to_thread(
            item["concepts_file"].read_text, encoding="utf-8"
        )
        user_prompt = self.build_prompt(item, concepts_json_raw)

        cfg_enable = self.cfg.get("enable_thinking") if hasattr(self, "cfg") else None
        disable_cli = getattr(self.args, "disable_thinking", False)
        enable_cli = getattr(self.args, "enable_thinking", False)

        if disable_cli or (cfg_enable is False and not enable_cli):
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
                "repetition_penalty": 1.0,
            }
            temperature = 0.7
            top_p = 0.80
            presence_penalty = 1.5
        elif enable_cli or (cfg_enable is True):
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
            temperature = 0.1
            top_p = 0.95
            presence_penalty = 0.0
        else:
            extra_body = None
            temperature = 0.0
            top_p = 0.95
            presence_penalty = 0.0

        response = await client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=self.settings.max_tokens or 4096,
            extra_body=extra_body,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("Verification response truncated: hit max_tokens limit")
        return self.parse_verification(choice.message.content, item["key"])

    # ── streaming persistence ──────────────────────────────────────────────
    async def persist_result(self, item: Dict[str, Any],
                             result: VerificationResult) -> None:
        """Write one chunk's audit result and advance the checkpoint."""
        payload = result.model_dump_json(indent=2)
        await asyncio.to_thread(atomic_write_text, item["result_file"], payload)
        self.checkpoint.mark(item["key"], True)
        if self.checkpoint.needs_flush():
            await self.checkpoint.flush()

    # ── reporting ──────────────────────────────────────────────────────────
    def generate_summary_report(self) -> None:
        """Print both runs and, for phase 2, the measured improvement on the evaluated sample."""
        items = self.prepare_verification_items()
        target_keys = {i["key"] for i in items}

        before = collect_metrics(self.staging_dir, label=PASS1_LABEL,
                                 result_filename=PASS1_RESULT,
                                 target_keys=target_keys)
        print("\n" + format_report(self.domain, before) + "\n")

        if self.phase == 1:
            summary_path = write_summary(self.base_wiki_dir, self.domain, before)
            print(f"Saved Pass 1 verification summary to: {summary_path}")
            print("Phase 2 (after Pass 2 refinement) will report the improvement.\n")
            return

        after = collect_metrics(self.staging_dir, label=PASS2_LABEL,
                                result_filename=PASS2_RESULT,
                                target_keys=target_keys)
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
              f"Stage budget: {self.settings.budget_minutes:g} min")

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
                    debug=None,
                    attempt_timeout=self.settings.profile.attempt_timeout,
                    on_result=self.persist_result,
                )
        finally:
            await self.checkpoint.flush()

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
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of random chunks to audit (default: 50; set 0 to audit all chunks).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic chunk sampling across phases (default: 42).")
    args = parser.parse_args()

    verifier = WikiVerifierPipeline(args)
    asyncio.run(verifier.run())

if __name__ == "__main__":
    main()
