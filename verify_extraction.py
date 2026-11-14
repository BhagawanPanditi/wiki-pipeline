#!/usr/bin/env python3
"""Stage 2 & Stage 4: Extraction Verification & Audit.
----------------------------------------------------
Audits extracted concepts against raw document chunks for factual completeness,
reporting missing concepts, omitted details, and hallucinations.
- Phase 1 (Stage 2): Audits Pass 1 output and produces repair feedback for Pass 2.
- Phase 2 (Stage 4): Audits Pass 2 output and computes measured delta improvements.

Usage:
    python3 verify_extraction.py --wiki_path /path/to/wiki --phase 1
    python3 verify_extraction.py --wiki_path /path/to/wiki --phase 2
"""


import asyncio
import json
import re
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, model_validator

from config import _read_config
from utils.atomic_io import atomic_write_json, atomic_write_text, read_json
from utils.checkpoint import load_bool_log, save_log_async, save_log_sync
from utils.cli import build_parser, resolve_settings
from utils.corpus import load_cached_chunks
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage
from utils.manifest import verify_stage_manifest, write_stage_manifest

PASS1_RESULT = "verification_pass1.json"
PASS2_RESULT = "verification_pass2.json"
PASS1_LABEL = "PASS 1 (raw extraction)"
PASS2_LABEL = "PASS 2 (after refinement)"
SUMMARY_FILENAME = "verification_summary.json"


class CheckpointLog:
    """Tracks per-item completion and flushes atomically every ``flush_every``."""

    def __init__(self, path: Path, *, flush_every: int = 25):
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        self.data: Dict[str, bool] = load_bool_log(self.path)
        self._dirty = 0

    def is_done(self, key: str) -> bool:
        return bool(self.data.get(key, False))

    def mark(self, key: str, completed: bool = True) -> None:
        self.data[key] = completed
        self._dirty += 1

    def needs_flush(self) -> bool:
        return self._dirty >= self.flush_every

    def pending(self, keys: Iterable[str]) -> List[str]:
        return [k for k in keys if not self.is_done(k)]

    def reset_incomplete(self, on_reset: Optional[Callable[[str], None]] = None) -> int:
        removed = 0
        for key, completed in list(self.data.items()):
            if not completed:
                if on_reset is not None:
                    on_reset(key)
                del self.data[key]
                removed += 1
        return removed

    async def flush(self) -> None:
        if self._dirty:
            await save_log_async(self.path, dict(self.data))
            self._dirty = 0

    def flush_sync(self) -> None:
        if self._dirty:
            save_log_sync(self.path, dict(self.data))
            self._dirty = 0


@dataclass
class VerificationMetrics:
    """Aggregate quality metrics for one verification run."""
    label: str
    result_filename: str
    total_chunks: int = 0
    average_coverage: float = 0.0
    sufficient_count: int = 0
    pass_rate: float = 0.0
    missed_concepts: int = 0
    missed_facts: int = 0
    synthetic_count: int = 0
    sample_missed_concepts: List[str] = None  # type: ignore[assignment]
    sample_missed_facts: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sample_missed_concepts is None:
            self.sample_missed_concepts = []
        if self.sample_missed_facts is None:
            self.sample_missed_facts = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_metrics(staging_dir: Path, *, label: str,
                    result_filename: str = PASS1_RESULT,
                    target_keys: Optional[set] = None) -> VerificationMetrics:
    metrics = VerificationMetrics(label=label, result_filename=result_filename)
    scores: List[float] = []
    missed_concepts: List[str] = []
    missed_facts: List[str] = []

    staging_dir = Path(staging_dir)
    if not staging_dir.is_dir():
        return metrics

    for chunk_dir in sorted(staging_dir.glob("*/*")):
        key = f"{chunk_dir.parent.name}::{chunk_dir.name}"
        if target_keys is not None and key not in target_keys:
            continue

        result_file = chunk_dir / result_filename
        if not result_file.is_file():
            if result_filename == "verification_pass1.json":
                legacy_file = chunk_dir / "verification.json"
                if legacy_file.is_file():
                    result_file = legacy_file
                else:
                    continue
            else:
                continue

        data = read_json(result_file, default=None)
        if not isinstance(data, dict) or "coverage_score" not in data:
            continue
        if data.get("synthetic", False):
            metrics.synthetic_count += 1
            continue
        metrics.total_chunks += 1
        try:
            scores.append(float(data.get("coverage_score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
        if data.get("is_sufficient", False):
            metrics.sufficient_count += 1
        chunk_concepts = data.get("missing_concepts", []) or []
        chunk_facts = data.get("missing_facts", []) or []
        missed_concepts.extend(str(c) for c in chunk_concepts)
        missed_facts.extend(str(f) for f in chunk_facts)

    if metrics.total_chunks:
        metrics.average_coverage = sum(scores) / len(scores)
        metrics.pass_rate = (metrics.sufficient_count / metrics.total_chunks) * 100.0
    metrics.missed_concepts = len(missed_concepts)
    metrics.missed_facts = len(missed_facts)
    metrics.sample_missed_concepts = missed_concepts[:5]
    metrics.sample_missed_facts = missed_facts[:5]
    return metrics


def format_report(domain: str, metrics: VerificationMetrics) -> str:
    lines = [
        "=" * 60,
        f"VERIFICATION SUMMARY REPORT [{domain.upper()}] - {metrics.label}",
        "=" * 60,
    ]
    if metrics.total_chunks == 0:
        lines.append(f"No {metrics.result_filename} results found.")
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.extend([
        f"Total Chunks Verified : {metrics.total_chunks}",
        f"Average Coverage Score: {metrics.average_coverage:.3f} / 1.000",
        f"Pass Rate             : {metrics.pass_rate:.1f}% ({metrics.sufficient_count}/{metrics.total_chunks} sufficient)",
        f"Missed Concepts Found : {metrics.missed_concepts}",
        f"Missed Facts Found    : {metrics.missed_facts}",
        "=" * 60,
    ])
    if metrics.sample_missed_concepts:
        lines.append("Sample Missed Concepts:")
        lines.extend(f"  - {c}" for c in metrics.sample_missed_concepts)
    if metrics.sample_missed_facts:
        lines.append("\nSample Missed Facts:")
        lines.extend(f"  - {f}" for f in metrics.sample_missed_facts)
    return "\n".join(lines)


def _delta_pct(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return ((after - before) / before) * 100.0


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def build_improvement(before: VerificationMetrics, after: VerificationMetrics) -> Dict[str, Any]:
    return {
        "coverage_before": round(before.average_coverage, 4),
        "coverage_after": round(after.average_coverage, 4),
        "coverage_delta": round(after.average_coverage - before.average_coverage, 4),
        "pass_rate_before": round(before.pass_rate, 2),
        "pass_rate_after": round(after.pass_rate, 2),
        "pass_rate_delta_points": round(after.pass_rate - before.pass_rate, 2),
        "sufficient_before": before.sufficient_count,
        "sufficient_after": after.sufficient_count,
        "chunks_verified": after.total_chunks,
        "missed_concepts_before": before.missed_concepts,
        "missed_concepts_after": after.missed_concepts,
        "missed_concepts_delta": after.missed_concepts - before.missed_concepts,
        "missed_concepts_change_pct": _delta_pct(before.missed_concepts, after.missed_concepts),
        "missed_facts_before": before.missed_facts,
        "missed_facts_after": after.missed_facts,
        "missed_facts_delta": after.missed_facts - before.missed_facts,
        "missed_facts_change_pct": _delta_pct(before.missed_facts, after.missed_facts),
    }


def format_improvement(domain: str, improvement: Dict[str, Any]) -> str:
    cov_b = improvement["coverage_before"]
    cov_a = improvement["coverage_after"]
    cov_d = improvement["coverage_delta"]
    pr_b = improvement["pass_rate_before"]
    pr_a = improvement["pass_rate_after"]
    pr_d = improvement["pass_rate_delta_points"]
    mc_b = improvement["missed_concepts_before"]
    mc_a = improvement["missed_concepts_after"]
    mf_b = improvement["missed_facts_before"]
    mf_a = improvement["missed_facts_after"]
    chunks = improvement["chunks_verified"]

    return "\n".join([
        "=" * 60,
        f"VERIFICATION IMPROVEMENT [{domain.upper()}]  PASS 1 -> PASS 2",
        "=" * 60,
        f"Chunks Compared       : {chunks}",
        f"{'Metric':<22}{'Before':>10}{'After':>10}{'Delta':>14}",
        "-" * 60,
        f"{'Average Coverage':<22}{cov_b:>10.3f}{cov_a:>10.3f}{cov_d:>+14.3f}",
        f"{'Pass Rate (%)':<22}{pr_b:>10.1f}{pr_a:>10.1f}{pr_d:>+13.1f}p",
        f"{'Sufficient Chunks':<22}{improvement['sufficient_before']:>10}"
        f"{improvement['sufficient_after']:>10}"
        f"{improvement['sufficient_after'] - improvement['sufficient_before']:>+14}",
        f"{'Missed Concepts':<22}{mc_b:>10}{mc_a:>10}"
        f"{improvement['missed_concepts_delta']:>+10} "
        f"{_fmt_pct(improvement['missed_concepts_change_pct']):>3}",
        f"{'Missed Facts':<22}{mf_b:>10}{mf_a:>10}"
        f"{improvement['missed_facts_delta']:>+10} "
        f"{_fmt_pct(improvement['missed_facts_change_pct']):>3}",
        "=" * 60,
    ])


def write_summary(domain_dir: Path, domain: str,
                  before: VerificationMetrics,
                  after: Optional[VerificationMetrics] = None) -> Path:
    summary: Dict[str, Any] = {
        "domain": domain,
        "pass1": before.to_dict(),
    }
    if after is not None:
        summary["pass2"] = after.to_dict()
        summary["improvement"] = build_improvement(before, after)
    out_path = Path(domain_dir) / SUMMARY_FILENAME
    atomic_write_json(out_path, summary)
    return out_path


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
    synthetic: bool = Field(
        default=False,
        description="True if this verification result was synthesized as a fallback rather than evaluated by the LLM."
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
You are an expert knowledge auditor evaluating a Knowledge Wiki extraction.
Your task is to compare a RAW TEXT CHUNK against the EXTRACTED CONCEPTS to verify factual completeness, accuracy, and core concept coverage.

AUDIT DEFINITIONS & EXPECTATIONS:
1. SOLID CONCEPTS ONLY (NO FRAGMENTATION):
   - A concept is a canonical, self-contained unit of knowledge capturing a meaningful subject and its relevant facts, properties, and relationships.
   - Minor parameters, mechanisms, sub-steps, and narrow variants should NOT be standalone concepts; they should be absorbed into the comprehensive description of their parent concept (with narrow sub-topics preserved in <subtopics>).
   - Do NOT penalize the extraction for omitting meta-labels ("Introduction", "Summary") or for chunk-boundary artifacts (e.g. cut-off section headers with no body text in the current chunk).

2. VERBATIM FIDELITY & FACT LOSS:
   - Check that as much as possible, the extraction captured the source text VERBATIM without generalizing away specific information.
   - Flag as missing facts whenever specific names, direct quotes, attributions, technical parameters, arguments, or code/robotics details (identifiers, terminal commands, file paths, launch files, configs, formulas, and units) present in the text were omitted or excessively generalized into vague summaries.
   - For prose explanations, verify that core assertions, mechanisms, and attributions are preserved.

3. NOISE & NON-FACTS (SILENTLY IGNORE):
   - Silently ignore HTML tags, web scraping artifacts, navigation menus, headers/footers, cookie notices, pure bibliography/references metadata, and boilerplate.
   - Silently ignore raw telemetry logs, packet sequence numbers (e.g. `seq: 1593`), raw timestamp numbers, and raw numeric matrix / covariance array dumps (`[5001.06..., 1405.96...]`). Do NOT enumerate individual matrix numbers or raw dump values as missing facts. Do NOT penalize the extraction for omitting them.

4. SCORING CRITERIA:
- `coverage_score` (0.0 to 1.0):
  - 1.0: Perfect retention. All substantive themes, verbatim specifics, and key facts captured.
  - 0.8 - 0.9: Main themes captured; missed minor secondary details.
  - 0.5 - 0.7: Omitted a primary thematic concept or several key details.
  - < 0.5: Severe omissions of main body content.
- `is_sufficient`: `true` if coverage_score >= 0.85 and NO major primary concept was dropped. Otherwise `false`.

OUTPUT FORMAT:
Output ONLY the final <verification>...</verification> XML block with no introductory or concluding text:
<verification>
<coverage_score>0.95</coverage_score>
<is_sufficient>true</is_sufficient>
<missing_concepts>
- Name of missing PRIMARY theme (if any, max 8)
</missing_concepts>
<missing_facts>
- Specific missing fact or detail (if any, max 20)
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

        upstream = "extract-concepts" if self.phase == 1 else "refine-concepts"
        verify_stage_manifest(self.base_wiki_dir, upstream)

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
Evaluate whether the extraction accurately captured the core concepts and substantive facts from the text.
Identify any genuinely missing primary concepts (max {MAX_MISSING_CONCEPTS}), missing facts (max {MAX_MISSING_FACTS}), or hallucinations (max {MAX_HALLUCINATIONS}).
Output ONLY the <verification>...</verification> block.
"""

    def parse_verification(self, raw_text: Optional[str], key: str) -> VerificationResult:
        """Parse an audit response from XML (with JSON fallback) into VerificationResult."""
        if not raw_text:
            raise ValueError("Empty verification response")

        # 1. Check for XML <verification> block
        if "<verification>" in raw_text:
            block_m = re.search(r"<verification>((?:(?!<verification>).)*?)</verification>", raw_text, re.DOTALL | re.IGNORECASE)
            block = block_m.group(1) if block_m else raw_text

            score_m = re.search(r"<coverage_score>(.*?)</coverage_score>", block, re.DOTALL | re.IGNORECASE)
            coverage_score = None
            if score_m:
                val_str = re.search(r"([0-9]*\.?[0-9]+)", score_m.group(1).strip())
                if val_str:
                    try:
                        coverage_score = float(val_str.group(1))
                        if coverage_score > 1.0 and coverage_score <= 100.0:
                            coverage_score /= 100.0
                        coverage_score = max(0.0, min(1.0, coverage_score))
                    except ValueError:
                        coverage_score = None

            if coverage_score is None:
                raise ValueError("Missing or invalid coverage_score in verification output")

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
            except Exception as e:
                raise ValueError(f"Could not validate verification output for {key}: {e}")

        raise ValueError(f"Could not parse verification XML for {key}")

    async def verify_chunk(self, client: Any, item: Dict[str, Any],
                           server_index: int) -> VerificationResult:
        """One audit call using call_llm (task='judge'). Retries are owned by the server pool."""
        concepts_json_raw = await asyncio.to_thread(
            item["concepts_file"].read_text, encoding="utf-8"
        )
        user_prompt = self.build_prompt(item, concepts_json_raw)

        max_model_len = getattr(self.settings, "max_model_len", 65536)
        content = await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16000,
            args=self.args,
            cfg=self.cfg,
            task="judge",
            max_growth=2,
            max_model_len=max_model_len,
        )
        return self.parse_verification(content, item["key"])

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

        result.raise_if_failed(self.stage)
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
        write_stage_manifest(self.base_wiki_dir, self.stage, expected=len(items), completed=len(items), failed=0)

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
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of random chunks to audit (default: 50; set 0 to audit all chunks).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic chunk sampling across phases (default: 42).")
    args = parser.parse_args()

    verifier = WikiVerifierPipeline(args)
    asyncio.run(verifier.run())

if __name__ == "__main__":
    main()
