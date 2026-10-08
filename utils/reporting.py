"""Verification metrics and the Pass 1 -> Pass 2 improvement report.

Verification runs twice: once over Pass 1 output and again over the concepts
Pass 2 merged in. Both runs score the same chunks against the same source text,
so the two averages are directly comparable and their difference is the
measurable improvement Pass 2 bought.

Pass 2 must not read the second run's results - it consumes the first run's
``missing_concepts`` / ``missing_facts`` as its repair instructions - so the
two runs write to separate filenames and are summarised together at the end.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .atomic_io import atomic_write_json, read_json

PASS1_RESULT = "verification.json"
PASS2_RESULT = "verification_pass2.json"
PASS1_LABEL = "PASS 1 (raw extraction)"
PASS2_LABEL = "PASS 2 (after refinement)"
SUMMARY_FILENAME = "verification_summary.json"


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
    sample_missed_concepts: List[str] = None  # type: ignore[assignment]
    sample_missed_facts: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Replace the mutable-default sentinels with empty lists.

        ``None`` is used as the field default because a dataclass cannot safely default
        to a shared list. This runs after ``__init__`` so callers may omit both sample
        fields and still get independent, mutable lists.

        Returns:
            None."""
        if self.sample_missed_concepts is None:
            self.sample_missed_concepts = []
        if self.sample_missed_facts is None:
            self.sample_missed_facts = []

    def to_dict(self) -> Dict[str, Any]:
        """Render the metrics as a plain dict for JSON output.

        Returns:
            Every field of this dataclass, via ``dataclasses.asdict``. Written into
            ``verification_summary.json`` so the two phases can be compared offline."""
        return asdict(self)


def collect_metrics(staging_dir: Path, *, label: str,
                    result_filename: str = PASS1_RESULT) -> VerificationMetrics:
    """Aggregate every per-chunk verification result under ``staging_dir``."""
    metrics = VerificationMetrics(label=label, result_filename=result_filename)
    scores: List[float] = []
    missed_concepts: List[str] = []
    missed_facts: List[str] = []

    staging_dir = Path(staging_dir)
    if not staging_dir.is_dir():
        return metrics

    for chunk_dir in sorted(staging_dir.glob("*/*")):
        result_file = chunk_dir / result_filename
        if not result_file.is_file():
            continue
        data = read_json(result_file, default=None)
        if not isinstance(data, dict) or "coverage_score" not in data:
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
    """Render one run in the established summary format."""
    lines = [
        "=" * 60,
        f"VERIFICATION SUMMARY REPORT [{domain.upper()}] - {metrics.label}",
        "=" * 60,
    ]
    if metrics.total_chunks == 0:
        lines.append(f"No {metrics.result_filename} results found to summarize.")
        return "\n".join(lines)

    lines.extend([
        f"Total Chunks Verified : {metrics.total_chunks}",
        f"Average Coverage Score: {metrics.average_coverage:.3f} / 1.000",
        f"Pass Rate (Sufficient): {metrics.pass_rate:.1f}% "
        f"({metrics.sufficient_count}/{metrics.total_chunks})",
        f"Total Missed Concepts : {metrics.missed_concepts}",
        f"Total Missed Facts    : {metrics.missed_facts}",
    ])
    if metrics.sample_missed_concepts:
        lines.append("\nSample Missed Concepts:")
        lines.extend(f"  - {c}" for c in metrics.sample_missed_concepts)
    if metrics.sample_missed_facts:
        lines.append("\nSample Missed Facts:")
        lines.extend(f"  - {f}" for f in metrics.sample_missed_facts)
    return "\n".join(lines)


def _delta_pct(before: float, after: float) -> Optional[float]:
    """Express a change as a percentage of the earlier value.

    Args:
        before: The Pass 1 value.
        after: The Pass 2 value.

    Returns:
        The signed percentage change, or ``None`` when ``before`` is zero -- a
        division that would be meaningless, which the report renders as "n/a"
        rather than as an infinite improvement."""
    if before == 0:
        return None
    return ((after - before) / before) * 100.0


def _fmt_pct(value: Optional[float]) -> str:
    """Format a percentage delta for the improvement table.

    Args:
        value: A signed percentage, or ``None`` when the delta is undefined.

    Returns:
        ``"+12.3%"``-style text, or ``"n/a"`` for ``None``."""
    return "n/a" if value is None else f"{value:+.1f}%"


def build_improvement(before: VerificationMetrics, after: VerificationMetrics) -> Dict[str, Any]:
    """Compute the Pass 1 -> Pass 2 deltas."""
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
    """Render the side-by-side improvement block."""
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
    """Persist both runs plus the improvement to ``verification_summary.json``."""
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


def load_summary(domain_dir: Path) -> Optional[Dict[str, Any]]:
    """Read a previously written summary, if any."""
    data = read_json(Path(domain_dir) / SUMMARY_FILENAME, default=None)
    return data if isinstance(data, dict) else None
