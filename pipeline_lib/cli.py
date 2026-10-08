"""Shared CLI plumbing and per-stage tuning profiles.

Every stage script repeated the same argparse block and the same hardcoded
httpx timeout. This module owns both, and adds a per-stage concurrency default
derived from the serving configuration rather than a single global 50.

Concurrency is bounded by KV cache, not by the client. On 2x A40 with
tensor-parallel-size 2 and --max-model-len 32768, a 27B model in BF16 leaves
roughly 32 GB for KV cache (~130k tokens). Each in-flight request reserves
prompt + max_tokens, so the sustainable fan-out scales inversely with
max_tokens:

    pass1/pass2 (8192 out, ~10k prompt)  -> ~8 per server
    briefs / reverse relations (256 out) -> ~40-48 per server

vLLM prints the exact figures at startup as "GPU KV cache size" and "Maximum
concurrency for 32,768 tokens per request"; override these defaults in
config.json under "stage_concurrency" to match your real numbers.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pipeline_config import (DEFAULT_CONCURRENCY, DEFAULT_MODEL, DEFAULT_SERVERS,
                             load_stage_overrides)

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_STAGE_BUDGET_MINUTES = 45.0
DEFAULT_DATA_DIR = "/home/compiling-ganesh/24m0829/long_document/data/long_documents"
DEFAULT_WIKI_BASE_DIR = "/home/compiling-ganesh/24m0829/long_document/wiki"


@dataclass(frozen=True)
class StageProfile:
    """Tuning for one stage. Timeouts match the values each stage used before."""

    key: str
    concurrency_per_server: int
    request_timeout: float
    connect_timeout: Optional[float] = None
    max_tokens: Optional[int] = None
    attempts: int = DEFAULT_MAX_ATTEMPTS
    budget_minutes: float = DEFAULT_STAGE_BUDGET_MINUTES

    @property
    def attempt_timeout(self) -> float:
        """Backstop above the httpx read timeout, for calls that never return."""
        return self.request_timeout + 30.0


STAGE_PROFILES: Dict[str, StageProfile] = {
    # ── extraction: 8192-token outputs dominate KV, keep fan-out low ────────
    "pass1-extraction":       StageProfile("pass1-extraction", 8, 300.0, 30.0, 8192),
    "pass2-refinement":       StageProfile("pass2-refinement", 8, 300.0, 10.0, 8192),
    "document-consolidation": StageProfile("document-consolidation", 6, 300.0, None, 8192),
    "stub-resolution":        StageProfile("stub-resolution", 10, 45.0, None, 8192),

    # ── verification: long prompt, structured JSON out ─────────────────────
    "verification":           StageProfile("verification", 12, 450.0, 10.0, 4096),
    "verification-pass2":     StageProfile("verification-pass2", 12, 450.0, 10.0, 4096),

    # ── consolidation: medium outputs ──────────────────────────────────────
    "global-consolidation":   StageProfile("global-consolidation", 16, 300.0, None, 4096),
    "final-macro-scan":       StageProfile("final-macro-scan", 12, 300.0, None, 4096),
    "final-deep-verify":      StageProfile("final-deep-verify", 24, 60.0, None, 1000),

    # ── graph enrichment: small outputs, high fan-out ──────────────────────
    "thin-concept-folding":   StageProfile("thin-concept-folding", 24, 45.0, None, 1024),
    "brief-generation":       StageProfile("brief-generation", 40, 40.0, None, 256),
    "island-audit":           StageProfile("island-audit", 32, 60.0, None, 400),
    "reverse-relations":      StageProfile("reverse-relations", 48, 60.0, None, 256),
}


def get_profile(stage: str) -> StageProfile:
    """Return a stage's profile, falling back to a conservative default."""
    return STAGE_PROFILES.get(stage) or StageProfile(
        stage, DEFAULT_CONCURRENCY, 300.0, None, None,
    )


@dataclass
class StageSettings:
    """Resolved runtime settings after CLI and config.json overrides."""

    profile: StageProfile
    servers: Any
    concurrency_per_server: int
    model_name: str
    max_tokens: Optional[int]
    attempts: int
    budget_minutes: Optional[float]


def resolve_settings(args: argparse.Namespace, stage: str) -> StageSettings:
    """Merge the stage profile with config.json overrides and CLI flags.

    Precedence: explicit CLI flag > config.json stage override > profile
    default. ``--concurrency_per_server 0`` means "use the stage default",
    which keeps the flag meaningful now that defaults differ per stage.
    """
    profile = get_profile(stage)
    overrides = load_stage_overrides()
    stage_override = overrides.get(stage, {}) if isinstance(overrides, dict) else {}

    def pick(cli_value: Any, override_key: str, default: Any) -> Any:
        if cli_value not in (None, 0):
            return cli_value
        if isinstance(stage_override, dict) and override_key in stage_override:
            return stage_override[override_key]
        return default

    concurrency = int(pick(getattr(args, "concurrency_per_server", 0),
                           "concurrency_per_server", profile.concurrency_per_server))
    attempts = int(pick(getattr(args, "max_attempts", 0), "attempts", profile.attempts))
    budget = getattr(args, "stage_budget_minutes", None)
    if budget in (None, 0):
        budget = stage_override.get("budget_minutes", profile.budget_minutes) if isinstance(
            stage_override, dict) else profile.budget_minutes
    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens in (None, 0):
        max_tokens = profile.max_tokens

    return StageSettings(
        profile=profile,
        servers=getattr(args, "servers", None) or DEFAULT_SERVERS,
        concurrency_per_server=max(1, concurrency),
        model_name=getattr(args, "model_name", None) or DEFAULT_MODEL,
        max_tokens=max_tokens,
        attempts=max(1, attempts),
        budget_minutes=float(budget) if budget else None,
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    stage: str,
    description: str,
    include_data_dir: bool = False,
    include_max_tokens: bool = True,
    include_resume: bool = False,
    include_model: bool = True,
) -> argparse.ArgumentParser:
    """Attach the argument block shared by every stage script."""
    profile = get_profile(stage)

    parser.add_argument("--domain", type=str, required=True,
                        help="Target domain (e.g. biology, robotics)")
    parser.add_argument("--wiki_base_dir", type=str, default=DEFAULT_WIKI_BASE_DIR)
    if include_data_dir:
        parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    if include_model:
        parser.add_argument("--model_name", "--model", dest="model_name", type=str,
                            default=DEFAULT_MODEL,
                            help="Model name served by the vLLM endpoints.")
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS,
                        help="vLLM OpenAI-compatible base URLs; work is spread evenly across them.")
    parser.add_argument("--concurrency_per_server", type=int, default=0,
                        help=f"In-flight requests per server. 0 = stage default "
                             f"({profile.concurrency_per_server} for {stage}).")
    if include_max_tokens:
        parser.add_argument("--max_tokens", type=int, default=0,
                            help=f"Max generation length. 0 = stage default "
                                 f"({profile.max_tokens} for {stage}).")
    parser.add_argument("--max_attempts", type=int, default=0,
                        help=f"Retries per item for transient errors. 0 = default "
                             f"({profile.attempts}). Permanent errors never retry.")
    parser.add_argument("--stage_budget_minutes", type=float, default=0.0,
                        help=f"Wall-clock ceiling for this stage. 0 = default "
                             f"({profile.budget_minutes:g} min). Stops backoff from "
                             f"running indefinitely; checkpoints stay resumable.")
    parser.add_argument("--no_failover", action="store_true",
                        help="Do not move work off a server that stops responding.")
    if include_resume:
        parser.add_argument("--resume", action="store_true",
                            help="Clear incomplete marks and continue from the checkpoint.")
    return parser


def build_parser(*, stage: str, description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """Create a parser preloaded with the shared stage arguments."""
    return add_common_args(argparse.ArgumentParser(description=description),
                           stage=stage, description=description, **kwargs)
