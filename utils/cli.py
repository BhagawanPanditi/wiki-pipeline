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
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import _read_config

from .corpus import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_RUN_NAME

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_STAGE_BUDGET_MINUTES = 300.0


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
    # ── extraction & refinement (structured concept generation) ────────
    # max_tokens set to 16000, timeouts increased for large chunk batches
    "extract-concepts":       StageProfile("extract-concepts", 64, 2400.0, 50.0, 16000),
    "verify-pass1":           StageProfile("verify-pass1", 12, 2400.0, 50.0, 16000),
    "refine-concepts":        StageProfile("refine-concepts", 8, 2400.0, 50.0, 16000),
    "verify-pass2":           StageProfile("verify-pass2", 12, 2400.0, 50.0, 16000),

    # ── document & global consolidation ─────────────────────────────────
    "consolidate-documents":  StageProfile("consolidate-documents", 8, 4500.0, 30.0, 32000),
    "consolidate-global":     StageProfile("consolidate-global", 8, 800.0, 30.0, 8192),


    # ── stub resolution & folding ───────────────────────────────────────
    # resolve-stubs fans out chunks 5x per stub: concurrency 2 = 10 in-flight requests per server
    "resolve-stubs":          StageProfile("resolve-stubs", 2, 600.0, 30.0, 6144),
    "fold-thin-concepts":     StageProfile("fold-thin-concepts", 12, 360.0, 20.0, 6144),

    # ── graph enrichment & links (lightweight output payloads) ──────────
    # Concurrency 8 per server prevents TCP connection queue saturation on vLLM
    "write-briefs":           StageProfile("write-briefs", 32, 240.0, 60.0, 1536),
    "audit-islands":          StageProfile("audit-islands", 16, 360.0, 20.0, 3072),
    "add-reverse-links":      StageProfile("add-reverse-links", 32, 240.0, 60.0, 1536),

    # ── final consolidation (macro scan & deep verify) ──────────────────
    "final-macro-scan":       StageProfile("final-macro-scan", 8, 600.0, 20.0, 6144),
    "final-deep-verify":      StageProfile("final-deep-verify", 12, 600.0, 20.0, 3072),
}


class StageArgumentParser(argparse.ArgumentParser):
    """Argument parser that normalises the optional ``--domain`` flag.

    ``--domain`` serves two different jobs, so it is split on the way in:

    * ``args.domain`` -- always a usable directory/label name. When the caller
      omits ``--domain`` this becomes ``"default"``, so every existing
      ``<wiki_base_dir>/<domain>/`` path expression keeps working unchanged.
    * ``args.domain_label`` -- the *optional* hint itself, ``None`` when it was
      not supplied. LLM prompts use this, so a corpus with no declared domain
      does not get a meaningless "in the field of None" instruction.
    """

    def parse_args(self, args=None, namespace=None):
        """Parse arguments and apply the domain normalisation described above.

        Args:
            args: Argument list, defaulting to ``sys.argv[1:]``.
            namespace: Optional pre-populated namespace.

        Returns:
            The parsed ``argparse.Namespace`` with ``domain`` and
            ``domain_label`` populated.
        """
        parsed = super().parse_args(args, namespace)
        if hasattr(parsed, "domain"):
            parsed.domain_label = parsed.domain or None
            parsed.domain = parsed.domain or DEFAULT_RUN_NAME
        return parsed


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


_warned_stage_names: set = set()


def _stage_override(stage: str) -> Dict[str, Any]:
    """Return one stage's ``stage_concurrency`` entry from config.json as a dict.

    A plain number is shorthand for ``{"concurrency_per_server": number}``. An
    object is used as it is. Anything else is an error, so a typo in config.json
    stops the run rather than being ignored.

    Args:
        stage: Stage key, for example ``"extract-concepts"``.

    Returns:
        A dict of overrides, which may be empty.

    Raises:
        ValueError: If the entry is neither a number nor an object.
    """
    raw = load_stage_overrides().get(stage)
    if raw is None:
        return {}
    if isinstance(raw, bool) or not isinstance(raw, (int, float, dict)):
        raise ValueError(
            f"config.json stage_concurrency['{stage}'] must be a number or an object, "
            f"not {type(raw).__name__}.")
    if isinstance(raw, dict):
        return raw
    return {"concurrency_per_server": int(raw)}


def _warn_unknown_stage_overrides() -> None:
    """Report config.json entries that name no LLM stage, once per name.

    Such an entry would otherwise be ignored without a word, and the stage would
    quietly run with its built-in limits.

    Returns:
        None. Prints one warning to stderr for each unknown name.
    """
    for name in load_stage_overrides():
        if name not in STAGE_PROFILES and name not in _warned_stage_names:
            _warned_stage_names.add(name)
            print(f"WARNING: config.json stage_concurrency names '{name}', which is not a "
                  f"stage that calls the LLM, so it is ignored. Valid names: "
                  f"{', '.join(sorted(STAGE_PROFILES))}.", file=sys.stderr)


def resolve_settings(args: argparse.Namespace, stage: str) -> StageSettings:
    """Resolve runtime settings ONLY from CLI flags or config.json.

    Precedence: CLI flag > config.json (stage-specific or top-level).
    There are NO hardcoded python fallbacks; all settings must be explicitly defined.
    """
    from config import _read_config
    cfg = _read_config()
    stage_cfg = cfg.get("stage_concurrency", {}).get(stage, {}) if isinstance(cfg.get("stage_concurrency"), dict) else {}
    if not isinstance(stage_cfg, dict):
        stage_cfg = {"concurrency_per_server": int(stage_cfg)}

    def get_setting(cli_val: Any, key: str, err_name: str) -> Any:
        if cli_val not in (None, 0, 0.0, ""):
            return cli_val
        if key in stage_cfg:
            return stage_cfg[key]
        if key in cfg:
            return cfg[key]
        raise ValueError(
            f"Missing required parameter '{err_name}'! "
            f"Specify via CLI argument or set '{key}' in config.json."
        )

    # 1. Servers
    servers = getattr(args, "servers", None) or cfg.get("servers")
    if not servers:
        raise ValueError("Missing 'servers'! Specify via --servers or set 'servers' in config.json.")

    # 2. Model
    model = getattr(args, "model", None) or cfg.get("model_name") or cfg.get("model")
    if not model:
        raise ValueError("Missing 'model'! Specify via --model or set 'model_name' in config.json.")

    # 3. Concurrency
    concurrency = int(get_setting(
        getattr(args, "concurrency_per_server", None),
        "concurrency_per_server",
        "--concurrency_per_server"
    ))

    # 4. Max Tokens
    max_tokens = int(get_setting(
        getattr(args, "max_tokens", None),
        "max_tokens",
        "--max_tokens"
    ))

    # 5. Request Timeout
    timeout = float(get_setting(
        getattr(args, "request_timeout", None),
        "request_timeout",
        "--request_timeout"
    ))

    # Optional / Tuning parameters (attempts & budget)
    attempts = getattr(args, "max_attempts", 0) or stage_cfg.get("attempts") or cfg.get("attempts") or 4
    budget = getattr(args, "stage_budget_minutes", 0.0) or stage_cfg.get("budget_minutes") or cfg.get("budget_minutes") or 300.0

    profile = StageProfile(
        key=stage,
        concurrency_per_server=concurrency,
        request_timeout=timeout,
        connect_timeout=30.0,
        max_tokens=max_tokens,
        attempts=int(attempts),
        budget_minutes=float(budget),
    )

    return StageSettings(
        profile=profile,
        servers=servers,
        concurrency_per_server=max(1, concurrency),
        model_name=model,
        max_tokens=max_tokens,
        attempts=max(1, int(attempts)),
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
    include_corpus: bool = False,
) -> argparse.ArgumentParser:
    """Attach the argument block shared by every stage script."""
    profile = get_profile(stage)

    parser.add_argument("--wiki_path", "--wiki_dir", dest="wiki_path", type=str, default=None,
                        help="Output folder where wiki data is stored. "
                             "Directory is created automatically if it does not exist.")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to input corpus file (.json or .jsonl).")
    parser.add_argument("--disable_thinking", action="store_true", default=False,
                        help="Disable Qwen3.8-27B thinking mode (Instruct mode with enable_thinking=False).")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Explicitly enable Qwen3.8-27B thinking mode.")
    parser.add_argument("--llm_log", type=str, choices=("full", "compact", "errors"),
                        default="compact",
                        help="How much of each LLM call to keep in log/<stage>.llm.jsonl. "
                             "'compact' (default), 'full', or 'errors'.")
    if include_corpus:
        chunk_group = parser.add_mutually_exclusive_group()
        chunk_group.add_argument("--chunking", dest="chunking", action="store_true",
                                 default=True,
                                 help="Split long documents into overlapping token windows (default).")
        chunk_group.add_argument("--no-chunking", dest="chunking", action="store_false",
                                 help="Treat each document as one unit.")
        parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                            help=f"Chunk window in tokens (default {DEFAULT_CHUNK_SIZE}).")
        parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                            help=f"Overlap/context in tokens (default {DEFAULT_CHUNK_OVERLAP}).")
    if include_model:
        parser.add_argument("--model", dest="model", type=str, default=None,
                            help="Model name (e.g. Qwen/Qwen3.8-27B). Overrides config.json.")
        parser.add_argument("--tokenizer", dest="tokenizer", type=str, default=None,
                            help="Optional tokenizer path or HuggingFace repo name.")
    parser.add_argument("--servers", nargs="+", default=None,
                        help="vLLM OpenAI-compatible base URLs; overrides config.json.")
    parser.add_argument("--concurrency_per_server", type=int, default=0,
                        help="In-flight requests per server. Overrides config.json.")
    if include_max_tokens:
        parser.add_argument("--max_tokens", type=int, default=0,
                            help="Max generation length in tokens. Overrides config.json.")
    parser.add_argument("--max_attempts", type=int, default=0,
                        help="Retries per item for transient errors. Overrides config.json.")
    parser.add_argument("--stage_budget_minutes", type=float, default=0.0,
                        help="Wall-clock ceiling for this stage. Overrides config.json.")
    parser.add_argument("--request_timeout", type=float, default=0.0,
                        help="Per-attempt request timeout in seconds. Overrides config.json.")
    parser.add_argument("--no_failover", action="store_true",
                        help="Do not move work off a server that stops responding.")
    if include_resume:
        parser.add_argument("--resume", action="store_true",
                            help="Clear incomplete marks and continue from checkpoint.")
    return parser


def build_parser(*, stage: str, description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """Create a parser preloaded with the shared stage arguments."""
    return add_common_args(StageArgumentParser(description=description),
                           stage=stage, description=description, **kwargs)
