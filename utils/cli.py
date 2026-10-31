"""Shared CLI Configuration & Dynamic Per-Stage Tuning Profiles.
-------------------------------------------------------------
Provides unified argument parsing, default stage profiles, and dynamic
per-stage settings resolution (concurrencies, max_tokens, timeouts) from config.json.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import _read_config

from .corpus import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_RUN_NAME

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_STAGE_BUDGET_MINUTES = 300.0
DEFAULT_CONCURRENCY = 16


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
    # ── Stage 1: Extraction & Stage 3: Refinement ───────────────────────
    "extract-concepts":       StageProfile("extract-concepts", 64, 2400.0, 50.0, 16000),
    "verify-pass1":           StageProfile("verify-pass1", 12, 2400.0, 50.0, 16000),
    "refine-concepts":        StageProfile("refine-concepts", 8, 2400.0, 50.0, 16000),
    "verify-pass2":           StageProfile("verify-pass2", 12, 2400.0, 50.0, 16000),

    # ── Stage 5: Document & Stage 6: Global Consolidation ───────────────
    "consolidate-documents":  StageProfile("consolidate-documents", 8, 4500.0, 30.0, 32000),
    "consolidate-global":     StageProfile("consolidate-global", 8, 800.0, 30.0, 8192),

    # ── Stage 7: Briefs & Entities, Stage 8: Semantic Jaccard Links ─────
    "write-briefs":           StageProfile("write-briefs", 32, 90.0, 30.0, 1536),
    "link-entities":          StageProfile("link-entities", 32, 60.0, 30.0, 512),

    # ── Stage 9: Final Consolidation (Macro Scan & Deep Verify) ─────────
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
    """Return one stage's ``stage_concurrency`` entry from config.json as a dict."""
    cfg = _read_config() or {}
    stage_concurrencies = cfg.get("stage_concurrency", {})
    raw = stage_concurrencies.get(stage)
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
    """Report config.json entries that name no LLM stage, once per name."""
    cfg = _read_config() or {}
    stage_concurrencies = cfg.get("stage_concurrency", {})
    for name in stage_concurrencies:
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
    # Read config.json
    cfg = _read_config()
    stage_concurrency_cfg = cfg.get("stage_concurrency", {}) if isinstance(cfg.get("stage_concurrency"), dict) else {}
    stage_cfg = stage_concurrency_cfg.get(stage, {}) if isinstance(stage_concurrency_cfg, dict) else {}
    if not isinstance(stage_cfg, dict):
        stage_cfg = {"concurrency_per_server": int(stage_cfg)}

    default_profile = STAGE_PROFILES.get(stage)

    def get_setting(cli_val: Any, key: str, default_val: Any = None) -> Any:
        if cli_val not in (None, 0, 0.0, ""):
            return cli_val
        if key in stage_cfg:
            return stage_cfg[key]
        if key in cfg:
            return cfg[key]
        if default_val is not None:
            return default_val
        raise ValueError(
            f"Missing required parameter for stage '{stage}': '{key}'! "
            f"Specify via CLI argument or set '{key}' in config.json."
        )

    # 1. Servers (Accepts list of URLs, or dict of {url: concurrency})
    raw_servers = getattr(args, "servers", None) or cfg.get("servers")
    if not raw_servers:
        raise ValueError("Missing 'servers'! Specify via --servers or set 'servers' in config.json.")

    # 2. Model
    model = getattr(args, "model", None) or cfg.get("model_name") or cfg.get("model")
    if not model:
        raise ValueError("Missing 'model'! Specify via --model or set 'model_name' in config.json.")

    # 3. Determine Concurrency
    cli_concurrency = getattr(args, "concurrency_per_server", None)
    if cli_concurrency not in (None, 0, 0.0, ""):
        default_concurrency = int(cli_concurrency)
    elif "concurrency_per_server" in stage_cfg:
        default_concurrency = int(stage_cfg["concurrency_per_server"])
    elif isinstance(stage_cfg, dict) and stage_cfg.get("concurrency"):
        default_concurrency = int(stage_cfg["concurrency"])
    elif "concurrency_per_server" in cfg:
        default_concurrency = int(cfg["concurrency_per_server"])
    elif isinstance(raw_servers, dict) and raw_servers:
        default_concurrency = max(int(c) for c in raw_servers.values())
    elif default_profile:
        default_concurrency = default_profile.concurrency_per_server
    else:
        default_concurrency = 16

    # Build per-server concurrency mapping for this specific stage:
    # 1. stage_cfg has specific server URLs: {"http://cn-dgx...": 32, "http://cn-a40...": 16}
    # 2. raw_servers is a list of URLs: ["http://cn-a40:8000/v1", "http://cn-dgx:8000/v1"]
    # 3. raw_servers is a dict: {"http://cn-a40:8000/v1": 16, "http://cn-dgx:8000/v1": 32}
    if isinstance(stage_cfg, dict) and any(isinstance(v, (int, float)) and k.startswith("http") for k, v in stage_cfg.items()):
        server_list = list(raw_servers.keys()) if isinstance(raw_servers, dict) else list(raw_servers)
        servers = {s: int(stage_cfg.get(s, default_concurrency)) for s in server_list}
        concurrency = max(servers.values())
    elif isinstance(raw_servers, dict):
        if "concurrency_per_server" in stage_cfg or (isinstance(stage_cfg, dict) and stage_cfg.get("concurrency")):
            servers = {s: default_concurrency for s in raw_servers}
            concurrency = default_concurrency
        else:
            servers = raw_servers
            concurrency = default_concurrency
    else:
        servers = raw_servers
        concurrency = default_concurrency

    # 4. Max Tokens (Checks CLI -> stage_cfg -> cfg -> default STAGE_PROFILES)
    default_max_tokens = default_profile.max_tokens if default_profile else 4096
    max_tokens = int(get_setting(
        getattr(args, "max_tokens", None),
        "max_tokens",
        default_val=default_max_tokens
    ))

    # 5. Request Timeout (Checks CLI -> stage_cfg -> cfg -> default STAGE_PROFILES)
    default_timeout = default_profile.request_timeout if default_profile else 600.0
    timeout = float(get_setting(
        getattr(args, "request_timeout", None),
        "request_timeout",
        default_val=default_timeout
    ))

    # Optional / Tuning parameters (attempts & budget)
    attempts = getattr(args, "max_attempts", 0) or stage_cfg.get("attempts") or cfg.get("attempts") or (default_profile.attempts if default_profile else 4)
    budget = getattr(args, "stage_budget_minutes", 0.0) or stage_cfg.get("budget_minutes") or cfg.get("budget_minutes") or (default_profile.budget_minutes if default_profile else 300.0)

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
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable verbose debugging, including printing raw LLM outputs on parse failure.")
    if include_resume:
        parser.add_argument("--resume", action="store_true",
                            help="Clear incomplete marks and continue from checkpoint.")
    return parser


def build_parser(*, stage: str, description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """Create a parser preloaded with the shared stage arguments."""
    return add_common_args(StageArgumentParser(description=description),
                           stage=stage, description=description, **kwargs)
