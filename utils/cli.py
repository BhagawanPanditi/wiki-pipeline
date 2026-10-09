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

from config import (DEFAULT_CONCURRENCY, DEFAULT_DATA_DIR, DEFAULT_WIKI_BASE_DIR,
                    load_global_config, load_stage_overrides)

from .corpus import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_RUN_NAME

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_STAGE_BUDGET_MINUTES = 45.0


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
    "extract-concepts":       StageProfile("extract-concepts", 32, 1200.0, 30.0, 16384 ),
    "refine-concepts":       StageProfile("refine-concepts", 32, 12000.0, 10.0, 16384),
    "consolidate-documents": StageProfile("consolidate-documents", 32, 1200.0, None, 16384),
    # Each stub fans out over its own source chunks 5 at a time, so 2 stubs in
    # flight is ~10 concurrent 8192-token requests per server - the KV-correct
    # figure. The old code ran 15 stubs x 5 chunks = 75 in flight, which
    # oversubscribed KV cache ~7x and caused preemption and recompute.  
    "resolve-stubs":        StageProfile("resolve-stubs", 2, 1200.0, None, 16384),

    # ── verification: long prompt, structured JSON out ─────────────────────
    "verify-pass1":           StageProfile("verify-pass1", 32, 1200.0, 10.0, 16384),
    "verify-pass2":     StageProfile("verify-pass2", 32, 1200.0, 10.0, 16384),

    # ── consolidation: medium outputs ──────────────────────────────────────
    "consolidate-global":   StageProfile("consolidate-global", 16, 600.0, None, 8192),
    "final-macro-scan":       StageProfile("final-macro-scan", 12, 600.0, None, 8192),
    "final-deep-verify":      StageProfile("final-deep-verify", 24, 600.0, None, 4096),

    # ── graph enrichment: small outputs, high fan-out ──────────────────────
    # max_tokens is 4096 because that is what this stage has always requested;
    # a smaller budget would truncate fold decisions and change output quality.
    "fold-thin-concepts":   StageProfile("fold-thin-concepts", 16, 600.0, None, 8192),
    "write-briefs":       StageProfile("write-briefs", 40, 600.0, None, 1024),
    "audit-islands":           StageProfile("audit-islands", 32, 600.0, None, 4096),
    "add-reverse-links":      StageProfile("add-reverse-links", 48, 600.0, None, 1024),
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
    """Merge the stage profile with config.json overrides and CLI flags.

    Precedence: explicit CLI flag > config.json stage override > profile
    default. ``--concurrency_per_server 0`` means "use the stage default",
    which keeps the flag meaningful now that defaults differ per stage.
    """
    profile = get_profile(stage)
    _warn_unknown_stage_overrides()
    stage_override = _stage_override(stage)
    servers_default, model_default = load_global_config()

    def pick(cli_value: Any, override_key: str, default: Any) -> Any:
        """Resolve one tunable by precedence.

        Args:
            cli_value: The value from the command line. ``None`` and ``0`` both mean
                "not specified", which is what makes ``--concurrency_per_server 0``
                usable as "give me the stage default".
            override_key: Key to look for in this stage's ``config.json`` override.
            default: The profile's built-in value.

        Returns:
            The first of the three that was actually specified: CLI, then config.json,
            then the profile default."""
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
        servers=getattr(args, "servers", None) or servers_default,
        concurrency_per_server=max(1, concurrency),
        model_name=getattr(args, "model_name", None) or model_default,
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
    include_corpus: bool = False,
) -> argparse.ArgumentParser:
    """Attach the argument block shared by every stage script.

    Args:
        parser: Parser to extend; mutated in place and returned.
        stage: Stage profile key, used to look up concurrency defaults.
        description: Shown in ``--help``.
        include_data_dir: Add the legacy ``--data_dir`` flag.
        include_max_tokens: Add ``--max_tokens`` (off for stages whose output
            length is fixed by the prompt).
        include_resume: Add ``--resume``.
        include_model: Add ``--model_name`` with a ``--model`` alias.
        include_corpus: Add ``--corpus`` plus the chunking flags, for the stage
            that reads the source documents.

    Returns:
        The same parser, with all requested arguments attached.
    """
    profile = get_profile(stage)

    parser.add_argument("--domain", type=str, default=None,
                        help="Optional domain hint (e.g. biology, robotics). Names the "
                             "output folder and, when given, focuses the LLM prompts. "
                             f"Omit it for an undeclared corpus: outputs go to "
                             f"'{DEFAULT_RUN_NAME}/' and prompts skip the domain clause.")
    parser.add_argument("--wiki_base_dir", type=str, default=DEFAULT_WIKI_BASE_DIR,
                        help="Folder that holds every run. This run writes to "
                             "<wiki_base_dir>/<domain or 'default'>/. Default: "
                             f"{DEFAULT_WIKI_BASE_DIR} (the cluster location; on another "
                             "machine pass your own path).")
    parser.add_argument("--llm_log", type=str, choices=("full", "compact", "errors"),
                        default="compact",
                        help="How much of each LLM call to keep in log/<stage>.llm.jsonl. "
                             "'compact' (default) cuts each prompt message and each reply "
                             "to 2000 characters and records the true length. 'full' keeps "
                             "every character. 'errors' keeps failed calls only. Failure "
                             "details in the debug log are always complete.")
    if include_corpus:
        parser.add_argument("--corpus", type=str, default=None,
                            help="Corpus of {id, contents} records: one JSON array (.json) "
                                 "or one object per line (.jsonl). Replaces the legacy "
                                 "--data_dir/--domain lookup.")
        chunk_group = parser.add_mutually_exclusive_group()
        chunk_group.add_argument("--chunking", dest="chunking", action="store_true",
                                 default=True,
                                 help="Split long documents into overlapping token windows "
                                      "(default). Needs a tokenizer.")
        chunk_group.add_argument("--no-chunking", dest="chunking", action="store_false",
                                 help="Treat each document as one unit. No tokenizer is "
                                      "loaded and per-document chunk consolidation becomes "
                                      "a pass-through.")
        parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                            help=f"Chunk window in tokens (default {DEFAULT_CHUNK_SIZE}); "
                                 "ignored with --no-chunking.")
        parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                            help=f"Overlap/context in tokens (default "
                                 f"{DEFAULT_CHUNK_OVERLAP}); ignored with --no-chunking.")
    if include_data_dir:
        parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                            help="Folder that holds <domain>.jsonl corpus files for the "
                                 "legacy layout. Used only when --corpus is not given. "
                                 f"Default: {DEFAULT_DATA_DIR} (the cluster location).")
    if include_model:
        parser.add_argument("--model_name", "--model", dest="model_name", type=str,
                            default=load_global_config()[1],
                            help="Model name served by the vLLM endpoints.")
    parser.add_argument("--servers", nargs="+", default=load_global_config()[0],
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
    return add_common_args(StageArgumentParser(description=description),
                           stage=stage, description=description, **kwargs)
