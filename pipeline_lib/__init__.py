"""Shared infrastructure for the wiki knowledge-graph pipeline.

The stage scripts at the repository root used to each carry their own copy of
JSON repair, checkpoint persistence, retry loops, server-pool setup and debug
dumping. That logic lives here once:

``errors``       classify a failure as transient / recoverable / permanent,
                 plus the per-stage wall-clock budget
``retry``        bounded retry with short capped backoff and jitter
``llm_pool``     spread work evenly across vLLM servers (item i -> server i % N)
                 with automatic failover off a dead server
``checkpoint``   atomic completion logs and resume support
``debug_log``    streaming JSONL sink; nothing is ever truncated
``json_repair``  extract JSON from raw model output
``atomic_io``    crash-safe writes for graphs, caches and reports
``cli``          shared argparse block and per-stage tuning profiles
``reporting``    verification metrics and the Pass 1 -> Pass 2 improvement
"""

from .atomic_io import atomic_write_json, atomic_write_text, read_json
from .checkpoint import (CheckpointLog, load_bool_log, load_json_log,
                         save_log_async, save_log_sync)
from .chunks import load_cached_chunks
from .cli import (STAGE_PROFILES, StageProfile, StageSettings, build_parser,
                  get_profile, resolve_settings)
from .debug_log import DebugLogger, iter_jsonl
from .errors import (PERMANENT, RECOVERABLE, TRANSIENT, AllServersUnhealthy,
                     PermanentFailure, StageBudget, StageBudgetExceeded,
                     TransientFailureExhausted, classify_error, describe_error)
from .json_repair import clean_json_string
from .llm_pool import PoolResult, ServerPool
from .reporting import (PASS1_LABEL, PASS1_RESULT, PASS2_LABEL, PASS2_RESULT,
                        SUMMARY_FILENAME, VerificationMetrics, build_improvement,
                        collect_metrics, format_improvement, format_report,
                        load_summary, write_summary)
from .retry import backoff_delay, retry_async

__all__ = [
    "atomic_write_json", "atomic_write_text", "read_json",
    "CheckpointLog", "load_bool_log", "load_json_log", "load_cached_chunks",
    "save_log_async", "save_log_sync",
    "STAGE_PROFILES", "StageProfile", "StageSettings", "build_parser",
    "get_profile", "resolve_settings",
    "DebugLogger", "iter_jsonl",
    "PERMANENT", "RECOVERABLE", "TRANSIENT", "AllServersUnhealthy",
    "PermanentFailure", "StageBudget", "StageBudgetExceeded",
    "TransientFailureExhausted", "classify_error", "describe_error",
    "clean_json_string",
    "PoolResult", "ServerPool",
    "PASS1_LABEL", "PASS1_RESULT", "PASS2_LABEL", "PASS2_RESULT",
    "SUMMARY_FILENAME", "VerificationMetrics", "build_improvement",
    "collect_metrics", "format_improvement", "format_report", "load_summary",
    "write_summary",
    "backoff_delay", "retry_async",
]
