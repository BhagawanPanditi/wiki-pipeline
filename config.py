#!/usr/bin/env python3
"""Pipeline configuration: servers, models, defaults and per-stage overrides.

Reads ``config.json`` from the repository root when it exists, and falls back to
the defaults below when it does not. The file is git-ignored, so every machine
keeps its own.

``servers`` and ``model_name`` apply to every stage, to ``check_servers.py`` and
to the runner's preflight. A file that exists but is not valid JSON is an error,
not a silent fall-back to the defaults.

Example::

    {
      "servers": ["http://cn27-a40:8000/v1", "http://cn30-a40:8001/v1"],
      "model_name": "Qwen/Qwen3.8-27B",
      "stage_concurrency": {
        "extract-concepts": 8,
        "add-reverse-links": {"concurrency_per_server": 48, "attempts": 6}
      }
    }

``stage_concurrency`` overrides the built-in limits in ``utils/cli.py``
(``STAGE_PROFILES``) for one stage. A plain number sets that stage's concurrency.
An object may set ``concurrency_per_server``, ``attempts`` and
``budget_minutes``. Use it to match the KV-cache budget that vLLM reports at
startup ("Maximum concurrency for 32,768 tokens per request") without editing code.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

DEFAULT_SERVERS = [
    "http://cn27-a40:8000/v1",
    "http://cn30-a40:8001/v1",
    "http://cn31-a40:8002/v1"
]
# Fallback only. Every stage resolves its own default from STAGE_PROFILES,
# because sustainable fan-out depends on that stage's max_tokens.
DEFAULT_CONCURRENCY = 50
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# Where runs are written and where legacy corpora are read when the command line
# does not say otherwise. These are the cluster locations. On any other machine,
# pass --wiki-base-dir and --data-dir, or set WIKI_BASE_DIR and DATA_DIR for
# run_pipeline.sh.
DEFAULT_WIKI_BASE_DIR = "/home/compiling-ganesh/24m0829/long_document/wiki"
DEFAULT_DATA_DIR = "/home/compiling-ganesh/24m0829/long_document/data/long_documents"


def _read_config() -> Dict[str, Any]:
    """Return the parsed config.json, or an empty mapping when there is no file.

    A file that exists but cannot be read, is not valid JSON, or is not a JSON
    object is an error. Falling back to the built-in cluster servers in that case
    would send every stage to the wrong machines without a word.

    Returns:
        The parsed object, or ``{}`` when config.json does not exist.

    Raises:
        ValueError: If config.json exists but is unreadable, not valid JSON, or
            not a JSON object.
    """
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{CONFIG_FILE} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_FILE} must contain a JSON object at the top level.")
    return data


_warned_config_keys: set = set()


def load_global_config() -> Tuple[List[str], str]:
    """Return the server list and model name that every stage uses.

    Each value comes from config.json when the file sets it, and from the built-in
    default otherwise. A top-level ``concurrency_per_server`` is not read: each
    stage's concurrency comes from its profile and from ``stage_concurrency``. When
    that key is present, a warning is printed once so that it is not ignored silently.

    Returns:
        ``(servers, model_name)``.

    Raises:
        ValueError: If config.json exists but cannot be parsed (see ``_read_config``).
    """
    data = _read_config()
    if "concurrency_per_server" in data and "concurrency_per_server" not in _warned_config_keys:
        _warned_config_keys.add("concurrency_per_server")
        print("WARNING: config.json 'concurrency_per_server' is no longer used. Set "
              "'stage_concurrency' per stage instead (see the README).", file=sys.stderr)
    return data.get("servers", DEFAULT_SERVERS), data.get("model_name", DEFAULT_MODEL)


def load_stage_overrides() -> Dict[str, Any]:
    """Return the per-stage tuning overrides from config.json, keyed by stage name.

    Returns:
        The ``stage_concurrency`` object, or ``{}`` when config.json does not set it.

    Raises:
        ValueError: If config.json exists but ``stage_concurrency`` is not an object,
            so that a mistyped list is reported rather than silently ignored.
    """
    data = _read_config()
    if "stage_concurrency" not in data:
        return {}
    overrides = data["stage_concurrency"]
    if not isinstance(overrides, dict):
        raise ValueError(f"{CONFIG_FILE}: 'stage_concurrency' must be an object keyed by stage name.")
    return overrides
