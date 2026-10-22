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
    """Return the server list and model name from config.json.

    Returns:
        (servers, model_name) loaded from config.json.

    Raises:
        ValueError: If config.json is missing or unreadable.
    """
    data = _read_config()
    servers = data.get("servers") or []
    model = data.get("model_name") or data.get("model") or ""
    return servers, model


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
