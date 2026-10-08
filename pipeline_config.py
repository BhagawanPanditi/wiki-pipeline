#!/usr/bin/env python3
"""
Pipeline Configuration Loader
-----------------------------
Reads global configuration from `config.json` if available, or falls back to
defaults. Centralises vLLM server URLs, the global concurrency fallback, model
names, and per-stage tuning overrides.

Embedding support has been removed: the pipeline is LLM-only, so there is no
embedding model, no vector cache and no GPU bridging stage to configure.

Optional `config.json` shape:

    {
      "servers": ["http://cn27-a40:8000/v1", "http://cn30-a40:8001/v1"],
      "concurrency_per_server": 50,
      "model_name": "Qwen/Qwen3.8-27B",
      "stage_concurrency": {
        "pass1-extraction": 8,
        "reverse-relations": 48
      }
    }

`stage_concurrency` overrides the per-stage defaults in
`pipeline_lib/cli.py`. Use it to match the KV-cache budget vLLM reports at
startup ("Maximum concurrency for 32,768 tokens per request: Nx") without
editing code.
"""

import json
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


def _read_config() -> Dict[str, Any]:
    """Return parsed config.json, or an empty mapping when unusable."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def load_global_config() -> Tuple[List[str], int, str]:
    """Returns (servers, concurrency_per_server, model_name) from config.json."""
    data = _read_config()
    if data:
        servers = data.get("servers", DEFAULT_SERVERS)
        concurrency = data.get("concurrency_per_server", DEFAULT_CONCURRENCY)
        model = data.get("model_name", DEFAULT_MODEL)
        return servers, concurrency, model
    return DEFAULT_SERVERS, DEFAULT_CONCURRENCY, DEFAULT_MODEL


def load_stage_overrides() -> Dict[str, Any]:
    """Per-stage tuning overrides keyed by stage name."""
    data = _read_config()
    overrides = data.get("stage_concurrency", {})
    return overrides if isinstance(overrides, dict) else {}
