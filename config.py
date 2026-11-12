#!/usr/bin/env python3
"""Pipeline configuration: servers, models, and per-stage overrides.

Reads ``config.json`` from the repository root.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def _read_config() -> Dict[str, Any]:
    """Return the parsed config.json, or an empty mapping when there is no file."""
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


def load_global_config() -> Tuple[List[str], str]:
    """Return the flat list of all server URLs and the model name from config.json."""
    data = _read_config()
    raw_servers = data.get("servers", {})
    if isinstance(raw_servers, dict):
        servers = []
        for urls in raw_servers.values():
            if isinstance(urls, list):
                servers.extend(urls)
            elif isinstance(urls, str):
                servers.append(urls)
    elif isinstance(raw_servers, list):
        servers = raw_servers
    else:
        servers = []
    model = data.get("model_name") or data.get("model") or ""
    return servers, model
