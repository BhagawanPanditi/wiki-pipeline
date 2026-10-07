#!/usr/bin/env python3
"""
Pipeline Configuration Loader
-----------------------------
Reads global configuration from `config.json` if available, or falls back to defaults.
Allows dynamically setting vLLM server URLs, concurrency, and model names in one central place.
"""

import json
from pathlib import Path
from typing import List, Tuple

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

DEFAULT_SERVERS = [
    "http://cn27-a40:8000/v1",
    "http://cn30-a40:8001/v1",
    "http://cn31-a40:8002/v1"
]
DEFAULT_CONCURRENCY = 50
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

def load_global_config() -> Tuple[List[str], int, str]:
    """Returns (servers, concurrency_per_server, model_name) from config.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                servers = data.get("servers", DEFAULT_SERVERS)
                concurrency = data.get("concurrency_per_server", DEFAULT_CONCURRENCY)
                model = data.get("model_name", DEFAULT_MODEL)
                return servers, concurrency, model
        except Exception:
            pass
    return DEFAULT_SERVERS, DEFAULT_CONCURRENCY, DEFAULT_MODEL

def load_embedding_model() -> str:
    """Returns embedding_model from config.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        except Exception:
            pass
    return DEFAULT_EMBEDDING_MODEL
