"""Checkpoint & State Logging Utilities.
--------------------------------------
Manages durable JSON completion logs for pipeline stages, ensuring completed units
are safely skipped on resume.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .atomic_io import atomic_write_json


def load_json_log(path: Path, *, corrupt_message: Optional[str] = None,
                  coerce_bool: bool = False) -> Dict[str, Any]:
    """Load a checkpoint dict, tolerating absence and corruption."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        if corrupt_message:
            print(f"Warning: {corrupt_message}")
        return {}
    if not isinstance(data, dict):
        return {}
    if coerce_bool:
        return {str(k): bool(v) for k, v in data.items()}
    return data


def load_bool_log(path: Path, *, corrupt_message: Optional[str] = None) -> Dict[str, bool]:
    """Load a ``{key: bool}`` completion log, tolerating absence and corruption."""
    return load_json_log(path, corrupt_message=corrupt_message, coerce_bool=True)


def save_log_sync(path: Path, data: Dict[str, Any]) -> None:
    """Atomic synchronous checkpoint write."""
    atomic_write_json(path, data)


async def save_log_async(path: Path, data: Dict[str, Any]) -> None:
    """Offload the atomic write to a thread so the event loop keeps streaming."""
    await asyncio.to_thread(save_log_sync, path, data)
