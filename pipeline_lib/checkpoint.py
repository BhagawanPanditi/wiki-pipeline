"""Completion checkpoints shared by the extraction-style stages.

Pass 1, verification, Pass 2 and document consolidation each carried their own
copies of ``_load_log`` / ``_save_log_sync`` / ``save_log_async``. The copies
were functionally identical, so they collapse into one implementation here
plus a small ``CheckpointLog`` helper that owns the flush cadence.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .atomic_io import atomic_write_json, read_json


def load_json_log(path: Path, *, corrupt_message: Optional[str] = None,
                  coerce_bool: bool = False) -> Dict[str, Any]:
    """Load a checkpoint dict, tolerating absence and corruption.

    ``corrupt_message`` reproduces the warning each stage used to print when its
    log existed but could not be parsed, so the console output is unchanged.
    ``coerce_bool`` is for the pure ``{key: bool}`` logs; stages that mix other
    value types into the same log (document consolidation records concept
    counts) must leave it off or those values would be flattened to booleans.
    """
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


class CheckpointLog:
    """Tracks per-item completion and flushes atomically every ``flush_every``.

    A crash loses at most ``flush_every`` marks, and because the write is
    atomic the log on disk is always parseable. Items are re-derived from the
    log on resume, so an unflushed mark costs a redo rather than corruption.
    """

    def __init__(self, path: Path, *, flush_every: int = 25):
        """Open a checkpoint log, loading any state already on disk.

        Args:
            path: JSON file holding ``{item_key: completed}``.
            flush_every: How many marks may accumulate before a flush is required. A
                crash loses at most this many marks, and because the write is atomic
                the file on disk is always parseable."""
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        self.data: Dict[str, bool] = load_bool_log(self.path)
        self._dirty = 0

    def is_done(self, key: str) -> bool:
        """Report whether one item is already complete.

        Args:
            key: Item key.

        Returns:
            ``True`` when the key is present and marked complete. A key recorded as
            ``False`` (started but not finished) counts as not done, so resume retries it."""
        return bool(self.data.get(key, False))

    def mark(self, key: str, completed: bool = True) -> None:
        """Record an item's completion state.

        Args:
            key: Item key.
            completed: ``True`` for a finished item. Passing ``False`` marks an item as
                started-but-incomplete, which :meth:`reset_incomplete` later clears.

        Returns:
            None. The write is deferred until :meth:`flush`."""
        self.data[key] = completed
        self._dirty += 1

    def needs_flush(self) -> bool:
        """Whether enough marks have accumulated to warrant a write.

        Returns:
            ``True`` once ``flush_every`` marks are pending."""
        return self._dirty >= self.flush_every

    def pending(self, keys: Iterable[str]) -> List[str]:
        """Keys that are not yet marked complete."""
        return [k for k in keys if not self.is_done(k)]

    def reset_incomplete(self, on_reset: Optional[Callable[[str], None]] = None) -> int:
        """Drop every non-complete entry so resume retries it.

        ``on_reset`` lets the caller clean the item's partial output while
        keeping its debug evidence, matching the extraction stages' behaviour.
        """
        removed = 0
        for key, completed in list(self.data.items()):
            if not completed:
                if on_reset is not None:
                    on_reset(key)
                del self.data[key]
                removed += 1
        return removed

    async def flush(self) -> None:
        """Atomically write pending marks without blocking the loop.

        Returns:
            None. Does nothing when no mark is pending, so calling it unconditionally
            at the end of a stage is cheap."""
        if self._dirty:
            await save_log_async(self.path, dict(self.data))
            self._dirty = 0

    def flush_sync(self) -> None:
        """Synchronous counterpart of :meth:`flush`.

        For use outside a coroutine, such as a ``finally`` block in synchronous code or
        an interpreter shutdown hook.

        Returns:
            None. Does nothing when no mark is pending."""
        if self._dirty:
            save_log_sync(self.path, dict(self.data))
            self._dirty = 0
