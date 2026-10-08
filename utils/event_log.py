"""Streaming, untruncated JSONL event log.

One class serves every log a stage writes: decision records ("stub absorbed into
X"), failure evidence (full prompt, raw response, traceback) and any other
event. Each event is written the moment it happens rather than buffered until
the end of a multi-hour run, and nothing is truncated, so a malformed 8k-token
payload can be replayed offline.

Records are newline-delimited JSON, which keeps a partially written last line
from invalidating the rest of the file and lets ``tail -f`` follow a live run.
"""

import asyncio
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


class EventLog:
    """Append-only JSONL sink with immediate, untruncated writes."""

    def __init__(self, path: Path, *, enabled: bool = True, fsync_every: int = 20,
                 label: str = "debug"):
        """Prepare a debug sink; the file is opened on first write.

        Args:
            path: JSONL file to append to.
            enabled: When ``False`` every write becomes a no-op, so a caller can disable
                debug output without branching at each call site.
            fsync_every: How many records may be written before forcing an ``fsync``.
            label: Stage name recorded on each line."""
        self.path = Path(path)
        self.enabled = enabled
        self.fsync_every = max(1, fsync_every)
        self.label = label
        self._lock = threading.Lock()
        self._handle = None
        self._since_fsync = 0
        self.records_written = 0
        self._warned = False
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── writing ────────────────────────────────────────────────────────────
    def record(self, event: str, **fields: Any) -> None:
        """Write one event immediately. Never truncates any field."""
        if not self.enabled:
            return
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": self.label,
            "event": event,
        }
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            try:
                if self._handle is None:
                    self._handle = open(self.path, "a", encoding="utf-8")
                self._handle.write(line + "\n")
                self._handle.flush()
                self._since_fsync += 1
                self.records_written += 1
                if self._since_fsync >= self.fsync_every:
                    os.fsync(self._handle.fileno())
                    self._since_fsync = 0
            except OSError as exc:
                # A lost log line must never kill a stage that is otherwise
                # making progress, but it must not vanish silently either.
                if not self._warned:
                    self._warned = True
                    print(f"WARNING: could not write to {self.path}: {exc}. "
                          f"Further write errors to this log are suppressed.")

    async def arecord(self, event: str, **fields: Any) -> None:
        """Thread-offloaded ``record`` so large payloads do not block the loop."""
        await asyncio.to_thread(self.record, event, **fields)

    def error(self, context: str, exc: BaseException, *, attempt: Optional[int] = None,
              attempts: Optional[int] = None, classification: Optional[str] = None,
              raw_response: Optional[str] = None, prompt: Optional[str] = None,
              server: Optional[str] = None, **fields: Any) -> None:
        """Record a failure with full context: traceback, prompt and raw output."""
        self.record(
            "error",
            context=context,
            attempt=attempt,
            attempts=attempts,
            classification=classification,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            server=server,
            raw_response=raw_response,
            prompt=prompt,
            **fields,
        )

    async def aerror(self, context: str, exc: BaseException, **kwargs: Any) -> None:
        """Awaitable form of :meth:`error`, offloaded to a thread.

        Args:
            context: Free-form description of where the failure happened.
            exc: The exception to record.
            **kwargs: Extra fields, forwarded to :meth:`error`.

        Returns:
            None. The blocking write and optional ``fsync`` happen off the event loop so
            a failing stage cannot stall the requests still in flight."""
        await asyncio.to_thread(self.error, context, exc, **kwargs)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self) -> None:
        """Flush, fsync and close the file if it was opened.

        Returns:
            None. Safe to call repeatedly and safe when nothing was ever written. An
            ``OSError`` during the final flush is ignored so teardown cannot mask the
            error that ended the stage."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
                except OSError:
                    pass
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "EventLog":
        """Return the logger for use in a ``with`` block.

        Returns:
            This logger."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the logger when leaving a ``with`` block.

        Args:
            *exc_info: Ignored exception context.

        Returns:
            None. Never suppresses an exception."""
        self.close()


def iter_jsonl(path: Path):
    """Yield records from a JSONL file, skipping any torn final line."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
