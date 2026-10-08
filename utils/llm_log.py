"""Append-only record of every LLM call a stage makes.

The debug log answers "what went wrong". This answers "what did we ask the model,
and what did it say", for every call, successful or not. Each stage writes its own
file at::

    <wiki_base_dir>/<domain>/log/<stage>.llm.jsonl

One JSON object per line, one line per HTTP request actually issued, so a retried
item produces one line per attempt. Lines are flushed as they happen, so a killed
or preempted run still leaves a readable trail.

``--llm_log`` controls how much of each call is kept:

* ``compact`` (default) -- each prompt message and the reply are cut to
  :data:`COMPACT_CHAR_LIMIT` characters, and the true lengths are recorded, so a
  cut is always visible.
* ``full`` -- every character of the prompt and of the reply.
* ``errors`` -- failed calls only. Successful traffic is not written.

Each line has an ``event`` field. ``"call"`` is the HTTP exchange itself.
``"rejection"`` is the stage deciding that a response was unusable: bad JSON, a
schema mismatch, or a ``finish_reason`` of ``length``. The distinction matters
because the most common failure is a *successful* HTTP response whose body the
stage cannot parse. ``ok`` is false for both transport failures and rejections,
which is what ``errors`` mode keeps.
"""

import contextvars
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_log import EventLog
from .layout import log_path, run_dir_for

# The three accepted --llm_log values, in the order shown in --help.
MODES = ("full", "compact", "errors")

# Character budget per side when --llm_log compact is requested.
COMPACT_CHAR_LIMIT = 2000

# ServerPool.map sets this around each handler invocation so a wrapped client can
# attribute a request to the item it belongs to. Stages never have to pass item
# identity down through their own call stack.
CURRENT_CALL: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "CURRENT_CALL", default={})


def truncate(text: Optional[str], limit: int = COMPACT_CHAR_LIMIT) -> Dict[str, Any]:
    """Optionally shorten a string, reporting that it was shortened.

    Args:
        text: The text to store, or ``None``.
        limit: Maximum characters to keep. Values at or below zero mean "keep
            everything".

    Returns:
        A dict with ``chars`` (the original length) and ``text``. When the text
        was cut, ``truncated`` is ``True`` and an explicit ``... [truncated N of
        M chars]`` marker is appended, so a reader can never mistake a compact
        record for a complete one.
    """
    if text is None:
        return {"chars": 0, "text": None, "truncated": False}
    total = len(text)
    if limit <= 0 or total <= limit:
        return {"chars": total, "text": text, "truncated": False}
    return {
        "chars": total,
        "text": text[:limit] + f"\n... [truncated {total - limit} of {total} chars]",
        "truncated": True,
    }


class LLMCallLog:
    """JSONL sink for one stage's LLM traffic.

    Instances are cheap and safe to share across the asyncio tasks of a single
    stage; writes are serialised by a lock and each line is flushed immediately.

    Attributes:
        path: Destination file. Parent directories are created on first write.
        mode: One of :data:`MODES`.
        stage: Stage label stored on every line.
    """

    def __init__(self, path: Path, *, mode: str = "compact", stage: str = ""):
        """Open a call log for one stage.

        Args:
            path: JSONL file to append to. Created lazily on the first record so
                a stage that makes no calls leaves no empty file behind.
            mode: ``"full"``, ``"compact"`` or ``"errors"``. An unrecognised
                value falls back to ``"compact"`` rather than losing data.
            stage: Stage name recorded on every line, e.g. ``"extract-concepts"``.
        """
        self.path = Path(path)
        self.mode = mode if mode in MODES else "compact"
        self.stage = stage
        self._events = EventLog(self.path, label=stage)
        self.calls = 0
        self.failures = 0
        self.rejections = 0

    @classmethod
    def from_args(cls, args: Any, stage: str) -> "LLMCallLog":
        """Build the conventional log path for a stage from its parsed CLI args.

        Args:
            args: Parsed namespace carrying ``wiki_base_dir``, ``domain`` and
                ``llm_log``.
            stage: Stage label, also used as the file name.

        Returns:
            An :class:`LLMCallLog` writing to
            ``<wiki_base_dir>/<domain>/log/<stage>.llm.jsonl``.
        """
        return cls(log_path(run_dir_for(args), stage, "llm"),
                   mode=getattr(args, "llm_log", "compact") or "compact",
                   stage=stage)

    @property
    def writes_successes(self) -> bool:
        """Whether successful calls are recorded at all.

        Returns:
            ``False`` only in ``errors`` mode.
        """
        return self.mode != "errors"

    def _limit(self) -> int:
        """Return the per-side character budget for the current mode.

        Returns:
            ``0`` (unlimited) in ``full`` mode, otherwise
            :data:`COMPACT_CHAR_LIMIT`.
        """
        return 0 if self.mode == "full" else COMPACT_CHAR_LIMIT

    def _write(self, event: str, fields: Dict[str, Any]) -> None:
        """Append one event through the shared :class:`EventLog`.

        Args:
            event: ``"call"`` or ``"rejection"``.
            fields: The event's payload. The stage name and timestamp are added
                by :class:`EventLog`, so they are not repeated here.

        Returns:
            None.
        """
        self._events.record(event, **fields)

    @staticmethod
    def _request_summary(kwargs: Dict[str, Any], limit: int) -> Dict[str, Any]:
        """Reduce a ``chat.completions.create`` call to loggable fields.

        Args:
            kwargs: Keyword arguments the handler passed to the SDK.
            limit: Per-message character budget; ``0`` keeps everything.

        Returns:
            A dict with ``model``, ``max_tokens``, ``temperature`` and
            ``messages``. Each message keeps its ``role`` and its (possibly
            truncated) ``content``, so the prompt is reconstructable.
        """
        messages: List[Dict[str, Any]] = []
        for message in kwargs.get("messages") or []:
            if isinstance(message, dict):
                role = message.get("role", "?")
                content = message.get("content")
            else:  # pragma: no cover - SDK param objects
                role = getattr(message, "role", "?")
                content = getattr(message, "content", None)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            messages.append({"role": role, **truncate(content, limit)})
        return {
            "model": kwargs.get("model"),
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
            "messages": messages,
        }

    @staticmethod
    def _response_text(completion: Any) -> Optional[str]:
        """Pull the generated text out of a chat completion object.

        Args:
            completion: The object returned by ``chat.completions.create``.

        Returns:
            The first choice's message content, or ``None`` when the response has
            no usable choice (which is itself worth logging).
        """
        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _finish_reason(completion: Any) -> Optional[str]:
        """Return why generation stopped, e.g. ``"stop"`` or ``"length"``.

        Args:
            completion: The object returned by ``chat.completions.create``.

        Returns:
            The first choice's ``finish_reason``, or ``None`` if unavailable.
            ``"length"`` means the response hit ``max_tokens`` and is truncated.
        """
        try:
            return completion.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            return None

    def _context(self) -> Dict[str, Any]:
        """Return the item identity ServerPool attached to the running call.

        Returns:
            A dict that may contain ``item``, ``index``, ``server_index`` and
            ``attempt``. Empty when a call happens outside ``ServerPool.map``.
        """
        return dict(CURRENT_CALL.get() or {})

    def success(self, *, server: str, kwargs: Dict[str, Any], completion: Any,
                latency_ms: float) -> None:
        """Record a completed LLM call.

        Args:
            server: Base URL that served the request.
            kwargs: The keyword arguments passed to ``create``.
            completion: The SDK response object.
            latency_ms: Wall-clock milliseconds the request took.

        Returns:
            None. Does nothing in ``errors`` mode.
        """
        self.calls += 1
        if not self.writes_successes:
            return
        limit = self._limit()
        text = self._response_text(completion)
        fields = {
            "ok": True,
            "server": server,
            "latency_ms": round(latency_ms, 2),
            "finish_reason": self._finish_reason(completion),
            "request": self._request_summary(kwargs, limit),
            "response": truncate(text, limit),
        }
        fields.update(self._context())
        self._write("call", fields)

    def failure(self, *, server: str, kwargs: Dict[str, Any], error: BaseException,
                latency_ms: float, classification: Optional[str] = None) -> None:
        """Record an LLM call that raised.

        Always written, in every mode: ``errors`` exists precisely to keep these.

        Args:
            server: Base URL the request was sent to.
            kwargs: The keyword arguments passed to ``create``.
            error: The exception raised.
            latency_ms: Wall-clock milliseconds before the failure.
            classification: Optional error class (``TRANSIENT``/``RECOVERABLE``/
                ``PERMANENT``) when the caller already knows it.

        Returns:
            None.
        """
        self.calls += 1
        self.failures += 1
        fields = {
            "ok": False,
            "server": server,
            "latency_ms": round(latency_ms, 2),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "classification": classification,
            "request": self._request_summary(kwargs, self._limit()),
            "response": None,
        }
        fields.update(self._context())
        self._write("call", fields)

    def rejection(self, *, server: str, error: BaseException,
                  classification: Optional[str] = None) -> None:
        """Record that a stage refused a response the transport delivered fine.

        This is the common failure in this pipeline: HTTP 200 with a body that is
        not parseable JSON, does not match the Pydantic schema, or was cut off at
        ``max_tokens``. It is written in *every* mode, including ``errors``, so
        that mode is actually useful.

        Args:
            server: Base URL that served the rejected response.
            error: The exception the stage raised when it rejected the response.
            classification: Optional error class (``TRANSIENT``/``RECOVERABLE``/
                ``PERMANENT``) when the caller has already classified it.

        Returns:
            None. The rejected body itself is not repeated here -- it was already
            written by :meth:`success` on the same attempt, in the same mode.
        """
        self.rejections += 1
        fields = {
            "ok": False,
            "server": server,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "classification": classification,
        }
        fields.update(self._context())
        self._write("rejection", fields)

    def close(self) -> None:
        """Flush and close the underlying file, if it was ever opened.

        Returns:
            None. Safe to call more than once and safe to call when nothing was
            written.
        """
        self._events.close()

    def __enter__(self) -> "LLMCallLog":
        """Return self so the log can be used in a ``with`` block.

        Returns:
            This log.
        """
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the log when leaving a ``with`` block.

        Args:
            *exc_info: Ignored exception context.

        Returns:
            None. Never suppresses an exception.
        """
        self.close()


def wrap_client(client: Any, server_url: str, log: Optional[LLMCallLog]) -> Any:
    """Make every ``chat.completions.create`` call on a client write to a log.

    The wrapper is installed once per client by :class:`ServerPool` when it
    builds its connections, which is what makes the log complete: a stage cannot
    forget to record a call, because recording happens below the stage.

    Args:
        client: An ``AsyncOpenAI`` instance.
        server_url: Base URL to attribute calls to.
        log: Destination log. When ``None`` the client is returned untouched, so
            disabling logging costs nothing.

    Returns:
        The same client object, with ``chat.completions.create`` replaced by a
        logging wrapper. The original call signature, return value and exception
        behaviour are preserved exactly.
    """
    if log is None:
        return client

    completions = client.chat.completions  # touch once: cached_property
    original = completions.create
    if getattr(original, "_llm_log_wrapped", False):
        return client

    async def logged_create(*args: Any, **kwargs: Any) -> Any:
        """Call the real ``create``, recording the outcome either way.

        Args:
            *args: Positional arguments for ``create`` (the SDK takes none in
                practice, but they are forwarded untouched).
            **kwargs: Keyword arguments for ``create``, including ``messages``.

        Returns:
            Whatever the underlying ``create`` returns.

        Raises:
            Whatever the underlying ``create`` raises, after the failure has been
            logged. Classification is deliberately left to the caller so retry
            policy stays in one place.
        """
        started = time.monotonic()
        try:
            completion = await original(*args, **kwargs)
        except BaseException as exc:
            log.failure(server=server_url, kwargs=kwargs, error=exc,
                        latency_ms=(time.monotonic() - started) * 1000.0)
            raise
        log.success(server=server_url, kwargs=kwargs, completion=completion,
                    latency_ms=(time.monotonic() - started) * 1000.0)
        return completion

    logged_create._llm_log_wrapped = True
    completions.create = logged_create
    return client
