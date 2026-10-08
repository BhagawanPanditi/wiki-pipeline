"""Error classification and wall-clock budgeting for pipeline stages.

The pipeline talks to vLLM over HTTP, and its failures fall into three classes
that need very different responses. Conflating them is what made the old retry
loops simultaneously slow and fragile: a 400 Bad Request was retried ten times
with exponential backoff even though it can never succeed, while a genuine
network timeout deserved more patience than it got.

TRANSIENT    network / server faults      -> retry with capped backoff
RECOVERABLE  malformed or truncated model
             output                       -> retry quickly (resampling helps)
PERMANENT    request / config errors      -> fail immediately, dump full debug

Unknown exceptions are treated as RECOVERABLE rather than PERMANENT. That is
deliberate: it preserves the previous behaviour where any parse or schema
failure was retried, while still failing fast on the errors that provably
cannot succeed.
"""

import asyncio
import json
import time
from typing import Optional

try:  # httpx is a hard runtime dependency, but keep unit tests importable.
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    ValidationError = None


TRANSIENT = "transient"
RECOVERABLE = "recoverable"
PERMANENT = "permanent"


class StageBudgetExceeded(RuntimeError):
    """The stage exhausted its wall-clock budget and must stop retrying."""


class PermanentFailure(RuntimeError):
    """An item failed for a reason that retrying cannot fix."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.cause = cause


class TransientFailureExhausted(RuntimeError):
    """An item kept failing transiently and ran out of attempts."""

    def __init__(self, message: str, *, attempts: int, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


class AllServersUnhealthy(RuntimeError):
    """Every configured vLLM server stopped accepting connections."""


# Our own control-flow exceptions must never be retried.
_CONTROL_EXCEPTIONS = (StageBudgetExceeded, PermanentFailure, TransientFailureExhausted,
                       AllServersUnhealthy)


def classify_error(exc: BaseException) -> str:
    """Return TRANSIENT, RECOVERABLE or PERMANENT for an exception."""
    if isinstance(exc, _CONTROL_EXCEPTIONS):
        return PERMANENT

    # ── asyncio / stdlib network faults ────────────────────────────────────
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return TRANSIENT
    if isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
        # Socket-level faults (ECONNRESET, EPIPE, ETIMEDOUT, ...).
        return TRANSIENT

    # ── httpx transport faults ─────────────────────────────────────────────
    if httpx is not None:
        if isinstance(exc, httpx.TimeoutException):
            return TRANSIENT
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                            httpx.RemoteProtocolError, httpx.NetworkError)):
            return TRANSIENT
        if isinstance(exc, httpx.HTTPStatusError):
            return _classify_status(exc.response.status_code)

    # ── openai SDK faults ──────────────────────────────────────────────────
    if openai is not None:
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return TRANSIENT
        if isinstance(exc, openai.RateLimitError):
            return TRANSIENT
        if isinstance(exc, openai.InternalServerError):
            return TRANSIENT
        if isinstance(exc, openai.APIStatusError):
            return _classify_status(getattr(exc, "status_code", 500))

    # ── malformed / truncated model output ─────────────────────────────────
    if isinstance(exc, json.JSONDecodeError):
        return RECOVERABLE
    if ValidationError is not None and isinstance(exc, ValidationError):
        return RECOVERABLE
    if isinstance(exc, (ValueError, KeyError, TypeError, AttributeError, IndexError,
                        StopIteration, UnicodeDecodeError)):
        # Includes the "Response truncated: Hit max_tokens limit" ValueErrors
        # raised by the extraction stages.
        return RECOVERABLE

    return RECOVERABLE


def _classify_status(status_code: Optional[int]) -> str:
    """Map an HTTP status code onto a retry class."""
    if status_code is None:
        return TRANSIENT
    if status_code == 429 or status_code >= 500:
        return TRANSIENT
    if 400 <= status_code < 500:
        # 400/401/403/404/422 are deterministic: the request itself is wrong
        # (bad payload, max_tokens over context length, unknown model, auth).
        return PERMANENT
    return TRANSIENT


def describe_error(exc: BaseException) -> str:
    """Full, untruncated one-line description used in logs and error messages."""
    return f"{type(exc).__name__}: {exc}"


class StageBudget:
    """Wall-clock ceiling for one stage.

    A stage must survive slow servers without idling forever on backoff, so
    every retry path checks this budget. On expiry the stage raises
    StageBudgetExceeded, leaves its checkpoint intact, and is resumable.
    """

    def __init__(self, minutes: Optional[float]):
        self.minutes = minutes
        self.started_at = time.monotonic()
        self.deadline: Optional[float] = None
        if minutes is not None and minutes > 0:
            self.deadline = self.started_at + (minutes * 60.0)

    @property
    def enabled(self) -> bool:
        return self.deadline is not None

    def remaining(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def check(self, context: str = "stage") -> None:
        """Raise StageBudgetExceeded once the budget is gone."""
        if self.expired():
            raise StageBudgetExceeded(
                f"{context} exhausted its {self.minutes:g}-minute budget after "
                f"{self.elapsed() / 60.0:.1f} min. Checkpoints are preserved; "
                f"rerun with --resume after addressing the cause, or raise "
                f"--stage_budget_minutes."
            )

    def cap_sleep(self, delay: float) -> float:
        """Clamp a backoff sleep so it cannot overshoot the deadline."""
        remaining = self.remaining()
        if remaining is None:
            return delay
        return max(0.0, min(delay, remaining))
