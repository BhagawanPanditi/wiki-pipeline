"""Bounded retry with error classification.

Replaces the ``for attempt in range(10)`` + ad-hoc ``asyncio.sleep`` loops that
were copy-pasted through eight stages. Three properties matter here:

* permanent errors (400/401/404/422) fail on the first attempt instead of
  burning ten rounds of backoff on a request that can never succeed;
* backoff is short and capped (2s -> 20s transient, 0.5s -> 4s recoverable)
  with jitter, so a stage degrades gracefully without idling for minutes;
* every attempt consults the stage's wall-clock budget, so no stage can run
  indefinitely on waiting alone.

Each attempt is streamed to the debug sink untruncated before it is retried.
"""

import asyncio
import random
from typing import Any, Awaitable, Callable, Optional

from .errors import (PERMANENT, RECOVERABLE, TRANSIENT, PermanentFailure,
                     StageBudget, TransientFailureExhausted, classify_error,
                     describe_error)

TRANSIENT_BASE_DELAY = 2.0
TRANSIENT_MAX_DELAY = 20.0
RECOVERABLE_BASE_DELAY = 0.5
RECOVERABLE_MAX_DELAY = 4.0


def backoff_delay(classification: str, attempt: int, *, base: Optional[float] = None,
                  ceiling: Optional[float] = None) -> float:
    """Capped exponential backoff with +/-20% jitter to avoid herd retries."""
    if base is None:
        base = TRANSIENT_BASE_DELAY if classification == TRANSIENT else RECOVERABLE_BASE_DELAY
    if ceiling is None:
        ceiling = TRANSIENT_MAX_DELAY if classification == TRANSIENT else RECOVERABLE_MAX_DELAY
    raw = min(base * (2 ** max(0, attempt - 1)), ceiling)
    return raw * random.uniform(0.8, 1.2)


async def retry_async(
    call: Callable[[], Awaitable[Any]],
    *,
    attempts: int,
    label: str,
    budget: Optional[StageBudget] = None,
    debug: Optional[Any] = None,
    attempt_timeout: Optional[float] = None,
    server: Optional[str] = None,
    context: Optional[dict] = None,
    on_attempt_error: Optional[Callable[[int, BaseException, str], Awaitable[None]]] = None,
) -> Any:
    """Run ``call`` until it succeeds, is classified permanent, or runs out.

    Returns the call's result. Raises ``PermanentFailure`` immediately for
    non-retryable errors, ``TransientFailureExhausted`` when every attempt
    failed, and ``StageBudgetExceeded`` when the stage ran out of time.
    """
    attempts = max(1, int(attempts))
    last_exc: Optional[BaseException] = None
    last_classification = TRANSIENT
    extra = context or {}

    for attempt in range(1, attempts + 1):
        if budget is not None:
            budget.check(label)
        try:
            if attempt_timeout:
                return await asyncio.wait_for(call(), timeout=attempt_timeout)
            return await call()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - classification decides the response
            last_exc = exc
            classification = classify_error(exc)
            last_classification = classification

            if debug is not None:
                debug.error(
                    label, exc,
                    attempt=attempt, attempts=attempts,
                    classification=classification, server=server, **extra,
                )
            if on_attempt_error is not None:
                await on_attempt_error(attempt, exc, classification)

            if classification == PERMANENT:
                raise PermanentFailure(
                    f"{label} failed permanently on attempt {attempt}: "
                    f"{describe_error(exc)}", cause=exc,
                ) from exc

            if attempt >= attempts:
                break

            delay = backoff_delay(classification, attempt)
            if budget is not None:
                delay = budget.cap_sleep(delay)
                if budget.expired():
                    budget.check(label)
            if delay > 0:
                await asyncio.sleep(delay)

    raise TransientFailureExhausted(
        f"{label} failed after {attempts} attempt(s); last error "
        f"({last_classification}): {describe_error(last_exc) if last_exc else 'unknown'}",
        attempts=attempts, cause=last_exc,
    )
