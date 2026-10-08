"""Evenly spread LLM work across a pool of vLLM servers.

The old pattern built ``len(servers) * concurrency`` workers that all pulled
from one shared queue. Workers were pinned to a server, but *dispatch* was a
race, so the tail of a run could pile onto whichever server happened to have
free workers - five remaining items could all land on server 1 while servers
2-5 idled.

This pool partitions the work up front: item ``i`` goes to server ``i % N``.
The spread is exact and deterministic (each server receives within one item of
the others), each server's in-flight count is capped by its own worker count,
and progress plus per-item results stream out as they complete.

If a server stops accepting connections its remaining queue is handed to the
healthy servers instead of failing, and the run only aborts when every server
is down.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

from .event_log import EventLog
from .llm_log import CURRENT_CALL, LLMCallLog, wrap_client
from .errors import (AllServersUnhealthy, PermanentFailure, StageBudget,
                     TransientFailureExhausted, describe_error)
from .retry import retry_async

IDLE_POLL_SECONDS = 0.25


@dataclass
class PoolResult:
    """Outcome of a pooled run."""

    results: List[Any] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    per_server: Dict[str, int] = field(default_factory=dict)
    attempted: int = 0
    completed: int = 0

    @property
    def failed_count(self) -> int:
        """Number of items that did not complete.

        Returns:
            The length of :attr:`failures`. Provided so callers can test for a clean
            run without touching the failure records themselves."""
        return len(self.failures)

    def raise_if_failed(self, label: str, *, examples: int = 10) -> None:
        """Fail the stage loudly, naming the items that did not complete."""
        if not self.failures:
            return
        sample = [f.get("key", f.get("index")) for f in self.failures[:examples]]
        first = self.failures[0].get("error", "unknown error")
        raise RuntimeError(
            f"{label} left {len(self.failures)} item(s) incomplete after retries. "
            f"Examples: {sample}. First error: {first}"
        )


class ServerPool:
    """Round-robin vLLM client pool with per-server concurrency caps.

    Work is partitioned deterministically (item ``i`` goes to server ``i % N``)
    and each server has its own queue and concurrency cap, so a slow tail cannot
    pile onto whichever worker happens to be free.

    Args:
        servers: vLLM OpenAI-compatible base URLs.
        concurrency_per_server: In-flight requests allowed per server.
        request_timeout: Per-request read timeout in seconds.
        connect_timeout: Optional separate connection timeout in seconds.
        max_connections_slack: Extra httpx connections above the concurrency cap,
            so a retry can start before the previous socket is released.
        label: Stage name, used in log messages and retry labels.
        failover: Whether to move work off a server that stops responding.
        unhealthy_threshold: Consecutive failures before a server is marked
            unhealthy.
        llm_log: Optional :class:`~pipeline_lib.llm_log.LLMCallLog`. When given,
            every request issued through this pool is appended to it, and the
            log is closed when the pool closes.
    """

    def __init__(
        self,
        servers: Sequence[str],
        *,
        concurrency_per_server: int,
        request_timeout: float,
        connect_timeout: Optional[float] = None,
        max_connections_slack: int = 10,
        label: str = "stage",
        failover: bool = True,
        unhealthy_threshold: int = 3,
        llm_log: Optional[LLMCallLog] = None,
    ):
        """Configure a pool; no connection is opened yet.

        Args:
            servers: vLLM OpenAI-compatible base URLs. Blank entries are dropped.
            concurrency_per_server: In-flight requests allowed per server.
            request_timeout: Per-request read timeout in seconds.
            connect_timeout: Optional separate connection timeout in seconds.
            max_connections_slack: Extra httpx connections above the concurrency cap.
            label: Stage name used in log messages and retry labels.
            failover: Whether to move work off a server that stops responding.
            unhealthy_threshold: Consecutive transient failures before a server is
                retired for the rest of this map.
            llm_log: Optional call log; every request is appended to it when given.

        Raises:
            RuntimeError: If no usable server URL was supplied.
            ValueError: If ``concurrency_per_server`` is below 1."""
        cleaned = [s for s in servers if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise RuntimeError(f"{label}: no usable LLM servers configured.")
        if concurrency_per_server < 1:
            raise ValueError(f"{label}: concurrency_per_server must be at least 1.")

        self.servers = cleaned
        self.concurrency_per_server = int(concurrency_per_server)
        self.request_timeout = float(request_timeout)
        self.connect_timeout = connect_timeout
        self.max_connections_slack = max_connections_slack
        self.label = label
        self.failover = failover
        self.unhealthy_threshold = max(1, unhealthy_threshold)
        # When supplied, every request these clients issue is appended to this
        # log. Wrapping happens in _build_clients, below the stage, so a stage
        # cannot forget to record a call.
        self.llm_log = llm_log

        self._http: List[httpx.AsyncClient] = []
        self._clients: List[AsyncOpenAI] = []
        self._consecutive_failures: List[int] = []
        self._healthy: List[bool] = []
        self._served: List[int] = []
        self._requeue_cursor = 0
        self._lock = asyncio.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def _build_clients(self) -> None:
        """Create one httpx client and one SDK client per server.

        Called on entering the async context, not in ``__init__``, because building an
        ``httpx.AsyncClient`` binds it to the running event loop. Also resets the
        per-server health and request counters, so a pool reused for a second ``map``
        starts from a clean slate.

        Returns:
            None. Populates ``_http``, ``_clients``, ``_healthy``, ``_served`` and
            ``_consecutive_failures``."""
        timeout = (httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
                   if self.connect_timeout else httpx.Timeout(self.request_timeout))
        limits = httpx.Limits(max_connections=self.concurrency_per_server + self.max_connections_slack)
        for url in self.servers:
            http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
            self._http.append(http_client)
            client = AsyncOpenAI(base_url=url, api_key="EMPTY",
                                 http_client=http_client, max_retries=0)
            self._clients.append(wrap_client(client, url, self.llm_log))
        self._consecutive_failures = [0] * len(self.servers)
        self._healthy = [True] * len(self.servers)
        self._served = [0] * len(self.servers)

    async def __aenter__(self) -> "ServerPool":
        """Build the clients and return the pool for use in ``async with``.

        Returns:
            This pool, ready to :meth:`map`."""
        self._build_clients()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close every connection when leaving an ``async with`` block.

        Args:
            *exc_info: Ignored exception context.

        Returns:
            None. Never suppresses an in-flight exception."""
        await self.aclose()

    async def aclose(self) -> None:
        """Release all httpx connections and flush the call log.

        Safe to call more than once. A failure to close one client is swallowed so
        teardown cannot mask the error that ended the stage.

        Returns:
            None."""
        for client in self._http:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._http.clear()
        self._clients.clear()
        if self.llm_log is not None:
            # Flushing here guarantees the trail survives a stage that dies
            # after the pool closes. Re-opening is cheap if a later pool in the
            # same process reuses the log.
            self.llm_log.close()

    @property
    def healthy_servers(self) -> List[str]:
        """Server URLs still considered usable.

        Returns:
            The subset of :attr:`servers` not retired by repeated transient failures."""
        return [s for s, ok in zip(self.servers, self._healthy) if ok]

    @property
    def per_server_counts(self) -> Dict[str, int]:
        """Requests actually issued to each server.

        Returns:
            ``{server_url: request_count}`` for every configured server, including
            those that served nothing. Counts attempts, not items, so a retried item
            is counted on each server that tried it."""
        return dict(zip(self.servers, self._served))

    def _mark_failure(self, index: int, classification: str) -> None:
        """Track connection-level failures and retire a server when it dies."""
        if classification != "transient" or not self.failover:
            return
        self._consecutive_failures[index] += 1
        if self._consecutive_failures[index] >= self.unhealthy_threshold and self._healthy[index]:
            self._healthy[index] = False

    def _next_healthy(self) -> Optional[int]:
        """Pick the next server to receive requeued work.

        Rotates through healthy servers so failover does not dump everything onto the
        first one in the list.

        Returns:
            The index of a healthy server, or ``None`` when every server has been
            retired -- which the caller turns into an ``AllServersUnhealthy`` failure."""
        healthy = [i for i, ok in enumerate(self._healthy) if ok]
        if not healthy:
            return None
        self._requeue_cursor = (self._requeue_cursor + 1) % len(healthy)
        return healthy[self._requeue_cursor]

    # ── dispatch ───────────────────────────────────────────────────────────
    async def map(
        self,
        items: Sequence[Any],
        handler: Callable[[AsyncOpenAI, Any, int], Awaitable[Any]],
        *,
        attempts: int = 4,
        desc: Optional[str] = None,
        key_fn: Optional[Callable[[Any], str]] = None,
        budget: Optional[StageBudget] = None,
        debug: Optional[EventLog] = None,
        attempt_timeout: Optional[float] = None,
        on_result: Optional[Callable[[Any, Any], Awaitable[None]]] = None,
        show_progress: bool = True,
    ) -> PoolResult:
        """Process ``items`` spread evenly across servers.

        ``handler(client, item, server_index)`` performs one logical unit of
        work and returns its result, or raises. Retries, classification,
        budget enforcement, progress streaming and failure collection are all
        handled here so stages never reimplement them.

        ``on_result(item, result)`` is awaited as each item completes, which is
        how stages checkpoint and stream partial output to disk.
        """
        result = PoolResult(per_server={s: 0 for s in self.servers})
        total = len(items)
        if total == 0:
            return result

        server_count = len(self.servers)
        queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(server_count)]

        # Deterministic partition: item i -> server i % N. Every server receives
        # within one item of the others regardless of how few items remain.
        for index, item in enumerate(items):
            queues[index % server_count].put_nowait((index, item))

        outstanding = total
        pbar = tqdm(total=total, desc=desc or self.label, disable=not show_progress,
                    smoothing=0.1, mininterval=0.5)

        async def run_item(server_index: int, index: int, item: Any) -> None:
            """Drive one item to completion, with retries and accounting.

            Applies ``retry_async`` around :func:`call`, then records the outcome: on
            success the value is appended and ``on_result`` is awaited; on exhaustion the
            failure is recorded against the item's index so the caller can recover the item
            from its own list.

            Args:
                server_index: Index of the server owning this item's queue.
                index: Position of the item in the caller's sequence.
                item: The work unit itself.

            Returns:
                None. Mutates ``result`` and the progress bar; never raises for an ordinary
                item failure.

            Raises:
                AllServersUnhealthy: If every server was retired while this item was in
                flight, which aborts the whole map rather than losing the remaining work."""
            nonlocal outstanding
            key = key_fn(item) if key_fn is not None else str(index)
            client = self._clients[server_index]
            url = self.servers[server_index]

            attempts_made = {"n": 0}

            async def call() -> Any:
                """Issue one attempt at this item and publish its identity.

                Increments the server's request counter, then sets the ``CURRENT_CALL``
                contextvar so a wrapped client can attribute the request to this item, server
                and attempt number in the call log. A response the handler rejects is recorded
                separately, because the HTTP exchange itself may well have succeeded.

                Returns:
                    Whatever the handler returns.

                Raises:
                    Whatever the handler raises, after logging a rejection if an ``llm_log``
                    is attached. ``retry_async`` decides whether that is retryable."""
                self._served[server_index] += 1
                attempts_made["n"] += 1
                # Publish the item identity so the wrapped client can attribute
                # this request without the handler threading it through.
                token = CURRENT_CALL.set({
                    "item": key, "index": index, "server": url,
                    "server_index": server_index, "attempt": attempts_made["n"],
                })
                try:
                    return await handler(client, item, server_index)
                except Exception as exc:
                    # The transport may have succeeded while the stage rejected
                    # the body (bad JSON, schema mismatch, truncation). Record
                    # that separately, or --llm_log errors would stay silent
                    # about the most common failure in this pipeline.
                    if self.llm_log is not None:
                        self.llm_log.rejection(server=url, error=exc)
                    raise
                finally:
                    CURRENT_CALL.reset(token)

            try:
                value = await retry_async(
                    call,
                    attempts=attempts,
                    label=f"{self.label}[{key}]",
                    budget=budget,
                    debug=debug,
                    attempt_timeout=attempt_timeout,
                    server=url,
                    context={"key": key, "index": index},
                )
                async with self._lock:
                    self._consecutive_failures[server_index] = 0
                result.results.append(value)
                result.completed += 1
                result.per_server[url] = result.per_server.get(url, 0) + 1
                if on_result is not None:
                    await on_result(item, value)
            except (PermanentFailure, TransientFailureExhausted) as exc:
                classification = "permanent" if isinstance(exc, PermanentFailure) else "transient"
                self._mark_failure(server_index, classification)
                cause = getattr(exc, "cause", None)
                result.failures.append({
                    "key": key, "index": index, "server": url,
                    "classification": classification,
                    "error": describe_error(cause) if cause else str(exc),
                })
            except AllServersUnhealthy as exc:
                result.failures.append({"key": key, "index": index, "server": url,
                                        "classification": "permanent", "error": str(exc)})
                raise
            finally:
                outstanding -= 1
                pbar.update(1)

        async def worker(server_index: int) -> None:
            """Consume one server's queue until the map is finished.

            Each server has its own worker and its own queue, which is what keeps the
            spread deterministic: item ``i`` is only ever offered to server ``i % N``. A
            worker exits when no items remain outstanding and its queue is empty. If its
            server has been retired, remaining items are handed to another healthy server
            instead of being dropped.

            Args:
                server_index: Index of the server this worker serves.

            Returns:
                None. Runs until the queue is drained.

            Raises:
                StageBudgetExceeded: Propagated from the budget check, so an expired
                    wall-clock ceiling stops the map instead of idling on backoff."""
            nonlocal outstanding
            queue = queues[server_index]
            while True:
                if budget is not None:
                    budget.check(self.label)
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=IDLE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    if outstanding <= 0 and queue.empty():
                        return
                    if not self._healthy[server_index]:
                        # Nothing left for a retired server to do.
                        if queue.empty():
                            return
                    continue

                index, item = entry

                if not self._healthy[server_index]:
                    target = self._next_healthy()
                    if target is None:
                        outstanding -= 1
                        pbar.update(1)
                        result.failures.append({
                            "key": key_fn(item) if key_fn else str(index),
                            "index": index, "server": self.servers[server_index],
                            "classification": "permanent",
                            "error": "all configured servers are unhealthy",
                        })
                        queue.task_done()
                        continue
                    queues[target].put_nowait((index, item))
                    queue.task_done()
                    continue

                await run_item(server_index, index, item)
                queue.task_done()

        workers = [asyncio.create_task(worker(s))
                   for s in range(server_count)
                   for _ in range(self.concurrency_per_server)]
        try:
            await asyncio.gather(*workers)
        finally:
            pbar.close()
            for task in workers:
                if not task.done():
                    task.cancel()

        if not any(self._healthy):
            raise AllServersUnhealthy(
                f"{self.label}: every configured vLLM server became unreachable "
                f"({', '.join(self.servers)}). Check the serving jobs, then rerun "
                f"with --resume."
            )
        return result
