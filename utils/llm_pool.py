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
        llm_log: Optional :class:`~utils.llm_log.LLMCallLog`. When given,
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
        parsed_servers: List[str] = []
        parsed_concurrencies: List[int] = []

        if isinstance(servers, dict):
            for s_url, s_conc in servers.items():
                if isinstance(s_url, str) and s_url.strip():
                    parsed_servers.append(s_url.strip())
                    parsed_concurrencies.append(max(1, int(s_conc)))
        else:
            default_conc = int(concurrency_per_server) if isinstance(concurrency_per_server, (int, float)) and concurrency_per_server > 0 else 32
            for s in servers:
                if not isinstance(s, str) or not s.strip():
                    continue
                s_str = s.strip()
                # Check for inline concurrency URL:CONCURRENCY (e.g. http://host:8000/v1:64)
                m = re.search(r':(\d+)$', s_str)
                if m and not s_str.endswith('/v1'):
                    conc = int(m.group(1))
                    url = s_str[:m.start()]
                elif isinstance(concurrency_per_server, dict):
                    url = s_str
                    conc = int(concurrency_per_server.get(url, default_conc))
                else:
                    url = s_str
                    conc = default_conc
                parsed_servers.append(url)
                parsed_concurrencies.append(max(1, conc))

        if not parsed_servers:
            raise RuntimeError(f"{label}: no usable LLM servers configured.")

        self.servers = parsed_servers
        self.server_concurrencies = parsed_concurrencies
        self.concurrency_per_server = max(parsed_concurrencies)  # For legacy property access
        self.request_timeout = float(request_timeout)
        self.connect_timeout = connect_timeout
        self.max_connections_slack = max_connections_slack
        self.label = label
        self.failover = failover
        self.unhealthy_threshold = max(1, unhealthy_threshold)
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
        """Create one httpx client and one SDK client per server with per-server limits."""
        timeout = (httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
                   if self.connect_timeout else httpx.Timeout(self.request_timeout))
        for url, conc in zip(self.servers, self.server_concurrencies):
            limits = httpx.Limits(max_connections=conc + self.max_connections_slack)
            http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
            self._http.append(http_client)
            client = AsyncOpenAI(base_url=url, api_key="EMPTY",
                                 http_client=http_client, max_retries=0)
            self._clients.append(client)
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
        # Shared dynamic queue: workers pull the next item as soon as their server
        # is free, preventing idle GPUs and balancing unequal workloads.
        shared_queue: asyncio.Queue = asyncio.Queue()
        for index, item in enumerate(items):
            shared_queue.put_nowait((index, item))

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
            """Consume items dynamically as this server becomes ready."""
            nonlocal outstanding
            while True:
                if budget is not None:
                    budget.check(self.label)
                if not self._healthy[server_index]:
                    return
                try:
                    entry = await asyncio.wait_for(shared_queue.get(), timeout=IDLE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    if outstanding <= 0 and shared_queue.empty():
                        return
                    continue

                index, item = entry
                if not self._healthy[server_index]:
                    # Server retired; return item to shared_queue for other healthy servers
                    shared_queue.put_nowait((index, item))
                    shared_queue.task_done()
                    return

                await run_item(server_index, index, item)
                shared_queue.task_done()

        workers = [asyncio.create_task(worker(s))
                   for s in range(server_count)
                   for _ in range(self.server_concurrencies[s])]
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


def pool_for_stage(args: Any, settings: Any, stage: str) -> ServerPool:
    """Build the ServerPool a stage sends its requests through.

    Every stage configures its pool the same way: the resolved servers,
    concurrency and timeouts, its call log, and whether to fail over. Building it
    here keeps those choices in one place.

    Args:
        args: Parsed CLI namespace. Supplies the call-log path and ``--no_failover``.
        settings: The stage's resolved settings from ``resolve_settings``.
        stage: Stage key, used as the pool's label and as its log name.

    Returns:
        An unopened ServerPool. Enter it with ``async with``.
    """
    return ServerPool(
        settings.servers,
        concurrency_per_server=settings.concurrency_per_server,
        request_timeout=settings.profile.request_timeout,
        connect_timeout=settings.profile.connect_timeout,
        max_connections_slack=20,
        label=stage,
        failover=not getattr(args, "no_failover", False),
    )
