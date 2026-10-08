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

from .debug_log import DebugLogger
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
    """Round-robin vLLM client pool with per-server concurrency caps."""

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
    ):
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

        self._http: List[httpx.AsyncClient] = []
        self._clients: List[AsyncOpenAI] = []
        self._consecutive_failures: List[int] = []
        self._healthy: List[bool] = []
        self._served: List[int] = []
        self._requeue_cursor = 0
        self._lock = asyncio.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def _build_clients(self) -> None:
        timeout = (httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
                   if self.connect_timeout else httpx.Timeout(self.request_timeout))
        limits = httpx.Limits(max_connections=self.concurrency_per_server + self.max_connections_slack)
        for url in self.servers:
            http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
            self._http.append(http_client)
            self._clients.append(AsyncOpenAI(base_url=url, api_key="EMPTY",
                                             http_client=http_client, max_retries=0))
        self._consecutive_failures = [0] * len(self.servers)
        self._healthy = [True] * len(self.servers)
        self._served = [0] * len(self.servers)

    async def __aenter__(self) -> "ServerPool":
        self._build_clients()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for client in self._http:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._http.clear()
        self._clients.clear()

    @property
    def healthy_servers(self) -> List[str]:
        return [s for s, ok in zip(self.servers, self._healthy) if ok]

    @property
    def per_server_counts(self) -> Dict[str, int]:
        return dict(zip(self.servers, self._served))

    def _mark_failure(self, index: int, classification: str) -> None:
        """Track connection-level failures and retire a server when it dies."""
        if classification != "transient" or not self.failover:
            return
        self._consecutive_failures[index] += 1
        if self._consecutive_failures[index] >= self.unhealthy_threshold and self._healthy[index]:
            self._healthy[index] = False

    def _next_healthy(self) -> Optional[int]:
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
        debug: Optional[DebugLogger] = None,
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
            nonlocal outstanding
            key = key_fn(item) if key_fn is not None else str(index)
            client = self._clients[server_index]
            url = self.servers[server_index]

            async def call() -> Any:
                self._served[server_index] += 1
                return await handler(client, item, server_index)

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
