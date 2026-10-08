"""Unit tests for the shared utils infrastructure.

These exercise the parts that replaced copy-pasted logic: error classification,
bounded retry, deterministic server spreading with failover, checkpointing,
JSON repair parity with the nine original implementations, and the Pass 1 ->
Pass 2 improvement arithmetic. No network access is required - the pool is
built normally but handlers never use the client.
"""

import argparse
import asyncio
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.atomic_io import atomic_write_json, read_json
from utils.checkpoint import CheckpointLog, load_bool_log
from utils.cli import STAGE_PROFILES
from utils.errors import (PERMANENT, RECOVERABLE, TRANSIENT,
                                 PermanentFailure, StageBudget,
                                 StageBudgetExceeded, TransientFailureExhausted,
                                 classify_error)
from utils.json_repair import clean_json_string
from utils.llm_pool import AllServersUnhealthy, ServerPool
from utils.reporting import (PASS1_RESULT, PASS2_RESULT,
                                    build_improvement, collect_metrics,
                                    format_improvement, format_report,
                                    write_summary)
from utils.retry import backoff_delay, retry_async

SERVERS = ["http://s0/v1", "http://s1/v1", "http://s2/v1", "http://s3/v1", "http://s4/v1"]


class ErrorClassificationTests(unittest.TestCase):
    def test_network_faults_are_transient(self):
        """Network-level errors are classified as transient."""
        import httpx
        for exc in (asyncio.TimeoutError(), TimeoutError(), ConnectionError(),
                    httpx.ConnectError("down"), httpx.ReadTimeout("slow"),
                    httpx.RemoteProtocolError("bad")):
            self.assertEqual(classify_error(exc), TRANSIENT, msg=type(exc).__name__)

    def test_http_status_mapping(self):
        """HTTP status codes map to the expected error classes."""
        import httpx
        import openai
        request = httpx.Request("POST", "http://s0/v1/chat/completions")
        for status, expected in ((400, PERMANENT), (401, PERMANENT), (404, PERMANENT),
                                 (422, PERMANENT), (429, TRANSIENT), (500, TRANSIENT),
                                 (503, TRANSIENT)):
            exc = openai.APIStatusError("err", response=httpx.Response(status, request=request),
                                        body=None)
            self.assertEqual(classify_error(exc), expected, msg=f"status {status}")

    def test_malformed_output_is_recoverable_not_permanent(self):
        """A bad JSON payload may succeed on resampling, so it must still retry."""
        from pydantic import BaseModel, ValidationError

        class Schema(BaseModel):
            brief: str

        with self.assertRaises(ValidationError) as ctx:
            Schema.model_validate_json('{"wrong": 1}')
        self.assertEqual(classify_error(ctx.exception), RECOVERABLE)
        self.assertEqual(classify_error(json.JSONDecodeError("x", "y", 0)), RECOVERABLE)
        self.assertEqual(classify_error(ValueError("Response truncated: Hit max_tokens")),
                         RECOVERABLE)

    def test_control_exceptions_never_retry(self):
        """Budget and failure-control exceptions are never retried."""
        for exc in (StageBudgetExceeded("out of time"), PermanentFailure("bad request"),
                    TransientFailureExhausted("gave up", attempts=4),
                    AllServersUnhealthy("all down")):
            self.assertEqual(classify_error(exc), PERMANENT, msg=type(exc).__name__)

    def test_stage_budget_expires_and_caps_sleep(self):
        """An expired stage budget stops the stage and caps each backoff sleep."""
        budget = StageBudget(minutes=0.0005)  # 30 ms
        self.assertFalse(budget.expired())
        self.assertAlmostEqual(budget.cap_sleep(999.0), budget.remaining(), delta=0.01)
        import time
        time.sleep(0.06)  # budget is 30 ms
        self.assertTrue(budget.expired())
        with self.assertRaises(StageBudgetExceeded):
            budget.check("test-stage")

    def test_unbounded_budget_never_expires(self):
        """A budget of None never expires."""
        budget = StageBudget(minutes=None)
        self.assertFalse(budget.enabled)
        self.assertIsNone(budget.remaining())
        self.assertEqual(budget.cap_sleep(12.0), 12.0)
        budget.check("no-budget")  # must not raise


class RetryTests(unittest.TestCase):
    def test_backoff_is_capped(self):
        """Backoff delays grow but never exceed the configured cap."""
        self.assertLessEqual(backoff_delay(TRANSIENT, 1), 24.0)
        self.assertLessEqual(backoff_delay(TRANSIENT, 20), 24.0)
        self.assertLessEqual(backoff_delay(RECOVERABLE, 20), 4.8)
        self.assertLess(backoff_delay(RECOVERABLE, 1), backoff_delay(TRANSIENT, 1) + 1)

    def test_succeeds_after_transient_failures(self):
        """A call that fails transiently and then succeeds returns its value."""
        calls = {"n": 0}

        async def flaky():
            """Fail with a connection error on the first two calls, then return "ok".

            Returns:
                The string ``"ok"``, returned from the third call onward."""
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("server reset")
            return "ok"

        got = asyncio.run(retry_async(flaky, attempts=4, label="t"))
        self.assertEqual(got, "ok")
        self.assertEqual(calls["n"], 3)

    def test_permanent_error_fails_on_first_attempt(self):
        """The core fix: a 400 must not burn ten rounds of backoff."""
        import httpx
        import openai
        calls = {"n": 0}

        async def bad_request():
            """Raise a 400 Bad Request from the SDK, a permanent error that must never be retried.

            Raises:
                openai.BadRequestError: Always. retry_async must surface this as
                PermanentFailure after a single attempt."""
            calls["n"] += 1
            request = httpx.Request("POST", "http://s0/v1")
            raise openai.BadRequestError("max_tokens too large",
                                         response=httpx.Response(400, request=request),
                                         body=None)

        with self.assertRaises(PermanentFailure):
            asyncio.run(retry_async(bad_request, attempts=10, label="t"))
        self.assertEqual(calls["n"], 1)

    def test_exhausted_transient_raises_with_attempt_count(self):
        """When every attempt fails transiently, the error reports how many attempts were made."""
        async def always_down():
            """Always fail with a transient error, so every retry is used up.

            Raises:
                httpx.ConnectError: Always."""
            raise ConnectionError("refused")

        with self.assertRaises(TransientFailureExhausted) as ctx:
            asyncio.run(retry_async(always_down, attempts=3, label="t"))
        self.assertEqual(ctx.exception.attempts, 3)

    def test_budget_stops_retrying(self):
        """A spent budget stops retries before the attempts are used up."""
        import time

        async def always_down():
            """Always fail with a transient error, so every retry is used up.

            Raises:
                httpx.ConnectError: Always."""
            raise ConnectionError("refused")

        budget = StageBudget(minutes=0.0008)
        time.sleep(0.06)
        with self.assertRaises(StageBudgetExceeded):
            asyncio.run(retry_async(always_down, attempts=50, label="t", budget=budget))


class JsonRepairParityTests(unittest.TestCase):
    """The nine original copies were not identical; parity must be preserved."""

    def test_default_matches_standard_variant(self):
        """The default JSON repair matches the standard variant."""
        raw = 'prefix ```json {"a": 1} ``` suffix'
        self.assertEqual(clean_json_string(raw), '{"a": 1}')
        # Standard variant does NOT fall back to a bare object.
        self.assertEqual(clean_json_string('bare {"a": 1} text'), 'bare {"a": 1} text')

    def test_pass2_variant_accepts_bare_object(self):
        """The Pass 2 JSON repair accepts a bare object."""
        raw = 'here is the result {"a": 1} thanks'
        self.assertEqual(clean_json_string(raw, allow_bare_object=True), '{"a": 1}')

    def test_final_consolidate_variant_accepts_arrays(self):
        """The final-consolidation JSON repair accepts a top-level array."""
        raw = '```json [1, 2, 3] ```'
        self.assertEqual(clean_json_string(raw), '```json [1, 2, 3] ```')
        self.assertEqual(clean_json_string(raw, allow_array=True), '[1, 2, 3]')

    def test_empty_and_none_safe(self):
        """The JSON repair returns a safe value for empty and None input."""
        self.assertEqual(clean_json_string(""), "")


class ServerPoolBalanceTests(unittest.TestCase):
    """Proves requests are spread evenly instead of racing onto one server."""

    def _run(self, items, *, concurrency=4, servers=None, handler=None, **kwargs):
        """Run one ServerPool map synchronously and return its result.

        Args:
            items: The items to process.
            concurrency: In-flight requests per server.
            servers: Fake server URLs; defaults to the module's SERVERS.
            handler: The handler to run per item.
            **kwargs: Forwarded to :meth:`ServerPool.map`.

        Returns:
            The ``PoolResult``."""
        servers = servers or SERVERS

        async def go():
            """Drive one ServerPool.map to completion and return its result."""
            async with ServerPool(servers, concurrency_per_server=concurrency,
                                  request_timeout=5.0, label="test") as pool:
                return await pool.map(items, handler, desc=None, show_progress=False,
                                      key_fn=lambda x: str(x), **kwargs)

        return asyncio.run(go())

    def test_five_items_over_five_servers_get_one_each(self):
        """The exact scenario from the bug report."""
        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            return server_index

        result = self._run(list(range(5)), concurrency=50, handler=handler)
        counts = [0] * 5
        for server_index in result.results:
            counts[server_index] += 1
        self.assertEqual(counts, [1, 1, 1, 1, 1])
        self.assertEqual(result.failed_count, 0)

    def test_uneven_item_count_still_balances(self):
        """An uneven number of items still spreads within one of each server."""
        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            return server_index

        result = self._run(list(range(7)), concurrency=2, servers=SERVERS[:3],
                           handler=handler)
        counts = [0] * 3
        for server_index in result.results:
            counts[server_index] += 1
        self.assertEqual(sorted(counts), [2, 2, 3])

    def test_single_server_tail_item_uses_that_server(self):
        """A final lone item is dispatched to its designated server."""
        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            return server_index

        result = self._run([99], concurrency=50, handler=handler)
        self.assertEqual(result.results, [0])

    def test_concurrency_cap_is_respected_per_server(self):
        """No server should ever exceed its configured in-flight limit."""
        peak = [0] * len(SERVERS)
        live = [0] * len(SERVERS)
        lock = asyncio.Lock()

        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            async with lock:
                live[server_index] += 1
                peak[server_index] = max(peak[server_index], live[server_index])
            await asyncio.sleep(0.005)
            async with lock:
                live[server_index] -= 1
            return server_index

        self._run(list(range(60)), concurrency=3, handler=handler)
        self.assertTrue(all(p <= 3 for p in peak), msg=f"peaks {peak}")
        self.assertTrue(all(p >= 1 for p in peak), msg=f"every server used: {peak}")

    def test_transient_failure_is_retried_then_succeeds(self):
        """A transient failure is retried, and the item then succeeds."""
        attempts = {}

        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            attempts[item] = attempts.get(item, 0) + 1
            if attempts[item] < 2:
                raise ConnectionError("reset")
            return item

        result = self._run(list(range(5)), concurrency=2, handler=handler, attempts=3)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(sorted(result.results), list(range(5)))

    def test_permanent_failure_recorded_without_retry(self):
        """A permanent failure is recorded once, without retrying."""
        import httpx
        import openai
        attempts = {}

        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            attempts[item] = attempts.get(item, 0) + 1
            request = httpx.Request("POST", "http://s0/v1")
            raise openai.BadRequestError("bad", response=httpx.Response(400, request=request),
                                         body=None)

        result = self._run(list(range(3)), concurrency=2, handler=handler, attempts=5)
        self.assertEqual(result.failed_count, 3)
        self.assertTrue(all(v == 1 for v in attempts.values()), msg=f"attempts {attempts}")
        self.assertTrue(all(f["classification"] == "permanent" for f in result.failures))

    def test_raise_if_failed_surfaces_examples(self):
        """A failed map raises an error that names example failures."""
        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            raise ValueError("Response truncated")

        result = self._run(list(range(2)), concurrency=1, handler=handler, attempts=1)
        with self.assertRaises(RuntimeError) as ctx:
            result.raise_if_failed("test-stage")
        self.assertIn("2 item(s) incomplete", str(ctx.exception))

    def test_failover_moves_work_off_a_dead_server(self):
        """A retired server's queue is served by healthy ones, not dropped."""
        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            if server_index == 0:
                raise ConnectionError("server 0 is down")
            return server_index

        async def go():
            """Drive one ServerPool.map to completion and return its result."""
            async with ServerPool(SERVERS[:3], concurrency_per_server=1, request_timeout=5.0,
                                  label="test", unhealthy_threshold=1) as pool:
                return await pool.map(list(range(9)), handler, desc=None,
                                      show_progress=False, key_fn=str, attempts=1)

        result = asyncio.run(go())
        # Items originally assigned to server 0 were requeued to servers 1/2.
        self.assertTrue(all(idx != 0 for idx in result.results), msg=result.results)
        self.assertGreater(len(result.results), 0)

    def test_on_result_streams_each_completion(self):
        """Each completion is reported to the callback as it finishes."""
        streamed = []

        async def handler(client, item, server_index):
            """Test handler for the enclosing test. Follows the ServerPool contract:
            (client, item, server_index) -> result. It raises to signal a failure the pool
            should classify and retry or record."""
            return item * 10

        async def on_result(item, value):
            """Pool callback, called once per completed item in completion order.

            Args:
                item: The item that completed.
                value: The handler's result for that item."""
            streamed.append((item, value))

        self._run(list(range(6)), concurrency=2, handler=handler, on_result=on_result)
        self.assertEqual(sorted(streamed), [(i, i * 10) for i in range(6)])


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_resume(self):
        """A checkpoint written to disk reloads with the same marks, so resume picks up where it stopped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log = CheckpointLog(path, flush_every=2)
            log.mark("a::1", True)
            log.mark("a::2", False)
            self.assertTrue(log.needs_flush())
            asyncio.run(log.flush())

            reloaded = CheckpointLog(path)
            self.assertTrue(reloaded.is_done("a::1"))
            self.assertFalse(reloaded.is_done("a::2"))
            self.assertEqual(reloaded.pending(["a::1", "a::2", "a::3"]), ["a::2", "a::3"])

    def test_reset_incomplete_keeps_completed(self):
        """Resetting incomplete items keeps the completed ones."""
        with tempfile.TemporaryDirectory() as tmp:
            log = CheckpointLog(Path(tmp) / "log.json")
            log.mark("keep", True)
            log.mark("retry", False)
            seen = []
            removed = log.reset_incomplete(on_reset=seen.append)
            self.assertEqual(removed, 1)
            self.assertEqual(seen, ["retry"])
            self.assertTrue(log.is_done("keep"))
            self.assertNotIn("retry", log.data)

    def test_corrupt_checkpoint_degrades_to_empty(self):
        """A corrupt checkpoint loads as empty instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text('{"half": ', encoding="utf-8")
            self.assertEqual(load_bool_log(path), {})
            self.assertIsNone(read_json(path, default=None))

    def test_atomic_write_leaves_no_temp_files(self):
        """An atomic write leaves no temporary files behind."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.json"
            atomic_write_json(path, {"concepts": [{"title": "x"}]})
            self.assertEqual(json.loads(path.read_text())["concepts"][0]["title"], "x")
            self.assertEqual(list(path.parent.glob(".*tmp")), [])


class VerificationReportTests(unittest.TestCase):
    def _write_chunk(self, staging, doc, chunk, filename, score, sufficient,
                     concepts=(), facts=()):
        """Write one chunk's verification result file for a metrics test.

        Args:
            staging: Staging directory.
            doc: Document id.
            chunk: Chunk id.
            filename: Result file name, which decides the phase.
            score: Coverage score to store.
            sufficient: Whether the chunk is marked sufficient.
            concepts: Missed concepts to store.
            facts: Missed facts to store.

        Returns:
            None."""
        chunk_dir = staging / doc / chunk
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / filename).write_text(json.dumps({
            "coverage_score": score, "is_sufficient": sufficient,
            "missing_concepts": list(concepts), "missing_facts": list(facts),
        }), encoding="utf-8")

    def test_metrics_aggregate_and_improvement_is_computed(self):
        """Per-chunk scores aggregate correctly, and the improvement between phases is computed."""
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            # Pass 1: poor coverage.
            self._write_chunk(staging, "d1", "c1", PASS1_RESULT, 0.5, False,
                              concepts=["a", "b"], facts=["f1"])
            self._write_chunk(staging, "d1", "c2", PASS1_RESULT, 0.9, True, concepts=["c"])
            # Pass 2: same chunks, repaired.
            self._write_chunk(staging, "d1", "c1", PASS2_RESULT, 0.95, True, concepts=["b"])
            self._write_chunk(staging, "d1", "c2", PASS2_RESULT, 1.0, True)

            before = collect_metrics(staging, label="PASS 1", result_filename=PASS1_RESULT)
            after = collect_metrics(staging, label="PASS 2", result_filename=PASS2_RESULT)

            self.assertEqual(before.total_chunks, 2)
            self.assertAlmostEqual(before.average_coverage, 0.7, places=3)
            self.assertEqual(before.sufficient_count, 1)
            self.assertAlmostEqual(before.pass_rate, 50.0, places=1)
            self.assertEqual(before.missed_concepts, 3)
            self.assertEqual(before.missed_facts, 1)

            self.assertAlmostEqual(after.average_coverage, 0.975, places=3)
            self.assertEqual(after.sufficient_count, 2)
            self.assertEqual(after.missed_concepts, 1)

            improvement = build_improvement(before, after)
            self.assertAlmostEqual(improvement["coverage_delta"], 0.275, places=3)
            self.assertAlmostEqual(improvement["pass_rate_delta_points"], 50.0, places=1)
            self.assertEqual(improvement["missed_concepts_delta"], -2)
            self.assertLess(improvement["missed_facts_change_pct"], 0)

            report = format_report("robotics", before)
            self.assertIn("VERIFICATION SUMMARY REPORT [ROBOTICS]", report)
            self.assertIn("Total Chunks Verified : 2", report)
            self.assertIn("Average Coverage Score: 0.700 / 1.000", report)
            self.assertIn("Pass Rate (Sufficient): 50.0% (1/2)", report)
            self.assertIn("Total Missed Concepts : 3", report)
            self.assertIn("Total Missed Facts    : 1", report)

            block = format_improvement("robotics", improvement)
            self.assertIn("VERIFICATION IMPROVEMENT [ROBOTICS]", block)
            self.assertIn("PASS 1 -> PASS 2", block)

            summary_path = write_summary(Path(tmp), "robotics", before, after)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["domain"], "robotics")
            self.assertIn("pass1", saved)
            self.assertIn("pass2", saved)
            self.assertIn("improvement", saved)
            self.assertAlmostEqual(saved["improvement"]["coverage_after"], 0.975, places=3)

    def test_missing_pass2_still_writes_pass1_summary(self):
        """With no phase 2 results, the phase 1 summary is still written."""
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            before = collect_metrics(staging, label="PASS 1", result_filename=PASS1_RESULT)
            after = collect_metrics(staging, label="PASS 2", result_filename=PASS2_RESULT)
            self.assertEqual(before.total_chunks, 0)
            self.assertEqual(after.total_chunks, 0)

            summary_path = write_summary(Path(tmp), "biology", before)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("pass1", saved)
            self.assertNotIn("pass2", saved)
            self.assertNotIn("improvement", saved)

            # An empty run must still render, not crash.
            self.assertIn("No verification_pass2.json results found",
                          format_report("biology", after))


if __name__ == "__main__":
    unittest.main()


class StageProfileParityTests(unittest.TestCase):
    """max_tokens must match what each stage historically requested.

    Lowering a stage's max_tokens silently truncates its output and changes
    graph quality, so the historical value is pinned here. Concurrency is a
    recommendation and may be tuned; max_tokens is behaviour.
    """

    HISTORICAL_MAX_TOKENS = {
        "extract-concepts": 8192,      # extract_concepts --max_tokens
        "refine-concepts": 8192,      # refine_concepts --max_tokens
        "consolidate-documents": 8192,
        "resolve-stubs": 8192,
        "verify-pass1": 4096,          # hardcoded in the original verifier
        "verify-pass2": 4096,
        "consolidate-global": 4096,
        "final-macro-scan": 4096,      # hardcoded macro pass
        "final-deep-verify": 1000,     # hardcoded deep-verify pass
        "fold-thin-concepts": 4096,
        "write-briefs": 256,
        "audit-islands": 400,           # hardcoded
        "add-reverse-links": 256,      # hardcoded
    }

    def test_every_profile_matches_its_historical_max_tokens(self):
        """Every stage profile keeps the max_tokens the stage always used."""
        self.assertEqual(set(STAGE_PROFILES), set(self.HISTORICAL_MAX_TOKENS),
                         "a stage profile was added or renamed without pinning "
                         "its historical max_tokens")
        for stage, expected in self.HISTORICAL_MAX_TOKENS.items():
            self.assertEqual(STAGE_PROFILES[stage].max_tokens, expected,
                             f"{stage}: max_tokens drifted from the historical value")

    def test_every_profile_is_bounded_so_no_stage_can_idle_forever(self):
        """Every stage profile has finite attempts, a positive budget and a bounded attempt timeout."""
        for stage, profile in STAGE_PROFILES.items():
            self.assertGreater(profile.attempts, 0, stage)
            self.assertLessEqual(profile.attempts, 8, f"{stage}: retry ladder too long")
            self.assertGreater(profile.budget_minutes, 0, stage)
            self.assertIsNotNone(profile.request_timeout, stage)
            self.assertGreater(profile.attempt_timeout, profile.request_timeout,
                               f"{stage}: attempt timeout must exceed request timeout")

    def test_concurrency_scales_inversely_with_max_tokens(self):
        """Big generations need fewer concurrent requests: KV is the constraint."""
        heavy = [p for p in STAGE_PROFILES.values() if p.max_tokens >= 4096]
        light = [p for p in STAGE_PROFILES.values() if p.max_tokens <= 1024]
        self.assertLess(max(p.concurrency_per_server for p in heavy),
                        min(p.concurrency_per_server for p in light))


class AttemptHistoryTests(unittest.TestCase):
    """The per-item attempt history that stages write next to failed items."""

    def test_attempts_are_numbered_and_flag_truncation(self):
        """Each attempt is numbered in order, and a length stop is flagged as truncated."""
        from utils.attempts import AttemptHistory

        history = AttemptHistory()
        history.record("doc::chunk", finish_reason="length", raw_text="cut off",
                       error_type="TokenLimitExceeded", error_message="too long")
        history.record("doc::chunk", finish_reason="stop", raw_text="{bad", error_type="JSONDecodeError",
                       error_message="Expecting value", server="http://s0/v1")
        attempts = history.get("doc::chunk")
        self.assertEqual([a["attempt"] for a in attempts], [1, 2])
        self.assertTrue(attempts[0]["truncated"])
        self.assertFalse(attempts[1]["truncated"])
        self.assertEqual(attempts[1]["server"], "http://s0/v1")
        self.assertEqual(attempts[0]["raw_response"], "cut off")

    def test_unknown_items_have_no_history_and_reads_are_copies(self):
        """An item with no attempts yields an empty list, and a returned list is a copy."""
        from utils.attempts import AttemptHistory

        history = AttemptHistory()
        self.assertEqual(history.get("nothing"), [])
        history.record("x", error_type="E", error_message="m")
        history.get("x").clear()
        self.assertEqual(len(history.get("x")), 1)


class StageOverrideTests(unittest.TestCase):
    """config.json's stage_concurrency: both accepted forms, and what a bad entry does."""

    def _settings(self, overrides, stage="extract-concepts"):
        """Resolve one stage's settings as if config.json held the given overrides.

        Args:
            overrides: The ``stage_concurrency`` mapping config.json would contain.
            stage: Stage key to resolve.

        Returns:
            The resolved StageSettings.
        """
        from utils import cli

        cli._warned_stage_names.clear()
        args = argparse.Namespace(servers=["http://s0/v1"])
        with mock.patch.object(cli, "load_stage_overrides", return_value=overrides):
            return cli.resolve_settings(args, stage)

    def test_a_plain_number_sets_the_concurrency(self):
        """A number in stage_concurrency is that stage's concurrency_per_server."""
        self.assertEqual(self._settings({"extract-concepts": 3}).concurrency_per_server, 3)

    def test_an_object_can_set_several_values(self):
        """The object form sets concurrency and attempts together."""
        settings = self._settings({"add-reverse-links": {"concurrency_per_server": 9, "attempts": 6}},
                                  stage="add-reverse-links")
        self.assertEqual((settings.concurrency_per_server, settings.attempts), (9, 6))

    def test_a_malformed_entry_is_an_error_not_a_silent_default(self):
        """A string where a number belongs stops the stage with a clear message."""
        with self.assertRaises(ValueError) as caught:
            self._settings({"extract-concepts": "eight"})
        self.assertIn("must be a number or an object", str(caught.exception))

    def test_a_retired_stage_name_is_reported_and_not_applied(self):
        """An override named after a retired stage is reported and has no effect."""
        from utils.cli import STAGE_PROFILES

        buffer = io.StringIO()
        with redirect_stderr(buffer):
            settings = self._settings({"island-audit": 5}, stage="audit-islands")
        self.assertIn("island-audit", buffer.getvalue())
        self.assertIn("ignored", buffer.getvalue())
        self.assertEqual(settings.concurrency_per_server,
                         STAGE_PROFILES["audit-islands"].concurrency_per_server)

    def test_the_llm_call_log_defaults_to_compact(self):
        """Without --llm_log the call log keeps compact records."""
        from utils.cli import build_parser

        args = build_parser(stage="extract-concepts", description="t").parse_args([])
        self.assertEqual(args.llm_log, "compact")


class ConfigJsonTests(unittest.TestCase):
    """config.json's servers and model_name reach every stage; a broken file is an error."""

    def _with_config(self, text):
        """Point config.CONFIG_FILE at a temporary file holding ``text``.

        Args:
            text: Contents to write to the temporary config.json.

        Returns:
            A context manager under which ``config`` reads the temporary file.
        """
        import config

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "config.json"
        path.write_text(text, encoding="utf-8")
        return mock.patch.object(config, "CONFIG_FILE", path)

    def test_servers_and_model_from_config_json_reach_the_parser_and_settings(self):
        """A stage gets the servers and model listed in config.json unless the command line overrides them."""
        from utils import cli

        text = json.dumps({"servers": ["http://a:1/v1", "http://b:2/v1"], "model_name": "m/x"})
        with self._with_config(text):
            args = cli.build_parser(stage="extract-concepts", description="t").parse_args([])
            settings = cli.resolve_settings(argparse.Namespace(), "extract-concepts")
        self.assertEqual(args.servers, ["http://a:1/v1", "http://b:2/v1"])
        self.assertEqual(args.model_name, "m/x")
        self.assertEqual(settings.servers, ["http://a:1/v1", "http://b:2/v1"])
        self.assertEqual(settings.model_name, "m/x")

    def test_a_malformed_config_json_is_an_error_not_a_silent_default(self):
        """A config.json that is not valid JSON raises instead of using the cluster defaults."""
        import config

        with self._with_config("{servers: ["):
            with self.assertRaises(ValueError) as caught:
                config.load_global_config()
        self.assertIn("not readable JSON", str(caught.exception))

    def test_a_stage_concurrency_that_is_not_an_object_is_an_error(self):
        """A list where stage_concurrency belongs is reported, not silently ignored."""
        import config

        with self._with_config(json.dumps({"stage_concurrency": [8]})):
            with self.assertRaises(ValueError) as caught:
                config.load_stage_overrides()
        self.assertIn("must be an object keyed by stage name", str(caught.exception))

    def test_a_top_level_concurrency_key_is_flagged_once(self):
        """The unused top-level concurrency_per_server is reported, once per process."""
        import config

        config._warned_config_keys.clear()
        buffer = io.StringIO()
        with self._with_config(json.dumps({"concurrency_per_server": 50})), redirect_stderr(buffer):
            config.load_global_config()
            config.load_global_config()
        self.assertEqual(buffer.getvalue().count("'concurrency_per_server' is no longer used"), 1)
