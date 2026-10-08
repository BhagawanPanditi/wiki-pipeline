"""Integration tests for concept_extraction_pass_1.py's ServerPool refactor.

Pass 1 needs `transformers` for its tokenizer, which is a cluster-only
dependency, so the tokenizer is stubbed. Everything under test - the LLM call,
retry classification, truncation fail-fast, even spreading, per-chunk
persistence and debug artifacts - is the real code path over real HTTP.
"""
import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if "transformers" not in sys.modules:
    stub = types.ModuleType("transformers")

    class _Tok:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def encode(self, text, add_special_tokens=False):
            return list(range(len(str(text).split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    stub.AutoTokenizer = _Tok
    sys.modules["transformers"] = stub

import concept_extraction_pass_1 as pass1  # noqa: E402
from pipeline_lib.llm_pool import ServerPool
from tests.fake_vllm import FakeVLLMCluster, FakeVLLMServer  # noqa: E402

GOOD = {"concepts": [{"title": "Real Concept", "description": "A description.",
                      "entities": ["e"], "outbound_relations": []}]}


def make_pipeline(root: Path, servers):
    parser = pass1.build_parser(stage=pass1.STAGE, description="t",
                                include_data_dir=True, include_resume=True)
    args = parser.parse_args(["--domain", "biology", "--wiki_base_dir", str(root),
                              "--servers", *servers, "--max_attempts", "2",
                              "--stage_budget_minutes", "2"])
    pipe = pass1.WikiExtractorPipeline(args)
    pipe.settings = pass1.resolve_settings(args, pass1.STAGE)
    pipe.debug = pass1.DebugLogger(pipe.base_wiki_dir / "pass1_extraction_debug.jsonl",
                                   label=pass1.STAGE)
    pipe.budget = pass1.StageBudget(pipe.settings.budget_minutes)
    pipe.attempt_errors = {}
    pipe.flush_count = 0
    return pipe


def chunks(n):
    return [{"doc_id": "canon_doc", "chunk_id": f"chunk_{i:03d}",
             "current_chunk": "Some raw text about a mechanism.",
             "context_before": "", "context_after": ""} for i in range(n)]


class Pass1PoolTests(unittest.TestCase):
    def _run(self, pipe, items, servers):
        pool = ServerPool(servers,
                          concurrency_per_server=pipe.settings.concurrency_per_server,
                          request_timeout=pipe.settings.profile.request_timeout,
                          connect_timeout=pipe.settings.profile.connect_timeout,
                          label=pass1.STAGE)

        async def go():
            try:
                async with pool:
                    return await pool.map(items, pipe.extract_chunk,
                                          attempts=pipe.settings.attempts,
                                          key_fn=lambda c: f"{c['doc_id']}::{c['chunk_id']}",
                                          budget=pipe.budget, debug=pipe.debug,
                                          attempt_timeout=pipe.settings.profile.attempt_timeout,
                                          on_result=pipe.persist_extraction)
            finally:
                pipe.debug.close()
        return asyncio.run(go())

    def test_chunks_persist_and_spread_evenly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with FakeVLLMCluster(3, responder=lambda p, i: GOOD) as cluster:
                pipe = make_pipeline(root, cluster.urls)
                items = chunks(7)
                result = self._run(pipe, items, cluster.urls)
                cluster.assert_balanced(self, expected_total=7)
                self.assertEqual(result.failed_count, 0)
                for c in items:
                    f = pipe.staging_dir / c["doc_id"] / c["chunk_id"] / "concepts.json"
                    self.assertTrue(f.is_file(), f)
                    self.assertEqual(json.loads(f.read_text())["concepts"][0]["title"],
                                     "Real Concept")
                    self.assertTrue(pipe.extraction_log[f"{c['doc_id']}::{c['chunk_id']}"])

    def test_token_limit_truncation_fails_fast_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = FakeVLLMServer(
                responder=lambda p, i: ("cut off mid sent", "length")).start()
            try:
                pipe = make_pipeline(root, [server.url])
                items = chunks(2)
                result = self._run(pipe, items, [server.url])
                # Truncation is permanent: exactly one request per chunk, no retries.
                self.assertEqual(server.requests, 2)
                self.assertEqual(result.failed_count, 2)
                for c in items:
                    d = pipe.staging_dir / c["doc_id"] / c["chunk_id"] / "debug.json"
                    self.assertFalse(d.is_file())  # written by the caller, not the handler
                    self.assertFalse((d.parent / "concepts.json").exists())
                    attempts = pipe.attempt_errors[f"{c['doc_id']}::{c['chunk_id']}"]
                    self.assertEqual(len(attempts), 1)
                    self.assertEqual(attempts[0]["error_type"], "TokenLimitExceeded")
                    self.assertTrue(attempts[0]["truncated"])
                    self.assertEqual(attempts[0]["raw_response"], "cut off mid sent")
            finally:
                server.stop()

    def test_unparseable_response_is_retried_then_dumps_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with FakeVLLMCluster(1, responder=lambda p, i: "{not json") as cluster:
                pipe = make_pipeline(root, cluster.urls)
                items = chunks(1)
                result = self._run(pipe, items, cluster.urls)
                self.assertEqual(result.failed_count, 1)
                # max_attempts=2 -> two real requests for the one chunk.
                self.assertEqual(cluster.total_requests, 2)
                failure = result.failures[0]
                asyncio.run(pipe.write_failure_debug(items[0], failure))
                d = json.loads((pipe.staging_dir / "canon_doc" / "chunk_000"
                                / "debug.json").read_text())
                self.assertEqual(d["total_attempts"], 2)
                self.assertEqual(len(d["attempts"]), 2)
                self.assertEqual(d["attempts"][0]["error_type"], "ValidationError")
                # Full untruncated evidence is retained.
                self.assertEqual(d["attempts"][0]["raw_response"], "{not json")
                self.assertEqual(d["final_classification"], "transient")
                self.assertEqual(d["chunk_metadata"]["current_chunk_chars"],
                                 len("Some raw text about a mechanism."))

    def test_transient_http_fault_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with FakeVLLMCluster(1, responder=lambda p, i: GOOD, fail_first=1) as cluster:
                pipe = make_pipeline(root, cluster.urls)
                items = chunks(1)
                result = self._run(pipe, items, cluster.urls)
                self.assertEqual(result.failed_count, 0)
                self.assertEqual(cluster.total_requests, 2)
                self.assertTrue((pipe.staging_dir / "canon_doc" / "chunk_000"
                                 / "concepts.json").is_file())


if __name__ == "__main__":
    unittest.main()
