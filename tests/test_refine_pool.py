"""End-to-end tests for refine_concepts.py on ServerPool.

Pass 2 needs no cluster-only dependencies, so these run the real script as a
subprocess against real HTTP: the merge logic, the Pass 1 backup, checkpointing,
resume, even spreading, and the wiring that feeds verification phase 1's audit
feedback into the Pass 2 prompt.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tests.fake_vllm import FakeVLLMCluster
from tests.helpers import run_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "refine_concepts.py"

DELTA = {
    "new_concepts": [
        {"title": "Brand New Concept",
         "description": "A concept Pass 1 missed entirely, with 12 ms latency."}
    ],
    "concept_updates": [
        {"title": "Existing Concept",
         "additional_aliases": ["EC"],
         "additional_description_sentences": ["It also reaches 94.2% accuracy."],
         "additional_entities": ["v2.1"]}
    ],
}


def run_refine(args, *, expect_rc=0, timeout=300):
    """Run refine_concepts.py as a subprocess and assert its exit code.

    Args:
        args: Command-line arguments after the script name.
        expect_rc: The exit code the run should produce.
        timeout: Seconds to wait before the run is abandoned.

    Returns:
        The completed process.
    """
    return run_script(SCRIPT, args, expect_rc=expect_rc, timeout=timeout)


class Pass2Tests(unittest.TestCase):
    def _build(self, root: Path, *, chunks=7, insufficient=True) -> Path:
        """Write a Pass 1 output tree, optionally with chunks flagged for refinement.

        Args:
            root: Temporary directory to build in.
            chunks: How many chunks to create.
            insufficient: Whether the chunks are flagged as needing refinement.

        Returns:
            The run directory."""
        base = root / "wiki" / "biology"
        (base / "cache").mkdir(parents=True, exist_ok=True)
        (base / "staging").mkdir(parents=True, exist_ok=True)

        cache = []
        for i in range(chunks):
            chunk_id = f"chunk_{i:03d}"
            cache.append({"doc_id": "canon_doc", "chunk_id": chunk_id,
                          "current_chunk": f"Raw text {i} about a mechanism.",
                          "context_before": "", "context_after": ""})
            d = base / "staging" / "canon_doc" / chunk_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "concepts.json").write_text(json.dumps({"concepts": [
                {"title": "Existing Concept", "aliases": [],
                 "description": "Original description.", "entities": [],
                 "outbound_relations": []}]}), encoding="utf-8")
            if insufficient:
                (d / "verification.json").write_text(json.dumps({
                    "coverage_score": 0.4, "is_sufficient": False,
                    "missing_concepts": ["Omitted Subsystem"],
                    "missing_facts": ["Omitted metric 77%"],
                    "critique": "Pass 1 missed a subsystem.",
                }), encoding="utf-8")
        (base / "cache" / "chunks_cache.json").write_text(json.dumps(cache),
                                                          encoding="utf-8")
        return base

    def _args(self, base, servers, **extra):
        """Build the CLI arguments for a Pass 2 run.

        Args:
            base: Run directory. Its parent becomes ``--wiki_base_dir``.
            servers: Fake server URLs.
            **extra: ``flag=value`` pairs appended as ``--flag value``.

        Returns:
            A list of CLI argument strings. The store_true flags are appended bare."""
        out = ["--domain", "biology", "--wiki_base_dir", str(base.parent),
               "--servers", *servers, "--stage_budget_minutes", "2"]
        for k, v in extra.items():
            # store_true flags are passed with an empty value.
            out.append(f"--{k}")
            if v != "":
                out.append(str(v))
        return out

    def test_merges_delta_backs_up_pass1_and_spreads_evenly(self):
        """The merge backs up Pass 1 first, and the requests are spread evenly across servers."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=7)
            with FakeVLLMCluster(3, responder=lambda p, i: DELTA) as cluster:
                out = run_refine(self._args(base, cluster.urls, max_attempts=2))
                cluster.assert_balanced(self, expected_total=7)
            self.assertIn("Requests served per server", out.stdout)

            d = base / "staging" / "canon_doc" / "chunk_000"
            merged = json.loads((d / "concepts.json").read_text())
            titles = [c["title"] for c in merged["concepts"]]
            self.assertIn("Brand New Concept", titles)
            self.assertIn("Existing Concept", titles)

            existing = next(c for c in merged["concepts"]
                            if c["title"] == "Existing Concept")
            self.assertIn("It also reaches 94.2% accuracy.", existing["description"])
            self.assertIn("Original description.", existing["description"])
            self.assertEqual(existing["aliases"], ["EC"])
            self.assertEqual(existing["entities"], ["v2.1"])

            # Pass 1 is preserved exactly once, unmodified.
            backup = json.loads((d / "concepts_pass1_backup.json").read_text())
            self.assertEqual(backup["concepts"][0]["description"], "Original description.")
            self.assertEqual([c["title"] for c in backup["concepts"]], ["Existing Concept"])
            self.assertTrue((d / "pass2_refinement.json").is_file())

            log = json.loads((base / "pass2_log.json").read_text())
            self.assertTrue(all(log.values()) and len(log) == 7)

    def test_audit_feedback_from_verification_reaches_the_prompt(self):
        """The missing concepts and facts from verification appear in the Pass 2 prompt."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=1)
            seen = []

            def responder(prompt, i):
                """Record the prompt and return the fixed Pass 2 delta used by the test.

                Args:
                    prompt: The prompt text sent by the stage.
                    i: Request index, unused.

                Returns:
                    The Pass 2 delta, a dict."""
                seen.append(prompt)
                return DELTA

            with FakeVLLMCluster(1, responder=responder) as cluster:
                run_refine(self._args(base, cluster.urls))

            self.assertEqual(len(seen), 1)
            prompt = seen[0]
            self.assertIn("AUDIT FEEDBACK (PREVIOUS VERIFICATION FOUND OMISSIONS)", prompt)
            self.assertIn("Omitted Subsystem", prompt)
            self.assertIn("Omitted metric 77%", prompt)
            self.assertIn("Pass 1 missed a subsystem.", prompt)
            self.assertIn("EXISTING EXTRACTED CONCEPTS (PASS 1)", prompt)

    def test_sufficient_chunks_send_no_audit_block(self):
        """A chunk that verification found sufficient gets no audit block in its prompt."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=1, insufficient=False)
            seen = []
            with FakeVLLMCluster(1, responder=lambda p, i: (seen.append(p), DELTA)[1]) as cluster:
                run_refine(self._args(base, cluster.urls))
            self.assertNotIn("AUDIT FEEDBACK", seen[0])

    def test_resume_skips_completed_chunks(self):
        """A second run skips chunks that already have a Pass 2 result."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=3)
            with FakeVLLMCluster(1, responder=lambda p, i: DELTA) as cluster:
                run_refine(self._args(base, cluster.urls))
                first = cluster.total_requests
                out = run_refine(self._args(base, cluster.urls, resume=""))
                self.assertEqual(cluster.total_requests, first)
            self.assertIn("All chunks are already refined in Pass 2!", out.stdout)

    def test_truncation_is_retried_here_unlike_pass1(self):
        """Pass 2 has always treated a token-limit truncation as retryable."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=3)
            with FakeVLLMCluster(3, responder=lambda p, i: ("cut off", "length")) as cluster:
                proc = run_refine(self._args(base, cluster.urls, max_attempts=2),
                                 expect_rc=1)
                # 3 chunks x 2 attempts, spread one chunk per server.
                self.assertEqual(cluster.total_requests, 6)
                self.assertEqual(cluster.counts, [2, 2, 2])
            self.assertIn("chunk(s) remain incomplete after Pass 2 retries",
                          proc.stdout + proc.stderr)

            d = base / "staging" / "canon_doc" / "chunk_000"
            self.assertFalse((d / "concepts.json").read_text().count("Brand New") > 0)
            debug = json.loads((d / "pass2_debug.json").read_text())
            self.assertEqual(len(debug["attempts"]), 2)
            self.assertEqual(debug["attempts"][0]["error_type"], "ValueError")
            self.assertIn("Response truncated", debug["attempts"][0]["error_message"])
            self.assertEqual(debug["attempts"][0]["raw_response"], "cut off")
            self.assertEqual(debug["final_classification"], "transient")
            self.assertIn("server", debug["attempts"][0])

    def test_transient_http_fault_recovers(self):
        """A transient HTTP fault is retried, and the chunk is refined."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=1)
            with FakeVLLMCluster(1, responder=lambda p, i: DELTA, fail_first=1) as cluster:
                run_refine(self._args(base, cluster.urls, max_attempts=3))
                self.assertEqual(cluster.total_requests, 2)
            self.assertIn("Brand New Concept",
                          json.loads((base / "staging" / "canon_doc" / "chunk_000"
                                      / "concepts.json").read_text())["concepts"][1]["title"])

    def test_missing_pass1_concepts_fails_before_any_request(self):
        """Pass 2 without Pass 1 concepts fails before making any request."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), chunks=2)
            (base / "staging" / "canon_doc" / "chunk_001" / "concepts.json").unlink()
            with FakeVLLMCluster(1, responder=lambda p, i: DELTA) as cluster:
                proc = run_refine(self._args(base, cluster.urls), expect_rc=1)
                self.assertEqual(cluster.total_requests, 0)
            self.assertIn("missing valid Pass 1 concepts", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
