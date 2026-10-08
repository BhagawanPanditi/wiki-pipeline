"""Tests for document consolidation when the corpus was not chunked.

``doc_concept_consolidation.py`` exists to merge duplicate concepts found in
*different chunks of the same document*. With ``--no-chunking`` there is exactly
one unit per document, so there is nothing to merge -- and the stage must not
spend LLM calls discovering that.

These tests point ``--servers`` at a port nothing is listening on. If the stage
tried to call the model the run would fail, so a clean exit is itself the
assertion that no request was made.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fake_vllm import FakeVLLMCluster

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "doc_concept_consolidation.py"
DEAD_SERVER = "http://127.0.0.1:9/v1"

LONG = ("A dense factual description with enough words to look like a real concept "
        "brief including measured throughput of 1240 requests per second at 61 ms.")


def concept(title, chunk):
    """Build one extracted concept in the shape Pass 1 writes.

    Args:
        title: Concept title.
        chunk: Chunk id used to form the ``source_chunk`` citation key.

    Returns:
        A concept dict with the fields consolidation and later stages read.
    """
    return {"title": title, "aliases": [], "description": LONG, "entities": ["e"],
            "source_chunks": [f"canon_doc::{chunk}"], "outbound_relations": []}


def run_stage(args, *, expect_rc=0, timeout=300):
    """Run document consolidation as a subprocess and assert the exit code.

    Args:
        args: CLI arguments after the script name.
        expect_rc: Expected return code.
        timeout: Seconds to wait.

    Returns:
        The completed ``subprocess.CompletedProcess``.

    Raises:
        AssertionError: On an unexpected return code, with stdout and stderr
            included so the failure is diagnosable.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / SCRIPT), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
        f"STDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")
    return proc


class NoChunkingPassThroughTests(unittest.TestCase):
    def _stage_docs(self, base: Path, chunking: bool, docs=("A", "B")):
        """Lay out a staging tree and a run_config for a finished Pass 1.

        Args:
            base: Wiki base directory; the run writes to ``base/biology/``.
            chunking: Value to record in ``run_config.json``.
            docs: Document letters to create, one canonical document each.

        Returns:
            The run directory, ``base/biology``.
        """
        run_dir = base / "biology"
        for letter in docs:
            chunk_dir = run_dir / "staging" / f"canon_{letter}" / "chunk_000"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / "concepts.json").write_text(json.dumps(
                {"concepts": [concept(f"{letter}lpha Concept", "chunk_000")]}),
                encoding="utf-8")
        (run_dir / "run_config.json").write_text(json.dumps(
            {"corpus": "x.jsonl", "chunking": chunking, "domain": "biology",
             "run_name": "biology"}), encoding="utf-8")
        return run_dir

    def _args(self, base, servers):
        """Build consolidation arguments for a temp wiki base.

        Args:
            base: Wiki base directory.
            servers: Server URLs to hand the stage.

        Returns:
            A list of CLI argument strings.
        """
        return ["--domain", "biology", "--wiki_base_dir", str(base), "--servers", *servers,
                "--max_attempts", "2", "--stage_budget_minutes", "2"]

    def test_unchunked_run_makes_no_llm_call_and_writes_output(self):
        """With chunking off, a dead server is never contacted and output is written."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_docs(base, chunking=False)

            out = run_stage(self._args(base, [DEAD_SERVER]))
            self.assertIn("Chunking was disabled", out.stdout)
            self.assertIn("without any LLM call", out.stdout)

            for letter in ("A", "B"):
                doc_dir = run_dir / "consolidated" / f"canon_{letter}"
                concepts = json.loads((doc_dir / "concepts.json").read_text())
                self.assertEqual(concepts["doc_id"], f"canon_{letter}")
                self.assertEqual(len(concepts["concepts"]), 1)
                self.assertEqual(concepts["concepts"][0]["title"], f"{letter}lpha Concept")
                # An empty merge decision is still written, so the shape matches
                # a chunked run and later stages can read it unconditionally.
                merges = json.loads((doc_dir / "merge_clusters.json").read_text())
                self.assertEqual(merges["merge_clusters"], [])

            self.assertIn("Concepts Before: 2", out.stdout)
            self.assertIn("Concepts After:  2", out.stdout)

            log = json.loads((run_dir / "doc_consolidation_log.json").read_text())
            self.assertTrue(log["canon_A"] and log["canon_B"])
            self.assertEqual(log["total_concepts_before"], 2)
            self.assertEqual(log["total_concepts_after"], 2)

    def test_unchunked_rerun_is_idempotent(self):
        """A second pass finds nothing pending and still exits cleanly."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            self._stage_docs(base, chunking=False)
            run_stage(self._args(base, [DEAD_SERVER]))
            out = run_stage(self._args(base, [DEAD_SERVER]))
            self.assertIn("Passing", out.stdout)
            self.assertIn("0 document(s) through", out.stdout)

    def test_chunked_run_still_calls_the_model(self):
        """Guard the other branch: with chunking on the LLM is used."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_docs(base, chunking=True)
            verdict = {"merge_clusters": []}
            with FakeVLLMCluster(2, responder=lambda p, i: verdict) as servers:
                run_stage(self._args(base, servers.urls))
                self.assertEqual(servers.total_requests, 2)
            self.assertNotIn("Chunking was disabled",
                             (run_dir / "doc_consolidation_log.json").read_text())

    def test_missing_run_config_defaults_to_chunking(self):
        """Output trees predating run_config.json keep the historical behaviour."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_docs(base, chunking=True)
            (run_dir / "run_config.json").unlink()

            verdict = {"merge_clusters": []}
            with FakeVLLMCluster(2, responder=lambda p, i: verdict) as servers:
                out = run_stage(self._args(base, servers.urls))
                # No config means "assume chunked", so the model is consulted.
                self.assertEqual(servers.total_requests, 2)
            self.assertNotIn("Chunking was disabled", out.stdout)


if __name__ == "__main__":
    unittest.main()
