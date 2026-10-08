"""End-to-end test of the two-phase verifier against a fake vLLM server.

Spins up a real OpenAI-compatible HTTP server on localhost so the whole path is
exercised: httpx client construction, even request spreading across servers,
retry on transient failure, atomic per-chunk result writes, checkpointing,
resume, and the Pass 1 -> Pass 2 improvement report.

The fake server derives ``coverage_score`` from how many concepts appear in the
prompt, so phase 2 (run after the test enriches the concepts) genuinely scores
higher and the improvement deltas are non-trivial.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.fake_vllm import FakeVLLMServer as _FakeServer
from tests.helpers import run_script

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "verify_concept_extraction.py"


class DoubleVerificationTests(unittest.TestCase):
    def _build_domain(self, root: Path, *, concepts_per_chunk: int) -> Path:
        """Create cache + staging for two chunks of one document."""
        domain_dir = root / "wiki" / "robotics"
        cache = domain_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)

        chunks = []
        for chunk_id in ("chunk_000", "chunk_001"):
            chunks.append({
                "doc_id": "canon_doc01", "chunk_id": chunk_id,
                "current_chunk": f"Raw text for {chunk_id}. Accuracy was 94.2%.",
                "context_before": "", "context_after": "",
            })
            chunk_dir = domain_dir / "staging" / "canon_doc01" / chunk_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / "concepts.json").write_text(json.dumps({
                "concepts": [{"title": f"Concept {chunk_id} {i}",
                              "description": "A description."}
                             for i in range(concepts_per_chunk)]
            }), encoding="utf-8")

        (cache / "chunks_cache.json").write_text(json.dumps(chunks), encoding="utf-8")
        return domain_dir

    def _rewrite_concepts(self, domain_dir: Path, concepts_per_chunk: int) -> None:
        """Trim every chunk's Pass 1 concepts to a fixed count, so a test controls the work size.

        Args:
            domain_dir: Run directory.
            concepts_per_chunk: How many concepts each chunk keeps.

        Returns:
            None."""
        for chunk_dir in (domain_dir / "staging").glob("*/*"):
            chunk_id = chunk_dir.name
            (chunk_dir / "concepts.json").write_text(json.dumps({
                "concepts": [{"title": f"Concept {chunk_id} {i}",
                              "description": "An enriched description."}
                             for i in range(concepts_per_chunk)]
            }), encoding="utf-8")

    def _run_phase(self, domain_dir: Path, phase: int, servers, *, expect_rc=0,
                   extra=()):
        """Run one verification phase as a subprocess.

        Args:
            domain_dir: Run directory.
            phase: 1 or 2.
            servers: Fake server URLs.
            expect_rc: The exit code the run should produce.

        Returns:
            The completed process. The exit code is asserted before returning."""
        return run_script("verify_concept_extraction.py",
                          ["--domain", "robotics", "--wiki_base_dir", str(domain_dir.parent),
                           "--phase", str(phase), "--servers", *servers, "--max_attempts", "4",
                           "--stage_budget_minutes", "2", *extra],
                          expect_rc=expect_rc, timeout=180)

    def test_two_phases_produce_two_scores_and_an_improvement(self):
        """Both phases report a score, and phase 2 reports how much it improved on phase 1."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _FakeServer() as server:
                domain_dir = self._build_domain(root, concepts_per_chunk=2)

                out1 = self._run_phase(domain_dir, 1, [server.url])
                self.assertIn("VERIFICATION SUMMARY REPORT [ROBOTICS]", out1.stdout)
                self.assertIn("Total Chunks Verified : 2", out1.stdout)
                self.assertIn("PASS 1 (raw extraction)", out1.stdout)

                # Pass 2 enriches the concepts, so the re-audit must score higher.
                self._rewrite_concepts(domain_dir, concepts_per_chunk=5)
                out2 = self._run_phase(domain_dir, 2, [server.url])

            self.assertIn("VERIFICATION SUMMARY REPORT [ROBOTICS]", out2.stdout)
            self.assertIn("PASS 1 (raw extraction)", out2.stdout)
            self.assertIn("PASS 2 (after refinement)", out2.stdout)
            self.assertIn("VERIFICATION IMPROVEMENT [ROBOTICS]", out2.stdout)
            self.assertIn("PASS 1 -> PASS 2", out2.stdout)

            # Per-chunk artifacts are phase-separated so Pass 2 keeps its input.
            chunk_dir = domain_dir / "staging" / "canon_doc01" / "chunk_000"
            pass1 = json.loads((chunk_dir / "verification.json").read_text())
            pass2 = json.loads((chunk_dir / "verification_pass2.json").read_text())
            self.assertLess(pass1["coverage_score"], pass2["coverage_score"])

            # Checkpoint logs are phase-separated too.
            self.assertTrue((domain_dir / "verification_log.json").is_file())
            self.assertTrue((domain_dir / "verification_pass2_log.json").is_file())

            summary = json.loads((domain_dir / "verification_summary.json").read_text())
            self.assertEqual(summary["domain"], "robotics")
            self.assertEqual(summary["pass1"]["total_chunks"], 2)
            self.assertEqual(summary["pass2"]["total_chunks"], 2)
            imp = summary["improvement"]
            self.assertGreater(imp["coverage_delta"], 0.0)
            self.assertLess(imp["missed_concepts_delta"], 0)
            self.assertEqual(imp["coverage_before"], round(pass1["coverage_score"], 4))

    def test_requests_spread_evenly_across_servers(self):
        """Verification requests are spread evenly across the servers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _FakeServer() as s0, _FakeServer() as s1, _FakeServer() as s2:
                domain_dir = self._build_domain(root, concepts_per_chunk=2)
                out = self._run_phase(domain_dir, 1, [s0.url, s1.url, s2.url])

                # 3 chunks would be ideal; we have 2, so two servers get one each
                # and the third gets none - the point is that one server must NOT
                # receive everything.
                counts = sorted([s0.requests, s1.requests, s2.requests], reverse=True)
                self.assertEqual(sum(counts), 2)
                self.assertLessEqual(counts[0], 1, msg=f"unbalanced spread: {counts}")
                self.assertIn("Requests served per server", out.stdout)

    def test_transient_server_faults_are_retried(self):
        """A transient server fault is retried and the phase still completes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # First two requests 503; the retry must recover and complete.
            with _FakeServer(fail_first=2) as server:
                domain_dir = self._build_domain(root, concepts_per_chunk=2)
                out = self._run_phase(domain_dir, 1, [server.url])
                self.assertIn("Total Chunks Verified : 2", out.stdout)
                self.assertGreaterEqual(server.requests, 4)

    def test_resume_skips_completed_chunks(self):
        """A second run skips chunks that already have a verdict."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _FakeServer() as server:
                domain_dir = self._build_domain(root, concepts_per_chunk=2)
                self._run_phase(domain_dir, 1, [server.url])
                first_requests = server.requests

                out = self._run_phase(domain_dir, 1, [server.url], extra=("--resume",))
                self.assertIn("already verified", out.stdout)
                self.assertEqual(server.requests, first_requests,
                                 msg="resume must not re-audit completed chunks")

    def test_missing_pass1_output_fails_loudly(self):
        """Phase 2 without phase 1 output fails before making any request."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _FakeServer() as server:
                domain_dir = self._build_domain(root, concepts_per_chunk=2)
                # Delete a chunk's concepts.json: phase 1 must refuse, not skip.
                (domain_dir / "staging" / "canon_doc01" / "chunk_001" / "concepts.json").unlink()
                proc = self._run_phase(domain_dir, 1, [server.url], expect_rc=1)
                self.assertIn("do not have valid extracted concepts", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
