"""Integration tests for stages refactored onto pipeline_lib.ServerPool.

These run the real stage scripts as subprocesses against real HTTP servers, so
they cover the parts a unit test cannot: argument resolution, even spreading of
work across a variable number of servers, the retry ladder, and the stage's own
failure contract.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fake_vllm import FakeVLLMCluster, FakeVLLMServer

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_stage(script, args, *, expect_rc=0, timeout=240):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, str(REPO_ROOT / script), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"{script} exited {proc.returncode} (expected {expect_rc})\n"
        f"ARGS: {args}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


class ReverseRelationsTests(unittest.TestCase):
    """synthesize_bidirectional_relations.py"""

    SCRIPT = "synthesize_bidirectional_relations.py"

    def _build_graph(self, root: Path, chain=("A", "B", "C", "D", "E", "F")) -> Path:
        """A one-directional chain, so every link needs a synthesized back-link."""
        domain_dir = root / "wiki" / "robotics"
        domain_dir.mkdir(parents=True, exist_ok=True)
        concepts = []
        for i, name in enumerate(chain):
            rel = []
            if i + 1 < len(chain):
                rel.append({"target_concept": f"Concept {chain[i + 1]}",
                            "relation": f"{name} drives {chain[i + 1]}.",
                            "source_chunks": ["chunk_000"]})
            concepts.append({"title": f"Concept {name}",
                             "brief": f"Brief for {name}.",
                             "outbound_relations": rel})
        (domain_dir / "global_concepts_refined.json").write_text(
            json.dumps({"concepts": concepts}), encoding="utf-8")
        return domain_dir

    def _edges(self, domain_dir: Path):
        data = json.loads((domain_dir / "global_concepts_refined.json").read_text())
        return {c["title"]: [r["target_concept"] for r in c["outbound_relations"]]
                for c in data["concepts"]}

    def test_every_edge_gains_a_reverse_link_and_work_is_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domain_dir = self._build_graph(root)
            with FakeVLLMCluster(3, responder=lambda p, i: "Reverse sentence.") as cluster:
                out = run_stage(self.SCRIPT, [
                    "--domain", "robotics", "--wiki_base_dir", str(root / "wiki"),
                    "--servers", *cluster.urls, "--max_attempts", "2",
                    "--stage_budget_minutes", "2"])
                # 5 forward links in a 6-concept chain -> 5 missing reverse links.
                cluster.assert_balanced(self, expected_total=5)
                self.assertIn("Requests served per server", out.stdout)

            edges = self._edges(domain_dir)
            self.assertIn("Concept A", edges["Concept B"])
            self.assertIn("Concept F", edges["Concept E"])
            # Original forward links must survive.
            self.assertIn("Concept B", edges["Concept A"])
            self.assertIn("Residual stubs purged: 0", out.stdout)

    def test_exhausted_retries_fail_loudly_with_the_stage_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domain_dir = self._build_graph(root)
            # Empty completions are recoverable, so they exhaust the retry ladder
            # without the servers being marked unreachable.
            with FakeVLLMCluster(3, responder=lambda p, i: "") as cluster:
                proc = run_stage(self.SCRIPT, [
                    "--domain", "robotics", "--wiki_base_dir", str(root / "wiki"),
                    "--servers", *cluster.urls, "--max_attempts", "1",
                    "--stage_budget_minutes", "2"], expect_rc=1)
            self.assertIn("Reverse-relation synthesis failed for 5 edge(s)",
                          proc.stdout + proc.stderr)
            # The stage raises before persisting, so a failed run must leave the
            # graph exactly as it was rather than half-linked. (The deterministic
            # fallback wording is attached in memory only, as it always was.)
            edges = self._edges(domain_dir)
            self.assertEqual(edges["Concept B"], ["Concept C"])
            self.assertNotIn("Concept A", edges["Concept B"])

    def test_fast_mode_needs_no_servers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domain_dir = self._build_graph(root)
            out = run_stage(self.SCRIPT, [
                "--domain", "robotics", "--wiki_base_dir", str(root / "wiki"), "--fast"])
            self.assertIn("template-based reverse relation", out.stdout)
            self.assertIn("Concept A", self._edges(domain_dir)["Concept B"])

    def test_transient_faults_are_retried_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domain_dir = self._build_graph(root)
            # One server injects two 503s; failover plus retry must still finish.
            bad = FakeVLLMServer(fail_first=2).start()
            with FakeVLLMCluster(2, responder=lambda p, i: "Reverse sentence.") as good:
                try:
                    out = run_stage(self.SCRIPT, [
                        "--domain", "robotics", "--wiki_base_dir", str(root / "wiki"),
                        "--servers", bad.url, *good.urls, "--max_attempts", "3",
                        "--stage_budget_minutes", "2"])
                finally:
                    bad.stop()
            self.assertIn("Concept A", self._edges(domain_dir)["Concept B"])
            self.assertNotIn("Traceback", out.stderr)


if __name__ == "__main__":
    unittest.main()
