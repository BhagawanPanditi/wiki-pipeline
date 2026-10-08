"""Regression tests for consolidate_global.run_llm_fuzzy_pass.

These exist because the accumulator lists this function shares between its
result callback and its post-map failure sweep (``approved_merges``,
``llm_evaluations``, ``failed_clusters``) were once used without ever being
initialised. ``py_compile`` and ``--help`` both passed, and no test imported the
module, so the NameError would have fired on the first real run that reached the
LLM fuzzy pass.

They also pin the failure contract: a cluster the model never returns usable
JSON for is recorded with empty decisions *and* still fails the stage loudly,
rather than being silently dropped.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import consolidate_global as gcc  # noqa: E402
from tests.fake_vllm import FakeVLLMCluster  # noqa: E402

LONG = ("A dense factual description with enough words to survive the similarity "
        "heuristics including measured throughput of 1240 requests per second.")


def cluster(*titles):
    """Build a fuzzy candidate cluster in the shape the stage produces.

    Args:
        *titles: Concept titles to place in the cluster.

    Returns:
        A list of concept dicts with the keys ``verify_cluster_llm`` reads:
        ``title``, ``aliases`` and ``description``.
    """
    return [{"title": t, "aliases": [], "description": LONG} for t in titles]


class FuzzyPassTests(unittest.TestCase):
    def _args(self, root: Path, servers, **extra):
        """Parse stage arguments for the consolidate-global profile.

        Args:
            root: Wiki base directory; outputs go to ``root/<domain>/``.
            servers: Fake vLLM base URLs.
            **extra: ``flag=value`` pairs appended as ``--flag value``.

        Returns:
            A parsed ``argparse.Namespace`` with ``domain`` normalised.
        """
        parser = gcc.build_parser(stage=gcc.STAGE, description="t", include_resume=True)
        out = ["--domain", "biology", "--wiki_base_dir", str(root), "--servers", *servers,
               "--max_attempts", "2", "--stage_budget_minutes", "2"]
        for key, value in extra.items():
            out.append(f"--{key}")
            out.append(str(value))
        args = parser.parse_args(out)
        (Path(root) / args.domain).mkdir(parents=True, exist_ok=True)
        return args

    def test_approved_merges_and_evaluations_are_populated(self):
        """A clean run returns the approvals and one evaluation per cluster."""
        verdict = {"merge_clusters": [{"primary_title": "Protein Folding",
                                       "merged_titles": ["Protein Folding Process"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with FakeVLLMCluster(2, responder=lambda p, i: verdict) as servers:
                args = self._args(root, servers.urls)
                approvals, evaluations = asyncio.run(gcc.run_llm_fuzzy_pass(
                    [cluster("Protein Folding", "Protein Folding Process"),
                     cluster("Enzyme Kinetics", "Enzyme Kinetics Rate")], args))
                self.assertEqual(servers.total_requests, 2)

            self.assertEqual(len(approvals), 2)
            self.assertEqual(approvals[0].primary_title, "Protein Folding")
            self.assertEqual(approvals[0].merged_titles, ["Protein Folding Process"])
            # Every cluster sent is recorded, so the audit trail is complete.
            self.assertEqual(len(evaluations), 2)
            for record in evaluations:
                self.assertIn("candidates_sent", record)
                self.assertIn("decisions", record)
                self.assertEqual(len(record["decisions"]), 1)

    def test_clusters_spread_evenly_across_servers(self):
        """Four clusters over two servers means two requests each."""
        verdict = {"merge_clusters": []}
        with tempfile.TemporaryDirectory() as tmp:
            with FakeVLLMCluster(2, responder=lambda p, i: verdict) as servers:
                args = self._args(Path(tmp), servers.urls)
                approvals, evaluations = asyncio.run(gcc.run_llm_fuzzy_pass(
                    [cluster(f"Concept {i}", f"Concept {i} variant") for i in range(4)],
                    args))
                self.assertEqual(servers.counts, [2, 2])
            self.assertEqual(approvals, [])
            self.assertEqual(len(evaluations), 4)
            self.assertTrue(all(r["decisions"] == [] for r in evaluations))

    def test_unparseable_response_fails_loudly_and_is_recorded(self):
        """Garbage JSON must raise, and still leave an evaluation trail."""
        with tempfile.TemporaryDirectory() as tmp:
            with FakeVLLMCluster(1, responder=lambda p, i: "not json at all {{{") as servers:
                args = self._args(Path(tmp), servers.urls)
                with self.assertRaises(RuntimeError) as caught:
                    asyncio.run(gcc.run_llm_fuzzy_pass(
                        [cluster("Broken Concept", "Broken Concept Two")], args))
                self.assertIn("LLM fuzzy review failed", str(caught.exception))
                self.assertIn("1 cluster", str(caught.exception))

            debug = Path(tmp) / "biology" / "log" / "consolidate-global.debug.jsonl"
            self.assertTrue(debug.is_file(), "no debug log written on failure")
            errors = [json.loads(line) for line in debug.read_text().splitlines() if line.strip()]
            self.assertTrue(errors, "debug log is empty after a failed cluster")

    def test_empty_candidate_list_short_circuits(self):
        """No clusters means no pool, no servers and no LLM traffic."""
        with tempfile.TemporaryDirectory() as tmp:
            approvals, evaluations = asyncio.run(gcc.run_llm_fuzzy_pass(
                [], self._args(Path(tmp), ["http://127.0.0.1:1/v1"])))
            self.assertEqual((approvals, evaluations), ([], []))

    def test_module_imports_without_nltk(self):
        """The stopword list falls back rather than making nltk mandatory."""
        self.assertTrue(gcc.STOP_WORDS, "STOP_WORDS is empty")
        self.assertIn("the", gcc.STOP_WORDS)


if __name__ == "__main__":
    unittest.main()
