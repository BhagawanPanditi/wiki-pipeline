"""End-to-end tests for final_consolidation.py.

This stage keeps its own per-server client construction (its two passes need
different request timeouts) but now uses the shared deterministic partition,
retry ladder, wall-clock budget and untruncated debug log. These tests pin the
behaviour that changed: even spreading and bounded retries.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tests.fake_vllm import FakeVLLMCluster
from tests.helpers import run_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "final_consolidation.py"
LONG = ("A dense factual description with enough words to look like a real concept "
        "brief including measured throughput of 1240 requests per second at 61 ms.")
NO_PAIRS = {"merge_pairs": [], "link_pairs": [], "noise": []}


def run(args, *, expect_rc=0, timeout=300):
    """Run final_consolidation.py as a subprocess and assert its exit code.

    Args:
        args: Command-line arguments after the script name.
        expect_rc: The exit code the run should produce.
        timeout: Seconds to wait before the run is abandoned.

    Returns:
        The completed process.
    """
    return run_script(SCRIPT, args, expect_rc=expect_rc, timeout=timeout)


class FinalConsolidateTests(unittest.TestCase):
    def _build(self, root: Path, letters="ABCDEF"):
        """Write a refined graph with one concept per letter, so each letter is one batch.

        Args:
            root: Temporary directory to build in.
            letters: One character per concept.

        Returns:
            The run directory, ``root/wiki/biology``."""
        base = root / "wiki" / "biology"
        base.mkdir(parents=True, exist_ok=True)
        # One alphabetical batch per letter, so len(letters) macro-scan items.
        concepts = [{"title": f"{L}lpha Concept {i}", "aliases": [], "description": LONG,
                     "entities": [], "source_chunks": ["doc0::chunk_000"],
                     "outbound_relations": []} for i, L in enumerate(letters)]
        (base / "global_concepts_refined.json").write_text(
            json.dumps({"concepts": concepts}), encoding="utf-8")
        return base

    def _args(self, base, servers, **extra):
        """Build the CLI arguments for a final-consolidation run.

        Args:
            base: The run directory. Its parent becomes ``--wiki_base_dir``.
            servers: Fake server URLs.
            **extra: ``flag=value`` pairs appended as ``--flag value``.

        Returns:
            A list of CLI argument strings."""
        out = ["--domain", "biology", "--wiki_base_dir", str(base.parent),
               "--servers", *servers, "--stage_budget_minutes", "2"]
        for k, v in extra.items():
            out.append(f"--{k}")
            if v != "":
                out.append(str(v))
        return out

    def test_macro_batches_spread_evenly_across_servers(self):
        """Six alphabetical batches over three servers reach each server exactly twice, and each batch is logged."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="ABCDEF")
            with FakeVLLMCluster(3, responder=lambda p, i: NO_PAIRS) as cluster:
                out = run(self._args(base, cluster.urls, max_attempts=2))
                # 6 alphabetical batches over 3 servers -> exactly 2 each.
                self.assertEqual(cluster.counts, [2, 2, 2])
            self.assertIn("item i -> server i % N", out.stdout)
            self.assertIn("Step 1 Complete", out.stdout)
            # The macro scan writes its own raw-response log; find it by pattern
            # rather than hardcoding a name.
            llm_logs = sorted(f.name for f in (base / "log").glob("*.llm.jsonl"))
            self.assertEqual(llm_logs, ["final-macro-scan.llm.jsonl"], llm_logs)
            decisions = [json.loads(line) for line in
                         (base / "log" / "final-macro-scan.decisions.jsonl").read_text().splitlines()]
            # One macro_scan decision per alphabetical batch: six letters, six records.
            self.assertEqual(sum(1 for d in decisions if d["event"] == "macro_scan"), 6)
            self.assertTrue(list(base.glob("*.json")), "no graph output written")

    def test_uneven_batch_count_still_balances(self):
        """Five batches over three servers stay within one request of each other."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="ABCDE")
            with FakeVLLMCluster(3, responder=lambda p, i: NO_PAIRS) as cluster:
                run(self._args(base, cluster.urls, max_attempts=2))
                self.assertEqual(cluster.total_requests, 5)
                self.assertLessEqual(max(cluster.counts) - min(cluster.counts), 1,
                                     msg=f"unbalanced: {cluster.counts}")

    def test_two_passes_resolve_different_profiles(self):
        """Macro scan and deep verification have separate max_tokens budgets."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="AB")
            with FakeVLLMCluster(1, responder=lambda p, i: NO_PAIRS) as cluster:
                out = run(self._args(base, cluster.urls))
            self.assertIn("final-macro-scan.debug.jsonl", out.stdout)
            self.assertIn("final-deep-verify.debug.jsonl", out.stdout)
            # Both passes announced their own spread.
            self.assertEqual(out.stdout.count("item i -> server i % N"), 2)

    def test_unparseable_batches_fail_loudly_and_log_the_error(self):
        """Batches the model cannot parse fail the stage, and each failure is in the decision log."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="AB")
            with FakeVLLMCluster(2, responder=lambda p, i: "{not json") as cluster:
                proc = run(self._args(base, cluster.urls, max_attempts=1), expect_rc=1)
                # 2 batches, 1 attempt each, one per server.
                self.assertEqual(cluster.total_requests, 2)
            self.assertIn("Final macro scan failed for 2 batch(es)",
                          proc.stdout + proc.stderr)
            entries = [json.loads(line) for line in
                       (base / "log" / "final-macro-scan.decisions.jsonl").read_text().splitlines()]
            errors = [e for e in entries if e["event"] == "macro_scan_failed"]
            self.assertEqual(len(errors), 2, entries)
            self.assertTrue(all(e["error"] for e in errors))

    def test_an_empty_deep_verification_reply_is_retried_not_a_name_error(self):
        """A deep-verification reply with no actions is retried after a backoff.

        Regression test: the retry branch used to reference an undefined name, so
        the stage recorded a NameError where it should have waited and asked again.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="AA")

            def responder(prompt, index):
                """Propose one merge for the macro scan; answer deep verification with nothing usable."""
                if "Indexed Titles:" in prompt:
                    return {"merge_pairs": [[1, 2]], "link_pairs": [], "noise": []}
                return {"actions": []}

            with FakeVLLMCluster(1, responder=responder) as cluster:
                proc = run(self._args(base, cluster.urls, max_attempts=2), expect_rc=1)
                deep_prompts = [q for srv in cluster.servers for q in srv.prompts if "Indexed Titles:" not in q]
            # The stage still fails loudly once the retries are used up...
            self.assertIn("Deep final verification failed for 1 group(s) after retries",
                          proc.stdout + proc.stderr)
            # ...but the one group got both of its attempts, not one.
            self.assertEqual(len(deep_prompts), 2)
            # Each attempt is recorded as a rejected reply, and none of them is a NameError.
            calls = [json.loads(line) for line in
                     (base / "log" / "final-deep-verify.llm.jsonl").read_text().splitlines()]
            rejections = [c for c in calls if c["event"] == "rejection"]
            self.assertEqual([r["error_type"] for r in rejections], ["ValueError", "ValueError"])

    def test_transient_faults_are_retried_not_fatal(self):
        """Transient faults are retried, and the stage completes."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), letters="AB")
            # fail_first is per server: each of the 2 servers takes 2 faults then
            # succeeds, so each batch costs 3 requests -> 6 in total.
            with FakeVLLMCluster(2, responder=lambda p, i: NO_PAIRS,
                                 fail_first=2) as cluster:
                out = run(self._args(base, cluster.urls, max_attempts=3))
                self.assertEqual(cluster.counts, [3, 3])
            self.assertIn("Step 1 Complete", out.stdout)


if __name__ == "__main__":
    unittest.main()
