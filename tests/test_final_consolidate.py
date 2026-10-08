"""End-to-end tests for final_consolidate_and_link.py.

This stage keeps its own per-server client construction (its two passes need
different request timeouts) but now uses the shared deterministic partition,
retry ladder, wall-clock budget and untruncated debug log. These tests pin the
behaviour that changed: even spreading and bounded retries.
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
SCRIPT = "final_consolidate_and_link.py"
LONG = ("A dense factual description with enough words to look like a real concept "
        "brief including measured throughput of 1240 requests per second at 61 ms.")
NO_PAIRS = {"merge_pairs": [], "link_pairs": [], "noise": []}


def run(args, *, expect_rc=0, timeout=240):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, str(REPO_ROOT / SCRIPT), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


class FinalConsolidateTests(unittest.TestCase):
    def _build(self, root: Path, letters="ABCDEF"):
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
        out = ["--domain", "biology", "--wiki_base_dir", str(base.parent),
               "--servers", *servers, "--stage_budget_minutes", "2"]
        for k, v in extra.items():
            out.append(f"--{k}")
            if v != "":
                out.append(str(v))
        return out

    def test_macro_batches_spread_evenly_across_servers(self):
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

    def test_transient_faults_are_retried_not_fatal(self):
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
