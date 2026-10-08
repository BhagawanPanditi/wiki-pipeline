"""Tests for the per-stage LLM call log (<wiki>/<domain>/logs/<stage>.jsonl).

The requirement is that every LLM call and its output is recorded per stage,
inside the wiki directory, with nothing silently truncated. These tests drive a
real stage over real HTTP against a fake cluster and inspect the JSONL:

  * the file lands at logs/<stage>.jsonl under the run directory
  * ``full`` (the default) keeps the complete prompt and the complete response
  * ``compact`` shortens both but says so, and records the true length
  * ``errors`` keeps failures and drops successes
  * failures are recorded in every mode
  * each line is attributable: item key, server and attempt number
  * a stage with two pools writes two logs (final consolidation)
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
DOC_STAGE = "doc_concept_consolidation.py"
FINAL_STAGE = "final_consolidate_and_link.py"

LONG = ("A dense factual description with enough words to look like a real concept "
        "brief including measured throughput of 1240 requests per second at 61 ms.")
# A response longer than pipeline_lib.llm_log.COMPACT_CHAR_LIMIT, so compact mode
# has something to cut and full mode has something to preserve.
WIDE = "W" * 6000


def run(script, args, *, expect_rc=0, timeout=300):
    """Run a stage script as a subprocess and assert its exit code.

    Args:
        script: Stage file name at the repo root.
        args: CLI arguments after the script name.
        expect_rc: Expected return code.
        timeout: Seconds to wait.

    Returns:
        The completed ``subprocess.CompletedProcess``.

    Raises:
        AssertionError: On an unexpected return code, with the tail of stdout and
            stderr included.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / script), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
        f"STDOUT:\n{proc.stdout[-2500:]}\nSTDERR:\n{proc.stderr[-2500:]}")
    return proc


def read_jsonl(path: Path):
    """Read a JSONL file into a list of dicts.

    Args:
        path: File to read.

    Returns:
        A list of decoded records, skipping blank lines.

    Raises:
        AssertionError: If the file does not exist, so a missing log is reported
            as a test failure rather than an empty result.
    """
    assert path.is_file(), f"expected a log at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class LLMCallLogTests(unittest.TestCase):
    def _stage_doc(self, base: Path):
        """Lay out one chunked document so consolidation calls the model.

        Args:
            base: Wiki base directory; the run writes to ``base/biology/``.

        Returns:
            The run directory ``base/biology``.
        """
        run_dir = base / "biology"
        chunk = run_dir / "staging" / "canon_A" / "chunk_000"
        chunk.mkdir(parents=True, exist_ok=True)
        (chunk / "concepts.json").write_text(json.dumps({"concepts": [
            {"title": "Alpha Concept", "aliases": [], "description": LONG,
             "entities": ["e"], "source_chunks": ["canon_A::chunk_000"],
             "outbound_relations": []}]}), encoding="utf-8")
        (run_dir / "run_config.json").write_text(json.dumps(
            {"corpus": "x.jsonl", "chunking": True, "domain": "biology",
             "run_name": "biology"}), encoding="utf-8")
        return run_dir

    def _args(self, base, servers, **extra):
        """Build document-consolidation arguments.

        Args:
            base: Wiki base directory.
            servers: Fake vLLM base URLs.
            **extra: ``flag=value`` pairs appended as ``--flag value``.

        Returns:
            A list of CLI argument strings.
        """
        out = ["--domain", "biology", "--wiki_base_dir", str(base), "--servers", *servers,
               "--max_attempts", "2", "--stage_budget_minutes", "2"]
        for key, value in extra.items():
            out.append(f"--{key}")
            out.append(str(value))
        return out

    def test_full_mode_records_untruncated_prompt_and_response(self):
        """The default keeps everything: a 6000-char response survives intact."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_doc(base)
            verdict = {"merge_clusters": [{"primary_title": WIDE, "merged_titles": []}]}
            with FakeVLLMCluster(1, responder=lambda p, i: verdict) as servers:
                run(DOC_STAGE, self._args(base, servers.urls))

            log = run_dir / "log" / "document-consolidation.llm.jsonl"
            records = read_jsonl(log)
            self.assertEqual(len(records), 1)
            rec = records[0]

            self.assertTrue(rec["ok"])
            self.assertEqual(rec["stage"], "document-consolidation")
            self.assertFalse(rec["response"]["truncated"])
            self.assertEqual(rec["response"]["chars"], len(json.dumps(verdict)))
            self.assertIn(WIDE, rec["response"]["text"])

            # The prompt is reconstructable: role plus complete content.
            messages = rec["request"]["messages"]
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertTrue(all(not m["truncated"] for m in messages))
            self.assertIn("Alpha Concept", messages[-1]["text"])

            # Attribution: which item, which server, which attempt.
            self.assertEqual(rec["item"], "canon_A")
            self.assertEqual(rec["attempt"], 1)
            self.assertEqual(rec["server"], servers.urls[0])
            self.assertGreater(rec["latency_ms"], 0)

    def test_compact_mode_truncates_but_says_so(self):
        """Compact shortens both sides, flags it, and keeps the true length."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_doc(base)
            verdict = {"merge_clusters": [{"primary_title": WIDE, "merged_titles": []}]}
            with FakeVLLMCluster(1, responder=lambda p, i: verdict) as servers:
                run(DOC_STAGE, self._args(base, servers.urls, llm_log="compact"))

            rec = read_jsonl(run_dir / "log" / "document-consolidation.llm.jsonl")[0]
            self.assertTrue(rec["response"]["truncated"])
            self.assertLess(len(rec["response"]["text"]), 2400)
            # The original size is still known, so truncation is never silent.
            self.assertEqual(rec["response"]["chars"], len(json.dumps(verdict)))
            self.assertIn("truncated", rec["response"]["text"])

    def test_errors_mode_keeps_only_failures(self):
        """In errors mode a successful call writes nothing at all."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_doc(base)
            with FakeVLLMCluster(1, responder=lambda p, i: {"merge_clusters": []}) as servers:
                run(DOC_STAGE, self._args(base, servers.urls, llm_log="errors"))
                self.assertEqual(servers.total_requests, 1)

            log = run_dir / "log" / "document-consolidation.llm.jsonl"
            self.assertFalse(log.exists(),
                             "errors mode must not write a file for a clean run")

    def test_failures_are_logged_in_every_mode(self):
        """A model that never returns usable JSON is recorded, with its prompt."""
        for mode in ("full", "compact", "errors"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "wiki"
                run_dir = self._stage_doc(base)
                with FakeVLLMCluster(1, responder=lambda p, i: "not json {{{") as servers:
                    run(DOC_STAGE, self._args(base, servers.urls, llm_log=mode),
                        expect_rc=1)
                    attempts = servers.total_requests

                records = read_jsonl(run_dir / "log" / "document-consolidation.llm.jsonl")
                self.assertTrue(records, f"{mode}: nothing logged for a failed run")

                # The HTTP exchange succeeded (200 with a garbage body), so the
                # stage's refusal is a separate "rejection" line. Without it,
                # errors mode would report a clean run.
                calls = [r for r in records if r["event"] == "call"]
                rejects = [r for r in records if r["event"] == "rejection"]
                # errors mode drops successful HTTP exchanges by design; the
                # other modes keep one call line per request actually issued.
                expected_calls = 0 if mode == "errors" else attempts
                self.assertEqual(len(calls), expected_calls,
                                 f"{mode}: expected {expected_calls} call line(s)")
                self.assertEqual(len(rejects), attempts,
                                 f"{mode}: one rejection line per refused response")
                self.assertTrue(all(r["ok"] for r in calls),
                                f"{mode}: the transport did succeed")
                self.assertTrue(all(not r["ok"] for r in rejects))

                for rec in rejects:
                    self.assertEqual(rec["item"], "canon_A")
                    self.assertTrue(rec["error_type"], f"{mode}: no error_type recorded")
                    self.assertTrue(rec["error_message"], f"{mode}: no error_message")

                if mode == "errors":
                    # Successes are dropped, but a refused response is not one.
                    self.assertEqual(len(records), attempts)
                    self.assertTrue(all(r["event"] == "rejection" for r in records))
                else:
                    for rec in calls:
                        self.assertIn("not json", rec["response"]["text"],
                                      f"{mode}: the rejected body was not kept")
                        self.assertTrue(rec["request"]["messages"],
                                        f"{mode}: failed call lost its prompt")

                # Retries are distinguishable by attempt number.
                self.assertEqual(sorted(r["attempt"] for r in rejects),
                                 list(range(1, attempts + 1)))

    def test_final_consolidation_writes_one_log_per_pool(self):
        """The macro scan and deep verify are separate stages, separate files."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = base / "biology"
            run_dir.mkdir(parents=True)
            concepts = [{"title": f"{L}lpha Concept {i}", "aliases": [], "description": LONG,
                         "entities": [], "source_chunks": ["canon_A::chunk_000"],
                         "outbound_relations": []} for i, L in enumerate("AB")]
            (run_dir / "global_concepts_refined.json").write_text(
                json.dumps({"concepts": concepts}), encoding="utf-8")

            with FakeVLLMCluster(2, responder=lambda p, i: {
                    "merge_pairs": [], "link_pairs": [], "noise": []}) as servers:
                run(FINAL_STAGE, ["--domain", "biology", "--wiki_base_dir", str(base),
                                  "--servers", *servers.urls,
                                  "--stage_budget_minutes", "2"])

            macro = read_jsonl(run_dir / "log" / "final-macro-scan.llm.jsonl")
            self.assertTrue(macro)
            self.assertTrue(all(r["stage"] == "final-macro-scan" for r in macro))
            self.assertTrue(all(r["item"].startswith("letter:") for r in macro))
            self.assertFalse((run_dir / "log" / "final-deep-verify.llm.jsonl").exists(),
                             "deep verify made no calls, so it should leave no log")

    def test_log_lives_inside_the_wiki_dir_not_the_repo(self):
        """The trail belongs to the run, next to its other outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "wiki"
            run_dir = self._stage_doc(base)
            with FakeVLLMCluster(1, responder=lambda p, i: {"merge_clusters": []}) as servers:
                run(DOC_STAGE, self._args(base, servers.urls))
            self.assertTrue((run_dir / "log").is_dir())
            self.assertEqual(sorted(p.name for p in (run_dir / "log").iterdir()),
                             ["document-consolidation.decisions.jsonl",
                              "document-consolidation.llm.jsonl"])
            # Nothing leaked into the repository working tree.
            self.assertFalse(list(REPO_ROOT.glob("log")))


if __name__ == "__main__":
    unittest.main()
