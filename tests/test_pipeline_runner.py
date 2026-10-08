"""Tests for run_pipeline.sh: one pipeline, resuming by default.

The runner is exercised with a fake ``python3`` first on PATH. It forwards the
runner's own inline validators (``python3 - <<PY``) to the real interpreter, and
dispatches every stage script to a helper that writes structurally valid outputs
-- including a chunk-cache manifest computed by the real
``pipeline_lib.corpus.cache_manifest``, so the runner's staleness check is tested
against the same code Pass 1 uses rather than a copy of it.

Pinned behaviour:
  * a first run executes every stage in order
  * a second run skips everything except the preflight health probe
  * deleting an upstream artifact reruns that stage and everything after it
  * a stage that fails is retried up to --max-stage-attempts, then the run stops
  * --fresh clears completion markers so every stage runs again
  * a missing corpus is reported before any stage starts
  * --no-chunking is forwarded, and the run directory is 'default' with no domain
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_pipeline.sh"

CONCEPTS = json.dumps({"concepts": [{"title": "Mock Concept", "aliases": [],
                                    "description": "A mock description.",
                                    "entities": [], "outbound_relations": []}]})

# The fake stage implementations. Each writes the outputs the runner validates
# for that step, so step_outputs_ready() is exercised for real.
HELPER = r'''
"""Fake stage implementations for the run_pipeline.sh tests."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("@REPO_ROOT@")
sys.path.insert(0, str(REPO_ROOT))

script = Path(sys.argv[1]).name
argv = sys.argv[2:]


def flag(name, default=None):
    """Return the value following --name, or default."""
    return argv[argv.index(name) + 1] if name in argv else default


def has(name):
    """True when --name was passed."""
    return name in argv


domain = flag("--domain", "default")
base = flag("--wiki_base_dir", ".")
run_dir = Path(base) / domain
run_dir.mkdir(parents=True, exist_ok=True)
# Record the attempt count so a test can make one stage fail exactly once.
attempts_file = Path(os.environ["FAKE_ATTEMPTS"])


def bump(key):
    """Increment and return a per-script attempt counter."""
    state = {}
    if attempts_file.is_file():
        state = json.loads(attempts_file.read_text())
    state[key] = int(state.get(key, 0)) + 1
    attempts_file.write_text(json.dumps(state))
    return state[key]


if script == "check_servers.py":
    sys.exit(0)

if script == "concept_extraction_pass_1.py":
    from pipeline_lib.corpus import cache_manifest, deduplicate, load_corpus, build_units

    corpus = Path(flag("--corpus"))
    chunking = not has("--no-chunking")
    chunk_size = int(flag("--chunk_size", "4500"))
    overlap = int(flag("--chunk_overlap", "400"))
    docs = load_corpus(corpus)
    canon, mapping = deduplicate(docs)
    # There is no tokenizer in the test environment, so the units are built
    # unchunked. The manifest, however, must describe what the runner asked for:
    # it is computed by the same cache_manifest() the runner calls, with the same
    # model name from the same config, so the staleness check is genuinely
    # exercised rather than trivially satisfied.
    tokenizer_name = None
    if chunking:
        from pipeline_config import load_global_config
        tokenizer_name = load_global_config()[2]
    units = build_units(canon, None, chunking=False)
    cache = run_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "chunks_cache.json").write_text(json.dumps(units))
    (cache / "doc_mapping.json").write_text(json.dumps(mapping))
    (cache / "chunks_cache_manifest.json").write_text(json.dumps(cache_manifest(
        corpus, tokenizer_name=tokenizer_name, chunking=chunking,
        chunk_size=chunk_size, overlap=overlap)))
    (run_dir / "extraction_log.json").write_text(
        json.dumps({f"{u['doc_id']}::{u['chunk_id']}": True for u in units}))
    (run_dir / "run_config.json").write_text(json.dumps(
        {"corpus": str(corpus), "chunking": chunking, "domain": None,
         "run_name": domain}))
    for unit in units:
        d = run_dir / "staging" / unit["doc_id"] / unit["chunk_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "concepts.json").write_text(
            json.dumps({"concepts": [{"title": "Mock Concept", "aliases": [],
                                      "description": "A mock description.",
                                      "entities": [], "outbound_relations": []}]}))
    sys.exit(0)

if script == "verify_concept_extraction.py":
    phase = flag("--phase", "1")
    chunks = json.loads((run_dir / "cache" / "chunks_cache.json").read_text())
    name = "verification.json" if phase == "1" else "verification_pass2.json"
    for c in chunks:
        d = run_dir / "staging" / c["doc_id"] / c["chunk_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(
            {"coverage_score": 0.9, "verdict": "sufficient",
             "audit_feedback": {"missing_concepts": [], "missing_facts": []}}))
    if phase == "1":
        (run_dir / "verification_log.json").write_text(
            json.dumps({f"{c['doc_id']}::{c['chunk_id']}": True for c in chunks}))
    else:
        (run_dir / "verification_pass2_log.json").write_text(
            json.dumps({f"{c['doc_id']}::{c['chunk_id']}": True for c in chunks}))
        (run_dir / "verification_summary.json").write_text(
            json.dumps({"phase1": {}, "phase2": {}, "improvement": {}}))
    sys.exit(0)

if script == "concept_refinement_pass_2.py":
    (run_dir / "pass2_log.json").write_text(json.dumps({"done": True}))
    sys.exit(0)

if script == "doc_concept_consolidation.py":
    consolidated = run_dir / "consolidated" / "canon_test"
    consolidated.mkdir(parents=True, exist_ok=True)
    (consolidated / "concepts.json").write_text(
        json.dumps({"doc_id": "canon_test",
                    "concepts": [{"title": "Mock Concept"}]}))
    (run_dir / "doc_consolidation_log.json").write_text(
        json.dumps({"canon_test": True, "total_concepts_before": 1,
                    "total_concepts_after": 1}))
    sys.exit(0)

if script == "global_concept_consolidation.py":
    # Fail exactly once so the runner's whole-stage retry is observable.
    if bump("global") == 1 and os.environ.get("FAKE_FAIL_GLOBAL") == "1":
        sys.exit(1)
    (run_dir / "global_concepts.json").write_text(
        json.dumps({"concepts": [{"title": "Mock Concept"}]}))
    sys.exit(0)

if script == "stub_resolution.py":
    (run_dir / "global_concepts_stubs_resolved.json").write_text(
        json.dumps({"concepts": [{"title": "Mock Concept"}]}))
    sys.exit(0)

if script == "analyze_islands.py":
    (run_dir / "islands_report.json").write_text(
        json.dumps({"summary": {}, "islands": []}))
    sys.exit(0)

if script == "final_consolidate_and_link.py":
    (run_dir / "global_concepts_refined.json").write_text(
        json.dumps({"concepts": [{"title": "Mock Concept"}]}))
    md = run_dir / "concepts_md"
    md.mkdir(parents=True, exist_ok=True)
    (md / "Mock Concept.md").write_text("# Mock Concept\n")
    sys.exit(0)

if script in ("thin_concept_handler.py", "add_concept_briefs.py",
              "link_entity_overlap.py", "audit_noise_islands.py",
              "synthesize_bidirectional_relations.py"):
    (run_dir / "global_concepts_refined.json").write_text(
        json.dumps({"concepts": [{"title": "Mock Concept"}]}))
    sys.exit(0)

sys.stderr.write("fake python: unhandled script %s\n" % script)
sys.exit(99)
'''

WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
REAL_PYTHON={real}
# The runner pipes its own validators in on stdin as `python3 - <<PY`; those must
# run for real or every output check would silently pass.
if [[ "${{1:-}}" == "-" ]]; then exec "$REAL_PYTHON" "$@"; fi
echo "$(basename "$1")" >> "$FAKE_CALL_LOG"
exec "$REAL_PYTHON" {helper} "$@"
"""


class RunPipelineTests(unittest.TestCase):
    def _env(self, root: Path, fail_global: bool = False):
        """Install a fake python3 and return (env, call_log, helper).

        Args:
            root: Temp directory to build the fake toolchain in.
            fail_global: When true, global consolidation fails on its first
                attempt so the runner's retry path is exercised.

        Returns:
            A 3-tuple of the environment dict, the call-log path and the helper
            script path.
        """
        real_python = shutil.which("python3")
        self.assertIsNotNone(real_python)

        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        helper = root / "fake_stage.py"
        helper.write_text(HELPER.replace("@REPO_ROOT@", str(REPO_ROOT)),
                          encoding="utf-8")

        fake_python = bin_dir / "python3"
        fake_python.write_text(WRAPPER.format(real=real_python, helper=helper),
                               encoding="utf-8")
        fake_python.chmod(0o755)

        # Retries sleep; stub it out so the tests stay fast.
        fake_sleep = bin_dir / "sleep"
        fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_sleep.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        env["FAKE_CALL_LOG"] = str(root / "calls.log")
        env["FAKE_ATTEMPTS"] = str(root / "attempts.json")
        env["FAKE_FAIL_GLOBAL"] = "1" if fail_global else "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env, Path(env["FAKE_CALL_LOG"]), helper

    def _corpus(self, root: Path) -> Path:
        """Write a small JSONL corpus and return its path.

        Args:
            root: Temp directory to write into.

        Returns:
            The corpus path, holding two documents with one shared body so
            deduplication is observable.
        """
        corpus = root / "corpus.jsonl"
        shared = "Photosynthesis converts light into chemical energy in plants."
        corpus.write_text(
            json.dumps({"id": "doc-A", "contents": shared}) + "\n"
            + json.dumps({"id": "doc-B", "contents": "The Krebs cycle oxidises acetyl-CoA."}) + "\n"
            + json.dumps({"id": "doc-C", "contents": shared}) + "\n",
            encoding="utf-8")
        return corpus

    def _run(self, env, args, *, expect_rc=0):
        """Invoke the runner and assert its exit code.

        Args:
            env: Environment from :meth:`_env`.
            args: Runner arguments.
            expect_rc: Expected return code.

        Returns:
            The completed process.

        Raises:
            AssertionError: On an unexpected return code, with output included.
        """
        proc = subprocess.run([str(RUNNER), *args], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == expect_rc, (
            f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
        return proc

    def _calls(self, call_log: Path):
        """Return the recorded stage script names, in order.

        Args:
            call_log: The fake interpreter's call log.

        Returns:
            A list of script basenames. Empty if nothing has run yet.
        """
        if not call_log.is_file():
            return []
        return call_log.read_text(encoding="utf-8").split()

    def test_first_run_executes_every_stage_in_order(self):
        """A clean run walks all fifteen steps and writes the final artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)
            corpus = self._corpus(root)
            wiki = root / "wiki"

            out = self._run(env, ["--corpus", str(corpus), "--domain", "biology",
                                  "--wiki-base-dir", str(wiki),
                                  "--max-stage-attempts", "2"])

            calls = self._calls(call_log)
            self.assertEqual(calls[0], "check_servers.py")
            for stage in ("concept_extraction_pass_1.py", "verify_concept_extraction.py",
                          "concept_refinement_pass_2.py", "doc_concept_consolidation.py",
                          "global_concept_consolidation.py", "stub_resolution.py",
                          "thin_concept_handler.py", "add_concept_briefs.py",
                          "link_entity_overlap.py", "audit_noise_islands.py",
                          "synthesize_bidirectional_relations.py",
                          "final_consolidate_and_link.py", "analyze_islands.py"):
                self.assertIn(stage, calls, f"{stage} never ran")
            # Verification runs twice: once per phase.
            self.assertEqual(calls.count("verify_concept_extraction.py"), 2)

            run_dir = wiki / "biology"
            self.assertTrue((run_dir / "concepts_md" / "Mock Concept.md").is_file())
            self.assertTrue((run_dir / "islands_report.json").is_file())
            self.assertIn("PIPELINE RUN COMPLETE", out.stdout)

    def test_second_run_skips_everything_but_preflight(self):
        """Resume is the default: a repeat run redoes no work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)
            corpus = self._corpus(root)
            args = ["--corpus", str(corpus), "--domain", "biology",
                    "--wiki-base-dir", str(root / "wiki"), "--max-stage-attempts", "2"]

            self._run(env, args)
            before = len(self._calls(call_log))

            # No --resume flag: resuming is the only behaviour.
            self._run(env, args)
            new = self._calls(call_log)[before:]
            self.assertEqual(new, ["check_servers.py"],
                             "a completed pipeline should only re-probe the servers")

    def test_missing_upstream_artifact_reruns_that_stage_and_downstream(self):
        """Deleting one output invalidates it and everything derived from it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)
            corpus = self._corpus(root)
            wiki = root / "wiki"
            args = ["--corpus", str(corpus), "--domain", "biology",
                    "--wiki-base-dir", str(wiki), "--max-stage-attempts", "2"]

            self._run(env, args)
            (wiki / "biology" / "global_concepts.json").unlink()
            before = len(self._calls(call_log))

            self._run(env, args)
            rebuilt = self._calls(call_log)[before:]
            self.assertEqual(rebuilt[0], "check_servers.py")
            self.assertEqual(rebuilt[1], "global_concept_consolidation.py")
            # Extraction is untouched, but everything downstream is redone.
            self.assertNotIn("concept_extraction_pass_1.py", rebuilt)
            self.assertIn("stub_resolution.py", rebuilt)
            self.assertIn("final_consolidate_and_link.py", rebuilt)
            self.assertIn("analyze_islands.py", rebuilt)

    def test_failing_stage_is_retried_then_succeeds(self):
        """The runner restarts a whole stage, independently of per-call retries."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root, fail_global=True)
            corpus = self._corpus(root)

            out = self._run(env, ["--corpus", str(corpus), "--domain", "biology",
                                  "--wiki-base-dir", str(root / "wiki"),
                                  "--max-stage-attempts", "3"])
            calls = self._calls(call_log)
            self.assertEqual(calls.count("global_concept_consolidation.py"), 2,
                             "the failing stage should have been restarted once")
            # The runner reports stage failures on stderr, progress on stdout.
            self.assertIn("failed with exit code 1", out.stderr)
            self.assertIn("Completed: global-consolidation (attempt 2)", out.stdout)
            self.assertIn("PIPELINE RUN COMPLETE", out.stdout)

    def test_fresh_clears_markers_and_reruns_everything(self):
        """--fresh drops completion markers but leaves generated data alone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)
            corpus = self._corpus(root)
            wiki = root / "wiki"
            args = ["--corpus", str(corpus), "--domain", "biology",
                    "--wiki-base-dir", str(wiki), "--max-stage-attempts", "2"]

            self._run(env, args)
            before = len(self._calls(call_log))
            out = self._run(env, args + ["--fresh"])

            self.assertIn("--fresh: cleared", out.stdout)
            fresh = self._calls(call_log)[before:]
            self.assertIn("concept_extraction_pass_1.py", fresh)
            self.assertIn("analyze_islands.py", fresh)
            # Markers only: the wiki data itself survives.
            self.assertTrue((wiki / "biology" / "concepts_md").is_dir())

    def test_no_chunking_is_forwarded_and_domain_is_optional(self):
        """Without --domain the run directory is 'default' and no hint is passed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, helper = self._env(root)
            corpus = self._corpus(root)
            wiki = root / "wiki"

            out = self._run(env, ["--corpus", str(corpus), "--no-chunking",
                                  "--wiki-base-dir", str(wiki),
                                  "--max-stage-attempts", "2"])

            self.assertIn("Domain hint:      none", out.stdout)
            self.assertIn("Chunking:         false", out.stdout)
            self.assertTrue((wiki / "default").is_dir(),
                            f"expected a 'default' run dir, have "
                            f"{sorted(p.name for p in wiki.iterdir())}")
            self.assertTrue((wiki / "default" / "concepts_md" / "Mock Concept.md").is_file())

            config = json.loads((wiki / "default" / "run_config.json").read_text())
            self.assertFalse(config["chunking"])
            self.assertIsNone(config["domain"])

    def test_missing_corpus_is_reported_before_any_stage_runs(self):
        """A bad input path fails fast with both supported spellings."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)

            out = self._run(env, ["--corpus", str(root / "nope.jsonl"),
                                  "--wiki-base-dir", str(root / "wiki")], expect_rc=2)
            self.assertIn("--corpus file not found", out.stderr)
            self.assertEqual(self._calls(call_log), [], "no stage should have started")

            out = self._run(env, ["--wiki-base-dir", str(root / "wiki")], expect_rc=2)
            self.assertIn("no corpus specified", out.stderr)
            self.assertIn("--corpus", out.stderr)

    def test_rejects_conflicting_chunk_parameters(self):
        """An overlap at least as large as the window would never terminate."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, call_log, _ = self._env(root)
            corpus = self._corpus(root)

            out = self._run(env, ["--corpus", str(corpus), "--chunk-size", "100",
                                  "--chunk-overlap", "100",
                                  "--wiki-base-dir", str(root / "wiki")], expect_rc=2)
            self.assertIn("must be smaller than", out.stderr)
            self.assertEqual(self._calls(call_log), [])

    def test_rejects_unknown_llm_log_mode(self):
        """--llm-log is validated up front, not by argparse deep in a stage."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _, _ = self._env(root)
            corpus = self._corpus(root)
            out = self._run(env, ["--corpus", str(corpus), "--llm-log", "verbose",
                                  "--wiki-base-dir", str(root / "wiki")], expect_rc=2)
            self.assertIn("--llm-log must be one of", out.stderr)


if __name__ == "__main__":
    unittest.main()
