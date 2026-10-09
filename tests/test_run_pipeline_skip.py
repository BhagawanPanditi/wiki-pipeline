"""Exercise runner skip flags with fake stage scripts, without an LLM server."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


STUB = '''import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).stem
args = sys.argv[1:]
root = Path(os.environ["TEST_WIKI_BASE"]) / "demo"
root.mkdir(parents=True, exist_ok=True)
with (root / "calls.txt").open("a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
chunk = root / "staging" / "doc" / "chunk"
if name == "extract_concepts":
    chunk.mkdir(parents=True, exist_ok=True)
    (chunk / "concepts.json").write_text('{"concepts": [{"title": "topic"}]}')
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "chunks_cache.json").write_text('[{"doc_id": "doc", "chunk_id": "chunk"}]')
    (cache / "chunks_cache_manifest.json").write_text('{}')
    (root / "doc_mapping.json").write_text('{}')
    (root / "extraction_log.json").write_text('{}')
elif name == "verify_extraction":
    phase = args[args.index("--phase") + 1]
    if phase == "2" and os.environ.get("FAIL_VERIFY_PASS2"):
        sys.exit(71)
    result = {"coverage_score": 0.9, "is_sufficient": True,
              "missing_concepts": [], "missing_facts": [],
              "hallucinations": [], "critique": "Complete."}
    file = "verification.json" if phase == "1" else "verification_pass2.json"
    (chunk / file).write_text(json.dumps(result))
    log = "verification_log.json" if phase == "1" else "verification_pass2_log.json"
    (root / log).write_text('{"doc::chunk": true}')
    if phase == "2":
        (root / "verification_summary.json").write_text('{}')
elif name == "refine_concepts":
    if os.environ.get("FAIL_REFINEMENT"):
        sys.exit(73)
    (root / "pass2_log.json").write_text('{"doc::chunk": true}')
elif name == "consolidate_documents":
    # Reaching this step proves neither audit is required downstream.
    sys.exit(85)
'''


class RunnerSkipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        source = Path(__file__).resolve().parents[1]
        shutil.copyfile(source / "run_pipeline.sh", self.bin / "run_pipeline.sh")
        (self.bin / "config.py").write_text(
            f'DEFAULT_WIKI_BASE_DIR = {str(self.root / "wiki")!r}\n'
            f'DEFAULT_DATA_DIR = {str(self.root)!r}\n'
            'def load_global_config(): return ([], "test-model")\n'
        )
        (self.bin / "utils").mkdir()
        (self.bin / "utils" / "corpus.py").write_text('def cache_manifest(*args, **kwargs): return {}\n')
        for name in ("check_servers", "extract_concepts", "verify_extraction",
                     "refine_concepts", "consolidate_documents"):
            (self.bin / f"{name}.py").write_text(STUB)
        self.corpus = self.root / "demo.jsonl"
        self.corpus.write_text('{"id": "doc", "contents": "topic"}\n')
        self.run_dir = self.root / "wiki" / "demo"

    def run_pipeline(self, *flags, fail_pass2=False, fail_refinement=False):
        env = {**os.environ, "TEST_WIKI_BASE": str(self.root / "wiki")}
        if fail_pass2:
            env["FAIL_VERIFY_PASS2"] = "1"
        if fail_refinement:
            env["FAIL_REFINEMENT"] = "1"
        return subprocess.run(
            ["bash", str(self.bin / "run_pipeline.sh"),
             "--corpus", str(self.corpus), "--domain", "demo",
             "--wiki-base-dir", str(self.root / "wiki"),
             "--max-stage-attempts", "1", *flags],
            env=env, text=True, capture_output=True, timeout=30,
        )

    def calls(self):
        return (self.run_dir / "calls.txt").read_text().splitlines()

    def test_skip_pass2_after_failed_audit_and_resume_it_later(self):
        failed = self.run_pipeline(fail_pass2=True)
        self.assertEqual(failed.returncode, 71, failed.stdout + failed.stderr)
        before = len([line for line in self.calls() if line.startswith("verify_extraction --")])
        skipped = self.run_pipeline("--skip-verify-pass2", fail_pass2=True)
        self.assertEqual(skipped.returncode, 85, skipped.stdout + skipped.stderr)
        self.assertIn("[SKIP] verify-pass2: requested", skipped.stdout)
        self.assertTrue(any(line.startswith("consolidate_documents ") for line in self.calls()))
        self.assertEqual(len([line for line in self.calls() if line.startswith("verify_extraction --")]), before)
        self.assertFalse((self.run_dir / ".pipeline_state" / "verify-pass2.done").exists())
        self.assertFalse((self.run_dir / ".pipeline_state" / "verify-pass2.failed").exists())
        resumed = self.run_pipeline()
        self.assertEqual(resumed.returncode, 85, resumed.stdout + resumed.stderr)
        self.assertTrue((self.run_dir / ".pipeline_state" / "verify-pass2.done").exists())

    def test_skip_both_passes_still_refines_without_feedback(self):
        result = self.run_pipeline("--skip-verification")
        self.assertEqual(result.returncode, 85, result.stdout + result.stderr)
        lines = self.calls()
        self.assertFalse(any(line.startswith("verify_extraction ") for line in lines))
        self.assertTrue(any(line.startswith("refine_concepts ") and
                            "--ignore_audit_feedback" in line for line in lines))
        self.assertTrue(any(line.startswith("consolidate_documents ") for line in lines))
        self.assertFalse((self.run_dir / ".pipeline_state" / "verify-pass1.done").exists())
        self.assertFalse((self.run_dir / ".pipeline_state" / "verify-pass2.done").exists())

    def test_skip_verify_pass1_and_refinement_proceeds_without_pass2_audit(self):
        result = self.run_pipeline("--skip-verify-pass1", "--skip-refinement")
        self.assertEqual(result.returncode, 85, result.stdout + result.stderr)
        self.assertIn("[SKIP] refine-concepts: requested", result.stdout)
        self.assertIn("[SKIP] verify-pass2: requested", result.stdout)
        lines = self.calls()
        self.assertFalse(any(line.startswith("verify_extraction ") or
                             line.startswith("refine_concepts ") for line in lines))
        self.assertTrue(any(line.startswith("consolidate_documents ") for line in lines))
        self.assertFalse((self.run_dir / ".pipeline_state" / "refine-concepts.done").exists())
        self.assertFalse((self.run_dir / "pass2_log.json").exists())

    def test_refinement_can_be_skipped_then_resumed(self):
        failed = self.run_pipeline(fail_refinement=True)
        self.assertEqual(failed.returncode, 73, failed.stdout + failed.stderr)
        skipped = self.run_pipeline("--skip-refinement", fail_refinement=True)
        self.assertEqual(skipped.returncode, 85, skipped.stdout + skipped.stderr)
        self.assertFalse((self.run_dir / ".pipeline_state" / "refine-concepts.failed").exists())
        self.assertFalse((self.run_dir / ".pipeline_state" / "refine-concepts.done").exists())
        self.assertFalse((self.run_dir / ".pipeline_state" / "verify-pass2.done").exists())
        self.assertTrue(any("--phase 1" in line for line in self.calls()))
        self.assertFalse(any("--phase 2" in line for line in self.calls()))
        resumed = self.run_pipeline()
        self.assertEqual(resumed.returncode, 85, resumed.stdout + resumed.stderr)
        self.assertTrue((self.run_dir / ".pipeline_state" / "refine-concepts.done").exists())
        self.assertTrue((self.run_dir / ".pipeline_state" / "verify-pass2.done").exists())


if __name__ == "__main__":
    unittest.main()
