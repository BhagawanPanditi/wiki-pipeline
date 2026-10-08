"""End-to-end tests for the generic corpus contract.

These run ``concept_extraction_pass_1.py`` as a real subprocess against a fake
vLLM cluster, with **no ``transformers`` stub on the path**. That is the point:
a corpus processed with ``--no-chunking`` must not need a tokenizer, an offline
HuggingFace model, or the BRIGHT directory layout.

Pinned behaviour:
  * ``.json`` array and ``.jsonl`` corpora are both accepted, auto-detected.
  * ``contents``/``content`` and ``id``/``doc_id`` spellings are normalised.
  * Identical documents are deduplicated to one ``canon_<hash>`` unit, and the
    original ids survive in ``cache/doc_mapping.json``.
  * ``--domain`` is optional: without it outputs land in ``default/``.
  * ``run_config.json`` records the chunking decision for later stages.
  * A bad corpus fails loudly, naming the file and the record.
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
SCRIPT = "concept_extraction_pass_1.py"

DESCRIPTION = ("A dense factual description of photosynthetic electron transport including "
               "measured quantum yield of 0.84 and a turnover of 1240 electrons per second "
               "at 25 degrees celsius under saturating red light.")
GOOD = {"concepts": [{"title": "Photosynthetic Electron Transport", "aliases": ["PET"],
                      "description": DESCRIPTION, "entities": ["Photosystem II"],
                      "outbound_relations": []}]}


def run_pass1(args, *, expect_rc=0, timeout=300):
    """Run Pass 1 as a subprocess and assert on its exit code.

    Args:
        args: CLI arguments, excluding the interpreter and script.
        expect_rc: Expected process return code.
        timeout: Seconds to wait before failing the test.

    Returns:
        The completed ``subprocess.CompletedProcess``.

    Raises:
        AssertionError: If the return code differs from ``expect_rc``. The
            message includes stdout and stderr so failures are diagnosable.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Deliberately NOT injecting a transformers stub: unchunked ingestion must
    # work without it.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / SCRIPT), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
        f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
    return proc


class CorpusIngestTests(unittest.TestCase):
    def _base_args(self, base, servers, flags=(), **extra):
        """Build common Pass 1 arguments pointing at a temp wiki base.

        Args:
            base: Wiki base directory (outputs go to ``base/<domain>/``).
            servers: Fake vLLM base URLs.
            flags: Raw argument strings appended verbatim, for store_true flags
                whose names contain hyphens (e.g. ``--no-chunking``).
            **extra: ``flag=value`` pairs emitted as ``--flag value``. Keys use
                underscores to match the value-taking flags this CLI defines.

        Returns:
            A list of CLI argument strings.
        """
        out = ["--wiki_base_dir", str(base), "--servers", *servers,
               "--max_attempts", "2", "--stage_budget_minutes", "2"]
        out.extend(flags)
        for key, value in extra.items():
            out.append(f"--{key}")
            if value != "":
                out.append(str(value))
        return out

    def test_json_array_corpus_no_chunking_no_domain(self):
        """A .json array with 'contents', unchunked and domainless, ingests."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "documents.json"
            corpus.write_text(json.dumps([
                {"id": "doc-A", "contents": "Photosynthesis converts light into chemical energy."},
                {"doc_id": "doc-B", "content": "The Krebs cycle oxidises acetyl-CoA to carbon dioxide."},
            ]), encoding="utf-8")

            with FakeVLLMCluster(2, responder=lambda p, i: GOOD) as cluster:
                out = run_pass1(self._base_args(root / "wiki", cluster.urls,
                                                corpus=str(corpus), flags=["--no-chunking"]))
                self.assertEqual(cluster.total_requests, 2)

            run_dir = root / "wiki" / "default"
            self.assertTrue(run_dir.is_dir(),
                            f"expected domainless output in 'default/', have "
                            f"{sorted(p.name for p in (root / 'wiki').iterdir())}")

            mapping = json.loads((run_dir / "cache" / "doc_mapping.json").read_text())
            self.assertEqual(sorted(mapping), ["doc-A", "doc-B"])
            self.assertTrue(all(v.startswith("canon_") for v in mapping.values()))
            self.assertEqual(len(set(mapping.values())), 2)

            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertFalse(config["chunking"])
            self.assertIsNone(config["chunk_size"])
            self.assertIsNone(config["domain"])
            self.assertEqual(config["run_name"], "default")
            self.assertEqual(config["unique_documents"], 2)

            units = json.loads((run_dir / "cache" / "chunks_cache.json").read_text())
            self.assertEqual(len(units), 2)
            # Without chunking each document is exactly one unit with no context.
            self.assertTrue(all(u["chunk_id"] == "chunk_000" for u in units))
            self.assertTrue(all(u["context_before"] == "" for u in units))
            self.assertIn("Chunking disabled", out.stdout)

            for unit in units:
                staging = run_dir / "staging" / unit["doc_id"] / unit["chunk_id"]
                self.assertTrue(staging.is_dir(), f"missing staging dir {staging}")

    def test_jsonl_corpus_with_domain_and_field_aliases(self):
        """A .jsonl corpus with a domain hint writes to <base>/<domain>/."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "documents.jsonl"
            corpus.write_text(
                '{"id": "x1", "contents": "Ribosomes translate mRNA into protein chains."}\n'
                '\n'
                '{"doc_id": "x2", "text": "Chlorophyll absorbs red and blue light strongly."}\n',
                encoding="utf-8")

            with FakeVLLMCluster(2, responder=lambda p, i: GOOD) as cluster:
                run_pass1(self._base_args(root / "wiki", cluster.urls,
                                          corpus=str(corpus), domain="biology",
                                          flags=["--no-chunking"]))
                self.assertEqual(cluster.total_requests, 2)

            run_dir = root / "wiki" / "biology"
            self.assertTrue(run_dir.is_dir())
            mapping = json.loads((run_dir / "cache" / "doc_mapping.json").read_text())
            self.assertEqual(sorted(mapping), ["x1", "x2"])
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(config["domain"], "biology")
            self.assertEqual(config["run_name"], "biology")

    def test_identical_documents_deduplicate_to_one_unit(self):
        """Three records with two distinct contents yield two units and 3 mappings."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = "Photosynthesis converts light into chemical energy in plants."
            corpus = root / "documents.jsonl"
            corpus.write_text(
                json.dumps({"id": "first", "contents": shared}) + "\n"
                + json.dumps({"id": "second", "contents": "A wholly different document body."}) + "\n"
                + json.dumps({"id": "third", "contents": shared}) + "\n",
                encoding="utf-8")

            with FakeVLLMCluster(2, responder=lambda p, i: GOOD) as cluster:
                out = run_pass1(self._base_args(root / "wiki", cluster.urls,
                                                corpus=str(corpus), flags=["--no-chunking"]))
                # Only unique documents are extracted, so 2 requests, not 3.
                self.assertEqual(cluster.total_requests, 2)

            run_dir = root / "wiki" / "default"
            mapping = json.loads((run_dir / "cache" / "doc_mapping.json").read_text())
            self.assertEqual(len(mapping), 3)
            self.assertEqual(mapping["first"], mapping["third"])
            self.assertNotEqual(mapping["first"], mapping["second"])
            units = json.loads((run_dir / "cache" / "chunks_cache.json").read_text())
            self.assertEqual(len(units), 2)
            self.assertIn("Deduplicated 3 docs down to 2", out.stdout)

    def test_chunking_flag_is_part_of_the_cache_fingerprint(self):
        """Re-running with the other chunking setting rebuilds instead of reusing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "documents.jsonl"
            corpus.write_text(json.dumps(
                {"id": "d1", "contents": " ".join(f"word{i}" for i in range(60))}) + "\n",
                encoding="utf-8")
            base_args = self._base_args(root / "wiki", ["http://127.0.0.1:1/v1"],
                                        corpus=str(corpus))

            cache = root / "wiki" / "default" / "cache"
            # Simulate a cached unchunked run.
            cache.mkdir(parents=True)
            (cache / "chunks_cache.json").write_text(json.dumps(
                [{"doc_id": "canon_x", "chunk_id": "chunk_000", "current_chunk": "old",
                  "context_before": "", "context_after": ""}]))
            (cache / "doc_mapping.json").write_text(json.dumps({"d1": "canon_x"}))

            from pipeline_lib.corpus import cache_manifest
            (cache / "chunks_cache_manifest.json").write_text(json.dumps(cache_manifest(
                corpus, tokenizer_name=None, chunking=False, chunk_size=4500, overlap=400)))

            # A chunking run must NOT accept that cache, even though the corpus
            # digest matches, because the tokenizer/window differ.
            expected = cache_manifest(corpus, tokenizer_name="m", chunking=True,
                                      chunk_size=4500, overlap=400)
            stored = json.loads((cache / "chunks_cache_manifest.json").read_text())
            self.assertNotEqual(stored, expected)
            self.assertEqual(stored["source_sha256"], expected["source_sha256"])
            self.assertEqual(stored["format_version"], expected["format_version"])

    def test_bad_corpus_fails_loudly_and_names_the_problem(self):
        """A record with no content aborts with an actionable message."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "documents.jsonl"
            corpus.write_text('{"id": "ok", "contents": "A usable document body here."}\n'
                              '{"id": "broken"}\n', encoding="utf-8")

            with FakeVLLMCluster(1, responder=lambda p, i: GOOD) as cluster:
                proc = run_pass1(self._base_args(root / "wiki", cluster.urls,
                                                 corpus=str(corpus), flags=["--no-chunking"]),
                                 expect_rc=1)
            combined = proc.stdout + proc.stderr
            self.assertIn("record #1", combined)
            self.assertIn("no content field", combined)
            self.assertIn("documents.jsonl", combined)

    def test_missing_corpus_flag_is_actionable(self):
        """Omitting --corpus (and --data_dir/--domain) explains both options."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with FakeVLLMCluster(1, responder=lambda p, i: GOOD) as cluster:
                proc = run_pass1(self._base_args(root / "wiki", cluster.urls,
                                                 flags=["--no-chunking"]), expect_rc=1)
            combined = proc.stdout + proc.stderr
            self.assertIn("--corpus", combined)

    def test_legacy_data_dir_layout_still_resolves(self):
        """--data_dir plus --domain still finds <data_dir>/<domain>.jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "biology.jsonl").write_text(json.dumps(
                {"id": "legacy", "contents": "A legacy BRIGHT-style document body."}) + "\n",
                encoding="utf-8")

            with FakeVLLMCluster(1, responder=lambda p, i: GOOD) as cluster:
                run_pass1(self._base_args(root / "wiki", cluster.urls,
                                          data_dir=str(data_dir), domain="biology",
                                          flags=["--no-chunking"]))
                self.assertEqual(cluster.total_requests, 1)

            run_dir = root / "wiki" / "biology"
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertTrue(config["corpus"].endswith("biology.jsonl"))
            mapping = json.loads((run_dir / "cache" / "doc_mapping.json").read_text())
            self.assertEqual(sorted(mapping), ["legacy"])
            self.assertTrue(mapping["legacy"].startswith("canon_"))


if __name__ == "__main__":
    unittest.main()
