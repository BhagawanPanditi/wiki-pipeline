"""Skipping Pass 1 verification must not feed stale audit files into refinement."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from refine_concepts import WikiSecondPassPipeline


class RefinementFeedbackTests(unittest.TestCase):
    def test_ignore_audit_feedback_even_when_old_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = root / "staging" / "doc" / "chunk"
            chunk.mkdir(parents=True)
            (chunk / "concepts.json").write_text('{"concepts": []}')
            (chunk / "verification.json").write_text(json.dumps({
                "is_sufficient": False, "missing_concepts": ["Old feedback"],
                "missing_facts": ["Old fact"], "critique": "Old audit"}))
            pipeline = WikiSecondPassPipeline.__new__(WikiSecondPassPipeline)
            pipeline.cache_dir = root / "cache"
            pipeline.staging_dir = root / "staging"
            chunks = {"doc::chunk": {"current_chunk": "example"}}
            with patch("refine_concepts.load_cached_chunks", return_value=chunks):
                pipeline.args = SimpleNamespace(ignore_audit_feedback=True)
                self.assertIsNone(pipeline.prepare_pass2_items()[0]["audit_feedback"])
                pipeline.args = SimpleNamespace(ignore_audit_feedback=False)
                self.assertEqual(pipeline.prepare_pass2_items()[0]["audit_feedback"]["missing_facts"],
                                 ["Old fact"])


if __name__ == "__main__":
    unittest.main()
