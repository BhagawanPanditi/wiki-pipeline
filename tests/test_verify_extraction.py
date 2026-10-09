"""Regression tests for bounded, schema-valid verification responses."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from utils.checkpoint import CheckpointLog
from verify_extraction import (MAX_MISSING_CONCEPTS, MAX_MISSING_FACTS,
                               WikiVerifierPipeline)


GOOD_RESULT = {
    "coverage_score": 0.9,
    "is_sufficient": True,
    "missing_concepts": [],
    "missing_facts": ["A substantive fact is absent."],
    "hallucinations": [],
    "critique": "The main themes are covered, but one detail was missed.",
}


class VerificationParsingTests(unittest.TestCase):
    def setUp(self):
        self.verifier = WikiVerifierPipeline.__new__(WikiVerifierPipeline)
        self.verifier.debug = MagicMock()

    def test_valid_report(self):
        result = self.verifier.parse_verification(json.dumps(GOOD_RESULT), "chunk")
        self.assertEqual(result.missing_facts, GOOD_RESULT["missing_facts"])

    def test_unterminated_json_is_logged_and_retried(self):
        with self.assertRaises(json.JSONDecodeError):
            self.verifier.parse_verification('{"missing_facts": ["unfinished', "chunk")
        self.verifier.debug.record.assert_called_once_with(
            "parse_failure", key="chunk", raw_response='{"missing_facts": ["unfinished'
        )

    def test_missing_critique_is_not_fabricated(self):
        response = {k: v for k, v in GOOD_RESULT.items() if k != "critique"}
        with self.assertRaises(ValidationError):
            self.verifier.parse_verification(json.dumps(response), "chunk")
        self.verifier.debug.record.assert_called_once()

    def test_re_extraction_is_not_fabricated_as_success(self):
        with self.assertRaises(ValidationError):
            self.verifier.parse_verification('{"concepts": []}', "chunk")

    def test_repetitions_and_overlong_reports_are_rejected(self):
        for facts in (["same", "same"], [str(n) for n in range(MAX_MISSING_FACTS + 1)]):
            with self.subTest(facts=facts):
                response = {**GOOD_RESULT, "missing_facts": facts}
                with self.assertRaises(ValidationError):
                    self.verifier.parse_verification(json.dumps(response), "chunk")

    def test_sufficiency_cannot_contradict_score_or_missing_primary_concepts(self):
        for changes in ({"coverage_score": 0.5},
                        {"missing_concepts": ["Missing theme"]}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    self.verifier.parse_verification(json.dumps({**GOOD_RESULT, **changes}), "chunk")
        self.assertGreater(MAX_MISSING_CONCEPTS, 0)

    def test_resume_reaudits_checkpointed_invalid_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = "doc::chunk"
            chunk_dir = root / "staging" / "doc" / "chunk"
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "concepts.json").write_text('{"concepts": []}', encoding="utf-8")
            result_file = chunk_dir / "verification_pass2.json"
            bad_result = {**GOOD_RESULT, "missing_concepts": ["Missing theme"]}
            result_file.write_text(json.dumps(bad_result), encoding="utf-8")
            self.verifier.cache_dir = root / "cache"
            self.verifier.staging_dir = root / "staging"
            self.verifier.result_filename = "verification_pass2.json"
            self.verifier.checkpoint = CheckpointLog(root / "checkpoint.json")
            self.verifier.checkpoint.mark(key, True)
            with patch("verify_extraction.load_cached_chunks", return_value={
                key: {"current_chunk": "Example text"}
            }):
                items = self.verifier.prepare_verification_items()
            self.assertEqual(len(items), 1)
            self.assertFalse(self.verifier.checkpoint.is_done(key))


class VerificationResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_length_finish_reason_rejected_even_if_json_is_complete(self):
        verifier = WikiVerifierPipeline.__new__(WikiVerifierPipeline)
        verifier.settings = SimpleNamespace(model_name="test", max_tokens=16384)
        verifier.debug = MagicMock()
        text = json.dumps(GOOD_RESULT)
        choice = SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=text))

        async def create(**kwargs):
            self.assertEqual(kwargs["max_tokens"], 16384)
            return SimpleNamespace(choices=[choice])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as tmp:
            concepts_file = Path(tmp) / "concepts.json"
            concepts_file.write_text('{"concepts": []}', encoding="utf-8")
            item = {"key": "doc::chunk", "current_chunk": "Example text",
                    "concepts_file": concepts_file}
            with self.assertRaisesRegex(ValueError, "hit max_tokens limit"):
                await verifier.verify_chunk(client, item, 0)
        verifier.debug.record.assert_called_once_with(
            "truncated_response", key="doc::chunk", max_tokens=16384, raw_response=text
        )


if __name__ == "__main__":
    unittest.main()
