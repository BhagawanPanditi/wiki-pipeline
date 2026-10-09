"""Large-document merge planning stays within context and checkpoint boundaries."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from openai import BadRequestError

from consolidate_documents import (MAX_MERGE_BATCH_CHARS, MAX_MERGE_BATCH_CONCEPTS,
                                   DocumentConsolidatorPipeline, make_merge_batches)
from utils.attempts import AttemptHistory


def client_for(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def response(clusters, *, finish_reason="stop"):
    content = json.dumps({"merge_clusters": clusters})
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=content))])


def pipeline():
    obj = DocumentConsolidatorPipeline.__new__(DocumentConsolidatorPipeline)
    obj.debug = MagicMock()
    obj.attempts = AttemptHistory()
    obj.settings = SimpleNamespace(model_name="test", max_tokens=16384)
    return obj


class MergeBatchTests(unittest.TestCase):
    def test_packing_bounds_and_preserves_every_concept(self):
        concepts = [{"title": f"Concept {i:03}", "description": "detail " * 200}
                    for i in reversed(range(175))]
        batches = make_merge_batches(concepts)
        self.assertGreater(len(batches), 1)
        self.assertEqual([c["title"] for batch in batches for c in batch],
                         sorted(c["title"] for c in concepts))
        for batch in batches:
            self.assertLessEqual(len(batch), MAX_MERGE_BATCH_CONCEPTS)
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False,
                                                separators=(",", ":"))), MAX_MERGE_BATCH_CHARS)

    def test_one_unfit_concept_is_not_dropped(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            make_merge_batches([{"title": "topic", "description": "x" * MAX_MERGE_BATCH_CHARS}])


class MergeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_batched_results_combined_for_one_document(self):
        obj = pipeline()
        raw = [{"title": f"Topic {i:03}", "description": "detail", "aliases": [],
                "outbound_relations": []} for i in reversed(range(170))]
        requests = []

        async def create(**kwargs):
            batch = json.loads(kwargs["messages"][1]["content"].split("CONCEPT LIST (only merge titles present in this list):\n")[1])
            requests.append(batch)
            self.assertLessEqual(len(batch), MAX_MERGE_BATCH_CONCEPTS)
            return response([{"primary_title": batch[0]["title"],
                              "merged_titles": [batch[1]["title"]]}])

        result = await obj.get_merge_clusters(client_for(create),
                                               {"doc_id": "doc", "raw_concepts": raw}, 0)
        self.assertEqual(len(result.merge_clusters), len(requests))
        self.assertGreater(len(requests), 1)
        self.assertEqual(len(obj.execute_programmatic_merge(raw, result)),
                         len(raw) - len(requests))

    async def test_server_context_rejection_splits_batch_without_losing_results(self):
        obj = pipeline()
        raw = [{"title": f"Topic {i:03}", "description": "detail",
                "outbound_relations": []} for i in range(24)]
        requests = []

        async def create(**kwargs):
            batch = json.loads(kwargs["messages"][1]["content"].split("CONCEPT LIST (only merge titles present in this list):\n")[1])
            requests.append(len(batch))
            if len(batch) > 8:
                req = httpx.Request("POST", "https://example.com/v1/chat/completions")
                resp = httpx.Response(400, request=req)
                raise BadRequestError("maximum context length exceeded (input_tokens)",
                                      response=resp, body=None)
            return response([])

        result = await obj.get_merge_clusters(client_for(create),
                                               {"doc_id": "doc", "raw_concepts": raw}, 0)
        self.assertEqual(result.merge_clusters, [])
        self.assertEqual(requests.count(24), 1)
        self.assertEqual(requests.count(6), 4)
        self.assertTrue(any(c.args[0] == "split_merge_batch"
                            for c in obj.debug.record.call_args_list))

    async def test_single_item_overflow_fails_not_silently_skipped(self):
        obj = pipeline()
        raw = [{"title": "Topic", "description": "detail", "outbound_relations": []}]

        async def create(**kwargs):
            return response([], finish_reason="length")

        with self.assertRaisesRegex(ValueError, "truncated"):
            await obj.get_merge_clusters(client_for(create),
                                         {"doc_id": "doc", "raw_concepts": raw}, 0)
        self.assertEqual(len(obj.attempts.get("doc")), 1)


if __name__ == "__main__":
    unittest.main()
