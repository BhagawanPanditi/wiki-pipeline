"""Bound global fuzzy prompts and allow an explicit exact-only global pass."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import networkx as nx
from openai import BadRequestError

from consolidate_global import (MAX_FUZZY_BATCH_CHARS, MAX_FUZZY_BATCH_CONCEPTS,
                                MergeCluster, balanced_cut_clusters,
                                build_fuzzy_candidate_clusters, format_fuzzy_review_batch,
                                global_consolidate, make_fuzzy_review_batches,
                                verify_cluster_llm)


def prompt_titles(kwargs):
    """Extract numbered titles from the sent *plain-text* user message."""
    return re.findall(r"(?m)^\d+\. (Topic \d+)", kwargs["messages"][1]["content"])


def client_for(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def response(clusters, reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=reason,
        message=SimpleNamespace(content=json.dumps({"merge_clusters": clusters})))])


class WeightedCutsTests(unittest.TestCase):
    def test_balanced_cut_prefers_weak_bridge_over_strong_clique_links(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(48))
        for start in (0, 24):
            for i in range(start, start + 24):
                for j in range(i + 1, start + 24):
                    graph.add_edge(i, j, weight=1.0)
        graph.add_edge(23, 24, weight=0.60)
        parts = balanced_cut_clusters(graph, set(graph))
        self.assertEqual(parts, [set(range(24)), set(range(24, 48))])
        self.assertTrue(all(1 < len(part) < 30 for part in parts))

    def test_prompt_budget_can_force_further_balanced_graph_cuts(self):
        graph = nx.path_graph(16)
        nx.set_edge_attributes(graph, 0.75, "weight")
        parts = balanced_cut_clusters(graph, set(graph), fits_prompt=lambda nodes: len(nodes) <= 8)
        self.assertEqual([len(part) for part in parts], [8, 8])

    def test_all_review_groups_under_30_and_metrics_remain_separate(self):
        connected = [{"title": f"Alpha Beta Concept {i}", "aliases": []}
                     for i in range(61)]
        groups = build_fuzzy_candidate_clusters(connected, "word")
        self.assertTrue(groups)
        self.assertTrue(all(1 < len(group) < 30 for group in groups))
        # Word containment does not connect titles that only look alike in
        # spelling; those are candidates of the later character-only pass.
        concepts = [{"title": title, "aliases": []} for title in
                    ("Alpha Beta Gamma", "Alpha Beta Gamma Process", "Alphaa Betaa Gammaa")]
        word = build_fuzzy_candidate_clusters(concepts, "word")
        self.assertEqual([[c["title"] for c in group] for group in word],
                         [["Alpha Beta Gamma", "Alpha Beta Gamma Process"]])
        char = build_fuzzy_candidate_clusters(concepts, "character")
        self.assertEqual(len(char), 1)
        self.assertEqual(len(char[0]), 3)


class FuzzyBatchesTests(unittest.TestCase):
    def test_oversized_cluster_is_split_and_evidence_is_bounded(self):
        concepts = [{"title": f"Topic {i:03}", "aliases": [f"Alias {n}" for n in range(100)],
                     "description": "word " * 1000} for i in reversed(range(96))]
        batches = make_fuzzy_review_batches(concepts)
        self.assertGreater(len(batches), 1)
        self.assertEqual([c["title"] for part in batches for c in part],
                         sorted(c["title"] for c in concepts))
        for part in batches:
            self.assertLessEqual(len(part), MAX_FUZZY_BATCH_CONCEPTS)
            self.assertLessEqual(len(format_fuzzy_review_batch(part)),
                                 MAX_FUZZY_BATCH_CHARS)
            self.assertEqual(len(part[0]["aliases"]), 8)
            self.assertLessEqual(len(part[0]["description_hint"]), 400)

    def test_numbered_text_not_json_is_sent_with_two_line_evidence(self):
        prompt = format_fuzzy_review_batch([{
            "title": "Long-Distance Neuronal Migration",
            "aliases": ["Neuronal Migration in Adult Brain", "Neuron movement"],
            "description_hint": "The movement of neurons in the adult brain.",
        }])
        self.assertIn("Return only the numbered IDs", prompt)
        self.assertIn("1. Long-Distance Neuronal Migration "
                      "(aliases: Neuronal Migration in Adult Brain; Neuron movement)\n"
                      "   The movement of neurons in the adult brain.", prompt)
        self.assertNotIn('"title":', prompt)
        self.assertNotIn('"aliases":', prompt)

    def test_no_concept_is_discarded_when_single_entry_is_oversized(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            make_fuzzy_review_batches([{"title": "Topic", "aliases": ["x" * 21000]}])

    def test_word_merges_feed_second_character_review_not_vice_versa(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            document = run / "consolidated" / "canon_doc"
            document.mkdir(parents=True)
            titles = ("Alpha Beta Gamma", "Alpha Beta Gamma Process",
                      "Alphaa Betaa Gammaa")
            (document / "concepts.json").write_text(json.dumps({"concepts": [
                {"title": title, "aliases": [], "description": title,
                 "source_chunks": [f"doc::chunk{i}"], "entities": [],
                 "outbound_relations": []} for i, title in enumerate(titles)
            ]}))
            seen = []

            async def fake_review(clusters, args):
                seen.append([[c["title"] for c in group] for group in clusters])
                if len(seen) == 1:
                    self.assertEqual(seen[0], [[titles[0], titles[1]]])
                    return [MergeCluster(primary_title=titles[0],
                                         merged_titles=[titles[1]])], []
                self.assertEqual(len(seen), 2)
                self.assertEqual(len(clusters), 1)
                self.assertEqual({c["title"] for c in clusters[0]}, {titles[0], titles[2]})
                self.assertIn(titles[1], next(c["aliases"] for c in clusters[0]
                                             if c["title"] == titles[0]))
                return [MergeCluster(primary_title=titles[0],
                                     merged_titles=[titles[2]])], []

            with patch("consolidate_global.run_llm_fuzzy_pass", side_effect=fake_review):
                global_consolidate(run, SimpleNamespace(skip_fuzzy_review=False))
            self.assertEqual(len(seen), 2)
            result = json.loads((run / "global_concepts.json").read_text())
            self.assertEqual(len(result["concepts"]), 1)
            audit = json.loads((run / "global_consolidation.json").read_text())
            self.assertEqual(audit["global_audit_summary"]["fuzzy_passes"], {
                "word": {"clusters": 1, "merges": 1},
                "character": {"clusters": 1, "merges": 1}})

    def test_exact_only_skips_fuzzy_and_still_writes_complete_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            document = run / "consolidated" / "canon_doc"
            document.mkdir(parents=True)
            (document / "concepts.json").write_text(json.dumps({"concepts": [
                {"title": "Same Topic", "aliases": [], "description": "First fact.",
                 "source_chunks": ["doc::chunk1"], "entities": [], "outbound_relations": []},
                {"title": "Same Topic", "aliases": [], "description": "Second fact.",
                 "source_chunks": ["doc::chunk2"], "entities": [], "outbound_relations": []},
            ]}))
            with patch("consolidate_global.build_fuzzy_candidate_clusters",
                       side_effect=AssertionError("should not build fuzzy candidates")), \
                 patch("consolidate_global.asyncio.run",
                       side_effect=AssertionError("should not call the LLM")):
                global_consolidate(run, SimpleNamespace(skip_fuzzy_review=True))
            result = json.loads((run / "global_concepts.json").read_text())
            self.assertEqual(len(result["concepts"]), 1)
            self.assertEqual(result["concepts"][0]["source_chunks"],
                             ["doc::chunk1", "doc::chunk2"])
            audit = json.loads((run / "global_consolidation.json").read_text())
            self.assertTrue(audit["global_audit_summary"]["fuzzy_review_skipped"])
            self.assertEqual(audit["global_audit_summary"]["llm_fuzzy_merges_applied"], 0)


class FuzzyRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_ids_are_ignored_without_losing_valid_merges(self):
        async def create(**kwargs):
            return response([{"primary_id": 1, "merged_ids": [2, 999]},
                             {"primary_id": 77, "merged_ids": [1]},
                             {"primary_id": 1, "merged_ids": [2]}])

        cluster = [{"title": "Topic   1", "aliases": [], "description": "Hint."},
                   {"title": "Topic 2", "aliases": [], "description": "Hint."}]
        with patch("builtins.print") as warning:
            result = await verify_cluster_llm(client_for(create), cluster,
                                              SimpleNamespace(model_name="test", max_tokens=8192))
        self.assertTrue(warning.called)
        self.assertEqual([m.model_dump() for m in result.merge_clusters], [
            {"primary_title": "Topic   1", "merged_titles": ["Topic 2"]}])

    async def test_splits_400_context_error_instead_of_failing_entire_cluster(self):
        seen = []

        async def create(**kwargs):
            batch = prompt_titles(kwargs)
            seen.append(len(batch))
            if len(batch) > 8:
                resp = httpx.Response(400, request=httpx.Request("POST", "https://example.com/v1"))
                raise BadRequestError("maximum context length exceeded (input_tokens)",
                                      response=resp, body=None)
            return response([])

        cluster = [{"title": f"Topic {i:03}", "aliases": [], "description": "Hint."}
                   for i in range(24)]
        result = await verify_cluster_llm(client_for(create), cluster,
                                          SimpleNamespace(model_name="test", max_tokens=8192))
        self.assertEqual(result.merge_clusters, [])
        self.assertEqual(seen.count(24), 1)
        self.assertEqual(seen.count(6), 4)

    async def test_split_truncated_generation_and_combine_results(self):
        async def create(**kwargs):
            batch = prompt_titles(kwargs)
            if len(batch) > 2:
                return response([], reason="length")
            return response([{"primary_id": 1, "merged_ids": [len(batch)]}])

        cluster = [{"title": f"Topic {i}", "aliases": [], "description": "Hint."}
                   for i in range(8)]
        result = await verify_cluster_llm(client_for(create), cluster,
                                          SimpleNamespace(model_name="test", max_tokens=8192))
        self.assertEqual(len(result.merge_clusters), 4)


if __name__ == "__main__":
    unittest.main()
