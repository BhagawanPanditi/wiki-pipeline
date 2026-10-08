"""End-to-end tests for the graph stages refactored onto ServerPool.

Covers stub_resolution.py (nested concurrency: stubs across servers, each stub's
chunks fanned out 5 at a time), thin_concept_handler.py (order-dependent folds)
and audit_noise_islands.py (networkx islands, per-island verdicts).
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
# Must exceed --min_concept_words (30) so these concepts are NOT thin.
LONG = ("This host concept is described in enough detail to exceed the minimum word "
        "threshold comfortably, covering the operating mechanism, the tuning parameters "
        "and the measured outcomes across several sentences of dense factual prose, and "
        "it reports a sustained throughput of 1240 requests per second at 61 ms latency.")


from tests.helpers import run_script as run


def base_args(base, servers, **extra):
    """Build the common arguments for a graph-stage test.

    Args:
        base: Run directory. Its parent becomes ``--wiki_base_dir``.
        servers: Fake server URLs.
        **extra: ``flag=value`` pairs appended as ``--flag value``.

    Returns:
        A list of CLI argument strings."""
    out = ["--wiki_base_dir", str(base.parent), "--servers", *servers,
           "--stage_budget_minutes", "2"]
    for k, v in extra.items():
        out.append(f"--{k}")
        if v != "":
            out.append(str(v))
    return out


class StubResolutionTests(unittest.TestCase):
    SCRIPT = "stub_resolution.py"

    def _build(self, root: Path, stubs=3, chunks_each=3):
        """Write a fixture graph in which several stubs are each cited by several chunks.

        Args:
            root: Temporary directory to build in.
            stubs: How many stubs to create.
            chunks_each: How many chunks cite each stub.

        Returns:
            The run directory."""
        base = root / "wiki" / "biology"
        (base / "cache").mkdir(parents=True, exist_ok=True)
        (base / "staging").mkdir(parents=True, exist_ok=True)

        cache, unresolved = [], []
        for i in range(stubs):
            doc = f"doc{i}"
            scs = []
            for j in range(chunks_each):
                cid = f"chunk_{j:03d}"
                scs.append(f"{doc}::{cid}")
                cache.append({"doc_id": doc, "chunk_id": cid,
                              "current_chunk": f"Text about stub {i} in chunk {j}.",
                              "context_before": "", "context_after": ""})
            unresolved.append({"stub_name": f"Stub {i}", "source_chunks": scs})

        (base / "cache" / "chunks_cache.json").write_text(json.dumps(cache), encoding="utf-8")
        (base / "global_concepts.json").write_text(json.dumps({"concepts": [
            {"title": "Parent Concept", "aliases": [], "description": LONG,
             "entities": [], "source_chunks": ["doc0::chunk_000"],
             "outbound_relations": [{"target_concept": "Stub 0", "relation": "mentions"}]}
        ]}), encoding="utf-8")
        (base / "global_consolidation.json").write_text(
            json.dumps({"unresolved_stubs": unresolved}), encoding="utf-8")
        return base

    def test_major_stubs_resolve_and_spread_across_servers(self):
        """Major stubs are defined by the LLM, spread evenly across servers, and recorded in the decision log."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp))
            responder = lambda p, i: {"is_valid_concept": True, "title": f"Resolved {i}",
                                      "description": LONG, "entities": ["e"]}
            with FakeVLLMCluster(3, responder=responder) as cluster:
                out = run(self.SCRIPT, ["--domain", "biology"] +
                          base_args(base, cluster.urls, max_attempts=2))
                # 3 stubs, one per server; each stub fans out over its 3 chunks.
                cluster.assert_balanced(self, expected_total=9)
                self.assertEqual(cluster.counts, [3, 3, 3])
            self.assertIn("Requests served per server", out.stdout)
            outfile = base / "global_concepts_stubs_resolved.json"
            self.assertTrue(outfile.is_file())
            concepts = json.loads(outfile.read_text())["concepts"]
            titles = [c["title"] for c in concepts]
            # Resolved stubs become real concepts, keeping the stub name as title.
            for i in range(3):
                self.assertIn(f"Stub {i}", titles)
            resolved = next(c for c in concepts if c["title"] == "Stub 0")
            self.assertIn("This host concept is described in enough detail",
                          resolved["description"])
            # They are no longer listed as unresolved.
            log = json.loads((base / "global_consolidation.json").read_text())
            self.assertIn("unresolved_stubs", log)
            # The decision log answers "what were the stubs, and what did each become".
            records = [json.loads(line) for line in
                       (base / "log" / "stub-resolution.decisions.jsonl").read_text().splitlines()]
            detected = next(r for r in records if r["event"] == "stubs_detected")
            self.assertEqual(sorted(s["stub"] for s in detected["major"]),
                             ["Stub 0", "Stub 1", "Stub 2"])
            synthesized = [r for r in records if r["event"] == "stub_synthesized"]
            self.assertEqual(sorted(r["title"] for r in synthesized),
                             ["Stub 0", "Stub 1", "Stub 2"])
            self.assertTrue(all(r["source_chunks"] for r in synthesized))

    def test_invalid_concept_downgrades_to_minor_stub(self):
        """An invalid LLM answer downgrades a stub to a minor stub rather than failing the run."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), stubs=3)
            with FakeVLLMCluster(3, responder=lambda p, i: {
                    "is_valid_concept": False, "title": "x", "description": "y"}) as cluster:
                out = run(self.SCRIPT, ["--domain", "biology"] +
                          base_args(base, cluster.urls, max_attempts=1))
            self.assertIn("Absorbing", out.stdout)
            self.assertTrue((base / "global_concepts_stubs_resolved.json").is_file())


class ThinConceptTests(unittest.TestCase):
    SCRIPT = "thin_concept_handler.py"

    def _build(self, root: Path, pairs=6):
        """Write a fixture graph of thin concepts, each paired with a host.

        Args:
            root: Temporary directory to build in.
            pairs: How many thin-concept and host pairs to create.

        Returns:
            The run directory."""
        base = root / "wiki" / "biology"
        base.mkdir(parents=True, exist_ok=True)
        concepts = []
        for i in range(pairs):
            concepts.append({"title": f"Host {i}", "aliases": [], "description": LONG,
                             "entities": [], "source_chunks": [f"doc0::chunk_{i:03d}"],
                             "outbound_relations": [{"target_concept": f"Thin {i}",
                                                     "relation": "describes"}]})
            concepts.append({"title": f"Thin {i}", "aliases": [f"T{i}"],
                             "description": "Too short.", "entities": [],
                             "source_chunks": [f"doc0::chunk_{i:03d}"],
                             "outbound_relations": []})
        (base / "global_concepts_stubs_resolved.json").write_text(
            json.dumps({"concepts": concepts}), encoding="utf-8")
        return base

    def test_thin_concepts_fold_into_hosts_and_spread_evenly(self):
        """Thin concepts are folded into their hosts, with the folds spread evenly across servers."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), pairs=6)

            def responder(prompt, i):
                # Fold into whichever host was offered as candidate 1.
                """Answer a thin-concept prompt with a fold decision into the host it was offered.

                Args:
                    prompt: The prompt text sent by the stage.
                    i: Request index, unused.

                Returns:
                    A fold decision naming the first candidate host in the prompt."""
                host = [l for l in prompt.splitlines() if l.startswith("Title: Host")]
                return {"should_fold": True,
                        "target_host_title": host[0].split("Title: ", 1)[1],
                        "synthesized_description": LONG + " Merged facts: 94.2%."}

            with FakeVLLMCluster(3, responder=responder) as cluster:
                out = run(self.SCRIPT, ["--domain", "biology"] +
                          base_args(base, cluster.urls, max_attempts=2))
                cluster.assert_balanced(self, expected_total=6)
            self.assertIn("Identified 6 thin concepts", out.stdout)

            data = json.loads((base / "global_concepts_refined.json").read_text())
            hosts = [c for c in data["concepts"] if c["title"].startswith("Host")]
            self.assertTrue(all("Merged facts: 94.2%." in h["description"] for h in hosts),
                            [h["description"][:40] for h in hosts])
            # Folded thin concepts contribute their aliases/entities to the host.
            self.assertTrue(any("Thin" in e for h in hosts for e in h.get("entities", [])))

    def test_failed_fold_decisions_fall_through_to_deterministic_rules(self):
        """A fold the LLM fails to decide falls through to the deterministic rules."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp), pairs=3)
            with FakeVLLMCluster(3, responder=lambda p, i: "{not json") as cluster:
                out = run(self.SCRIPT, ["--domain", "biology"] +
                          base_args(base, cluster.urls, max_attempts=1), expect_rc=0)
            self.assertIn("got no fold decision", out.stdout)
            # The stage still completes and writes its output.
            self.assertTrue((base / "global_concepts_refined.json").is_file())


class IslandAuditTests(unittest.TestCase):
    SCRIPT = "audit_noise_islands.py"

    def _build(self, root: Path):
        """Write a fixture graph with a two-concept island and a single lone concept.

        Args:
            root: Temporary directory to build in.

        Returns:
            The run directory."""
        base = root / "wiki" / "biology"
        base.mkdir(parents=True, exist_ok=True)
        concepts = []
        # Mainland chain of 5 -> the giant component.
        for i in range(5):
            rel = [{"target_concept": f"Concept {i + 1}", "relation": "links"}] if i < 4 else []
            concepts.append({"title": f"Concept {i}", "aliases": [], "description": LONG,
                             "entities": [], "source_chunks": ["doc0::chunk_000"],
                             "outbound_relations": rel})
        # A 2-node island and a 1-node island, both from doc0 so they have siblings.
        concepts.append({"title": "Island A", "aliases": [], "description": "Short.",
                         "entities": [], "source_chunks": ["doc0::chunk_001"],
                         "outbound_relations": [{"target_concept": "Island B", "relation": "pairs"}]})
        concepts.append({"title": "Island B", "aliases": [], "description": "Short.",
                         "entities": [], "source_chunks": ["doc0::chunk_001"],
                         "outbound_relations": []})
        concepts.append({"title": "Lone Node", "aliases": [], "description": "Short.",
                         "entities": [], "source_chunks": ["doc0::chunk_002"],
                         "outbound_relations": []})
        (base / "global_concepts_refined.json").write_text(
            json.dumps({"concepts": concepts}), encoding="utf-8")
        return base

    def test_islands_audited_and_spread_evenly(self):
        """Small islands are audited and the audit requests are spread evenly across servers."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp))
            responder = lambda p, i: {"decision": "LINK", "target_sibling": "Concept 0",
                                      "relation_to_sibling": "relates to",
                                      "relation_from_sibling": "is related to"}
            with FakeVLLMCluster(3, responder=responder) as cluster:
                out = run(self.SCRIPT, ["--domain", "biology"] +
                          base_args(base, cluster.urls, max_attempts=2))
                # 2 small islands (the 2-node one and the lone node).
                cluster.assert_balanced(self, expected_total=2)
            self.assertIn("Reviewing and bridging 2 small islands", out.stdout)

            # Every verdict is a decision record; the old island_audit_log.jsonl is gone.
            self.assertFalse((base / "island_audit_log.jsonl").exists())
            records = [json.loads(line) for line in
                       (base / "log" / "island-audit.decisions.jsonl").read_text().splitlines()]
            verdicts = [r for r in records if r["event"] == "island_verdict"]
            self.assertEqual(len(verdicts), 2)
            for rec in verdicts:
                self.assertEqual(rec["decision"], "LINK")
                self.assertTrue(rec["applied"])
                self.assertEqual(rec["target_sibling"], "Concept 0")

            data = json.loads((base / "global_concepts_refined.json").read_text())
            titles = {c["title"] for c in data["concepts"]}
            self.assertIn("Lone Node", titles)  # LINK keeps the concept
            lone = next(c for c in data["concepts"] if c["title"] == "Lone Node")
            self.assertIn("Concept 0",
                          [r["target_concept"] for r in lone["outbound_relations"]])

    def test_prune_decision_drops_noise(self):
        """A PRUNE verdict removes the island from the graph."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp))
            with FakeVLLMCluster(2, responder=lambda p, i: {"decision": "PRUNE"}) as cluster:
                run(self.SCRIPT, ["--domain", "biology"] +
                    base_args(base, cluster.urls, max_attempts=1))
            titles = {c["title"] for c in
                      json.loads((base / "global_concepts_refined.json").read_text())["concepts"]}
            self.assertNotIn("Lone Node", titles)
            self.assertNotIn("Island A", titles)
            self.assertIn("Concept 0", titles)

    def test_unparseable_verdicts_fail_loudly(self):
        """Island verdicts the model cannot parse fail the stage."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build(Path(tmp))
            with FakeVLLMCluster(2, responder=lambda p, i: "not json") as cluster:
                proc = run(self.SCRIPT, ["--domain", "biology"] +
                           base_args(base, cluster.urls, max_attempts=1), expect_rc=1)
            self.assertIn("island", (proc.stdout + proc.stderr).lower())


if __name__ == "__main__":
    unittest.main()
