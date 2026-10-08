"""Tests for the per-stage decision logs under <run>/log/.

The LLM call log says what the model was asked and answered. A decision log says
what the stage then did with that answer, so a reader can answer questions like
"which concepts were merged, into what, and why" without re-deriving it.

These run the real stage scripts as subprocesses against fixture graphs. None of
them need an LLM server: each one exercises a deterministic path, which is where
the graph-changing decisions are recorded.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from pipeline_lib.layout import LOG_DIR_NAME, log_path, run_dir_for

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG = ("A dense factual description with enough words to look like a real concept "
        "brief including measured throughput of 1240 requests per second at 61 ms.")


from tests.helpers import run_script as run


def read_events(path: Path):
    """Read a JSONL decision log into a list of records.

    Args:
        path: The ``.decisions.jsonl`` file.

    Returns:
        One dict per line, in file order.

    Raises:
        AssertionError: If the file is missing, so an absent log fails loudly.
    """
    assert path.is_file(), f"expected a decision log at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def concept(title, **extra):
    """Build one refined-graph concept with the fields the stages read.

    Args:
        title: Concept title.
        **extra: Overrides for any default field (aliases, entities, relations...).

    Returns:
        A concept dict.
    """
    base = {"title": title, "aliases": [], "description": LONG, "entities": [],
            "source_chunks": ["canon_a::chunk_000"], "outbound_relations": []}
    base.update(extra)
    return base


class LayoutTests(unittest.TestCase):
    def test_logs_live_in_one_folder_named_by_stage_and_kind(self):
        """Every log path is <run>/log/<stage>.<kind>.jsonl."""
        run_dir = Path("/wiki/biology")
        self.assertEqual(LOG_DIR_NAME, "log")
        self.assertEqual(log_path(run_dir, "stub-resolution", "decisions"),
                         Path("/wiki/biology/log/stub-resolution.decisions.jsonl"))

    def test_run_dir_uses_the_domain_or_default(self):
        """The run folder is <wiki_base_dir>/<domain>, with 'default' for no domain."""
        args = Namespace(wiki_base_dir="/wiki", domain="default")
        self.assertEqual(run_dir_for(args), Path("/wiki/default"))


class DecisionLogTests(unittest.TestCase):
    def test_entity_links_are_recorded_and_dry_run_records_nothing(self):
        """Links written to the graph are logged; a dry run writes no log at all."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "wiki"
            domain = root / "biology"
            domain.mkdir(parents=True)
            shared = ["Photosystem II", "Cytochrome b6f"]
            (domain / "global_concepts_refined.json").write_text(json.dumps({"concepts": [
                concept("Alpha Process", entities=shared),
                concept("Beta Process", entities=shared),
                concept("Gamma Unrelated", entities=["Other"]),
            ]}), encoding="utf-8")

            run("link_entity_overlap.py", ["--domain", "biology", "--wiki_base_dir", str(root),
                                           "--dry_run"])
            self.assertFalse(log_path(domain, "entity-overlap", "decisions").exists(),
                             "a dry run must not record decisions")

            run("link_entity_overlap.py", ["--domain", "biology", "--wiki_base_dir", str(root)])
            events = read_events(log_path(domain, "entity-overlap", "decisions"))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "entity_link_added")
            self.assertEqual({events[0]["a"], events[0]["b"]}, {"Alpha Process", "Beta Process"})
            # Entities are matched after normalisation (lowercased), and the record
            # shows the normalised form that actually matched.
            self.assertEqual(sorted(events[0]["shared_entities"]),
                             sorted(e.lower() for e in shared))

    def test_reverse_links_record_the_concept_they_were_added_to(self):
        """A one-way relation gains its back-link, and the log names both ends."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "wiki"
            domain = root / "biology"
            domain.mkdir(parents=True)
            (domain / "global_concepts_refined.json").write_text(json.dumps({"concepts": [
                concept("Alpha Process", outbound_relations=[
                    {"target_concept": "Beta Process", "relation": "feeds into",
                     "source_chunks": []}]),
                concept("Beta Process"),
            ]}), encoding="utf-8")

            run("synthesize_bidirectional_relations.py",
                ["--domain", "biology", "--wiki_base_dir", str(root), "--fast"])

            refined = json.loads((domain / "global_concepts_refined.json").read_text())
            beta = next(c for c in refined["concepts"] if c["title"] == "Beta Process")
            self.assertEqual([r["target_concept"] for r in beta["outbound_relations"]],
                             ["Alpha Process"])
            events = read_events(log_path(domain, "reverse-relations", "decisions"))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["method"], "template")
            self.assertEqual((events[0]["source"], events[0]["target"]),
                             ("Beta Process", "Alpha Process"))

    def test_exact_merges_name_the_canonical_title_and_what_was_absorbed(self):
        """Two documents describing one concept produce a merge record naming both."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "wiki"
            consolidated = root / "biology" / "consolidated"
            # "Folding of Proteins" carries "Protein Folding" as an alias, which is
            # what links the two into one group.
            for doc, title, aliases in [("canon_a", "Protein Folding", []),
                                        ("canon_b", "Folding of Proteins",
                                         ["Protein Folding"])]:
                (consolidated / doc).mkdir(parents=True)
                (consolidated / doc / "concepts.json").write_text(json.dumps({
                    "doc_id": doc,
                    "concepts": [concept(title, aliases=aliases,
                                         source_chunks=[f"{doc}::chunk_000"])]}),
                    encoding="utf-8")

            run("global_concept_consolidation.py",
                ["--domain", "biology", "--wiki_base_dir", str(root),
                 "--servers", "http://127.0.0.1:9/v1"])

            events = read_events(log_path(root / "biology", "global-consolidation",
                                          "decisions"))
            merges = [e for e in events if e["event"] == "exact_merge"]
            self.assertEqual(len(merges), 1, events)
            # Which title wins is a tie-break, so assert on the set of names instead.
            names = {merges[0]["canonical"], *merges[0]["merged"]}
            self.assertEqual(names, {"Protein Folding", "Folding of Proteins"})
            self.assertEqual(merges[0]["members"], 2)


if __name__ == "__main__":
    unittest.main()
