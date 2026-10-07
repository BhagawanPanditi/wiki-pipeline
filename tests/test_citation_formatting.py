import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from citation_utils import clean_brief_text, clean_chunk_citations

# tqdm is only used for export progress; keep the formatter tests runnable in
# minimal environments where optional CLI dependencies have not been installed.
try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub

from export_markdown_wiki import export_wiki_markdown, format_description_paragraphs


class CitationFormattingTests(unittest.TestCase):
    def test_brief_drops_provenance_and_flattens_newlines(self):
        brief = (
            "A phenomenon described here. [canon_9531246a9168::chunk_001]\n\n"
            "A second sentence without a citation. [canon_f3a24c7a38b0]"
        )

        cleaned = clean_brief_text(brief)

        self.assertEqual(
            cleaned,
            "A phenomenon described here. A second sentence without a citation.",
        )
        self.assertNotIn("canon_", cleaned)
        self.assertNotIn("chunk_", cleaned)
        self.assertNotIn("\n", cleaned)

    def test_chunk_citations_reduce_to_document_ids(self):
        description = "First [canon_abc123::chunk_000]. Second [canon_def456::chunk_019]."

        self.assertEqual(
            clean_chunk_citations(description),
            "First [canon_abc123]. Second [canon_def456].",
        )

    def test_each_paragraph_gets_a_document_level_citation(self):
        description = (
            "First paragraph has an inline source. [canon_10d15b3efade::chunk_000]\n\n"
            "Second paragraph is missing an inline source.\n\n"
            "Additionally, a final note has a chunk-level source. "
            "[canon_714a6891eb7::chunk_001]"
        )
        source_docs = ["canon_10d15b3efade", "canon_714a6891eb7"]

        formatted = format_description_paragraphs(description, source_docs)
        paragraphs = formatted.split("\n\n")

        self.assertEqual(len(paragraphs), 3)
        self.assertTrue(paragraphs[0].endswith("[canon_10d15b3efade]"))
        self.assertTrue(paragraphs[1].endswith("[canon_10d15b3efade] [canon_714a6891eb7]"))
        self.assertTrue(paragraphs[2].endswith("[canon_714a6891eb7]"))
        self.assertNotIn("chunk_", formatted)

    def test_export_keeps_brief_on_one_line_and_cites_each_body_paragraph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            domain_dir = Path(temp_dir) / "biology"
            domain_dir.mkdir()
            refined_file = domain_dir / "global_concepts_refined.json"
            refined_file.write_text(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "title": "Example Concept",
                                "aliases": [],
                                "brief": (
                                    "A clean summary. [canon_abc123::chunk_001]\n\n"
                                    "Second sentence."
                                ),
                                "description": (
                                    "First paragraph. [canon_abc123::chunk_001]\n\n"
                                    "Second paragraph with no inline citation."
                                ),
                                "entities": [],
                                "source_chunks": ["canon_abc123::chunk_001"],
                                "outbound_relations": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_dir = domain_dir / "concepts_md"
            export_wiki_markdown(domain_dir, output_dir)
            markdown = (output_dir / "example_concept.md").read_text(encoding="utf-8")

        brief_line = next(line for line in markdown.splitlines() if line.startswith("brief: "))
        self.assertEqual(brief_line, 'brief: "A clean summary. Second sentence."')
        self.assertNotIn("canon_", brief_line)

        body = markdown.split("---", 2)[2].split("\n# Related", 1)[0].strip()
        paragraphs = body.split("\n\n")
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(all(paragraph.endswith("[canon_abc123]") for paragraph in paragraphs))
        self.assertNotIn("chunk_", body)


if __name__ == "__main__":
    unittest.main()
