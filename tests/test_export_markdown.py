"""Tests for export_markdown.py's output folder handling.

The exporter regenerates a folder of Markdown pages. It must replace only the
pages it wrote itself: a mistyped --output_dir must never delete unrelated notes.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import run_script


def concept(title, brief):
    """Build one refined-graph concept for the export fixture.

    Args:
        title: Concept title.
        brief: One-line brief shown on the page.

    Returns:
        A concept dict with the fields the exporter reads.
    """
    return {"title": title, "aliases": [], "description": f"{title} is described here.",
            "entities": [], "source_chunks": ["canon_a::chunk_000"],
            "outbound_relations": [], "brief": brief}


class ExportFolderTests(unittest.TestCase):
    def test_second_export_replaces_only_its_own_pages(self):
        """Unrelated files in the output folder survive repeated exports."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "biology"
            run_dir.mkdir()
            (run_dir / "global_concepts_refined.json").write_text(json.dumps({"concepts": [
                concept("Alpha", "Alpha is first."), concept("Beta", "Beta is second.")]}))
            out = root / "notes"
            out.mkdir()
            (out / "my_own_notes.md").write_text("keep me")
            (out / "config.json").write_text("{}")

            for _ in range(2):
                run_script("export_markdown.py",
                           ["--domain", "biology", "--wiki_base_dir", str(root),
                            "--output_dir", str(out)])

            names = sorted(p.name for p in out.iterdir())
            self.assertIn("my_own_notes.md", names)
            self.assertIn("config.json", names)
            self.assertEqual(sorted(n for n in names if n.endswith(".md")
                                    and n != "my_own_notes.md"), ["alpha.md", "beta.md"])


if __name__ == "__main__":
    unittest.main()
