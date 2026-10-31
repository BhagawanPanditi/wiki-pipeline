"""Chunk Cache Loader.
-------------------
Loads cached sliding-window text chunks from disk, keyed by `doc_id::chunk_id`.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_cached_chunks(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load the tokenized sliding-window chunks, keyed by ``doc_id::chunk_id``."""
    chunks_cache_file = Path(cache_dir) / "chunks_cache.json"
    if not chunks_cache_file.exists():
        raise FileNotFoundError(
            f"Chunk cache file not found at {chunks_cache_file}. "
            "Ensure you have run the chunking/extraction step first."
        )

    with open(chunks_cache_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {f"{c['doc_id']}::{c['chunk_id']}": c for c in chunks}
