"""Corpus loading, normalisation, deduplication and chunking.

The pipeline is not tied to the BRIGHT benchmark files. A corpus is any file of
documents in one of two shapes, auto-detected:

* a single JSON array  -- ``[{"id": "a", "contents": "..."}, ...]``  (``.json``)
* JSON Lines           -- ``{"id": "a", "contents": "..."}`` per line (``.jsonl``)

Field names are normalised on the way in, so ``contents``/``content``/``text``
and ``id``/``doc_id``/``document_id`` are all accepted.

Documents with byte-identical content are collapsed into one *canonical*
document named ``canon_<hash>``, and every original id is recorded in a mapping
so a citation can be traced back to the source corpus. Deduplication happens
whether or not chunking is enabled.

Chunking is optional. With chunking off, each document becomes exactly one
extraction unit, so every downstream stage sees the same shape and no tokenizer
is needed at all -- which means a new corpus can be processed without
``transformers`` or any offline HuggingFace model.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Accepted spellings, in priority order.
CONTENT_FIELDS = ("contents", "content", "text", "body")
ID_FIELDS = ("id", "doc_id", "document_id")

# Chunking defaults. These are part of the cache fingerprint, so changing them
# invalidates a cached chunking run rather than silently mixing window sizes.
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_CHUNK_OVERLAP = 500

# Used as the output directory name when no domain hint is supplied.
DEFAULT_RUN_NAME = "default"

MANIFEST_VERSION = 2


class CorpusError(ValueError):
    """Raised when a corpus file cannot be read or is missing required fields.

    The message always names the file and, where applicable, the offending line
    or record index, so a bad corpus is diagnosable without a debugger.
    """


def short_hash(text: str) -> str:
    """Return the 12-character MD5 prefix used for canonical document ids.

    Args:
        text: Document content.

    Returns:
        The first 12 hex characters of the content's MD5 digest.

    Note:
        Citation ids are ``canon_<short_hash>``, so this algorithm is part of the
        on-disk format. Changing it would orphan every existing citation.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def canonical_id(content: str) -> str:
    """Return the canonical document id for a piece of content.

    Args:
        content: Document content.

    Returns:
        ``canon_<hash>`` -- the id used in staging paths and citations.
    """
    return f"canon_{short_hash(content)}"


def fix_mojibake(text: str) -> str:
    """Repair ids that were decoded as cp437 when they were really UTF-8.

    Some source corpora arrive with ids whose non-ASCII characters were decoded
    with the wrong codec. This reverses that when it can, and otherwise returns
    the text unchanged rather than raising.

    Args:
        text: A raw identifier from the corpus.

    Returns:
        The repaired identifier, or the original if it was not mis-decoded.
    """
    try:
        return text.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return text


def file_digest(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file, read in blocks so large corpora stream.

    Args:
        path: File to hash.
        block_size: Bytes read per iteration.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_format(path: Path) -> str:
    """Decide whether a corpus file is a JSON array or JSON Lines.

    The file extension is trusted first because it is what the caller declared;
    if the extension is ambiguous the first non-whitespace byte decides.

    Args:
        path: Corpus file.

    Returns:
        ``"json_array"`` or ``"jsonl"``.

    Raises:
        CorpusError: If the file is empty or starts with neither ``[`` nor ``{``.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json_array"

    with path.open("r", encoding="utf-8") as handle:
        head = handle.read(4096)
    first = next((c for c in head if not c.isspace()), "")
    if first == "[":
        return "json_array"
    if first == "{":
        return "jsonl"
    raise CorpusError(
        f"{path} does not look like a corpus: expected a JSON array starting "
        f"with '[' or JSON Lines starting with '{{', found {first!r}.")


def _normalise(record: Any, index: int, source: Path) -> Dict[str, str]:
    """Normalise one raw corpus record to ``{"id": ..., "content": ...}``.

    Args:
        record: A decoded JSON value from the corpus.
        index: Position of the record, used only in error messages.
        source: Corpus path, used only in error messages.

    Returns:
        A dict with exactly ``id`` (str) and ``content`` (str).

    Raises:
        CorpusError: If the record is not an object, has no content field, or
            has empty content.
    """
    if not isinstance(record, dict):
        raise CorpusError(f"{source} record #{index} is a {type(record).__name__}, "
                          f"expected a JSON object.")

    content = None
    for field in CONTENT_FIELDS:
        if field in record and record[field] is not None:
            content = record[field]
            break
    if content is None:
        raise CorpusError(
            f"{source} record #{index} has no content field. Looked for one of "
            f"{CONTENT_FIELDS}; keys present: {sorted(record)}.")
    if not isinstance(content, str):
        raise CorpusError(f"{source} record #{index} content is a "
                          f"{type(content).__name__}, expected a string.")
    if not content.strip():
        raise CorpusError(f"{source} record #{index} has empty content.")

    raw_id = None
    for field in ID_FIELDS:
        if field in record and record[field] is not None:
            raw_id = record[field]
            break
    if raw_id is None:
        # An id is not strictly required: fall back to the record's position so
        # the citation mapping still has a usable key.
        raw_id = f"doc_{index:06d}"

    return {"id": fix_mojibake(str(raw_id)), "content": content}


def load_corpus(path: Path) -> List[Dict[str, str]]:
    """Read a corpus file and normalise every record.

    Accepts a single JSON array or JSON Lines, auto-detected. Blank lines in
    JSONL input are skipped.

    Args:
        path: Corpus file (``.json`` or ``.jsonl``).

    Returns:
        A list of ``{"id": str, "content": str}`` dicts, in file order.

    Raises:
        CorpusError: If the file is missing, malformed, or a record lacks
            usable content. The message names the file and the record.
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"Corpus file not found: {path}")

    fmt = detect_format(path)
    raw: List[Any] = []

    if fmt == "json_array":
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            # A .json file that is really JSON Lines is a common mistake; say so.
            raise CorpusError(
                f"{path} is not valid JSON ({exc}). If it holds one JSON object "
                f"per line, rename it to .jsonl.") from exc
        if isinstance(data, dict):
            # Tolerate {"documents": [...]} / {"data": [...]} wrappers.
            for key in ("documents", "data", "docs", "records"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise CorpusError(f"{path} holds a {type(data).__name__}, expected a JSON array.")
        raw = data
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"{path} line {line_no} is not valid JSON: {exc}") from exc

    if not raw:
        raise CorpusError(f"{path} contains no records.")

    return [_normalise(record, index, path) for index, record in enumerate(raw)]


def deduplicate(docs: List[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Collapse byte-identical documents into canonical ones.

    Args:
        docs: Normalised records from :func:`load_corpus`.

    Returns:
        A 2-tuple of:
            * ``canon_docs`` -- ``{canonical_id: content}``, one entry per unique
              document, in first-seen order.
            * ``doc_mapping`` -- ``{original_id: canonical_id}`` for every record
              in the corpus, including duplicates. This is the citation mapping
              written to ``doc_mapping.json``.
    """
    canon_docs: Dict[str, str] = {}
    doc_mapping: Dict[str, str] = {}
    for doc in docs:
        cid = canonical_id(doc["content"])
        canon_docs.setdefault(cid, doc["content"])
        doc_mapping[doc["id"]] = cid
    return canon_docs, doc_mapping


def chunk_document(doc_id: str, content: str, tokenizer: Any,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Split one document into overlapping token windows with surrounding context.

    Each unit carries the window itself plus the tokens immediately before and
    after it, so the extractor can resolve references that straddle a boundary.

    Args:
        doc_id: Canonical document id.
        content: Full document text.
        tokenizer: Anything exposing ``encode(str) -> list`` and
            ``decode(list, skip_special_tokens=bool) -> str``.
        chunk_size: Window size in tokens.
        overlap: Tokens repeated on each side of a window, also used as the
            context window.

    Returns:
        A list of unit dicts with keys ``doc_id``, ``chunk_id``,
        ``current_chunk``, ``context_before`` and ``context_after``. Empty for a
        document that tokenises to nothing.

    Raises:
        ValueError: If ``overlap >= chunk_size``, which would make the stride
            non-positive and loop forever.
    """
    if overlap >= chunk_size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than "
                         f"chunk size ({chunk_size}).")

    tokens = tokenizer.encode(content)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return []

    chunks: List[Dict[str, Any]] = []
    step = chunk_size - overlap
    chunk_idx = 0

    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        before_tokens = tokens[max(0, start - overlap):start]
        after_tokens = tokens[end:min(total_tokens, end + overlap)]

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"chunk_{chunk_idx:03d}",
            "current_chunk": tokenizer.decode(chunk_tokens, skip_special_tokens=True),
            "context_before": tokenizer.decode(before_tokens, skip_special_tokens=True),
            "context_after": tokenizer.decode(after_tokens, skip_special_tokens=True),
        })
        chunk_idx += 1
        if end >= total_tokens:
            break

    return chunks


def whole_document_unit(doc_id: str, content: str) -> Dict[str, Any]:
    """Return a single extraction unit holding an entire document.

    Used when chunking is disabled. Keeping the same key names as
    :func:`chunk_document` means no downstream stage has to know whether the
    corpus was chunked.

    Args:
        doc_id: Canonical document id.
        content: Full document text.

    Returns:
        A unit dict with the same shape as a chunk, with empty context.
    """
    return {
        "doc_id": doc_id,
        "chunk_id": "chunk_000",
        "current_chunk": content,
        "context_before": "",
        "context_after": "",
    }


def build_units(canon_docs: Dict[str, str], tokenizer: Optional[Any], *,
                chunking: bool,
                chunk_size: int = DEFAULT_CHUNK_SIZE,
                overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Turn canonical documents into the extraction units Pass 1 consumes.

    Args:
        canon_docs: ``{canonical_id: content}`` from :func:`deduplicate`.
        tokenizer: Required only when ``chunking`` is true. When chunking is off
            this may be ``None``, so a corpus can be processed without
            ``transformers`` installed.
        chunking: Whether to split documents into overlapping token windows.
        chunk_size: Window size in tokens (ignored when ``chunking`` is false).
        overlap: Context/overlap in tokens (ignored when ``chunking`` is false).

    Returns:
        A list of unit dicts, grouped by document in ``canon_docs`` order.

    Raises:
        ValueError: If chunking is requested without a tokenizer.
    """
    if chunking and tokenizer is None:
        raise ValueError("Chunking requires a tokenizer; pass one or disable chunking.")

    from tqdm import tqdm
    units: List[Dict[str, Any]] = []
    desc = "Tokenizing & chunking documents" if chunking else "Preparing document units"
    for doc_id, content in tqdm(canon_docs.items(), desc=desc):
        if chunking:
            units.extend(chunk_document(doc_id, content, tokenizer,
                                        chunk_size=chunk_size, overlap=overlap))
        else:
            units.append(whole_document_unit(doc_id, content))
    return units


def cache_manifest(source_path: Path, *, tokenizer_name: Optional[str],
                   chunking: bool, chunk_size: int, overlap: int) -> Dict[str, Any]:
    """Fingerprint everything a cached chunking depends on.

    A cache is reused only when this manifest matches exactly, so editing the
    corpus, switching tokenizer, or changing the window size forces a rebuild
    instead of silently mixing old and new units.

    Args:
        source_path: Corpus file the units were built from.
        tokenizer_name: Tokenizer/model name, or ``None`` when not chunking.
        chunking: Whether chunking was enabled.
        chunk_size: Window size in tokens.
        overlap: Overlap in tokens.

    Returns:
        A JSON-serialisable dict suitable for writing to
        ``chunks_cache_manifest.json``.
    """
    return {
        "format_version": MANIFEST_VERSION,
        "source_sha256": file_digest(source_path),
        "chunking": bool(chunking),
        "tokenizer": tokenizer_name if chunking else None,
        "chunk_size": chunk_size if chunking else None,
        "overlap": overlap if chunking else None,
    }


def resolve_corpus_path(args: Any) -> Path:
    """Work out which corpus file a stage should read from --input_path.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The resolved corpus path.

    Raises:
        CorpusError: If --input_path is missing or not a file.
    """
    input_path = getattr(args, "input_path", None)
    if input_path:
        path = Path(input_path)
        if not path.is_file():
            raise CorpusError(f"Corpus file not found: {path}")
        return path

    raise CorpusError("No corpus specified. Pass --input_path /path/to/documents.json (or .jsonl).")


# ── Run configuration ────────────────────────────────────────────────────────
# Pass 1 records how the corpus was prepared so later stages do not have to be
# told the same thing twice -- and so a resumed run cannot silently disagree
# with the run that produced the cache.

RUN_CONFIG_NAME = "run_config.json"


def save_run_config(run_dir: Path, config: Dict[str, Any]) -> Path:
    """Persist how this run's corpus was prepared.

    Args:
        run_dir: The run's output directory (``<wiki_base_dir>/<domain>/``).
        config: JSON-serialisable settings; at minimum ``corpus``, ``domain``,
            ``chunking``, ``chunk_size`` and ``chunk_overlap``.

    Returns:
        The path written, ``<run_dir>/run_config.json``.
    """
    from .atomic_io import atomic_write_json

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_CONFIG_NAME
    atomic_write_json(path, config)
    return path


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Read a run's saved corpus configuration.

    Args:
        run_dir: The run's output directory.

    Returns:
        The saved config dict, or an empty dict when the file does not exist or
        cannot be decoded. Callers should treat a missing config as "unknown"
        rather than as an error, so stages still run on output trees produced
        before this file existed.
    """
    path = Path(run_dir) / RUN_CONFIG_NAME
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def chunking_enabled(run_dir: Path, default: bool = True) -> bool:
    """Report whether the run that produced ``run_dir`` chunked its corpus.

    Args:
        run_dir: The run's output directory.
        default: Value to return when no run config is recorded. Defaults to
            ``True`` so pre-existing BRIGHT trees keep their current behaviour.

    Returns:
        ``True`` when documents were split into chunks, ``False`` when each
        document was a single unit.
    """
    config = load_run_config(run_dir)
    if "chunking" not in config:
        return default
    return bool(config["chunking"])
