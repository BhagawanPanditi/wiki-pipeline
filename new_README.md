# Concept Knowledge Graph & Wiki Pipeline

This repository provides an end-to-end, high-throughput pipeline to convert unstructured long-document corpora into a dense, interconnected **Concept Knowledge Graph** and **Markdown Wiki**.

---

## 1. Stage 1: Concept Extraction (`extract_concepts.py`)

Stage 1 splits documents into overlapping token windows and uses an LLM with high-density XML formatting to extract concepts and directional relationships.

### What is Extracted and Stored?

For each substantive concept identified in a chunk, the LLM outputs:
1. **`title`**: Exact canonical concept title (1–6 words, singular).
2. **`aliases`**: Explicit acronyms, alternate spellings, or abbreviations.
3. **`description`**: A dense, factual paragraph retaining all raw quantitative metrics, numerical bounds, parameter ranges, formulas, and technical details from the chunk.
4. **`outbound_relations`**: A list of directed outbound relationships pointing to other concepts mentioned in the chunk:
   ```json
   {
     "target_concept": "Target Concept Title",
     "relation": "1 concise sentence describing how this concept affects, extends, utilizes, or relates outward to the target concept."
   }
   ```
5. **`source_chunks`**: Exact chunk IDs tracking provenance back to the source document.

*(Note: Named entities are omitted during extraction to maximize output token budget for dense definitions and graph edges. Dedicated entity synthesis runs during post-consolidation enrichment).*

---

## 2. XML Extraction Format

To prevent JSON quote escaping issues (common in code snippets and math equations) and to save 20–30% of generation tokens, extraction outputs compact XML tags:

```xml
<concept>
<title>Two Pointers</title>
<aliases>Two-Pointer Technique; Dual Pointer</aliases>
<description>Two Pointers is an algorithmic technique that utilizes two indices to traverse a data structure concurrently, commonly employed to optimize search time from O(N^2) to O(N) in sorted arrays or linked lists.</description>
<relations>
Sliding Window :: Is a special case of this technique where both pointers move in the same direction.
Binary Search :: Shares the idea of shrinking a sorted search range.
</relations>
</concept>
```

### Automatic Truncation Recovery via Continuation
If a long chunk produces extensive output and hits `finish_reason == "length"`:
1. All completed `<concept>...</concept>` blocks are salvaged immediately.
2. The extractor identifies the cut-off / incomplete concept.
3. It automatically issues a targeted continuation prompt with the already-generated prefix.
4. The remaining concepts are extracted, parsed, and merged seamlessly.

---

## 3. Chunking & Context Boundaries

Long documents are tokenized and chunked into sliding windows:

| Parameter | Default Value | CLI Flag | Description |
| :--- | :--- | :--- | :--- |
| **Chunk Size** | **5,000 tokens** | `--chunk_size` | Primary evaluated window (`current_chunk`). |
| **Chunk Overlap** | **500 tokens** | `--chunk_overlap` | Stride between consecutive windows (`step = 4,500` tokens). |
| **Context Before** | Up to **500 tokens** | Automatic | Tokens immediately preceding the chunk. Empty for the first chunk. Used **strictly** for pronoun/acronym resolution. |
| **Context After** | Up to **500 tokens** | Automatic | Tokens immediately following the chunk. Empty for the last chunk. Used **strictly** for reference resolution. |

> **Strict Boundary Constraint**: Facts, definitions, and metrics are extracted **exclusively from `current_chunk`**. `context_before` and `context_after` are strictly used to resolve pronouns (e.g. *"this algorithm"*) and expand abbreviations into canonical names.

---

## 4. Input & Output Structure

The pipeline operates on direct input and output paths without requiring domain names:

- **`--input_path`**: Path to the source corpus (`.json` or `.jsonl`).
- **`--wiki_path`**: Destination folder where all wiki outputs, staging chunks, cache, and logs are stored. Automatically created (`mkdir -p`) if it does not exist.

### Input File Format
Supports either:
1. **JSON array** (`.json`):
   ```json
   [
     {"id": "doc_01", "content": "...document text..."},
     {"id": "doc_02", "content": "...document text..."}
   ]
   ```
2. **JSON Lines** (`.jsonl`):
   ```json
   {"id": "doc_01", "content": "...document text..."}
   {"id": "doc_02", "content": "...document text..."}
   ```

### Output Directory Structure
Running `extract_concepts.py` creates:
```text
<wiki_path>/
├── staging/
│   └── canon_<hash>/
│       └── chunk_000/
│           └── concepts.json         # Extracted concepts for this chunk
├── cache/
│   ├── chunks_cache.json             # Tokenized chunk objects
│   └── chunks_cache_manifest.json    # Cache fingerprint bound to corpus & tokenizer
├── doc_mapping.json                  # Source doc ID -> canonical hash mapping
└── extraction_log.json               # Progress log tracking completed chunks
```

---

## 5. Configuration: Strict Two-Tier Resolution (Zero Python Hardcoding)

There are **NO hardcoded fallback values in Python code**. Every execution parameter must be explicitly supplied either via the Command Line or in `config.json`:

| Setting | CLI Flag | `config.json` Key | Description |
| :--- | :--- | :--- | :--- |
| **Model** | `--model <name>` | `"model_name"` | Name of model served on vLLM. |
| **Servers** | `--servers <urls...>` | `"servers"` | List of vLLM endpoint URLs. |
| **Concurrency** | `--concurrency_per_server <N>` | `"concurrency_per_server"` | Concurrent in-flight requests per server. |
| **Max Tokens** | `--max_tokens <N>` | `"max_tokens"` | Maximum output tokens per request (e.g. 16000). |
| **Timeout** | `--request_timeout <seconds>` | `"request_timeout"` | Per-attempt HTTP timeout in seconds (e.g. 2400.0). |

If any required parameter is missing from both the CLI and `config.json`, the script immediately exits with an informative error.

### Example `config.json` with Per-Server Concurrency:
You can specify custom concurrency per server as a dictionary (e.g. 64 for DGX nodes, 20 for A40 nodes):
```json
{
  "servers": {
    "http://cn13-dgx:8000/v1": 64,
    "http://cn43-l40:8005/v1": 32,
    "http://cn35-a40:8002/v1": 32,
    "http://cn26-a40:8000/v1": 20,
    "http://cn26-a40:8001/v1": 20
  },
  "model_name": "Qwen/Qwen3.8-27B",
  "concurrency_per_server": 32,
  "max_tokens": 16000,
  "request_timeout": 2400.0
}
```

Or pass them inline via CLI:
```bash
--servers http://cn13-dgx:8000/v1:64 http://cn26-a40:8000/v1:20 http://cn35-a40:8002/v1:32
```

---

## 6. Execution Examples

### Basic Execution (Using `config.json`)
```bash
python3 extract_concepts.py \
  --input_path /path/to/corpus.jsonl \
  --wiki_path /path/to/my_wiki
```

### Overriding Settings via CLI
```bash
python3 extract_concepts.py \
  --input_path /path/to/corpus.jsonl \
  --wiki_path /path/to/my_wiki \
  --model Qwen/Qwen3.8-27B \
  --servers http://cn12-dgx:8000/v1 http://cn13-dgx:8001/v1 \
  --concurrency_per_server 32 \
  --chunk_size 5000 \
  --chunk_overlap 500
```

### Resuming an Interrupted Run
If an extraction run was stopped mid-way, add `--resume` to continue:
```bash
python3 extract_concepts.py \
  --input_path /path/to/corpus.jsonl \
  --wiki_path /path/to/my_wiki \
  --resume
```
It reads `extraction_log.json`, removes any incomplete chunks, and picks up immediately where it left off.
