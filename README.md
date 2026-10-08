# Wiki Knowledge Graph Pipeline

This repository builds an LLM-powered knowledge graph from technical and biological documents.

The pipeline processes documents from raw chunks into canonical concepts, merges aliases and near-duplicates, resolves stubs, and builds a connected concept graph with bi-directional links and inline chunk citations.

---

## Staged Execution Workflow (Handling vLLM & GPU Slurm Allocations)

Since you have 3 Slurm jobs (2x A40 GPUs each) for vLLM, and need to reallocate GPUs for Qwen Embedding, the pipeline is partitioned into **Two Stages**:

```
[3 vLLM Servers Active]
      │
      ▼
STAGE 1: ./run_stage1_pre_embed.sh biology
   1. global_concept_consolidation.py   (Exact lemmatization, boilerplate scrub, LLM fuzzy merge)
   2. stub_resolution.py                (Synthesize major stubs, repoint minor stubs)
   3. thin_concept_handler.py           (Fold thin sub-concepts < 30 words)
   4. add_concept_briefs.py             (Generate 2-3 sentence factual briefs)
   5. link_entity_overlap.py            (Connect islands sharing named entities)
      │
      ▼
[Pause: Kill 1 vLLM job -> Submit run_embedding.sbatch]
   6. embed_and_bridge_islands.py       (QwenEmbedding-8B dense cosine similarity bridging)
      │
      ▼
[Resume: Restart vLLM server -> 3 vLLM Servers Active]
      │
      ▼
STAGE 2: ./run_stage2_post_embed.sh biology
   7. synthesize_bidirectional_relations.py (Extract back-link descriptions B -> A)
   8. export_markdown_wiki.py           (Export concepts/*.md with frontmatter, [[slug|Title]], citations)
   9. analyze_islands.py                (Final topological island & giant component audit)
```

---

## Detailed Step-by-Step Breakdown

### 1. `global_concept_consolidation.py`
- Reuses all raw concepts already extracted in Pass 1 / Pass 2.
- **Title Lemmatization & Normalization**: Strips plural suffixes (`hallucination` vs `hallucinations`) and hyphenation splits (`hand washing` vs `handwashing`).
- **Scraping & Boilerplate Filter**: Automatically detects and drops paper meta-headers (`Study Participants`, `Statistical Methods`, cookie notices).
- **Paragraph Citation Mapping**: Uses chunk provenance internally, while exported description paragraphs end with document-level citations such as `[canon_doc01]` (never a `::chunk_...` suffix).

### 2. `stub_resolution.py`
- Major stubs (> 2 source chunks) are synthesized into full concepts.
- Minor stubs ($\le 2$ chunks) are absorbed into parent descriptions, and **outbound relations are repointed** rather than deleted, preserving graph edges.

### 3. `thin_concept_handler.py`
- Folds thin concepts (< 30 words) into candidate host concepts via LLM validation.

### 4. `add_concept_briefs.py`
- Generates a 2-3 sentence (40–60 word) factual `brief` for each concept.
- Strips internal document/chunk IDs from briefs and flattens them to a single line.

### 5. `link_entity_overlap.py`
- Discovers latent edges between concepts sharing common named entities (genes, tools, pathways, proteins).

### 6. `embed_and_bridge_islands.py`
- Uses `qwen-embed` (`Alibaba-NLP/gte-Qwen2-7B-instruct` or `Qwen2-Embedding-8B`) to compute cosine similarity across low-degree nodes and bridge isolated islands into the mainland.

### 7. `synthesize_bidirectional_relations.py`
- Turns all directed edges into **bidirectional relationships**.
- Uses an LLM with Concept A's brief, Concept B's brief, and the forward relation to generate a 1-sentence description for the reverse relation.

### 8. `export_markdown_wiki.py`
- Generates clean Markdown files in `concepts_md/<sluggified_title>.md`.
- **Frontmatter**: `brief` is a single-line summary without internal provenance IDs; `sources` contains document-level IDs.
  ```yaml
  ---
  title: "Closed-Eye Hallucination"
  aliases:
    - "Closed-eye hallucinations"
  brief: "Closed-eye hallucination is an entoptic visual phenomenon..."
  entities:
    - "Phosphenes"
    - "Visual cortex"
  sources:
    - "canon_doc01"
  ---
  ```
- **Body**: Every paragraph carries document-level citations. Chunk suffixes are removed, and paragraphs missing an inline citation use the concept's known source documents:
  ```markdown
  Closed-eye hallucination is a visual perception phenomenon occurring when the eyes are closed... [canon_doc01]
  ```
- **# Related Section**:
  ```markdown
  # Related
  - [[phosphene|Phosphene]] - Entoptic light patterns caused by mechanical stimulation or spontaneous retinal neuron firing.
  ```

### 9. `analyze_islands.py`
- Analyzes weakly connected components, giant component percentage, and reports small isolated islands.

---

## Unattended Runs, Resume, and Skipping Embeddings

For a single end-to-end run with embedding bridging disabled:

```bash
./run_entire_pipeline.sh biology --skip-embed --max-stage-attempts 5
```

The runner checkpoints each stage under `<wiki-base>/<domain>/.pipeline_state/` and writes a separate log for each stage under `pipeline_logs/`. If interrupted or if a stage exhausts its retries, restart with:

```bash
./run_entire_pipeline.sh biology --resume --skip-embed --max-stage-attempts 5
```

Completed stages with their expected outputs are skipped; the failed/incomplete stage is retried. Rerunning an upstream stage invalidates downstream completion markers, including those used by the separate stage wrappers and embedding job. Use `--retry-forever` only when you want the process to wait through outages indefinitely. It can also repeat permanent input/configuration failures, so the default bounded retries are safer for unattended jobs. Use `--fresh` to rerun the orchestration stages while preserving generated data, logs, and per-chunk extraction checkpoints; it resets only pipeline completion markers. `--skip-extraction` is accepted only when prior extraction and document consolidation produced valid, non-empty concept lists in `consolidated/canon_*/concepts.json`.

`--skip-embed` skips only embedding-based semantic island bridging. The rest of the graph pipeline still runs, but the graph may have fewer semantic bridge links. Embedding is optional, not required for export.

Retries cannot guarantee completion if servers, GPUs, storage, or input data remain unavailable or invalid. After the configured attempts, the runner exits non-zero, preserves checkpoints/logs, and can be resumed after fixing the cause.

## How to Cache Previous Stages & Run Anew

To preserve your previous run's output before running the updated pipeline:
```bash
# 1. On Prajna, backup your current domain directory
cd /home/compiling-ganesh/24m0829/long_document/wiki
cp -r biology biology_backup_run1

# 2. Push updated pipeline code from your local machine
./run_push.sh

# 3. Run Stage 1 (Pre-embedding)
./run_stage1_pre_embed.sh biology

# 4. Submit embedding job (qwen-embed)
sbatch run_embedding.sbatch

# 5. After embedding completes, run Stage 2 (Bidirectional relations & Markdown export)
./run_stage2_post_embed.sh biology
```
