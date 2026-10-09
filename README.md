# Wiki pipeline

Turns a corpus of documents into a wiki of Markdown pages. Each page is one
concept extracted from the corpus, with a short brief, its body paragraphs and
its relations to other concepts. Every paragraph cites the source document it came
from. The language model runs on your own vLLM servers.

```
corpus (JSON / JSONL)  ->  run_pipeline.sh  ->  <run>/concepts_md/*.md
                                             ->  <run>/doc_mapping.json   (citation ids -> your ids)
                                             ->  <run>/log/*.jsonl        (every LLM call and decision)
```

`<run>` is `<wiki-base-dir>/<domain>/`, or `<wiki-base-dir>/default/` when no domain is given.
Section 7.3 explains both paths.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick start](#2-quick-start)
3. [Input: the corpus](#3-input-the-corpus)
4. [Output: the run directory](#4-output-the-run-directory)
5. [The stages](#5-the-stages)
6. [Running one stage on its own](#6-running-one-stage-on-its-own)
7. [Options](#7-options)
8. [Chunking](#8-chunking)
9. [The domain hint](#9-the-domain-hint)
10. [Citations](#10-citations)
11. [Logs](#11-logs)
12. [Resuming, retries and time limits](#12-resuming-retries-and-time-limits)
13. [Concurrency](#13-concurrency)
14. [Verification and the improvement report](#14-verification-and-the-improvement-report)
15. [Code layout](#15-code-layout)
16. [Upgrading from the earlier names](#16-upgrading-from-the-earlier-names)
17. [Known limitations](#17-known-limitations)

---

## 1. Prerequisites

**Python.** Python 3.10 or newer. Tested with 3.11.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`transformers` is needed for the default chunked mode. It is not needed with
`--no-chunking`, and then the pipeline never loads a tokenizer.

**Language model servers.** One or more OpenAI-compatible endpoints, normally vLLM.
The serving setup this pipeline was tuned for:

```bash
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --enable-prefix-caching --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --host 0.0.0.0 --port 8000
```

A Slurm job for it:

```bash
#!/bin/bash
#SBATCH -J vllm_qwen -p a40 --qos=a40 --nodes=1 -x cn22-a40 --gres=gpu:2
#SBATCH --ntasks-per-node=1 --cpus-per-task=16 --mem=128G --time=24:00:00
#SBATCH --output=logs/vllm_qwen_%j.log
export TOKENIZERS_PARALLELISM=false VLLM_USE_MODELSCOPE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source ~/miniconda3/etc/profile.d/conda.sh
vllm serve Qwen/Qwen3.8-27B ...   # the command above
```

Start one job per node you want to use. Servers can be added or removed between
runs.

**`config.json`.** Copy `config.example.json` to `config.json` and list your
servers. The file is git-ignored, so each machine keeps its own. It has three keys:

| Key | Used by | When absent |
|---|---|---|
| `servers` | every stage, and the preflight check | the three cluster hosts in `config.py` |
| `model_name` | every stage (unless `--model_name` is given) | `Qwen/Qwen3.8-27B` |
| `stage_concurrency` | the stage it names | that stage's built-in profile (section 13) |

```json
{
  "servers": ["http://gpu-node-1:8000/v1", "http://gpu-node-1:8001/v1"],
  "model_name": "Qwen/Qwen3.8-27B",
  "stage_concurrency": {
    "extract-concepts": 8,
    "add-reverse-links": {"concurrency_per_server": 48, "attempts": 6}
  }
}
```

- A file that exists but is not valid JSON stops the run with an error. It is never
  silently replaced by the defaults, because that would send work to the wrong machines.
- A top-level `concurrency_per_server` is no longer used. A warning says so. Use
  `stage_concurrency` instead.
- A `stage_concurrency` key that names no LLM stage (for example a stage name from an
  older version) is ignored, with a warning on stderr that lists the valid names.
- A `stage_concurrency` entry that is neither a number nor an object, or a `stage_concurrency`
  that is not an object, stops the run in the preflight, before any stage starts.

Work is spread evenly over the servers in the list. Check that they are reachable
with `python3 check_servers.py`.

**Network.** The pipeline talks only to your model servers. With `--no-chunking`
it needs no HuggingFace access at all. With chunking it needs the model's
tokenizer available locally, which `HF_HUB_OFFLINE=1` respects once it is cached.

---

## 2. Quick start

```bash
# Corpus with a domain name (outputs go to /data/wiki/myset/)
./run_pipeline.sh --corpus /data/docs.jsonl --domain myset --wiki-base-dir /data/wiki

# Corpus with no domain (outputs go to /data/wiki/default/)
./run_pipeline.sh --corpus /data/docs.json --wiki-base-dir /data/wiki --no-chunking
```

The run takes every stage in order and ends with the Markdown wiki in
`<wiki-base-dir>/<domain>/concepts_md/`.

If you leave out `--wiki-base-dir`, the cluster default is used (section 7.3).

If the run stops (a crash, a preempted job, Ctrl-C, a server outage), run the
same command again. Finished stages are skipped, and the interrupted stage carries
on from where it stopped (section 12). There is no separate resume flag. `--fresh`
clears the completion markers so that every stage runs again. It does not delete
any output.

---

## 3. Input: the corpus

One file, in one of two shapes. The shape is detected automatically.

**A JSON array** (`.json`):

```json
[
  {"id": "doc-001", "contents": "Photosynthesis converts light into chemical energy..."},
  {"id": "doc-002", "contents": "The Krebs cycle oxidises acetyl-CoA..."}
]
```

**JSON Lines** (`.jsonl`), one object per line:

```
{"id": "doc-001", "contents": "Photosynthesis converts light into chemical energy..."}
{"id": "doc-002", "contents": "The Krebs cycle oxidises acetyl-CoA..."}
```

- The text field may be `contents`, `content`, `text` or `body`.
- The id field may be `id`, `doc_id` or `document_id`. If there is no id, the record's
  position is used.
- A top-level `{"documents": [...]}` (or `data`, `docs`, `records`) wrapper is accepted.
- Records with no text, or with blank text, stop the run with a message naming the
  file and the record number. Nothing is skipped quietly.
- Documents with identical text are merged into one. Every original id is kept in
  the citation map (section 10).

The legacy BRIGHT layout also works: `--data-dir DIR --domain NAME` reads
`DIR/NAME.jsonl` (section 7.3).

---

## 4. Output: the run directory

Everything for one run lives in `<wiki-base-dir>/<domain or default>/`:

```
<run>/
  concepts_md/                    the wiki: one .md page per concept      <- the main output
  global_concepts_refined.json    the concept graph as JSON
  doc_mapping.json                original document id -> citation id (canon_...)
  islands_report.json             how the graph splits into connected pieces
  verification_summary.json       before and after quality scores (section 14)
  run_config.json                 how the corpus was read: chunking, counts

  log/                            every log, in one folder (section 11)
    <stage>.llm.jsonl             every LLM call: prompt, reply, server, attempt
    <stage>.decisions.jsonl       what each stage merged, pruned, linked or folded
    <stage>.debug.jsonl           failures, with the traceback and the raw output
    <step>.run.log                console output of each pipeline step

  .pipeline_state/                completion markers, one per step (section 12)
  staging/<doc>/<chunk>/          per-unit results: concepts.json, verification.json,
                                  and debug.json for an item that failed for good
  consolidated/<doc>/             per-document merged concepts
  cache/                          chunk cache and its manifest (reused on resume)
  *_log.json                      resume checkpoints. These are not logs; the runner reads them.
  ...                             other intermediate graphs
```

The files named `*_log.json` are **resume checkpoints**. Everything that is a log is
in `log/`.

---

## 5. The stages

`run_pipeline.sh` runs these in order. The stage key is the name used by the runner,
by the log file names and by `stage_concurrency` in `config.json`.

| # | Step (runner name) | Stage key | Script | What it does | Reads | Writes |
|---|---|---|---|---|---|---|
| 0 | preflight | — | `check_servers.py` | Probes every server. | `config.json` | — |
| 1 | extract-concepts | `extract-concepts` | `extract_concepts.py` | Extracts concepts from every chunk or document. | corpus | `staging/`, `cache/`, `doc_mapping.json` |
| 2 | verify-pass1 | `verify-pass1` | `verify_extraction.py --phase 1` | Scores how well each extraction covers its source. | staging | `verification.json` per unit, `verification_log.json` |
| 3 | refine-concepts | `refine-concepts` | `refine_concepts.py` | Adds what verification found missing. Backs up Pass 1 first. | staging, verification | `pass2_refinement.json`, `concepts_pass1_backup.json`, updated `concepts.json` |
| 4 | verify-pass2 | `verify-pass2` | `verify_extraction.py --phase 2` | Scores the refined output against the same sources. | staging | `verification_pass2.json`, `verification_summary.json` |
| 5 | consolidate-documents | `consolidate-documents` | `consolidate_documents.py` | Merges duplicates **within** one document. Passes through when `--no-chunking`. | staging | `consolidated/<doc>/concepts.json` |
| 6 | consolidate-global | `consolidate-global` | `consolidate_global.py` | Merges duplicates **across** documents: exact matches, then fuzzy matches confirmed by the LLM. | consolidated | `global_concepts.json`, `global_consolidation.json` |
| 7 | resolve-stubs | `resolve-stubs` | `resolve_stubs.py` | Defines concepts that were only ever mentioned, and absorbs minor mentions into their parents. | global, cache | `global_concepts_stubs_resolved.json` |
| 8 | fold-thin-concepts | `fold-thin-concepts` | `fold_thin_concepts.py` | Folds very short concepts into the best host concept, or prunes them. | stubs resolved | `global_concepts_refined.json` |
| 9 | write-briefs | `write-briefs` | `write_briefs.py` | Writes the 2–3 sentence brief for each concept. | refined | refined (briefs added) |
| 10 | link-entities | `link-entities` | `link_entities.py` | Links concepts that share named entities. No LLM. | refined | refined (links added) |
| 11 | audit-islands | `audit-islands` | `audit_islands.py` | For small disconnected groups, decides PRUNE, MERGE, LINK or KEEP. | refined | refined |
| 12 | add-reverse-links | `add-reverse-links` | `add_reverse_links.py` | Adds the missing back-link for every one-way relation. | refined | refined |
| 13 | final-consolidation | `final-macro-scan`, `final-deep-verify` | `final_consolidation.py` | Whole-graph scan for duplicates, links and noise, then verification of each candidate group. Writes the Markdown wiki. | refined | refined, `concepts_md/` |
| 14 | island-report | `island-report` | `island_report.py` | Reports the connected pieces of the finished graph. | refined | `islands_report.json` |

Stages 1–8 read the output of the stage before them. Section 6 lists what each
stage needs.

---

## 6. Running one stage on its own

Set the two variables that every stage takes:

```bash
W=/data/wiki      # --wiki_base_dir
D=myset           # --domain (leave it out for an undeclared corpus)
```

The stage scripts take underscore flags (`--wiki_base_dir`, `--chunk_size`). The
runner takes hyphen flags (`--wiki-base-dir`, `--chunk-size`).

Each stage needs the outputs of the stages above it, listed under **Needs**. A stage
that finds a missing input stops with a message that names the file.

Every stage accepts `--help`, and the flags in section 7.2.

**0. Check the servers**

```bash
python3 check_servers.py
```

**1. Extract concepts**

```bash
python3 extract_concepts.py --domain $D --wiki_base_dir $W \
  --corpus /data/docs.jsonl                  # or --no-chunking, or --chunk_size N --chunk_overlap N
```

Needs: the corpus. Writes: `staging/`, `cache/`, `doc_mapping.json`, `run_config.json`.

**2. Verify Pass 1**

```bash
python3 verify_extraction.py --domain $D --wiki_base_dir $W --phase 1
```

Needs: step 1. Writes: `staging/*/*/verification.json`, `verification_log.json`.
Optional with `run_pipeline.sh --skip-verify-pass1`.

**3. Refine (Pass 2)**

```bash
python3 refine_concepts.py --domain $D --wiki_base_dir $W
```

Needs: step 1. It uses the missing concepts and facts from step 2 when available.
The runner passes `--ignore_audit_feedback` if Pass 1 verification is skipped, so
stale verification files do not influence new refinement work.
Writes: updated `staging/*/*/concepts.json`, `concepts_pass1_backup.json`, `pass2_log.json`.

**4. Verify Pass 2**

```bash
python3 verify_extraction.py --domain $D --wiki_base_dir $W --phase 2
```

Needs: step 3. Writes: `verification_pass2.json`, `verification_summary.json`, and prints the
improvement report (section 14). Optional with `run_pipeline.sh --skip-verify-pass2`.

**5. Consolidate within documents**

```bash
python3 consolidate_documents.py --domain $D --wiki_base_dir $W
```

Needs: step 3 (or 1). Writes: `consolidated/<doc>/concepts.json`. With `--no-chunking`
it writes the concepts unchanged and makes no LLM call. Merge-planning prompts
contain only each concept's title, up to 8 aliases, and a description hint of
up to 30 words (at most 400 characters); full descriptions and relations stay
in the saved output. For very large chunked
documents, the planner sorts titles and splits the list into context-bounded batches;
all batches must succeed before the document is checkpointed. Exact-title duplicates
still combine across batches, but unusual aliases spanning distant batches may need
later global consolidation to detect them.

**6. Consolidate across documents**

```bash
python3 consolidate_global.py --domain $D --wiki_base_dir $W
```

Needs: step 5. Writes: `global_concepts.json`, `global_consolidation.json`.
Fuzzy review is two sequential, LLM-confirmed passes. The first graph connects
concepts by word containment (at least two words shared, score >= 0.60). After
applying **only** the LLM-approved word merges, the second graph is rebuilt from
the merged concepts using character-trigram Jaccard (score >= 0.50). Each edge
is weighted with its pass's similarity score; there is no mixed score or Louvain
community detection. Non-singleton connected components with 30 or more nodes
are recursively split with a seeded, weighted Kernighan–Lin **balanced min-cut
heuristic** until every reviewed group has fewer than 30 nodes. Exact balanced
min-cut is computationally hard; an unconstrained min-cut tends to peel off
single nodes. Cut edges are not reviewed in that pass, so merges across splits
can be missed. Prompt-length limits can also require further splitting.

LLM requests show up to 8 aliases and a 30-word hint per concept in compact
numbered text, not input JSON; responses are JSON containing only those numeric
candidate IDs. Valid IDs are translated back to the stored titles; unknown or
duplicate suggestions are ignored with warnings rather than inventing concepts
or failing the whole cluster. Requests are limited to 20,000 rendered
characters, and context/response-length errors split a request further. All
requests must succeed. To proceed with exact and alias matches only, pass
`run_pipeline.sh --skip-global-fuzzy` (or
`consolidate_global.py --skip_fuzzy_review`). This skips candidate generation and
LLM review, but still fuses descriptions, citations and relations. The audit
summary records `fuzzy_review_skipped: true`, and the runner won't mistake an
exact-only output for a completed fuzzy run or vice versa.

**7. Resolve stubs**

```bash
python3 resolve_stubs.py --domain $D --wiki_base_dir $W
```

Needs: step 6, and `cache/chunks_cache.json` from step 1. Writes:
`global_concepts_stubs_resolved.json`.

**8. Fold thin concepts**

```bash
python3 fold_thin_concepts.py --domain $D --wiki_base_dir $W --min_concept_words 30
```

Needs: step 7. Writes: `global_concepts_refined.json`.

**9. Write briefs**

```bash
python3 write_briefs.py --domain $D --wiki_base_dir $W
```

Needs: step 8. Updates `global_concepts_refined.json`.

**10. Link by shared entities** (no LLM)

```bash
python3 link_entities.py --domain $D --wiki_base_dir $W \
  --min_shared_entities 2 --max_links_per_node 4 [--dry_run]
```

Needs: step 9. Updates `global_concepts_refined.json`. `--dry_run` reports the links and
writes nothing.

**11. Audit small islands**

```bash
python3 audit_islands.py --domain $D --wiki_base_dir $W --max_island_size 5
```

Needs: step 10. Updates `global_concepts_refined.json`.

**12. Add reverse relations**

```bash
python3 add_reverse_links.py --domain $D --wiki_base_dir $W [--fast]
```

Needs: step 11. Updates `global_concepts_refined.json`. `--fast` uses a template and no LLM.

**13. Final consolidation and Markdown**

```bash
python3 final_consolidation.py --domain $D --wiki_base_dir $W
python3 final_consolidation.py --domain $D --wiki_base_dir $W --from_log   # re-apply the saved decisions, no LLM
```

Needs: step 12. Overwrites `global_concepts_refined.json` and writes `concepts_md/`.
`--from_log` re-applies the decisions in `log/final-deep-verify.decisions.jsonl` without
calling the model. Use it to re-render after changing the code that applies decisions.

**14. Island report**

```bash
python3 island_report.py --domain $D --wiki_base_dir $W
```

Needs: step 13. Writes `islands_report.json`.

**Markdown only**

```bash
python3 export_markdown.py --domain $D --wiki_base_dir $W [--output_dir DIR]
```

Needs: `global_concepts_refined.json`. Writes one page per concept into `DIR` (default
`<run>/concepts_md/`). The exporter removes only the pages it wrote last time, as recorded
in `DIR/.export_manifest.json`. Other files in the folder are not touched.

---

## 7. Options

### 7.1 Runner (`run_pipeline.sh`)

| Option | Default | Meaning |
|---|---|---|
| `--corpus PATH` | — | The corpus file (section 3). Required, unless `--data-dir` and `--domain` are given. |
| `--domain NAME` | none | Optional hint. Names the output folder and focuses the prompts. |
| `--wiki-base-dir PATH` | `/home/compiling-ganesh/24m0829/long_document/wiki` | The folder that holds every run (section 7.3). Environment variable: `WIKI_BASE_DIR`. |
| `--data-dir PATH` | `/home/compiling-ganesh/24m0829/long_document/data/long_documents` | Legacy layout: `<data-dir>/<domain>.jsonl` (section 7.3). Environment variable: `DATA_DIR`. |
| `--chunking` / `--no-chunking` | chunking on | Split long documents into windows (section 8). |
| `--chunk-size N` / `--chunk-overlap N` | 4500 / 400 | Window and overlap, in tokens. |
| `--llm-log full\|compact\|errors` | `compact` | How much of each LLM call to record (section 11). |
| `--servers URL ...` | `config.json` | Replace the server list for this run. |
| `--item-attempts N` | stage default (4) | Tries per LLM call inside a stage. |
| `--stage-budget-minutes M` | stage default (45) | Wall-clock limit per stage (section 12). |
| `--max-stage-attempts N` | 3 | How many times the runner restarts a failed stage. |
| `--retry-forever` | off | Never give up on a stage. This also retries permanent errors, so use it with care. |
| `--skip-verify-pass2` | off | Bypass Pass 2 verification and continue to document consolidation. No verification summary is updated. |
| `--skip-verify-pass1` | off | Bypass Pass 1 verification; refinement runs without audit feedback unless also skipped. |
| `--skip-refinement` | off | Bypass Pass 2 refinement and implicitly bypass Pass 2 verification; consolidate the currently available concepts. Can be combined with `--skip-verify-pass1`. |
| `--skip-verification` | off | Bypass both extraction verification passes; refinement still runs unless `--skip-refinement` is also set. Does not skip final consolidation's candidate review. |
| `--skip-global-fuzzy` | off | In global consolidation, keep exact/alias merges but skip fuzzy candidate generation and LLM review. Near-duplicates may remain separate. |
| `--fresh` | off | Clear the completion markers so that every stage runs again. No output is deleted. |

`./run_pipeline.sh --help` prints these, with the resolved default paths at the end.

To continue a run stalled at Pass 2 verification, rerun the same pipeline command
with `--skip-verify-pass2` (for example, `./run_pipeline.sh --domain biology
--wiki-base-dir /path/to/wiki --skip-verify-pass2`). Skipping leaves existing
audit files intact but does not mark an audit as done. A later run without the
flag can resume it. An existing `verification_summary.json` may be stale; it is
not refreshed when either audit is skipped. Skipping Pass 1 also suppresses
reading old Pass 1 audit files during *new* refinement work; completed
refinements are not undone. To go directly from extraction to document
consolidation, use `--skip-verify-pass1 --skip-refinement` (the latter also
skips Pass 2 verification). Skipping refinement preserves the staging concepts
as they are: if an earlier run already refined them, the skip flag does not
restore the Pass 1 backup. Skipped stages leave their output data and logs
intact and can be resumed on a subsequent run without skip flags.

### 7.2 Stages

Every stage script accepts these flags.

| Flag | Meaning |
|---|---|
| `--domain`, `--wiki_base_dir` | As in 7.1. |
| `--servers URL ...` | Server list. Defaults to `config.json`. |
| `--concurrency_per_server N` | Requests in flight per server. `0` or omitted uses the stage's setting (section 13). |
| `--max_tokens N` | Generation limit. `0` or omitted uses the stage's setting. |
| `--max_attempts N` | Tries per item. |
| `--stage_budget_minutes M` | Wall-clock limit for this stage. `0` uses the stage default. |
| `--no_failover` | Do not move work away from a server that stops answering. |
| `--llm_log full\|compact\|errors` | How much of each LLM call to record. Default `compact`. |
| `--model_name NAME` (`--model`) | The model the endpoints serve. Defaults to `config.json`. |

Three stages also take `--resume` (`extract_concepts.py`, `refine_concepts.py` and
`verify_extraction.py`): it clears the incomplete marks and continues from the
checkpoint. The runner passes it where it exists.

Stage-specific flags: `--phase 1|2` (`verify_extraction.py`), `--skip_fuzzy_review`
(`consolidate_global.py`), `--min_concept_words`
(`fold_thin_concepts.py`), `--min_shared_entities`, `--max_links_per_node`, `--dry_run`
(`link_entities.py`), `--max_island_size` (`audit_islands.py`), `--fast`
(`add_reverse_links.py`), `--from_log` (`final_consolidation.py`), `--output_dir`
(`export_markdown.py`).

### 7.3 Where runs are written, and where legacy corpora are read

**`--wiki-base-dir`** (stage scripts: `--wiki_base_dir`) is the folder that holds every
run. A run writes to `<wiki-base-dir>/<domain>/`, or to `<wiki-base-dir>/default/` when no
domain is given.

- Default: `/home/compiling-ganesh/24m0829/long_document/wiki`. This is the cluster
  location, defined as `DEFAULT_WIKI_BASE_DIR` in `config.py`.
- On another machine, pass your own folder, or set `WIKI_BASE_DIR` before running
  `run_pipeline.sh`. The command-line flag wins over the variable.

**`--data-dir`** (stage scripts: `--data_dir`) is used only by the legacy BRIGHT layout.
When `--corpus` is not given and `--domain NAME` is, the corpus is `<data-dir>/NAME.jsonl`.
When `--corpus` is given, `--data-dir` is ignored.

- Default: `/home/compiling-ganesh/24m0829/long_document/data/long_documents`, defined as
  `DEFAULT_DATA_DIR` in `config.py`.
- Set `DATA_DIR` for `run_pipeline.sh` on another machine.

Both defaults are kept, so the existing cluster commands work without change. Neither
option is required.

```bash
# On the cluster, the defaults apply:
./run_pipeline.sh --corpus /data/docs.jsonl --domain myset

# Elsewhere, name the folder explicitly:
./run_pipeline.sh --corpus ~/docs.jsonl --domain myset --wiki-base-dir ~/wiki
```

---

## 8. Chunking

Chunking is on by default. Long documents are cut into overlapping token windows,
and each window is extracted separately. Each window carries the text just before
and just after it as context.

- `--chunk-size 4500 --chunk-overlap 400` are the defaults. The overlap must be
  smaller than the window.
- Changing either value, the tokenizer or the corpus invalidates the chunk cache. A
  stale cache is never reused.
- `--no-chunking` makes each document a single unit. The tokenizer is not loaded,
  document-level consolidation is skipped, and the pipeline does not need
  `transformers`.

---

## 9. The domain hint

`--domain` is optional. Without it the pipeline still works, and the outputs go to
`default/`.

With a domain, the prompts that judge relevance say so. The final consolidation
and the island audit use it to decide what is off-topic noise. Without a domain
they judge content by whether it fits the rest of the graph. No prompt is told
the domain is "None".

---

## 10. Citations

Each concept lists the document chunks it came from. In the Markdown, each body
paragraph ends with a citation such as `[canon_3f9a1c0b2d4e]`.

- `canon_<hash>` is the id of a unique document. Documents with identical text
  share one id.
- `doc_mapping.json` maps every original id in your corpus to its `canon_<hash>`:

  ```json
  {"doc-001": "canon_3f9a1c0b2d4e", "doc-417": "canon_3f9a1c0b2d4e"}
  ```

To find the source of a citation, look up the `canon_` id in the mapping, and fetch the
record with that id from your corpus.

---

## 11. Logs

Every log of a run is in `<run>/log/`. Each stage writes up to three files, and the
runner writes one per step.

**`<stage>.llm.jsonl`: every LLM call.** One line per HTTP request, so a retried item has
one line per attempt. The file is flushed as it grows, so a killed run still leaves a
readable trail.

| Field | Meaning |
|---|---|
| `timestamp`, `stage`, `event` | `event` is `call` for the HTTP exchange, or `rejection` when the stage refused a reply it received. |
| `ok` | Whether the request succeeded. A `rejection` line is always `ok: false`. |
| `server`, `latency_ms`, `finish_reason` | Where it went, how long it took, and why generation stopped (`length` means it was cut off). |
| `item`, `index`, `attempt` | Which item, and which try. |
| `request.messages[]` | Each prompt message, with `role` and `text`. |
| `response.text` | The reply. |
| `error_type`, `error_message`, `classification` | For failures. |

The three `--llm-log` modes:

- `compact` (default): each prompt message and the reply are cut to 2000 characters.
  The cut is marked in the text, and the true length is recorded.
- `full`: every character of every prompt and reply.
- `errors`: failed calls and rejected replies only.

The debug log and the per-item attempt histories are never truncated, whichever mode you use.

**`<stage>.decisions.jsonl`: what changed, and where.**

| Stage | Events |
|---|---|
| `resolve-stubs` | `stubs_detected` (every stub, major and minor), `stub_synthesized`, `stub_failed`, `stub_downgraded`, `stub_absorbed` (`into` is the concept that took the mention). |
| `consolidate-global` | `exact_merge` (`canonical`, `merged`), `fuzzy_merge` (LLM-approved). |
| `consolidate-documents` | `document_consolidated` (merges per document). |
| `refine-concepts` | `concept_refined` (per chunk). |
| `fold-thin-concepts` | `concept_folded` (`method` is `llm` or `deterministic`), `concept_pruned`. |
| `audit-islands` | `island_verdict` (`decision`, and whether it was `applied`). |
| `add-reverse-links` | `reverse_link_added`. |
| `link-entities` | `entity_link_added`. A dry run writes nothing. |
| `final-macro-scan`, `final-deep-verify` | `macro_scan`, `macro_scan_failed`, `deep_verification`, `deep_verification_failed`. |

To see where a concept went, search for its title:

```bash
grep -h '"Photosynthesis"' log/*.decisions.jsonl
jq -c 'select(.event=="stub_absorbed")' log/resolve-stubs.decisions.jsonl   # jq is optional
```

**`<stage>.debug.jsonl`: failures with full evidence.** Each line has the traceback, the
classification, the attempt number and, where there is one, the raw reply and the
prompt. Nothing is truncated. An item that failed for good also gets a `debug.json`
in its folder under `staging/`, with every attempt it made.

**`<step>.run.log`: console output.** Every attempt of the step, including the server
checks between attempts.

Any stage that could not write a log line prints one warning. It does not fail silently.

---

## 12. Resuming, retries and time limits

**Resuming.** Run the same command again after any interruption. The runner skips every
stage whose outputs are complete and valid, and restarts the stage that was interrupted.
What the restarted stage keeps depends on how it saves its work:

| Stage | Kept when interrupted |
|---|---|
| extract-concepts | Every finished chunk (`extraction_log.json`, `staging/`). It continues at the first unfinished chunk. |
| verify-pass1 | Every finished unit (`verification_log.json`). |
| refine-concepts | Every finished unit (`pass2_log.json`). |
| verify-pass2 | Every finished unit (`verification_pass2_log.json`). |
| consolidate-documents | Every finished document (`doc_consolidation_log.json`). |
| final-consolidation | Its checkpoint (`final_consolidation_log.json`). |
| write-briefs, consolidate-global, resolve-stubs, fold-thin-concepts, audit-islands, add-reverse-links | Nothing until the stage finishes. The stage starts again from its input. |
| link-entities, island-report, export-markdown | Nothing to keep. link-entities makes no LLM calls, and the other two are quick to re-run. |

Re-running a stage clears the completion markers of everything after it, because the
downstream output came from the older upstream output.

**Retries.** Each call is classified:

| Class | Examples | Behaviour |
|---|---|---|
| Transient | timeouts, connection resets, HTTP 429 and 5xx | Retried. Backoff starts at 2 s, doubles, and is capped at 20 s, with ±20% jitter. |
| Recoverable | a reply that is not valid JSON, fails the schema, holds no usable actions, or was cut off at `max_tokens` | Retried. Backoff starts at 0.5 s and is capped at 4 s. |
| Permanent | other HTTP 4xx, such as a request that is too large | Not retried. The item is recorded as failed. |

Each item gets 4 attempts by default. A server that fails three requests in a row is
taken out of rotation. Its work moves to the others, and the stage continues.

**Time limits.** Each stage has a wall-clock budget of 45 minutes by default. When the
budget runs out, the stage stops at once and keeps its checkpoint, so rerunning the
pipeline continues from there. Backoff never waits past the budget.

**The runner.** If a stage exits with an error, the runner checks the servers and
restarts the stage, up to `--max-stage-attempts` times. It waits 15 s before the first
retry, doubling up to 5 min between tries.

---

## 13. Concurrency

Requests are spread across the servers by item index: item `i` always goes to server
`i mod N`. Work is not raced to whichever worker is free, so a slow tail does not pile
onto one server.

Each stage has a built-in profile in `utils/cli.py` (`STAGE_PROFILES`). The concurrency
comes from how much KV cache a server has. A request reserves its prompt plus
`max_tokens`, so a stage with a large `max_tokens` can run fewer requests at once.

| Stage key | Requests per server | Request timeout (s) | `max_tokens` |
|---|---:|---:|---:|
| `extract-concepts` | 8 | 300 | 8192 |
| `refine-concepts` | 8 | 300 | 8192 |
| `consolidate-documents` | 6 | 300 | 8192 |
| `resolve-stubs` | 2 | 45 | 8192 |
| `verify-pass1` | 32 | 1200 | 16384 |
| `verify-pass2` | 32 | 1200 | 16384 |
| `consolidate-global` | 16 | 300 | 4096 |
| `final-macro-scan` | 12 | 300 | 4096 |
| `final-deep-verify` | 24 | 60 | 1000 |
| `fold-thin-concepts` | 16 | 45 | 4096 |
| `write-briefs` | 40 | 40 | 256 |
| `audit-islands` | 32 | 60 | 400 |
| `add-reverse-links` | 48 | 60 | 256 |

Every stage makes 4 attempts per item and has a 45-minute budget, unless you change them.

These numbers assume two A40 GPUs per server, tensor parallel 2, and a 32k context.
Stub resolution is kept low on purpose. Each stub fans out over up to five chunks, so
two stubs in flight means about ten requests.

**Tuning.** vLLM prints its real capacity at startup, in the lines "GPU KV cache size"
and "Maximum concurrency for 32,768 tokens per request". Raise a stage's value if there
is room, and lower it if requests queue or time out. Override a stage in `config.json`.
A number sets the requests per server:

```json
"stage_concurrency": {"extract-concepts": 12, "write-briefs": 64}
```

An object can also set `attempts` and `budget_minutes`:

```json
"stage_concurrency": {"add-reverse-links": {"concurrency_per_server": 48, "attempts": 6, "budget_minutes": 60}}
```

For one run, use `--concurrency_per_server N` on the stage script. The precedence is the
command line, then `config.json`, then the built-in profile. Adding or removing servers
needs no change: the work is re-spread over whatever is listed.

`max_tokens` is not a tuning knob. It is the length the stage was built for, and
changing it changes the output.

---

## 14. Verification and the improvement report

Verification scores each extraction against its source. Each unit gets a
`coverage_score`, a sufficiency flag, and the concepts and facts it missed.
The audit checks the whole chunk but reports only the highest-priority unique
omissions (up to 8 concepts, 20 facts, and 8 hallucinations per chunk), not a
copy of the source. These are feedback limits, not extraction limits; a chunk
with more than 20 important missing facts may need smaller verification units
for exhaustive feedback. Replies stopped by `max_tokens` or missing required
fields are retried rather than accepted. On `--resume`, checkpointed verdicts
that no longer pass the verifier's schema are re-audited.

The pipeline verifies twice. Verification 1 scores the Pass 1 concepts. Verification 2
re-scores the **same chunks** against the Pass 2 output. The report shows both scores and
the change between them. That change is the improvement Pass 2 bought.

Example layout (the numbers are illustrative, not a real run):

```
VERIFICATION SUMMARY REPORT [MYSET]
  Total Chunks Verified : 1204
  Average Coverage Score: 0.812
  Pass Rate (Sufficient): 71.4%
  Total Missed Concepts : 318
  Total Missed Facts    : 904

VERIFICATION IMPROVEMENT [MYSET]  PASS 1 -> PASS 2
  Metric              Before     After     Delta
  Average Coverage     0.812     0.901    +0.089
  Pass Rate (%)         71.4      88.2    +16.8p
  ...
```

Both runs are kept in `verification_summary.json` in the run directory.

---

## 15. Code layout

```
run_pipeline.sh                  the runner: every stage, in order, resumable
run_push.sh                      copies the repository to the cluster (rsync)
check_servers.py                 probes the servers; also the runner's preflight
extract_concepts.py              stage 1   extract-concepts (corpus -> staging)
verify_extraction.py             stages 2 and 4   verify-pass1 and verify-pass2 (--phase 1 | 2)
refine_concepts.py               stage 3   refine-concepts
consolidate_documents.py         stage 5   consolidate-documents (merge within a document)
consolidate_global.py            stage 6   consolidate-global (merge across documents)
resolve_stubs.py                 stage 7   resolve-stubs
fold_thin_concepts.py            stage 8   fold-thin-concepts
write_briefs.py                  stage 9   write-briefs
link_entities.py                 stage 10  link-entities (no LLM)
audit_islands.py                 stage 11  audit-islands
add_reverse_links.py             stage 12  add-reverse-links
final_consolidation.py           stage 13  final-consolidation; writes concepts_md/
island_report.py                 stage 14  island-report
export_markdown.py               Markdown export (called by stage 13; also runs alone)
config.py                        servers, model, default paths, stage overrides (reads config.json)
config.example.json              a template for config.json
requirements.txt                 Python dependencies
utils/                           shared code, one module per concern
  cli.py            argument parsing, stage profiles and per-stage settings
  corpus.py         corpus formats, deduplication, chunking, the cache manifest
  chunks.py         access to the chunk cache
  layout.py         where every file of one run lives
  llm_pool.py       ServerPool: even spread, concurrency caps, failover; pool_for_stage
  attempts.py       every attempt for each item, kept with the failed items
  llm_log.py        the LLM call log
  event_log.py      the JSONL writer behind every log
  retry.py          the retry ladder and backoff
  errors.py         error classes, classification, the stage budget
  checkpoint.py     resumable progress
  json_repair.py    tolerant JSON extraction from model replies
  citation_utils.py citation formatting
  prompts.py        domain-aware prompt wording
  reporting.py      verification report and improvement table
  text.py           shared text helpers
  atomic_io.py      atomic writes
```

---

## 16. Upgrading from the earlier names

The stages, files and folders were renamed so that each name says what the file does.
Data files were not renamed. Existing runs still resume.

**File and folder names**

| Earlier name | Now |
|---|---|
| `pipeline_lib/` | `utils/` |
| `pipeline_config.py` | `config.py` |
| `concept_extraction_pass_1.py` | `extract_concepts.py` |
| `verify_concept_extraction.py` | `verify_extraction.py` |
| `concept_refinement_pass_2.py` | `refine_concepts.py` |
| `doc_concept_consolidation.py` | `consolidate_documents.py` |
| `global_concept_consolidation.py` | `consolidate_global.py` |
| `stub_resolution.py` | `resolve_stubs.py` |
| `thin_concept_handler.py` | `fold_thin_concepts.py` |
| `add_concept_briefs.py` | `write_briefs.py` |
| `link_entity_overlap.py` | `link_entities.py` |
| `audit_noise_islands.py` | `audit_islands.py` |
| `synthesize_bidirectional_relations.py` | `add_reverse_links.py` |
| `final_consolidate_and_link.py` | `final_consolidation.py` |
| `analyze_islands.py` | `island_report.py` |
| `export_markdown_wiki.py` | `export_markdown.py` |

**Stage keys** (used in the runner, in log file names and in `config.json`)

| Earlier key | Now |
|---|---|
| `pass1-extraction` | `extract-concepts` |
| `verification` (phase 1) | `verify-pass1` |
| `pass2-refinement` | `refine-concepts` |
| `verification-pass2` | `verify-pass2` |
| `document-consolidation` | `consolidate-documents` |
| `global-consolidation` | `consolidate-global` |
| `stub-resolution` | `resolve-stubs` |
| `thin-concept-folding` | `fold-thin-concepts` |
| `brief-generation` | `write-briefs` |
| `entity-overlap` | `link-entities` |
| `island-audit`, `noise-audit` | `audit-islands` |
| `reverse-relations`, `bidirectional-relations` | `add-reverse-links` |
| `final-analysis` | `island-report` |
| `markdown-export` | `export-markdown` |

The sub-stages `final-macro-scan` and `final-deep-verify` keep their names.

**What you need to do**

1. **Re-key `stage_concurrency` in `config.json`** to the new stage keys. An entry under an
   old key is ignored, and a warning names it. The stage then uses its built-in profile.
2. **Remove a top-level `concurrency_per_server`** from `config.json`, or leave it. It is
   no longer used, and a warning says so. Use `stage_concurrency` instead.
3. **Change `--llm-log full` if you want the old behaviour.** The default is now `compact`.
4. **Update any imports and scripts** that refer to `pipeline_lib` or the old file names.
5. **Nothing else.** Completion markers written under the old step names are renamed
   automatically the first time the runner starts. Data files were not renamed:
   `verification.json`, `verification_log.json`, the `*_log.json` checkpoints,
   `global_concepts_refined.json` and the rest keep their names, so a run that stopped
   part-way resumes where it stopped.

`concepts.log` has been removed from the repository. It was an old log, not an input.

---

## 17. Known limitations

- **Restart granularity.** Six LLM stages (write-briefs, consolidate-global, resolve-stubs,
  fold-thin-concepts, audit-islands and add-reverse-links) save their results only when
  they finish. If one of them is interrupted, it starts again from its input. The runner
  still skips every stage that finished. Per-item checkpoints for these six would cut the
  lost work, and they are not built yet.
- **Not yet run against live model servers.** The stages have been exercised against fake
  vLLM servers only. The first run against your real servers is the first end-to-end test
  of this version, so watch the first run closely.
- **Long functions.** Twenty-four functions are longer than 80 lines. The longest is
  `audit_and_connect_islands` in `audit_islands.py`, at about 330 lines. They are harder to read than they should be. Splitting them is planned for after the first
  real run.
- **Preemption.** A preempted Slurm job is not restarted automatically. Submit the runner
  again, and it resumes.
- **Default paths.** The defaults in `config.py` point at the original cluster. Pass
  `--wiki-base-dir` and `--data-dir`, or set `WIKI_BASE_DIR` and `DATA_DIR`.
- **The chunk cache** is reused only when its manifest matches exactly. Nothing else is
  cached across runs.
- **Entity linking** can add many links on a large corpus. Lower `--max_links_per_node`
  if the graph becomes too dense.
