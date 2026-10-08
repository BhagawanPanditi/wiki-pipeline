# Wiki pipeline

Turns a corpus of documents into a wiki of Markdown pages. Each page is one
concept extracted from the corpus, with a short brief, its body paragraphs and
its relations to other concepts. Every paragraph cites the source document it came
from. The language model runs on your own vLLM servers.

```
corpus (JSON / JSONL)  ->  run_pipeline.sh  ->  <out>/concepts_md/*.md
                                             ->  <out>/doc_mapping.json   (citation ids -> your ids)
                                             ->  <out>/log/*.jsonl        (every LLM call and decision)
```

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
15. [Tests](#15-tests)
16. [Code layout](#16-code-layout)
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

Start one job per node you want to use. The servers can be added or removed
between runs.

**`config.json`.** Copy `config.example.json` to `config.json` and list your
servers. The file is git-ignored, so each machine keeps its own.

```json
{
  "servers": ["http://gpu-node-1:8000/v1", "http://gpu-node-1:8001/v1"],
  "concurrency_per_server": 50,
  "model_name": "Qwen/Qwen3.8-27B",
  "stage_concurrency": {"pass1-extraction": 8}
}
```

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

The run takes all the stages in order and ends with the Markdown wiki in
`<wiki>/<domain>/concepts_md/`.

If the run stops (a crash, a preempted job, Ctrl-C, a server outage), run the
same command again. Finished stages are skipped and the unfinished one carries on
from its checkpoint. There is no separate resume flag. `--fresh` clears the
completion markers so that every stage runs again. It does not delete any output.

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
`DIR/NAME.jsonl`.

---

## 4. Output: the run directory

Everything for one run lives in `<wiki-base-dir>/<domain or default>/`:

```
<run>/
  concepts_md/              the wiki: one .md page per concept           <- the main output
  global_concepts_refined.json   the concept graph as JSON
  doc_mapping.json          original document id -> citation id (canon_...)
  islands_report.json       how the graph splits into connected pieces
  verification_summary.json before/after quality scores (section 14)
  run_config.json           how the corpus was read: chunking, digest, counts

  log/                      every log, one folder (section 11)
    <stage>.llm.jsonl       every LLM call: prompt, response, server, attempt
    <stage>.decisions.jsonl what each stage merged, pruned, linked or folded
    <stage>.debug.jsonl     failures with the full traceback and raw output
    <step>.run.log          the console output of each pipeline step

  staging/<doc>/<chunk>/    per-unit results: concepts.json, verification.json, ...
  consolidated/<doc>/       per-document merged concepts
  cache/                    chunk cache and its manifest (reused on resume)
  extraction_log.json       resume checkpoints (not logs; the runner reads them)
  ...                       other checkpoints and intermediate graphs
```

Files named `*_log.json` at the top level are **resume checkpoints**. They are not
logs. Everything that is a log is in `log/`.

---

## 5. The stages

`run_pipeline.sh` runs these in order. The stage key in parentheses is the name
that tunes its concurrency and appears in the log file names.

| # | Step (runner name) | Stage key | Script | What it does | Reads | Writes |
|---|---|---|---|---|---|---|
| 0 | preflight | — | `check_servers.py` | Probes every server. | `config.json` | — |
| 1 | pass1-extraction | `pass1-extraction` | `concept_extraction_pass_1.py` | Extracts concepts from every chunk or document. | corpus | `staging/`, `cache/`, `doc_mapping.json` |
| 2 | verification | `verification` | `verify_concept_extraction.py --phase 1` | Scores how well each extraction covers its source. | staging | `verification.json` per unit, `verification_log.json` |
| 3 | pass2-refinement | `pass2-refinement` | `concept_refinement_pass_2.py` | Adds what verification found missing. Backs up Pass 1 first. | staging, verification | `pass2_refinement.json`, `concepts_pass1_backup.json`, updated `concepts.json` |
| 4 | verification-pass2 | `verification-pass2` | `verify_concept_extraction.py --phase 2` | Scores the refined output against the same sources. | staging | `verification_pass2.json`, `verification_summary.json` |
| 5 | document-consolidation | `document-consolidation` | `doc_concept_consolidation.py` | Merges duplicates **within** one document. Pass-through when `--no-chunking`. | staging | `consolidated/<doc>/concepts.json` |
| 6 | global-consolidation | `global-consolidation` | `global_concept_consolidation.py` | Merges duplicates **across** documents: exact matches, then fuzzy matches confirmed by the LLM. | consolidated | `global_concepts.json`, `global_consolidation.json` |
| 7 | stub-resolution | `stub-resolution` | `stub_resolution.py` | Defines concepts that were only ever mentioned, and absorbs minor mentions into their parents. | global, cache | `global_concepts_stubs_resolved.json` |
| 8 | thin-concept-folding | `thin-concept-folding` | `thin_concept_handler.py` | Folds very short concepts into the best host concept, or prunes them. | stubs-resolved | `global_concepts_refined.json` |
| 9 | brief-generation | `brief-generation` | `add_concept_briefs.py` | Writes the 2–3 sentence brief for each concept. | refined | refined (briefs added) |
| 10 | entity-overlap | `entity-overlap` | `link_entity_overlap.py` | Links concepts that share named entities. No LLM. | refined | refined (links added) |
| 11 | noise-audit | `island-audit` | `audit_noise_islands.py` | For small disconnected groups, decides PRUNE / MERGE / LINK / KEEP. | refined | refined |
| 12 | bidirectional-relations | `reverse-relations` | `synthesize_bidirectional_relations.py` | Adds the missing back-link for every one-way relation. | refined | refined |
| 13 | final-consolidation | `final-macro-scan`, `final-deep-verify` | `final_consolidate_and_link.py` | Whole-graph scan for duplicates, links and noise, then verification of each candidate group. Writes the Markdown wiki. | refined | refined, `concepts_md/` |
| 14 | final-analysis | — | `analyze_islands.py` | Reports the connected pieces of the finished graph. | refined | `islands_report.json` |

Stages 1–8 read the output of the stage before them. Each stage's input is
described in section 6.

---

## 6. Running one stage on its own

Set the two variables that every stage takes:

```bash
W=/data/wiki      # --wiki-base-dir
D=myset           # --domain (omit it for an undeclared corpus)
```

Each stage needs the outputs of the stages above it, listed under **Needs**. A
stage that finds a missing input stops with a message that names the file.

Every stage accepts `--help`, and the server, retry and log flags described in
section 7.

**0. Check the servers**

```bash
python3 check_servers.py
```

**1. Extract concepts**

```bash
python3 concept_extraction_pass_1.py --domain $D --wiki_base_dir $W \
  --corpus /data/docs.jsonl                  # or --no-chunking, or --chunk_size N --chunk_overlap N
```
Needs: the corpus. Writes: `staging/`, `cache/`, `doc_mapping.json`, `run_config.json`.

**2. Verify Pass 1**

```bash
python3 verify_concept_extraction.py --domain $D --wiki_base_dir $W --phase 1
```
Needs: step 1. Writes: `staging/*/*/verification.json`, `verification_log.json`.

**3. Refine (Pass 2)**

```bash
python3 concept_refinement_pass_2.py --domain $D --wiki_base_dir $W
```
Needs: steps 1 and 2 (it uses the missing concepts and facts that verification listed).
Writes: updated `staging/*/*/concepts.json`, `concepts_pass1_backup.json`, `pass2_log.json`.

**4. Verify Pass 2**

```bash
python3 verify_concept_extraction.py --domain $D --wiki_base_dir $W --phase 2
```
Needs: step 3. Writes: `verification_pass2.json`, `verification_summary.json`, and prints the
improvement report.

**5. Consolidate within documents**

```bash
python3 doc_concept_consolidation.py --domain $D --wiki_base_dir $W
```
Needs: step 3 (or 1). Writes: `consolidated/<doc>/concepts.json`. With `--no-chunking` it
writes the concepts unchanged and makes no LLM call.

**6. Consolidate across documents**

```bash
python3 global_concept_consolidation.py --domain $D --wiki_base_dir $W
```
Needs: step 5. Writes: `global_concepts.json`, `global_consolidation.json`.

**7. Resolve stubs**

```bash
python3 stub_resolution.py --domain $D --wiki_base_dir $W
```
Needs: step 6, and `cache/chunks_cache.json` from step 1. Writes:
`global_concepts_stubs_resolved.json`.

**8. Fold thin concepts**

```bash
python3 thin_concept_handler.py --domain $D --wiki_base_dir $W --min_concept_words 30
```
Needs: step 7. Writes: `global_concepts_refined.json`.

**9. Write briefs**

```bash
python3 add_concept_briefs.py --domain $D --wiki_base_dir $W
```
Needs: step 8. Updates `global_concepts_refined.json`.

**10. Link by shared entities** (no LLM)

```bash
python3 link_entity_overlap.py --domain $D --wiki_base_dir $W \
  --min_shared_entities 2 --max_links_per_node 4 [--dry_run]
```
Needs: step 9. Updates `global_concepts_refined.json`. `--dry_run` reports the links and writes
nothing.

**11. Audit small islands**

```bash
python3 audit_noise_islands.py --domain $D --wiki_base_dir $W --max_island_size 5
```
Needs: step 10. Updates `global_concepts_refined.json`.

**12. Add reverse relations**

```bash
python3 synthesize_bidirectional_relations.py --domain $D --wiki_base_dir $W [--fast]
```
Needs: step 11. Updates `global_concepts_refined.json`. `--fast` uses a template and no LLM.

**13. Final consolidation and Markdown**

```bash
python3 final_consolidate_and_link.py --domain $D --wiki_base_dir $W
python3 final_consolidate_and_link.py --domain $D --wiki_base_dir $W --from_log   # re-apply the saved decisions, no LLM
```
Needs: step 12. Overwrites `global_concepts_refined.json` and writes `concepts_md/`.
`--from_log` re-applies the decisions in `log/final-deep-verify.decisions.jsonl` without
calling the model. Use it to re-render after editing the code that applies decisions.

**14. Island report**

```bash
python3 analyze_islands.py --domain $D --wiki_base_dir $W
```
Needs: step 13. Writes `islands_report.json`.

**Markdown only**

```bash
python3 export_markdown_wiki.py --domain $D --wiki_base_dir $W [--output_dir DIR]
```
Needs: `global_concepts_refined.json`. Writes one page per concept into `DIR` (default
`<run>/concepts_md/`). The exporter removes only the pages it wrote last time, as recorded
in `DIR/.export_manifest.json`. Other files in the folder are not touched.

---

## 7. Options

### Runner (`run_pipeline.sh`)

| Option | Default | Meaning |
|---|---|---|
| `--corpus PATH` | — | The corpus file (section 3). |
| `--domain NAME` | none | Optional hint. Names the output folder and focuses the prompts. |
| `--wiki-base-dir PATH` | cluster default | Where runs are written. Also `WIKI_BASE_DIR` in the environment. |
| `--data-dir PATH` | cluster default | Legacy layout: `DIR/<domain>.jsonl`. Also `DATA_DIR`. |
| `--chunking` / `--no-chunking` | chunking | Split long documents into windows (section 8). |
| `--chunk-size N` / `--chunk-overlap N` | 4500 / 400 | Window and overlap in tokens. |
| `--llm-log full\|compact\|errors` | `full` | How much of each LLM call to record (section 11). |
| `--servers URL ...` | `config.json` | Override the server list for this run. |
| `--item-attempts N` | 4 | Tries per LLM call inside a stage. |
| `--stage-budget-minutes M` | 45 | Wall-clock limit per stage (section 12). |
| `--max-stage-attempts N` | 3 | How many times the runner restarts a whole failed stage. |
| `--retry-forever` | off | Never give up on a stage. Also retries permanent errors, so use with care. |
| `--fresh` | off | Clear completion markers so that every stage runs again. |

### Stages

Every stage accepts these:

| Flag | Meaning |
|---|---|
| `--domain`, `--wiki_base_dir` | As above. |
| `--servers URL ...` | Server list. Defaults to `config.json`. |
| `--concurrency_per_server N` | Requests in flight per server. `0` or omitted uses the stage default (section 13). |
| `--max_tokens N` | Generation limit. `0` or omitted uses the stage default. |
| `--max_attempts N` | Tries per item. |
| `--stage_budget_minutes M` | Wall-clock limit. |
| `--no_failover` | Do not move work away from a server that stops answering. |
| `--llm_log full\|compact\|errors` | Call-log detail. |
| `--model_name NAME` (`--model`) | Model served by the endpoints. |

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

To find the source of a citation, look up the `canon_` id in the mapping and fetch the
record with that id from your corpus.

---

## 11. Logs

Every log of a run is in `<run>/log/`. Each stage writes up to three files, and the
runner writes one per step.

**`<stage>.llm.jsonl`: every LLM call.** One line per HTTP request, so a retried
item has one line per attempt.

| Field | Meaning |
|---|---|
| `timestamp`, `stage`, `event` | `event` is `call` for the HTTP exchange, or `rejection` when the stage refused an answer it received. |
| `ok` | Whether the request succeeded. A `rejection` line is always `ok: false`. |
| `server`, `latency_ms`, `finish_reason` | Where it went, how long it took, and why generation stopped (`length` means it was cut off). |
| `item`, `index`, `attempt` | Which item and which try. |
| `request.messages[]` | Each prompt message with `role` and `text`. |
| `response.text` | The complete reply. |
| `error_type`, `error_message`, `classification` | For failures. |

`--llm-log full` (the default) keeps everything, untruncated. `compact` cuts each side
to 2000 characters and records the true length. `errors` keeps only failures.

**`<stage>.decisions.jsonl`: what changed, and where.**

| Stage | Events |
|---|---|
| `stub-resolution` | `stubs_detected` (every stub, major and minor), `stub_synthesized`, `stub_failed`, `stub_downgraded`, `stub_absorbed` (`into` = the concept that took the mention). |
| `global-consolidation` | `exact_merge` (`canonical`, `merged`), `fuzzy_merge` (LLM-approved). |
| `document-consolidation` | `document_consolidated` (merges per document). |
| `pass2-refinement` | `concept_refined` (per chunk). |
| `thin-concept-folding` | `concept_folded` (`method` `llm` or `deterministic`), `concept_pruned`. |
| `island-audit` | `island_verdict` (`decision`, and whether it was `applied`). |
| `reverse-relations` | `reverse_link_added`. |
| `entity-overlap` | `entity_link_added`. Dry runs write nothing. |
| `final-macro-scan`, `final-deep-verify` | `macro_scan`, `macro_scan_failed`, `deep_verification`, `deep_verification_failed`. |

To see where a concept went, search for its title:

```bash
jq -c 'select(.event=="stub_absorbed" or .event=="exact_merge")' log/stub-resolution.decisions.jsonl
grep -h '"Photosynthesis"' log/*.decisions.jsonl
```

**`<stage>.debug.jsonl`: failures with full evidence.** Each line has the traceback, the
classification, the attempt number and, where there is one, the raw response and the
prompt. Nothing is truncated.

**`<step>.run.log`: console output.** Every attempt of the step, including the server
checks between attempts.

Any stage that could not write a log line prints one warning. It does not fail
silently.

---

## 12. Resuming, retries and time limits

**Resuming.** Stages with per-item checkpoints (pass 1, both verification passes,
Pass 2, document consolidation) continue from the items that are not finished.
The runner skips a whole stage when its outputs are complete and valid, and
reruns a stage whose output is missing or corrupt. Rerunning a stage also clears
the completion markers of everything after it, because downstream output came
from the older upstream output.

**Retries.** Each call is classified:

| Class | Examples | Behaviour |
|---|---|---|
| Transient | timeouts, connection resets, HTTP 429 and 5xx | Retried. Backoff 2 s, doubling, capped at 20 s, with ±20% jitter. |
| Recoverable | an answer that is not valid JSON, fails the schema, or was cut off at `max_tokens` | Retried. Backoff 0.5 s, capped at 4 s. |
| Permanent | other HTTP 4xx, such as a request that is too large | Not retried. The item is recorded as failed. |

A server that fails three requests in a row is taken out of rotation. Its work moves to
the others, and the stage continues.

**Time limits.** Each stage has a wall-clock budget of 45 minutes by default. When the
budget runs out, the stage stops at once and keeps its checkpoint, so rerunning the
pipeline continues from there. Backoff never waits past the budget.

**The runner.** If a stage exits with an error, the runner checks the servers and
restarts the stage, up to `--max-stage-attempts` times, waiting 15 s and doubling up
to 5 min between tries.

---

## 13. Concurrency

Requests are spread across the servers by item index: item `i` always goes to server
`i mod N`. Work is not raced to whichever worker is free, so a slow tail does not pile
onto one server.

Each stage has a concurrency default in `pipeline_lib/cli.py` (`STAGE_PROFILES`). It
is set from how much KV cache a server has. A request reserves its prompt plus
`max_tokens`, so a stage with a large `max_tokens` can run fewer requests at once.

| Stage key | Requests per server | Request timeout (s) | `max_tokens` |
|---|---:|---:|---:|
| `pass1-extraction` | 8 | 300 | 8192 |
| `pass2-refinement` | 8 | 300 | 8192 |
| `document-consolidation` | 6 | 300 | 8192 |
| `stub-resolution` | 2 | 45 | 8192 |
| `verification` | 12 | 450 | 4096 |
| `verification-pass2` | 12 | 450 | 4096 |
| `global-consolidation` | 16 | 300 | 4096 |
| `final-macro-scan` | 12 | 300 | 4096 |
| `final-deep-verify` | 24 | 60 | 1000 |
| `thin-concept-folding` | 16 | 45 | 4096 |
| `brief-generation` | 40 | 40 | 256 |
| `island-audit` | 32 | 60 | 400 |
| `reverse-relations` | 48 | 60 | 256 |

These numbers assume two A40 GPUs per server, tensor parallel 2, and a 32k context.
Stub resolution is low on purpose: each stub fans out over up to five chunks, so
its effective fan-out is ten.

**Tuning.** vLLM prints its real capacity at startup, in the lines "GPU KV cache
size" and "Maximum concurrency for 32,768 tokens per request". Raise a stage's value
if there is room, lower it if requests queue or time out. Override any stage in
`config.json`:

```json
"stage_concurrency": {"pass1-extraction": 12, "brief-generation": 64}
```

or for one run with `--concurrency_per_server N`. The precedence is command line,
then `config.json`, then the table above. Adding or removing servers needs no
changes: the work is re-spread over whatever is listed.

`max_tokens` is not a tuning knob. It is the length the stage was built for, and
changing it changes the output.

---

## 14. Verification and the improvement report

Verification scores each extraction against its source. Each unit gets a
`coverage_score`, a sufficiency flag, and the concepts and facts it missed.

The pipeline verifies twice: once on the raw extraction (phase 1), then again
after Pass 2 has refined it (phase 2), on the same sources. The phase 2 report
shows how much Pass 2 improved the result.

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

## 15. Tests

No GPU and no model server are needed. The tests start fake vLLM servers and
run the real stage scripts against them.

```bash
python3 -m unittest discover -s tests -t . -v
```

The suite takes about two minutes. The tests cover retry classification and the
budget, the server pool and its even spread, every stage end to end, the resume
paths, the log and decision records, the corpus formats, and the runner's
behaviour with a fake stage set.

---

## 16. Code layout

```
run_pipeline.sh                 the runner: every stage, in order, resumable
check_servers.py                probes the servers (also the runner's preflight)
concept_extraction_pass_1.py    stage 1  extraction                        (corpus -> staging)
verify_concept_extraction.py    stages 2 and 4  verification               (--phase 1 | 2)
concept_refinement_pass_2.py    stage 3  refinement
doc_concept_consolidation.py    stage 5  merge within a document
global_concept_consolidation.py stage 6  merge across documents
stub_resolution.py              stage 7  stubs
thin_concept_handler.py         stage 8  thin concepts
add_concept_briefs.py           stage 9  briefs
link_entity_overlap.py          stage 10 entity links
audit_noise_islands.py          stage 11 island audit
synthesize_bidirectional_relations.py  stage 12  reverse relations
final_consolidate_and_link.py   stage 13 final consolidation, writes concepts_md/
analyze_islands.py              stage 14 island report
export_markdown_wiki.py         Markdown export (also called by stage 13)
pipeline_config.py              reads config.json
pipeline_lib/                   shared code
  cli.py            argument parsing and per-stage tuning
  corpus.py         corpus formats, deduplication, chunking, cache manifest
  layout.py         where every file of a run lives
  llm_pool.py       ServerPool: even spread, concurrency caps, failover
  llm_log.py        the LLM call log
  event_log.py      the JSONL writer behind every log
  retry.py          retry ladder
  errors.py         error classes, backoff, stage budget
  checkpoint.py     resumable progress
  json_repair.py    tolerant JSON extraction from model replies
  citation_utils.py citation formatting
  prompts.py        domain-aware prompt wording
  reporting.py      verification report and improvement table
  text.py           shared text helpers
  atomic_io.py      atomic writes
tests/                          the test suite
```

---

## 17. Known limitations

- The stages call only the model servers you give them. Nothing is cached across
  runs apart from the chunk cache, and that is reused only when its manifest matches
  exactly.
- A preempted Slurm job is not restarted automatically. Submit the runner again and
  it resumes.
- The default paths in `pipeline_lib/cli.py` point at the original cluster. Pass
  `--wiki-base-dir` and `--data-dir`, or set `WIKI_BASE_DIR` and `DATA_DIR`.
- `concepts.log` (154 KB) is an old log that is still tracked by git. It is not used by the
  pipeline and can be removed.
- Entity linking can add many links on a large corpus. Lower `--max_links_per_node`
  if the graph becomes too dense.
