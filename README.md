# Wiki Knowledge Graph Pipeline

Builds a canonical concept knowledge graph from long technical documents using a
pool of local vLLM servers. **The pipeline is LLM-only** — the former embedding
stage (`embed_and_bridge_islands.py`, `run_embedding.sbatch`) has been removed
entirely, so there is no vector cache, no second GPU allocation and no
`--skip-embed` flag. Islands are now bridged by the entity-overlap linker and the
bidirectional relation synthesizer instead.

Every stage:

- streams its results and progress as it goes,
- writes untruncated debug/error evidence for every failed attempt,
- can be resumed from its checkpoint,
- spreads work **evenly** across however many servers you hand it,
- and stops on a bounded retry ladder plus a wall-clock budget, so a stage can
  never idle indefinitely in backoff.

---

## Serving the models

Canonical single-node configuration (2× A40, tensor parallel):

```bash
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --host 0.0.0.0 --port 8000
```

Slurm submission:

```bash
sbatch -J vllm_qwen -p a40 --qos=a40 --nodes=1 -x cn22-a40 \
  --gres=gpu:2 --ntasks-per-node=1 --cpus-per-task=16 \
  --mem=128G --time=24:00:00 --output=logs/vllm_qwen_%j.log
```

Environment for the serving job:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_MODELSCOPE=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Check every endpoint before a run:

```bash
python3 check_servers.py
```

---

## Running

```bash
# whole pipeline
./run_entire_pipeline.sh biology

# the two halves, if you release the GPU allocation in between
./run_stage1_graph_build.sh biology      # extraction -> entity overlap
./run_stage2_graph_finalize.sh biology   # noise audit -> islands report
```

Useful flags (all three scripts):

| Flag | Meaning |
| --- | --- |
| `--resume` | Keep completed work, clear incomplete marks, continue. |
| `--skip-extraction` | Start from existing `consolidated/canon_*/concepts.json`. Exits 2 if those inputs are missing or empty. |
| `--servers URL [URL ...]` | Override the server list for this run. |
| `--max-stage-attempts N` | Shell-level retries for a whole stage. |
| `--max_attempts N` | Per-item LLM retries inside a stage. |
| `--stage_budget_minutes M` | Wall-clock ceiling per stage. |
| `--retry-forever` | Keep retrying a stage (ignores `--max-stage-attempts`). |

### Stage order

```
preflight → pass1-extraction → verification → pass2-refinement
          → verification-pass2 → document-consolidation → global-consolidation
          → stub-resolution → thin-concept-folding → brief-generation
          → entity-overlap → noise-audit → bidirectional-relations
          → final-consolidation → final-analysis
```

Rerunning any stage invalidates the checkpoint markers of every stage after it,
so a resumed run rebuilds downstream artifacts instead of trusting stale ones.

---

## Two verification scores, and the improvement between them

`verify_concept_extraction.py` runs **twice**:

| Phase | Scores | Writes |
| --- | --- | --- |
| `--phase 1` | Pass 1's raw extraction | `staging/<doc>/<chunk>/verification.json`, `verification_log.json` |
| `--phase 2` | Pass 2's refined output, same chunks | `verification_pass2.json`, `verification_pass2_log.json` |

Phase 1 runs *before* Pass 2 because Pass 2 consumes its
`audit_feedback.missing_concepts`. Phase 2 runs *after* Pass 2 has merged its
repairs into `concepts.json` (Pass 1's output is preserved alongside as
`concepts_pass1_backup.json`), so the two scores are directly comparable.

Both runs plus the deltas are written to `<wiki>/<domain>/verification_summary.json`,
and phase 2 prints:

```
============================================================
VERIFICATION SUMMARY REPORT [ROBOTICS] - PASS 1 (raw extraction)
============================================================
Total Chunks Verified : 128
Average Coverage Score: 0.812 / 1.000
Pass Rate (Sufficient): 61.7% (79/128)
Total Missed Concepts : 214
Total Missed Facts    : 96

============================================================
VERIFICATION SUMMARY REPORT [ROBOTICS] - PASS 2 (after refinement)
============================================================
Total Chunks Verified : 128
Average Coverage Score: 0.947 / 1.000
Pass Rate (Sufficient): 92.2% (118/128)
Total Missed Concepts : 41
Total Missed Facts    : 12

============================================================
VERIFICATION IMPROVEMENT [ROBOTICS]  PASS 1 -> PASS 2
============================================================
Chunks Compared       : 128
Metric                    Before     After         Delta
------------------------------------------------------------
Average Coverage           0.812     0.947        +0.135
Pass Rate (%)               61.7      92.2        +30.5p
Sufficient Chunks             79       118           +39
Missed Concepts              214        41       -173 -80.8%
Missed Facts                  96        12        -84 -87.5%
============================================================
```

---

## Concurrency per stage

Sustainable fan-out is set by KV cache, not by CPU. On 2× A40 (96 GB) with
tensor parallel 2, a 27B model in BF16 leaves roughly 32 GB for KV — about
130k tokens. Each in-flight request reserves `prompt + max_tokens`, so **the
larger a stage's `max_tokens`, the fewer concurrent requests it can sustain**.
The old global `concurrency_per_server = 50` was roughly 6× too high for the
8192-token extraction stages: vLLM queues rather than crashing, but KV
preemption and recompute lower throughput and start tripping request timeouts.

Defaults in `pipeline_lib/cli.py`:

| Stage | Concurrency / server | `max_tokens` | Request timeout |
| --- | --- | --- | --- |
| `pass1-extraction` | 8 | 8192 | 300 s |
| `pass2-refinement` | 8 | 8192 | 300 s |
| `document-consolidation` | 6 | 8192 | 300 s |
| `verification` / `verification-pass2` | 12 | 4096 | 450 s |
| `global-consolidation` | 16 | 4096 | 300 s |
| `final-macro-scan` | 12 | 4096 | 300 s |
| `stub-resolution` | 10 | 8192 | 45 s |
| `thin-concept-folding` | 24 | 1024 | 45 s |
| `final-deep-verify` | 24 | 1000 | 60 s |
| `island-audit` | 32 | 400 | 60 s |
| `brief-generation` | 40 | 256 | 40 s |
| `reverse-relations` | 48 | 256 | 60 s |

Calibrate against your own allocation: vLLM prints
`GPU KV cache size: N tokens` and
`Maximum concurrency for 32,768 tokens per request: X.XXx` at startup. Divide
that KV budget by your stage's `max_tokens` plus a typical prompt length.

Override without editing code — `config.json`:

```json
{
  "servers": ["http://cn27-a40:8000/v1", "http://cn30-a40:8001/v1"],
  "concurrency_per_server": 50,
  "model_name": "Qwen/Qwen3.8-27B",
  "stage_concurrency": {
    "pass1-extraction": 8,
    "brief-generation": 40
  }
}
```

Precedence is **CLI flag → `config.json` → stage profile**. Pass
`--concurrency_per_server 0` to fall back to the stage default. `config.json` is
gitignored.

### How work is spread

Item *i* goes to server *i mod N*, each server has its own queue and
`concurrency_per_server` workers, so 5 remaining items over 5 servers means one
item each — never five on the first server. If a server stops answering, its
queued items are re-dispatched to a healthy one (disable with `--no_failover`).
Every stage prints the requests each server actually served:

```
Requests served per server:
  http://cn27-a40:8000/v1                  43
  http://cn30-a40:8001/v1                  43
  http://cn31-a40:8002/v1                  42
```

You can change the server count between runs; nothing is pinned.

Eleven of the twelve LLM stages run through `ServerPool` directly.
`final_consolidate_and_link.py` is the exception: its two passes need different
request timeouts (300 s for the macro scan, 60 s for deep verification), so it
builds its own per-server clients — but it uses the same deterministic
partition, the same classified retry ladder, the same wall-clock budget and the
same untruncated debug log. No stage pulls from a single shared queue any more.

---

## Failure handling

Errors are classified, and the classification decides the response:

| Class | Examples | Behaviour |
| --- | --- | --- |
| **Transient** | connection reset, 429, 5xx, timeout | retry, backoff 2 s → 20 s |
| **Recoverable** | empty completion, unparseable JSON, schema validation | retry, backoff 0.5 s → 4 s |
| **Permanent** | 400/404, token-limit truncation, control errors | fail immediately, no retry |

Backoff carries ±20 % jitter. By default each item gets **4 attempts** and each
stage a **45-minute wall-clock budget**. When the budget expires the stage stops
with `StageBudgetExceeded`, leaves its checkpoint intact, and you rerun with
`--resume`. Backoff sleeps are capped by the remaining budget, so a stage can
never sit idle waiting.

Debug evidence is never truncated:

- `<domain>/<stage>_debug.jsonl` — one line per failed attempt, with the full raw
  response, prompt length, server, classification and attempt number.
- `staging/<doc>/<chunk>/debug.json` — Pass 1's per-chunk record, including every
  attempt's raw response and the final classification.

---

## Resume

Every LLM stage keeps a completion log and can be resumed:

| Stage | Checkpoint |
| --- | --- |
| Pass 1 | `extraction_log.json` |
| Verification | `verification_log.json`, `verification_pass2_log.json` |
| Pass 2 | `pass2_log.json` |
| Document consolidation | `consolidation_log.json` |

`--resume` clears *incomplete* marks (and their partial outputs) so they are
retried, while completed items are skipped. A chunk is only trusted as complete
when its log entry is set **and** its output file still parses — a marker whose
artifact went missing is redone. The shell wrappers honour `--resume` too, and
rerunning a stage invalidates the markers of every stage downstream of it.

---

## Repository layout

```
run_entire_pipeline.sh          full pipeline driver
run_stage1_graph_build.sh       extraction -> entity overlap
run_stage2_graph_finalize.sh    noise audit -> islands report
pipeline_shell_utils.sh         shared shell retry/marker helpers
run_push.sh                     rsync code to Prajna
check_servers.py                endpoint preflight

concept_extraction_pass_1.py    chunk + extract concepts
verify_concept_extraction.py    audit extraction (runs twice, --phase 1|2)
concept_refinement_pass_2.py    repair concepts using verification feedback
doc_concept_consolidation.py    per-document merge
global_concept_consolidation.py cross-document canonicalisation
stub_resolution.py              resolve stub concepts
thin_concept_handler.py         fold near-empty concepts
add_concept_briefs.py           2-3 sentence briefs
link_entity_overlap.py          bridge islands via shared entities
audit_noise_islands.py          flag low-signal islands
synthesize_bidirectional_relations.py  reverse-relation text
final_consolidate_and_link.py   final merge + link
export_markdown_wiki.py         concepts_md/ output
analyze_islands.py              islands_report.json

pipeline_lib/                   shared machinery (see below)
pipeline_config.py              config.json loader + defaults
tests/                          74 tests
```

### `pipeline_lib/`

| Module | Provides |
| --- | --- |
| `errors.py` | error classification, `StageBudget`, control exceptions |
| `json_repair.py` | `clean_json_string` — all five historical fence/bare-object/array variants behind two flags |
| `atomic_io.py` | `atomic_write_json` / `atomic_write_text` / `read_json` (temp file + `os.replace` + dir fsync) |
| `checkpoint.py` | `load_bool_log`, `load_json_log`, `save_log_sync/async`, `CheckpointLog` |
| `chunks.py` | `load_cached_chunks` |
| `debug_log.py` | `DebugLogger` — untruncated JSONL error evidence |
| `retry.py` | `backoff_delay`, `retry_async` |
| `llm_pool.py` | `ServerPool` — per-server queues, even spread, failover, streaming |
| `cli.py` | `STAGE_PROFILES`, `build_parser`, `resolve_settings` |
| `reporting.py` | verification metrics, both report blocks, improvement math |
| `citation_utils.py` | citation stripping helpers |

Stage scripts stay at the repository root with their original filenames and
original command lines (plus the new shared flags), so `run_push.sh` and any
existing `sbatch` invocations keep working unchanged.

---

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

74 tests, no GPU required. `tests/fake_vllm.py` serves a real OpenAI-compatible
HTTP endpoint on localhost, so pooling, even spreading, retries, truncation
fail-fast, resume and both verification phases are exercised end to end.
`transformers` and `nltk` are cluster-only dependencies; the Pass 1 tests stub
the tokenizer.

---

## Caching previous stages

```bash
# 1. On Prajna, back up the domain directory
cp -r /home/compiling-ganesh/24m0829/long_document/wiki/biology \
      /home/compiling-ganesh/24m0829/long_document/wiki/biology.bak

# 2. Push updated pipeline code from your local machine
./run_push.sh

# 3. Run Stage 1 (graph build)
./run_stage1_graph_build.sh biology

# 4. Run Stage 2 (finalize) once Stage 1 has finished
./run_stage2_graph_finalize.sh biology
```

The Pass 1 chunk cache is bound to a manifest of
`sha256(source) + tokenizer + chunk_size 4500 + overlap 400`. Changing the input
document, the tokenizer or the chunking parameters invalidates the cache and
forces re-chunking.
