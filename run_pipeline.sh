#!/usr/bin/env bash
# Single end-to-end runner for the Wiki knowledge graph pipeline.
#
# There is one pipeline, not two halves. Run it and it walks every stage in
# order, from corpus ingestion to Markdown export.
#
# Resuming is the default and the only behaviour: a stage whose outputs are
# already complete and valid is skipped, and a stage that is half done picks up
# from its own checkpoint instead of starting over. So re-running this script
# after a crash, a preemption or a Ctrl-C finishes whatever was left and
# continues -- no --resume flag, and nothing to remember about which stage
# stopped.
#
# Usage:
#   ./run_pipeline.sh [--corpus PATH] [--domain NAME]
#       [--wiki-base-dir PATH] [--data-dir PATH]
#       [--chunking | --no-chunking] [--chunk-size N] [--chunk-overlap N]
#       [--llm-log full|compact|errors] [--fresh]
#       [--servers URL [URL ...]] [--max-stage-attempts N | --retry-forever]
#       [--item-attempts N] [--stage-budget-minutes MIN]
#
# The corpus may also be given as the legacy BRIGHT pair: --data-dir plus a
# domain, which resolves to <data-dir>/<domain>.jsonl.
#
# --fresh clears only the completion markers, so every stage runs again. It does
# NOT delete generated wiki data, logs or checkpoints; the stages themselves
# still resume from their own internal state unless you remove it.
#
# Two levels of retry are independent:
#   --max-stage-attempts  how many times the RUNNER restarts a whole stage
#   --item-attempts       how many times a STAGE retries one LLM call
# Each stage also enforces --stage-budget-minutes so backoff can never idle
# forever; on expiry it fails fast and stays resumable.

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DEFAULT_DATA_DIR="/home/compiling-ganesh/24m0829/long_document/data/long_documents"
DEFAULT_RUN_NAME="default"
DEFAULT_CHUNK_SIZE=4500
DEFAULT_CHUNK_OVERLAP=400

DOMAIN=""                      # optional hint; empty means "no domain declared"
CORPUS=""                      # explicit corpus file, if given
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
FRESH=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3
ITEM_ATTEMPTS=""
STAGE_BUDGET_MINUTES=""
CHUNKING=true
CHUNK_SIZE="$DEFAULT_CHUNK_SIZE"
CHUNK_OVERLAP="$DEFAULT_CHUNK_OVERLAP"
LLM_LOG="full"
SERVERS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_pipeline.sh [--corpus PATH] [--domain NAME]
      [--wiki-base-dir PATH] [--data-dir PATH]
      [--chunking | --no-chunking] [--chunk-size N] [--chunk-overlap N]
      [--llm-log full|compact|errors] [--fresh]
      [--servers URL [URL ...]] [--max-stage-attempts N | --retry-forever]
      [--item-attempts N] [--stage-budget-minutes MIN]

Runs every stage in order:
  preflight, pass1-extraction, verification, pass2-refinement,
  verification-pass2, document-consolidation, global-consolidation,
  stub-resolution, thin-concept-folding, brief-generation, entity-overlap,
  noise-audit, bidirectional-relations, final-consolidation, final-analysis

Resuming is the default. Completed stages are skipped, half-finished stages
continue from their own checkpoint, and the run proceeds to the end. Use
--fresh to clear the completion markers and run every stage again.

Input:
  --corpus PATH        Corpus of {id, contents} records: one JSON array (.json)
                       or one object per line (.jsonl).
  --data-dir PATH      Legacy layout; with --domain NAME reads
                       <data-dir>/<NAME>.jsonl.
  --domain NAME        Optional hint. Names the output folder and focuses the
                       LLM prompts. Without it, outputs go to 'default/' and the
                       prompts omit the domain clause entirely.
  --no-chunking        Treat each document as one unit. No tokenizer is loaded,
                       and document consolidation becomes a pass-through.

Output goes to <wiki-base-dir>/<domain or 'default'>/.

Tuning:
  --llm-log MODE       full (default, complete prompt and response), compact,
                       or errors. Written to log/<stage>.llm.jsonl in the run dir.
  --servers URL ...    Override config.json; work is spread evenly across them.
  --item-attempts N    Retries per LLM call inside a stage.
  --stage-budget-minutes MIN
                       Wall-clock ceiling per stage; prevents indefinite backoff.
  --max-stage-attempts N
                       How many times this runner restarts a whole stage.
  --retry-forever      Keep waiting through failures. Can also loop on permanent
                       errors, so the bounded default is safer unattended.
EOF
}

# A bare first argument is still accepted as the domain, for convenience.
if [[ $# -gt 0 && "$1" != --* ]]; then
    DOMAIN="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)
            [[ $# -ge 2 ]] || { echo "Missing value for --corpus" >&2; exit 2; }
            CORPUS="$2"; shift 2 ;;
        --domain)
            [[ $# -ge 2 ]] || { echo "Missing value for --domain" >&2; exit 2; }
            DOMAIN="$2"; shift 2 ;;
        --fresh)
            FRESH=true; shift ;;
        --retry-forever)
            RETRY_FOREVER=true; shift ;;
        --no-chunking)
            CHUNKING=false; shift ;;
        --chunking)
            CHUNKING=true; shift ;;
        --chunk-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --chunk-size" >&2; exit 2; }
            CHUNK_SIZE="$2"; shift 2 ;;
        --chunk-overlap)
            [[ $# -ge 2 ]] || { echo "Missing value for --chunk-overlap" >&2; exit 2; }
            CHUNK_OVERLAP="$2"; shift 2 ;;
        --llm-log)
            [[ $# -ge 2 ]] || { echo "Missing value for --llm-log" >&2; exit 2; }
            LLM_LOG="$2"; shift 2 ;;
        --max-stage-attempts)
            [[ $# -ge 2 ]] || { echo "Missing value for --max-stage-attempts" >&2; exit 2; }
            MAX_STAGE_ATTEMPTS="$2"; shift 2 ;;
        --item-attempts)
            [[ $# -ge 2 ]] || { echo "Missing value for --item-attempts" >&2; exit 2; }
            ITEM_ATTEMPTS="$2"; shift 2 ;;
        --stage-budget-minutes)
            [[ $# -ge 2 ]] || { echo "Missing value for --stage-budget-minutes" >&2; exit 2; }
            STAGE_BUDGET_MINUTES="$2"; shift 2 ;;
        --wiki-base-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --wiki-base-dir" >&2; exit 2; }
            WIKI_BASE_DIR="$2"; shift 2 ;;
        --data-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --data-dir" >&2; exit 2; }
            DATA_DIR="$2"; shift 2 ;;
        --servers)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--servers needs at least one URL" >&2
                exit 2
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SERVERS+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

# ── Validate the command line before touching anything ───────────────────────
if [[ "$RETRY_FOREVER" != true && ! "$MAX_STAGE_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-stage-attempts must be a positive integer." >&2
    exit 2
fi
if [[ -n "$ITEM_ATTEMPTS" && ! "$ITEM_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--item-attempts must be a positive integer." >&2
    exit 2
fi
if [[ -n "$STAGE_BUDGET_MINUTES" ]] &&
    ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$STAGE_BUDGET_MINUTES" 2>/dev/null; then
    echo "--stage-budget-minutes must be a positive number." >&2
    exit 2
fi
case "$LLM_LOG" in
    full|compact|errors) ;;
    *) echo "--llm-log must be one of: full, compact, errors." >&2; exit 2 ;;
esac
if [[ "$CHUNKING" == true ]]; then
    for pair in "chunk-size:$CHUNK_SIZE" "chunk-overlap:$CHUNK_OVERLAP"; do
        name="${pair%%:*}"; value="${pair#*:}"
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "--$name must be a positive integer." >&2
            exit 2
        fi
    done
    if ((CHUNK_OVERLAP >= CHUNK_SIZE)); then
        echo "--chunk-overlap ($CHUNK_OVERLAP) must be smaller than --chunk-size ($CHUNK_SIZE)." >&2
        exit 2
    fi
fi

# Resolve the corpus once, here, so a missing input is reported before any stage
# starts rather than partway through Pass 1.
if [[ -n "$CORPUS" ]]; then
    if [[ ! -f "$CORPUS" ]]; then
        echo "ERROR: --corpus file not found: $CORPUS" >&2
        exit 2
    fi
elif [[ -n "$DOMAIN" ]]; then
    CORPUS="${DATA_DIR}/${DOMAIN}.jsonl"
    if [[ ! -f "$CORPUS" ]]; then
        echo "ERROR: no corpus found at $CORPUS" >&2
        echo "Pass --corpus /path/to/documents.json (or .jsonl), or --data-dir with" >&2
        echo "--domain for the legacy <data-dir>/<domain>.jsonl layout." >&2
        exit 2
    fi
else
    echo "ERROR: no corpus specified." >&2
    echo "Pass --corpus /path/to/documents.json (or .jsonl), or --domain together" >&2
    echo "with --data-dir for the legacy <data-dir>/<domain>.jsonl layout." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The folder name is always usable; the domain hint stays optional and is only
# forwarded to stages when the caller actually declared one, so an undeclared
# corpus never produces prompts about the domain "default".
RUN_NAME="${DOMAIN:-$DEFAULT_RUN_NAME}"
DOMAIN_DIR="${WIKI_BASE_DIR}/${RUN_NAME}"
STATE_DIR="${DOMAIN_DIR}/.pipeline_state"
LOG_DIR="${DOMAIN_DIR}/log"
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Arguments forwarded to every stage script so one command line controls the
# whole run. Per-stage concurrency is deliberately NOT set here: each stage
# resolves its own default from pipeline_lib.cli.STAGE_PROFILES, because the
# sustainable fan-out depends on that stage's max_tokens.
STAGE_ARGS=(--wiki_base_dir "$WIKI_BASE_DIR" --llm_log "$LLM_LOG")
if [[ -n "$DOMAIN" ]]; then
    STAGE_ARGS+=(--domain "$DOMAIN")
fi
if [[ ${#SERVERS[@]} -gt 0 ]]; then
    STAGE_ARGS+=(--servers "${SERVERS[@]}")
fi
if [[ -n "$ITEM_ATTEMPTS" ]]; then
    STAGE_ARGS+=(--max_attempts "$ITEM_ATTEMPTS")
fi
if [[ -n "$STAGE_BUDGET_MINUTES" ]]; then
    STAGE_ARGS+=(--stage_budget_minutes "$STAGE_BUDGET_MINUTES")
fi

# Pass 1 also needs to know how to chunk (or not to).
PASS1_ARGS=("${STAGE_ARGS[@]}" --corpus "$CORPUS" --chunk_size "$CHUNK_SIZE"
            --chunk_overlap "$CHUNK_OVERLAP")
if [[ "$CHUNKING" == true ]]; then
    PASS1_ARGS+=(--chunking)
else
    PASS1_ARGS+=(--no-chunking)
fi

# --fresh clears completion markers only. Generated data, logs and the stages'
# own checkpoints are left alone on purpose: deleting those is a separate,
# destructive decision that this script should not make implicitly.
if [[ "$FRESH" == true ]]; then
    shopt -s nullglob
    old_markers=("$STATE_DIR"/*.done "$STATE_DIR"/*.failed)
    if ((${#old_markers[@]})); then
        rm -f -- "${old_markers[@]}"
        echo ">>> --fresh: cleared ${#old_markers[@]} completion marker(s)."
    fi
    shopt -u nullglob
fi

# ── Output validators ────────────────────────────────────────────────────────
# A stage is skipped only when its outputs exist AND are structurally valid. An
# exit code of 0 with missing or corrupt output is treated as a failure, which
# is what makes resume trustworthy after an interrupted run.

json_has_concepts() {
    # True when the file holds a non-empty "concepts" list.
    local path="$1"
    python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path
try:
    with Path(sys.argv[1]).open(encoding="utf-8") as f:
        data = json.load(f)
    concepts = data.get("concepts")
    raise SystemExit(0 if isinstance(concepts, list) and bool(concepts) else 1)
except Exception:
    raise SystemExit(1)
PY
}

pass1_outputs_current() {
    # The chunk cache is reusable only if it was built from today's corpus with
    # today's tokenizer and chunking parameters. This asks pipeline_lib.corpus
    # for the expected manifest rather than restating the recipe here, so the
    # runner and Pass 1 cannot drift apart.
    python3 - "$SCRIPT_DIR" "$DOMAIN_DIR" "$CORPUS" "$CHUNKING" "$CHUNK_SIZE" "$CHUNK_OVERLAP" <<'PY'
import json
import sys
from pathlib import Path

script_dir, domain_dir, corpus, chunking, chunk_size, overlap = sys.argv[1:]
sys.dont_write_bytecode = True
sys.path.insert(0, script_dir)
try:
    from pipeline_config import load_global_config
    from pipeline_lib.corpus import cache_manifest

    cache_dir = Path(domain_dir) / "cache"
    chunks_path = cache_dir / "chunks_cache.json"
    manifest_path = cache_dir / "chunks_cache_manifest.json"
    log_path = Path(domain_dir) / "extraction_log.json"
    if not chunks_path.is_file() or chunks_path.stat().st_size == 0:
        raise ValueError("missing chunk cache")
    if not log_path.is_file():
        raise ValueError("missing extraction log")

    chunking_on = chunking == "true"
    model = load_global_config()[2]
    expected = cache_manifest(
        Path(corpus),
        tokenizer_name=model if chunking_on else None,
        chunking=chunking_on,
        chunk_size=int(chunk_size),
        overlap=int(overlap))
    with manifest_path.open(encoding="utf-8") as f:
        stored = json.load(f)
    raise SystemExit(0 if stored == expected else 1)
except Exception:
    raise SystemExit(1)
PY
}

verification_complete() {
    # A verification pass is complete only when every unit in the chunk cache has
    # a parseable result file carrying a coverage score. The aggregate summary is
    # written at the end of phase 2, not per chunk.
    local result_name="$1"
    python3 - "$DOMAIN_DIR/staging" "$DOMAIN_DIR/cache/chunks_cache.json" "$result_name" <<'PY'
import json
import sys
from pathlib import Path

staging, chunks_path, result_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
try:
    if not staging.is_dir() or not chunks_path.is_file():
        raise SystemExit(1)
    with chunks_path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit(1)
    missing = 0
    for chunk in chunks:
        result = staging / str(chunk["doc_id"]) / str(chunk["chunk_id"]) / result_name
        if not result.is_file():
            missing += 1
            continue
        try:
            with result.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            missing += 1
            continue
        if not isinstance(data, dict) or "coverage_score" not in data:
            missing += 1
    raise SystemExit(0 if missing == 0 else 1)
except Exception:
    raise SystemExit(1)
PY
}

# Ordered stage list. Rerunning any stage invalidates every marker after it,
# because downstream output was derived from the older upstream output.
PIPELINE_STEPS=(
    preflight pass1-extraction verification pass2-refinement verification-pass2
    document-consolidation global-consolidation stub-resolution thin-concept-folding
    brief-generation entity-overlap noise-audit bidirectional-relations
    final-consolidation final-analysis
)

step_outputs_ready() {
    # True when a step's outputs are present and structurally valid.
    local step="$1"
    local output_file
    case "$step" in
        preflight)
            return 1 # Always recheck live servers for each invocation.
            ;;
        pass1-extraction)
            [[ -s "$DOMAIN_DIR/cache/chunks_cache.json" && \
              -s "$DOMAIN_DIR/cache/chunks_cache_manifest.json" && \
              -s "$DOMAIN_DIR/doc_mapping.json" && \
              -s "$DOMAIN_DIR/extraction_log.json" ]] && pass1_outputs_current
            ;;
        verification)
            [[ -s "$DOMAIN_DIR/verification_log.json" ]] && \
                verification_complete "verification.json"
            ;;
        pass2-refinement)
            [[ -s "$DOMAIN_DIR/pass2_log.json" ]]
            ;;
        verification-pass2)
            [[ -s "$DOMAIN_DIR/verification_pass2_log.json" && \
              -s "$DOMAIN_DIR/verification_summary.json" ]] && \
                verification_complete "verification_pass2.json"
            ;;
        document-consolidation)
            [[ -s "$DOMAIN_DIR/doc_consolidation_log.json" && -d "$DOMAIN_DIR/consolidated" ]] || return 1
            shopt -s nullglob
            local consolidated_outputs=("$DOMAIN_DIR"/consolidated/canon_*/concepts.json)
            shopt -u nullglob
            for output_file in "${consolidated_outputs[@]}"; do
                [[ -s "$output_file" ]] && return 0
            done
            return 1
            ;;
        global-consolidation)
            json_has_concepts "$DOMAIN_DIR/global_concepts.json"
            ;;
        stub-resolution)
            json_has_concepts "$DOMAIN_DIR/global_concepts_stubs_resolved.json"
            ;;
        thin-concept-folding|brief-generation|entity-overlap|noise-audit|bidirectional-relations)
            json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json"
            ;;
        final-consolidation)
            json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json" && \
                [[ -d "$DOMAIN_DIR/concepts_md" ]] && \
                find "$DOMAIN_DIR/concepts_md" -maxdepth 1 -type f -name '*.md' -print -quit | grep -q .
            ;;
        final-analysis)
            [[ -s "$DOMAIN_DIR/islands_report.json" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

invalidate_downstream_markers() {
    # Drop the completion markers for every step after the one being rerun.
    local current="$1"
    local found=false
    local step
    for step in "${PIPELINE_STEPS[@]}"; do
        if [[ "$found" == true ]]; then
            rm -f -- "$STATE_DIR/${step}.done" "$STATE_DIR/${step}.failed"
        fi
        if [[ "$step" == "$current" ]]; then
            found=true
        fi
    done
}

run_step() {
    # Run one stage to completion, retrying the whole stage on failure.
    #
    # Skips the stage when its outputs are already valid. Otherwise the stage
    # runs with its own --resume so a half-finished stage continues rather than
    # restarting. On failure the stage is retried up to --max-stage-attempts
    # times with exponential backoff, probing server health first so an outage is
    # distinguishable from a deterministic bug.
    local step="$1"
    shift
    local marker="$STATE_DIR/${step}.done"
    local failed_marker="$STATE_DIR/${step}.failed"
    local step_log="$LOG_DIR/${step}.run.log"
    local attempt=1
    local delay=15
    local rc=0

    if [[ -s "$marker" ]] && step_outputs_ready "$step"; then
        echo ">>> [SKIP] $step: outputs already complete and valid."
        return 0
    fi
    if [[ -e "$marker" ]]; then
        echo ">>> [RERUN] $step: checkpoint present but outputs are missing or invalid."
        rm -f -- "$marker"
    fi
    if [[ "$step" != "preflight" ]]; then
        invalidate_downstream_markers "$step"
    fi

    while true; do
        {
            printf '\n===== %s | attempt %d | %s =====\n' "$step" "$attempt" "$(date -Is)"
            printf 'Command:'
            printf ' %q' "$@"
            printf '\n'
        } | tee -a "$step_log"

        # Unbuffered so a long stage streams into the log instead of appearing
        # to hang; stderr is merged so failures are captured in order.
        if PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$step_log"; then
            if [[ "$step" == "preflight" ]] || step_outputs_ready "$step"; then
                printf 'completed_at=%s\nattempts=%s\n' "$(date -Is)" "$attempt" > "${marker}.tmp"
                mv -f -- "${marker}.tmp" "$marker"
                rm -f -- "$failed_marker"
                echo ">>> Completed: $step (attempt $attempt)"
                return 0
            fi
            rc=1
            echo ">>> Step '$step' exited successfully but its required outputs are missing or invalid." >&2
        else
            rc=$?
        fi

        printf 'failed_at=%s\nattempt=%s\nexit_code=%s\n' "$(date -Is)" "$attempt" "$rc" > "${failed_marker}.tmp"
        mv -f -- "${failed_marker}.tmp" "$failed_marker"
        rm -f -- "$marker"
        echo ">>> Step '$step' failed with exit code $rc." >&2

        if [[ "$RETRY_FOREVER" != true && "$attempt" -ge "$MAX_STAGE_ATTEMPTS" ]]; then
            echo "ERROR: '$step' failed after $attempt attempt(s). Check $step_log, then rerun" >&2
            echo "       this script: it resumes from where it stopped." >&2
            return "$rc"
        fi

        # A health probe on retries distinguishes a service outage from a
        # deterministic stage failure. The stage still owns its own validation.
        if [[ "$step" != "preflight" ]]; then
            echo ">>> Checking configured LLM servers before retry..." | tee -a "$step_log"
            if ! python3 "$SCRIPT_DIR/check_servers.py" 2>&1 | tee -a "$step_log"; then
                echo ">>> One or more servers are still unhealthy; waiting before retry." | tee -a "$step_log"
            fi
        fi

        echo ">>> Retrying '$step' in ${delay}s..." | tee -a "$step_log"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
        if ((delay > 300)); then
            delay=300
        fi
    done
}

cat <<EOF
========================================================================
 STARTING WIKI PIPELINE
 Corpus:           ${CORPUS}
 Domain hint:      ${DOMAIN:-none (outputs to '${RUN_NAME}/', prompts omit it)}
 Run directory:    ${DOMAIN_DIR}
 Chunking:         ${CHUNKING} (size ${CHUNK_SIZE}, overlap ${CHUNK_OVERLAP})
 LLM call log:     ${LLM_LOG} (per stage, under ${LOG_DIR}/)
 Fresh run:        ${FRESH}
 Retry forever:    ${RETRY_FOREVER}
 Max stage tries:  ${MAX_STAGE_ATTEMPTS}
 Item attempts:    ${ITEM_ATTEMPTS:-stage default}
 Stage budget:     ${STAGE_BUDGET_MINUTES:-stage default} minutes
 Server override:  ${SERVERS[*]:-from config.json}
 Checkpoints:      ${STATE_DIR}
 Logs:              ${LOG_DIR}/ (stage LLM calls, decisions, debug, step output)
========================================================================
EOF

run_step preflight python3 "$SCRIPT_DIR/check_servers.py"

# Extraction and both verification passes. Verification runs twice: once over
# raw Pass 1 output, and again after Pass 2 has merged its repairs, so the
# improvement is measurable.
run_step pass1-extraction python3 "$SCRIPT_DIR/concept_extraction_pass_1.py" \
    "${PASS1_ARGS[@]}" --resume
run_step verification python3 "$SCRIPT_DIR/verify_concept_extraction.py" \
    "${STAGE_ARGS[@]}" --phase 1 --resume
run_step pass2-refinement python3 "$SCRIPT_DIR/concept_refinement_pass_2.py" \
    "${STAGE_ARGS[@]}" --resume
run_step verification-pass2 python3 "$SCRIPT_DIR/verify_concept_extraction.py" \
    "${STAGE_ARGS[@]}" --phase 2 --resume
# With --no-chunking this stage passes every document through without an LLM
# call; it still runs, because later stages read consolidated/.
run_step document-consolidation python3 "$SCRIPT_DIR/doc_concept_consolidation.py" \
    "${STAGE_ARGS[@]}"

run_step global-consolidation python3 "$SCRIPT_DIR/global_concept_consolidation.py" \
    "${STAGE_ARGS[@]}"
run_step stub-resolution python3 "$SCRIPT_DIR/stub_resolution.py" \
    "${STAGE_ARGS[@]}"
run_step thin-concept-folding python3 "$SCRIPT_DIR/thin_concept_handler.py" \
    "${STAGE_ARGS[@]}"
run_step brief-generation python3 "$SCRIPT_DIR/add_concept_briefs.py" \
    "${STAGE_ARGS[@]}"
run_step entity-overlap python3 "$SCRIPT_DIR/link_entity_overlap.py" \
    "${STAGE_ARGS[@]}" --min_shared_entities 2
run_step noise-audit python3 "$SCRIPT_DIR/audit_noise_islands.py" \
    "${STAGE_ARGS[@]}" --max_island_size 5
run_step bidirectional-relations python3 "$SCRIPT_DIR/synthesize_bidirectional_relations.py" \
    "${STAGE_ARGS[@]}"
run_step final-consolidation python3 "$SCRIPT_DIR/final_consolidate_and_link.py" \
    "${STAGE_ARGS[@]}"
run_step final-analysis python3 "$SCRIPT_DIR/analyze_islands.py" \
    "${STAGE_ARGS[@]}"

cat <<EOF
========================================================================
 PIPELINE RUN COMPLETE
 Output Markdown files: ${DOMAIN_DIR}/concepts_md/
 Output concept graph:  ${DOMAIN_DIR}/global_concepts_refined.json
 Citation doc mapping:  ${DOMAIN_DIR}/doc_mapping.json
 Verification scores:   ${DOMAIN_DIR}/verification_summary.json
 Logs:                 ${LOG_DIR}/
========================================================================
EOF
