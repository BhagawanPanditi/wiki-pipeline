#!/usr/bin/env bash
# Resumable, retrying end-to-end runner for the Wiki knowledge graph pipeline.
#
# The pipeline is LLM-only: there is no embedding stage, no vector cache and no
# GPU handoff, so a single run goes straight from extraction to Markdown export.
#
# Usage:
#   ./run_entire_pipeline.sh [domain] [--resume | --fresh]
#       [--skip-extraction] [--max-stage-attempts N | --retry-forever]
#       [--wiki-base-dir PATH] [--data-dir PATH] [--servers URL [URL ...]]
#       [--item-attempts N] [--stage-budget-minutes MIN]
#
# --resume skips completed steps recorded under <wiki-base-dir>/<domain>/.pipeline_state
# and retries the failed/incomplete step. --fresh clears only those completion
# markers; it does not delete generated wiki data or logs. --retry-forever waits
# through outages indefinitely but will also loop on permanent errors, so the
# bounded default is safer for unattended runs.
#
# Two levels of retry are independent:
#   --max-stage-attempts  how many times the RUNNER restarts a whole stage
#   --item-attempts       how many times a STAGE retries one LLM call
# Each stage also enforces --stage-budget-minutes so backoff can never idle
# forever; on expiry it fails fast and stays resumable.

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DEFAULT_DATA_DIR="/home/compiling-ganesh/24m0829/long_document/data/long_documents"

DOMAIN="biology"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
SKIP_EXTRACTION=false
RESUME=false
FRESH=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3
ITEM_ATTEMPTS=""
STAGE_BUDGET_MINUTES=""
SERVERS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_entire_pipeline.sh [domain] [--resume | --fresh]
      [--skip-extraction] [--max-stage-attempts N | --retry-forever]
      [--wiki-base-dir PATH] [--data-dir PATH] [--servers URL [URL ...]]
      [--item-attempts N] [--stage-budget-minutes MIN]

Stages: preflight, pass1-extraction, verification, pass2-refinement,
        verification-pass2, document-consolidation, global-consolidation,
        stub-resolution, thin-concept-folding, brief-generation,
        entity-overlap, noise-audit, bidirectional-relations,
        final-consolidation, final-analysis

--resume skips completed steps and retries the failed/incomplete step.
--servers overrides config.json for this run; work is spread evenly across them.
--item-attempts / --stage-budget-minutes tune per-call retries and the stage clock.
--retry-forever keeps waiting through failures; it can also loop on permanent errors.
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
    DOMAIN="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --fresh)
            FRESH=true
            shift
            ;;
        --retry-forever)
            RETRY_FOREVER=true
            shift
            ;;
        --max-stage-attempts)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --max-stage-attempts" >&2
                exit 2
            fi
            MAX_STAGE_ATTEMPTS="$2"
            shift 2
            ;;
        --item-attempts)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --item-attempts" >&2
                exit 2
            fi
            ITEM_ATTEMPTS="$2"
            shift 2
            ;;
        --stage-budget-minutes)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --stage-budget-minutes" >&2
                exit 2
            fi
            STAGE_BUDGET_MINUTES="$2"
            shift 2
            ;;
        --wiki-base-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --wiki-base-dir" >&2
                exit 2
            fi
            WIKI_BASE_DIR="$2"
            shift 2
            ;;
        --data-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --data-dir" >&2
                exit 2
            fi
            DATA_DIR="$2"
            shift 2
            ;;
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
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$RESUME" == true && "$FRESH" == true ]]; then
    echo "Choose either --resume or --fresh, not both." >&2
    exit 2
fi
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN_DIR="${WIKI_BASE_DIR}/${DOMAIN}"
STATE_DIR="${DOMAIN_DIR}/.pipeline_state"
LOG_DIR="${DOMAIN_DIR}/pipeline_logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Arguments forwarded to every stage script so one command line controls the
# whole run. Per-stage concurrency is deliberately NOT set here: each stage
# resolves its own default from pipeline_lib.cli.STAGE_PROFILES, because the
# sustainable fan-out depends on that stage's max_tokens.
STAGE_ARGS=(--domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR")
if [[ ${#SERVERS[@]} -gt 0 ]]; then
    STAGE_ARGS+=(--servers "${SERVERS[@]}")
fi
if [[ -n "$ITEM_ATTEMPTS" ]]; then
    STAGE_ARGS+=(--max_attempts "$ITEM_ATTEMPTS")
fi
if [[ -n "$STAGE_BUDGET_MINUTES" ]]; then
    STAGE_ARGS+=(--stage_budget_minutes "$STAGE_BUDGET_MINUTES")
fi

if [[ "$SKIP_EXTRACTION" == true ]]; then
    if ! python3 - "$DOMAIN_DIR/consolidated" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("canon_*/concepts.json")) if root.is_dir() else []
if not files:
    raise SystemExit(1)
concept_count = 0
for path in files:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        concepts = data.get("concepts")
        if not isinstance(concepts, list):
            raise ValueError("missing concepts list")
        if any(not isinstance(item, dict) or not str(item.get("title", "")).strip() for item in concepts):
            raise ValueError("concept entries must be objects with non-empty titles")
        concept_count += len(concepts)
    except Exception as exc:
        print(f"Invalid consolidated input {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(0 if concept_count else 1)
PY
    then
        echo "ERROR: --skip-extraction requires valid, non-empty concepts in existing inputs at $DOMAIN_DIR/consolidated/canon_*/concepts.json." >&2
        echo "Run extraction and document consolidation first, or remove --skip-extraction." >&2
        exit 2
    fi
fi

# Default behavior is a fresh run. Preserve generated artifacts, but don't let
# old completion markers accidentally make this new run skip work.
if [[ "$RESUME" != true ]]; then
    shopt -s nullglob
    old_markers=("$STATE_DIR"/*.done "$STATE_DIR"/*.failed)
    if ((${#old_markers[@]})); then
        rm -f -- "${old_markers[@]}"
    fi
    shopt -u nullglob
fi

json_has_concepts() {
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

# Ordered stage list. Rerunning any stage invalidates every marker after it.
PIPELINE_STEPS=(
    preflight pass1-extraction verification pass2-refinement verification-pass2
    document-consolidation global-consolidation stub-resolution thin-concept-folding
    brief-generation entity-overlap noise-audit bidirectional-relations
    final-consolidation final-analysis
)

invalidate_downstream_markers() {
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

pass1_outputs_current() {
    python3 - "$SCRIPT_DIR" "$DOMAIN_DIR/cache/chunks_cache.json" \
        "$DOMAIN_DIR/cache/chunks_cache_manifest.json" "$DOMAIN_DIR/extraction_log.json" \
        "$DATA_DIR/${DOMAIN}.jsonl" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

script_dir, chunks_path, manifest_path, log_path, source_path = map(Path, sys.argv[1:])
try:
    if not chunks_path.is_file() or chunks_path.stat().st_size == 0 or not log_path.is_file():
        raise ValueError("missing extraction cache or log")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    with log_path.open(encoding="utf-8") as f:
        extraction_log = json.load(f)
    if not isinstance(extraction_log, dict):
        raise ValueError("invalid extraction log")
    if not source_path.is_file():
        raise ValueError("source input is missing")

    digest = hashlib.sha256()
    with source_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(script_dir))
    from pipeline_config import load_global_config
    model = load_global_config()[2]
    expected = {
        "format_version": 1,
        "source_sha256": digest.hexdigest(),
        "tokenizer": model,
        "chunk_size": 4500,
        "overlap": 400,
    }
    raise SystemExit(0 if manifest == expected else 1)
except Exception:
    raise SystemExit(1)
PY
}

# A verification run is only complete when every chunk has a parseable result
# file; the aggregate summary is written at the end of the second run.
verification_complete() {
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

step_outputs_ready() {
    local step="$1"
    local output_file
    case "$step" in
        preflight)
            return 1 # Always recheck live servers for each invocation.
            ;;
        pass1-extraction)
            [[ -s "$DOMAIN_DIR/cache/chunks_cache.json" && -s "$DOMAIN_DIR/cache/chunks_cache_manifest.json" && \
                -s "$DOMAIN_DIR/extraction_log.json" ]] && pass1_outputs_current
            ;;
        verification)
            [[ -s "$DOMAIN_DIR/verification_log.json" ]] && verification_complete "verification.json"
            ;;
        pass2-refinement)
            [[ -s "$DOMAIN_DIR/pass2_log.json" ]]
            ;;
        verification-pass2)
            [[ -s "$DOMAIN_DIR/verification_pass2_log.json" && -s "$DOMAIN_DIR/verification_summary.json" ]] && \
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
        thin-concept-folding|brief-generation|entity-overlap)
            json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json"
            ;;
        noise-audit|bidirectional-relations)
            json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json"
            ;;
        final-consolidation)
            json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json" && [[ -d "$DOMAIN_DIR/concepts_md" ]] && \
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

run_step() {
    local step="$1"
    shift
    local marker="$STATE_DIR/${step}.done"
    local failed_marker="$STATE_DIR/${step}.failed"
    local step_log="$LOG_DIR/${step}.log"
    local attempt=1
    local delay=15
    local rc=0

    if [[ "$RESUME" == true && -s "$marker" ]] && step_outputs_ready "$step"; then
        echo ">>> [RESUME] Skipping completed step: $step"
        return 0
    fi
    if [[ "$RESUME" == true && -e "$marker" ]]; then
        echo ">>> [RESUME] Checkpoint for '$step' has missing outputs; rerunning it."
        rm -f -- "$marker"
    fi
    if [[ "$RESUME" == true && "$step" != "preflight" ]]; then
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
            echo "ERROR: '$step' failed after $attempt attempt(s). Check $step_log, then resume with --resume." >&2
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
 STARTING END-TO-END WIKI PIPELINE
 Domain:           ${DOMAIN}
 Wiki base:        ${WIKI_BASE_DIR}
 Data directory:   ${DATA_DIR}
 Skip extraction:  ${SKIP_EXTRACTION}
 Resume:           ${RESUME}
 Retry forever:    ${RETRY_FOREVER}
 Max stage tries:  ${MAX_STAGE_ATTEMPTS}
 Item attempts:    ${ITEM_ATTEMPTS:-stage default}
 Stage budget:     ${STAGE_BUDGET_MINUTES:-stage default} minutes
 Server override:  ${SERVERS[*]:-from config.json}
 Checkpoints:      ${STATE_DIR}
 Logs:             ${LOG_DIR}
========================================================================
EOF

run_step preflight python3 "$SCRIPT_DIR/check_servers.py"

if [[ "$SKIP_EXTRACTION" == false ]]; then
    run_step pass1-extraction python3 "$SCRIPT_DIR/concept_extraction_pass_1.py" \
        "${STAGE_ARGS[@]}" --data_dir "$DATA_DIR" --resume
    # Verification runs twice: once over raw Pass 1 output, and again after
    # Pass 2 has merged its repairs, so the improvement is measurable.
    run_step verification python3 "$SCRIPT_DIR/verify_concept_extraction.py" \
        "${STAGE_ARGS[@]}" --phase 1 --resume
    run_step pass2-refinement python3 "$SCRIPT_DIR/concept_refinement_pass_2.py" \
        "${STAGE_ARGS[@]}" --resume
    run_step verification-pass2 python3 "$SCRIPT_DIR/verify_concept_extraction.py" \
        "${STAGE_ARGS[@]}" --phase 2 --resume
    run_step document-consolidation python3 "$SCRIPT_DIR/doc_concept_consolidation.py" \
        "${STAGE_ARGS[@]}"
else
    echo ">>> Skipping extraction, both verification passes, Pass 2, and document consolidation."
fi

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
 PIPELINE RUN COMPLETE FOR DOMAIN: ${DOMAIN}
 Output Markdown files: ${DOMAIN_DIR}/concepts_md/
 Output Concept graph:  ${DOMAIN_DIR}/global_concepts_refined.json
 Verification scores:   ${DOMAIN_DIR}/verification_summary.json
 Logs/checkpoints:      ${LOG_DIR}/ and ${STATE_DIR}/
========================================================================
EOF
