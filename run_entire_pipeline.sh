#!/usr/bin/env bash
# Resumable, retrying end-to-end runner for the Wiki knowledge graph pipeline.
#
# Usage:
#   ./run_entire_pipeline.sh [domain] [--resume | --fresh]
#       [--skip-extraction] [--skip-embed]
#       [--max-stage-attempts N | --retry-forever]
#       [--wiki-base-dir PATH] [--data-dir PATH]
#
# --resume skips completed steps recorded under <wiki-base-dir>/<domain>/.pipeline_state
# and retries the failed/incomplete step. --fresh clears only those completion markers;
# it does not delete generated wiki data or logs. --retry-forever is useful for a
# monitored unattended run when transient service outages are expected, but will
# also keep retrying permanent configuration/input errors.

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DEFAULT_DATA_DIR="/home/compiling-ganesh/24m0829/long_document/data/long_documents"

DOMAIN="biology"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
SKIP_EXTRACTION=false
SKIP_EMBED=false
RESUME=false
FRESH=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3

usage() {
    cat <<'EOF'
Usage:
  ./run_entire_pipeline.sh [domain] [--resume | --fresh]
      [--skip-extraction] [--skip-embed]
      [--max-stage-attempts N | --retry-forever]
      [--wiki-base-dir PATH] [--data-dir PATH]

--resume skips completed steps and retries the failed/incomplete step.
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
        --skip-embed|--skip-embedding)
            SKIP_EMBED=true
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN_DIR="${WIKI_BASE_DIR}/${DOMAIN}"
STATE_DIR="${DOMAIN_DIR}/.pipeline_state"
LOG_DIR="${DOMAIN_DIR}/pipeline_logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

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

PIPELINE_STEPS=(
    preflight pass1-extraction verification pass2-refinement document-consolidation
    global-consolidation stub-resolution thin-concept-folding brief-generation entity-overlap
    embedding-bridging noise-audit bidirectional-relations final-consolidation final-analysis
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
            [[ -s "$DOMAIN_DIR/verification_log.json" ]]
            ;;
        pass2-refinement)
            [[ -s "$DOMAIN_DIR/pass2_log.json" ]]
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
        embedding-bridging)
            [[ -s "$DOMAIN_DIR/concept_embeddings.npy" && -s "$DOMAIN_DIR/concept_embeddings.meta.json" ]] && json_has_concepts "$DOMAIN_DIR/global_concepts_refined.json"
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

        if "$@" 2>&1 | tee -a "$step_log"; then
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

        # A health probe on retries helps distinguish service outages from a
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
 Skip embeddings:  ${SKIP_EMBED}
 Resume:           ${RESUME}
 Retry forever:    ${RETRY_FOREVER}
 Max stage tries:  ${MAX_STAGE_ATTEMPTS}
 Checkpoints:      ${STATE_DIR}
 Logs:             ${LOG_DIR}
========================================================================
EOF

run_step preflight python3 "$SCRIPT_DIR/check_servers.py"

if [[ "$SKIP_EXTRACTION" == false ]]; then
    run_step pass1-extraction python3 "$SCRIPT_DIR/concept_extraction_pass_1.py" \
        --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --data_dir "$DATA_DIR" --resume
    run_step verification python3 "$SCRIPT_DIR/verify_concept_extraction.py" \
        --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
    run_step pass2-refinement python3 "$SCRIPT_DIR/concept_refinement_pass_2.py" \
        --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --resume
    run_step document-consolidation python3 "$SCRIPT_DIR/doc_concept_consolidation.py" \
        --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
else
    echo ">>> Skipping extraction, verification, Pass 2, and document consolidation."
fi

run_step global-consolidation python3 "$SCRIPT_DIR/global_concept_consolidation.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step stub-resolution python3 "$SCRIPT_DIR/stub_resolution.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step thin-concept-folding python3 "$SCRIPT_DIR/thin_concept_handler.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step brief-generation python3 "$SCRIPT_DIR/add_concept_briefs.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step entity-overlap python3 "$SCRIPT_DIR/link_entity_overlap.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --min_shared_entities 2

if [[ "$SKIP_EMBED" == false ]]; then
    run_step embedding-bridging python3 "$SCRIPT_DIR/embed_and_bridge_islands.py" \
        --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
else
    echo ">>> Skipping embedding/island-bridging stage (--skip-embed)."
fi

run_step noise-audit python3 "$SCRIPT_DIR/audit_noise_islands.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --max_island_size 5
run_step bidirectional-relations python3 "$SCRIPT_DIR/synthesize_bidirectional_relations.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step final-consolidation python3 "$SCRIPT_DIR/final_consolidate_and_link.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
run_step final-analysis python3 "$SCRIPT_DIR/analyze_islands.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"

cat <<EOF
========================================================================
 PIPELINE RUN COMPLETE FOR DOMAIN: ${DOMAIN}
 Output Markdown files: ${DOMAIN_DIR}/concepts_md/
 Output Concept graph:  ${DOMAIN_DIR}/global_concepts_refined.json
 Logs/checkpoints:      ${LOG_DIR}/ and ${STATE_DIR}/
========================================================================
EOF
