#!/usr/bin/env bash
# ==============================================================================
# Complete Graph Enrichment Pipeline Runner
# ==============================================================================
# Executes all post-consolidation graph enrichment stages in order for a domain:
#   1. consolidate_global_exact.py
#   2. resynthesize_entities.py
#   3. resolve_stubs.py
#   4. fold_thin_concepts.py
#   5. write_briefs.py
#   6. link_entities.py
#   7. audit_islands.py
#   8. add_reverse_links.py
#   9. final_consolidation.py
#   10. island_report.py
#
# Usage:
#   ./run_enrichment.sh <domain> [--wiki_base_dir <path>] [--concurrency <N>]
# ==============================================================================

set -eo pipefail

DOMAIN="${1:-biology}"
shift || true

WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
CONCURRENCY=16
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVERS=()

MIN_WORDS=30
START_STAGE=1
RESUME_FLAG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wiki_base_dir|--wiki-base-dir)
            WIKI_BASE_DIR="$2"; shift 2 ;;
        --concurrency)
            CONCURRENCY="$2"; shift 2 ;;
        --min_concept_words|--min-concept-words)
            MIN_WORDS="$2"; shift 2 ;;
        --from_stage|--from-stage|--start_stage|--start-stage)
            START_STAGE="$2"; shift 2 ;;
        --resume)
            RESUME_FLAG=1; shift ;;
        --servers)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SERVERS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

COMMON_ARGS=(
    --domain "$DOMAIN"
    --wiki_base_dir "$WIKI_BASE_DIR"
    --concurrency_per_server "$CONCURRENCY"
    --no_failover
)

if [[ ${#SERVERS[@]} -gt 0 ]]; then
    COMMON_ARGS+=(--servers "${SERVERS[@]}")
fi

run_stage() {
    local stage_name="$1"
    shift
    echo ""
    echo "========================================================================"
    echo "  RUNNING STAGE: ${stage_name} [${DOMAIN}]"
    echo "  Time: $(date)"
    echo "========================================================================"
    "$@"
    echo " Stage ${stage_name} finished successfully."
}

echo "Starting Full Enrichment Pipeline for Domain: ${DOMAIN}"
echo "Base Wiki Dir: ${WIKI_BASE_DIR}"
echo "Concurrency:   ${CONCURRENCY} req/server"
echo "Starting Stage: ${START_STAGE}"

# 1. Exact Global Consolidation
if (( START_STAGE <= 1 )); then
run_stage "1. Consolidate Global Exact" \
    python3 "$SCRIPT_DIR/consolidate_global_exact.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
fi

# 2. Re-synthesize Clean Entities (<= 10 per concept)
if (( START_STAGE <= 2 )); then
run_stage "2. Resynthesize Entities" \
    python3 "$SCRIPT_DIR/resynthesize_entities.py" \
    "${COMMON_ARGS[@]}"
fi

# 3. Resolve Outbound Mention Stubs
if (( START_STAGE <= 3 )); then
run_stage "3. Resolve Stubs" \
    python3 "$SCRIPT_DIR/resolve_stubs.py" \
    "${COMMON_ARGS[@]}"
fi

# 4. Fold Thin Concepts
if (( START_STAGE <= 4 )); then
EXTRA_FOLD_ARGS=()
if (( RESUME_FLAG == 1 )); then
    EXTRA_FOLD_ARGS+=(--resume)
fi
run_stage "4. Fold Thin Concepts" \
    python3 "$SCRIPT_DIR/fold_thin_concepts.py" \
    "${COMMON_ARGS[@]}" --min_concept_words "$MIN_WORDS" "${EXTRA_FOLD_ARGS[@]}"
fi

# 5. Write Concept Briefs
if (( START_STAGE <= 5 )); then
run_stage "5. Write Briefs" \
    python3 "$SCRIPT_DIR/write_briefs.py" \
    "${COMMON_ARGS[@]}"
fi

# 6. Link Entities (No LLM, Instant)
if (( START_STAGE <= 6 )); then
run_stage "6. Link Entities" \
    python3 "$SCRIPT_DIR/link_entities.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --min_shared_entities 2
fi

# 7. Audit Small Graph Islands
if (( START_STAGE <= 7 )); then
run_stage "7. Audit Islands" \
    python3 "$SCRIPT_DIR/audit_islands.py" \
    "${COMMON_ARGS[@]}" --max_island_size 5
fi

# 8. Add Reciprocal Reverse Links (Instant)
if (( START_STAGE <= 8 )); then
run_stage "8. Add Reverse Links" \
    python3 "$SCRIPT_DIR/add_reverse_links.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
fi

# 9. Final Alphabet Macro-Scan & Markdown Export
if (( START_STAGE <= 9 )); then
run_stage "9. Final Alphabet Macro-Scan & Markdown Export" \
    python3 "$SCRIPT_DIR/final_consolidation.py" \
    "${COMMON_ARGS[@]}"
fi

# 10. Generate Final Island Report
if (( START_STAGE <= 10 )); then
run_stage "10. Island Report" \
    python3 "$SCRIPT_DIR/island_report.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
fi

echo ""
echo "========================================================================"
echo "  FULL ENRICHMENT PIPELINE COMPLETE FOR [${DOMAIN}]"
echo "  Finished at: $(date)"
echo "  Final Markdown Wiki: ${WIKI_BASE_DIR}/${DOMAIN}/concepts_md/"
echo "  Final Graph:        ${WIKI_BASE_DIR}/${DOMAIN}/global_concepts_refined.json"
echo "========================================================================"
