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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wiki_base_dir|--wiki-base-dir)
            WIKI_BASE_DIR="$2"; shift 2 ;;
        --concurrency)
            CONCURRENCY="$2"; shift 2 ;;
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

# 1. Exact Global Consolidation
run_stage "1. Consolidate Global Exact" \
    python3 "$SCRIPT_DIR/consolidate_global_exact.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"

# 2. Re-synthesize Clean Entities (<= 10 per concept)
run_stage "2. Resynthesize Entities" \
    python3 "$SCRIPT_DIR/resynthesize_entities.py" \
    "${COMMON_ARGS[@]}"

# 3. Resolve Outbound Mention Stubs
run_stage "3. Resolve Stubs" \
    python3 "$SCRIPT_DIR/resolve_stubs.py" \
    "${COMMON_ARGS[@]}"

# 4. Fold Thin Concepts
run_stage "4. Fold Thin Concepts" \
    python3 "$SCRIPT_DIR/fold_thin_concepts.py" \
    "${COMMON_ARGS[@]}"

# 5. Write Concept Briefs
run_stage "5. Write Briefs" \
    python3 "$SCRIPT_DIR/write_briefs.py" \
    "${COMMON_ARGS[@]}"

# 6. Link Entities (No LLM, Instant)
run_stage "6. Link Entities" \
    python3 "$SCRIPT_DIR/link_entities.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --min_shared_entities 2

# 7. Audit Small Graph Islands
run_stage "7. Audit Islands" \
    python3 "$SCRIPT_DIR/audit_islands.py" \
    "${COMMON_ARGS[@]}" --max_island_size 5

# 8. Add Reciprocal Reverse Links (Instant)
run_stage "8. Add Reverse Links" \
    python3 "$SCRIPT_DIR/add_reverse_links.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"

# 9. Final Alphabet Macro-Scan & Markdown Export
run_stage "9. Final Alphabet Macro-Scan & Markdown Export" \
    python3 "$SCRIPT_DIR/final_consolidation.py" \
    "${COMMON_ARGS[@]}"

# 10. Generate Final Island Report
run_stage "10. Island Report" \
    python3 "$SCRIPT_DIR/island_report.py" \
    --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"

echo ""
echo "========================================================================"
echo "  FULL ENRICHMENT PIPELINE COMPLETE FOR [${DOMAIN}]"
echo "  Finished at: $(date)"
echo "  Final Markdown Wiki: ${WIKI_BASE_DIR}/${DOMAIN}/concepts_md/"
echo "  Final Graph:        ${WIKI_BASE_DIR}/${DOMAIN}/global_concepts_refined.json"
echo "========================================================================"
