#!/usr/bin/env bash
# Stage 2: finalise the graph and export the Markdown wiki.
#
# Covers noise-island auditing, bidirectional relation synthesis, Markdown
# export and final topology analysis. Expects stage 1 to have produced
# global_concepts_refined.json.
#
# Usage: ./run_stage2_graph_finalize.sh [domain] [--resume]
#            [--max-stage-attempts N] [--retry-forever]
#            [--wiki-base-dir PATH] [--servers URL [URL ...]]
#            [--item-attempts N] [--stage-budget-minutes MIN]

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DOMAIN="biology"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
RESUME=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3
ITEM_ATTEMPTS=""
STAGE_BUDGET_MINUTES=""
SERVERS=()

if [[ $# -gt 0 && "$1" != --* ]]; then
    DOMAIN="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=true; shift ;;
        --retry-forever) RETRY_FOREVER=true; shift ;;
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
        --servers)
            shift
            [[ $# -gt 0 && "$1" != --* ]] || { echo "--servers needs at least one URL" >&2; exit 2; }
            while [[ $# -gt 0 && "$1" != --* ]]; do SERVERS+=("$1"); shift; done ;;
        -h|--help)
            echo "Usage: $0 [domain] [--resume] [--retry-forever] [--max-stage-attempts N]"
            echo "           [--wiki-base-dir PATH] [--servers URL ...] [--item-attempts N]"
            echo "           [--stage-budget-minutes MIN]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$RETRY_FOREVER" != true && ! "$MAX_STAGE_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-stage-attempts must be a positive integer." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/pipeline_shell_utils.sh"
DOMAIN_DIR="$WIKI_BASE_DIR/$DOMAIN"
PIPELINE_DOMAIN_DIR="$DOMAIN_DIR"
STATE_DIR="$DOMAIN_DIR/.pipeline_state"
LOG_DIR="$DOMAIN_DIR/pipeline_logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

STAGE_ARGS=(--domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR")
if [[ ${#SERVERS[@]} -gt 0 ]]; then STAGE_ARGS+=(--servers "${SERVERS[@]}"); fi
if [[ -n "$ITEM_ATTEMPTS" ]]; then STAGE_ARGS+=(--max_attempts "$ITEM_ATTEMPTS"); fi
if [[ -n "$STAGE_BUDGET_MINUTES" ]]; then STAGE_ARGS+=(--stage_budget_minutes "$STAGE_BUDGET_MINUTES"); fi

PIPELINE_STAGE_STEPS=(
    stage2-preflight stage2-noise-audit stage2-bidirectional-relations
    stage2-markdown-export stage2-final-analysis
    noise-audit bidirectional-relations final-consolidation final-analysis
)

if [[ "$RESUME" != true ]]; then
    shopt -s nullglob
    markers=("$STATE_DIR"/stage2-*.done "$STATE_DIR"/stage2-*.failed)
    ((${#markers[@]} == 0)) || rm -f -- "${markers[@]}"
    shopt -u nullglob
fi

cat <<EOF
========================================================================
 STAGE 2 (GRAPH FINALIZE) FOR DOMAIN: ${DOMAIN}
 Resume: ${RESUME} | Max attempts: ${MAX_STAGE_ATTEMPTS} | Retry forever: ${RETRY_FOREVER}
 Item attempts: ${ITEM_ATTEMPTS:-stage default} | Stage budget: ${STAGE_BUDGET_MINUTES:-stage default} min
 Servers: ${SERVERS[*]:-from config.json}
 State: ${STATE_DIR} | Logs: ${LOG_DIR}
========================================================================
EOF

pipeline_stage_step stage2-preflight "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/check_servers.py"
pipeline_stage_step stage2-noise-audit "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/audit_noise_islands.py" "${STAGE_ARGS[@]}" --max_island_size 5
pipeline_stage_step stage2-bidirectional-relations "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/synthesize_bidirectional_relations.py" "${STAGE_ARGS[@]}"
pipeline_stage_step stage2-markdown-export "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/export_markdown_wiki.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" \
        --output_dir "$WIKI_BASE_DIR/$DOMAIN/concepts_md"
pipeline_stage_step stage2-final-analysis "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/analyze_islands.py" "${STAGE_ARGS[@]}"

cat <<EOF
========================================================================
 STAGE 2 COMPLETE. Markdown output: ${DOMAIN_DIR}/concepts_md/
========================================================================
EOF
