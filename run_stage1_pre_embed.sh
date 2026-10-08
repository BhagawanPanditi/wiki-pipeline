#!/usr/bin/env bash
# Pre-embedding graph stages for the Slurm/GPU workflow.
# Usage: ./run_stage1_pre_embed.sh [domain] [--resume] [--max-stage-attempts N]

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DOMAIN="biology"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
RESUME=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3

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
        --wiki-base-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --wiki-base-dir" >&2; exit 2; }
            WIKI_BASE_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [domain] [--resume] [--retry-forever] [--max-stage-attempts N] [--wiki-base-dir PATH]"
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
PIPELINE_STAGE_STEPS=(
    stage1-preflight stage1-global-consolidation stage1-stub-resolution
    stage1-thin-concept-folding stage1-brief-generation stage1-entity-overlap
    stage2-preflight stage2-noise-audit stage2-bidirectional-relations
    stage2-markdown-export stage2-final-analysis
    global-consolidation stub-resolution thin-concept-folding brief-generation
    entity-overlap embedding-bridging noise-audit bidirectional-relations
    final-consolidation final-analysis
)

if [[ "$RESUME" != true ]]; then
    shopt -s nullglob
    markers=("$STATE_DIR"/stage1-*.done "$STATE_DIR"/stage1-*.failed)
    ((${#markers[@]} == 0)) || rm -f -- "${markers[@]}"
    shopt -u nullglob
fi

cat <<EOF
========================================================================
 STAGE 1 (PRE-EMBED) FOR DOMAIN: ${DOMAIN}
 Resume: ${RESUME} | Max attempts: ${MAX_STAGE_ATTEMPTS} | Retry forever: ${RETRY_FOREVER}
 State: ${STATE_DIR} | Logs: ${LOG_DIR}
========================================================================
EOF

pipeline_stage_step stage1-preflight "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/check_servers.py"
pipeline_stage_step stage1-global-consolidation "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/global_concept_consolidation.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
pipeline_stage_step stage1-stub-resolution "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/stub_resolution.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
pipeline_stage_step stage1-thin-concept-folding "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/thin_concept_handler.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
pipeline_stage_step stage1-brief-generation "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/add_concept_briefs.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
pipeline_stage_step stage1-entity-overlap "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/link_entity_overlap.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --min_shared_entities 2

cat <<EOF
========================================================================
 STAGE 1 COMPLETE. Embedding is optional; submit run_embedding.sbatch only
 if you want semantic island-bridging links. Then run stage 2 with --resume.
========================================================================
EOF
