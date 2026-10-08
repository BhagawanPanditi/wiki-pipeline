#!/usr/bin/env bash
# Post-embedding graph stages for the Slurm/GPU workflow.
# Usage: ./run_stage2_post_embed.sh [domain] [--resume] [--skip-embed]
# Embeddings are performed outside this script by run_embedding.sbatch; pass
# --skip-embed when intentionally continuing without that optional job.

set -Eeuo pipefail

DEFAULT_WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DOMAIN="biology"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-$DEFAULT_WIKI_BASE_DIR}"
RESUME=false
SKIP_EMBED=false
RETRY_FOREVER=false
MAX_STAGE_ATTEMPTS=3

if [[ $# -gt 0 && "$1" != --* ]]; then
    DOMAIN="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=true; shift ;;
        --skip-embed|--skip-embedding) SKIP_EMBED=true; shift ;;
        --retry-forever) RETRY_FOREVER=true; shift ;;
        --max-stage-attempts)
            [[ $# -ge 2 ]] || { echo "Missing value for --max-stage-attempts" >&2; exit 2; }
            MAX_STAGE_ATTEMPTS="$2"; shift 2 ;;
        --wiki-base-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --wiki-base-dir" >&2; exit 2; }
            WIKI_BASE_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [domain] [--resume] [--skip-embed] [--retry-forever] [--max-stage-attempts N] [--wiki-base-dir PATH]"
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
 STAGE 2 (POST-EMBED) FOR DOMAIN: ${DOMAIN}
 Resume: ${RESUME} | Skip embeddings: ${SKIP_EMBED}
 Max attempts: ${MAX_STAGE_ATTEMPTS} | Retry forever: ${RETRY_FOREVER}
 State: ${STATE_DIR} | Logs: ${LOG_DIR}
========================================================================
EOF

if [[ "$SKIP_EMBED" == true ]]; then
    echo ">>> Embedding skipped by request; continuing with graph audit and export."
else
    echo ">>> Embeddings are handled separately by run_embedding.sbatch. If not run, use --skip-embed to make that explicit."
fi

pipeline_stage_step stage2-preflight "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/check_servers.py"
pipeline_stage_step stage2-noise-audit "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/audit_noise_islands.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" --max_island_size 5
pipeline_stage_step stage2-bidirectional-relations "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/synthesize_bidirectional_relations.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"
pipeline_stage_step stage2-markdown-export "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/export_markdown_wiki.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR" \
        --output_dir "$WIKI_BASE_DIR/$DOMAIN/concepts_md"
pipeline_stage_step stage2-final-analysis "$STATE_DIR" "$LOG_DIR" "$RESUME" "$MAX_STAGE_ATTEMPTS" "$RETRY_FOREVER" \
    python3 "$SCRIPT_DIR/analyze_islands.py" --domain "$DOMAIN" --wiki_base_dir "$WIKI_BASE_DIR"

cat <<EOF
========================================================================
 STAGE 2 COMPLETE. Markdown output: ${DOMAIN_DIR}/concepts_md/
========================================================================
EOF
