#!/usr/bin/env bash
# ==============================================================================
# End-to-End Wiki Knowledge Graph Pipeline Runner
# ==============================================================================
# Runs all 9 pipeline stages sequentially for a single domain or all 7 domains:
#   Stage 1: extract_concepts.py       (Concept Extraction from Chunks)
#   Stage 2: verify_extraction.py      (Phase 1 Verification Audit)
#   Stage 3: refine_concepts.py        (Second-Pass Concept Refinement)
#   Stage 4: verify_extraction.py      (Phase 2 Verification Delta Audit)
#   Stage 5: consolidate_documents.py (Document-Level Deduplication)
#   Stage 6: consolidate_global.py    (Corpus-Level Global Consolidation)
#   Stage 7: write_briefs_and_entities.py (Briefs & Vital Entities Extraction)
#   Stage 8: link_entities_jaccard.py (Semantic Jaccard Linking & LLM Relations)
#   Stage 9: final_consolidation.py   (Macro/Deep Verify & Hybrid Wiki Export)
#
# Supported Domains:
#   1. sustainable_living
#   2. biology
#   3. earth_science
#   4. economics
#   5. psychology
#   6. robotics
#   7. stackoverflow
#
# Usage:
#   ./pipeline.sh <domain | all> [options]
#
# Examples:
#   ./pipeline.sh sustainable_living
#   ./pipeline.sh all
#   ./pipeline.sh sustainable_living --from-stage 7
#   ./pipeline.sh all --data-dir /custom/data --wiki-base-dir /custom/wiki
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default Paths (Adjustable via CLI flags or environment variables)
DATA_DIR="${DATA_DIR:-/home/compiling-ganesh/24m0829/long_document/wiki_pipeline/data}"
WIKI_BASE_DIR="${WIKI_BASE_DIR:-/home/compiling-ganesh/24m0829/long_document/wiki}"

ALL_DOMAINS=(
    "sustainable_living"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
    "stackoverflow"
)

TARGET_DOMAIN="${1:-sustainable_living}"
shift || true

FROM_STAGE=1
TO_STAGE=9
DISABLE_THINKING=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir|--data_dir)
            DATA_DIR="$2"; shift 2 ;;
        --wiki-base-dir|--wiki_base_dir)
            WIKI_BASE_DIR="$2"; shift 2 ;;
        --from-stage|--from_stage|--start-stage|--start_stage)
            FROM_STAGE="$2"; shift 2 ;;
        --to-stage|--to_stage)
            TO_STAGE="$2"; shift 2 ;;
        --enable-thinking|--enable_thinking)
            DISABLE_THINKING=false; shift ;;
        --disable-thinking|--disable_thinking)
            DISABLE_THINKING=true; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

THINKING_FLAG="--disable_thinking"
if [ "$DISABLE_THINKING" = false ]; then
    THINKING_FLAG="--enable_thinking"
fi

# Print banner
echo "=============================================================================="
echo "          END-TO-END CONCEPT KNOWLEDGE GRAPH PIPELINE RUNNER                  "
echo "=============================================================================="
echo "Script Directory  : $SCRIPT_DIR"
echo "Data Directory    : $DATA_DIR"
echo "Wiki Base Dir     : $WIKI_BASE_DIR"
echo "Target Domain     : $TARGET_DOMAIN"
echo "Stage Range       : Stage $FROM_STAGE -> Stage $TO_STAGE"
echo "Thinking Mode     : $THINKING_FLAG"
echo "=============================================================================="
echo ""

run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    shift 2
    local cmd=("$@")

    if [ "$stage_num" -lt "$FROM_STAGE" ] || [ "$stage_num" -gt "$TO_STAGE" ]; then
        echo ">>> [STAGE $stage_num: $stage_name] SKIPPED (Stage range: $FROM_STAGE-$TO_STAGE)"
        return 0
    fi

    echo ""
    echo "=============================================================================="
    echo ">>> [STAGE $stage_num: $stage_name] Starting at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Command: ${cmd[*]}"
    echo "=============================================================================="

    local start_time=$(date +%s)
    local retries=3
    local attempt=1
    while [ $attempt -le $retries ]; do
        if "${cmd[@]}"; then
            break
        else
            if [ $attempt -lt $retries ]; then
                echo ">>> [STAGE $stage_num WARNING] Command failed on attempt $attempt/$retries. Retrying with --resume in 5s..." >&2
                sleep 5
            else
                echo ">>> [STAGE $stage_num ERROR] Command failed after $retries attempts!" >&2
                exit 1
            fi
        fi
        attempt=$((attempt + 1))
    done
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    echo ">>> [STAGE $stage_num: $stage_name] COMPLETED in ${elapsed}s ($(date '+%Y-%m-%d %H:%M:%S'))"
}

run_domain_pipeline() {
    local domain="$1"
    local input_corpus="$DATA_DIR/${domain}.jsonl"
    local wiki_path="$WIKI_BASE_DIR/${domain}"

    echo ""
    echo "##############################################################################"
    echo "##############################################################################"
    echo "  PROCESSING DOMAIN: [${domain^^}]"
    echo "  Input File : $input_corpus"
    echo "  Wiki Path  : $wiki_path"
    echo "##############################################################################"
    echo "##############################################################################"

    if [ ! -f "$input_corpus" ]; then
        # Check if .json exists
        if [ -f "$DATA_DIR/${domain}.json" ]; then
            input_corpus="$DATA_DIR/${domain}.json"
        else
            echo "ERROR: Input file $input_corpus does not exist!" >&2
            exit 1
        fi
    fi

    mkdir -p "$wiki_path"

    # Stage 1: Concept Extraction (Pass 1)
    run_stage 1 "Concept Extraction (Pass 1)" \
        python3 extract_concepts.py \
            --input_path "$input_corpus" \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 2: Audit Verification (Phase 1)
    run_stage 2 "Verification Audit (Phase 1)" \
        python3 verify_extraction.py \
            --wiki_path "$wiki_path" \
            --phase 1 \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 3: Second-Pass Refinement
    run_stage 3 "Concept Refinement (Pass 2)" \
        python3 refine_concepts.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 4: Second-Pass Verification (Phase 2)
    run_stage 4 "Verification Delta (Phase 2)" \
        python3 verify_extraction.py \
            --wiki_path "$wiki_path" \
            --phase 2 \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 5: Document Consolidation
    run_stage 5 "Document Consolidation" \
        python3 consolidate_documents.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 6: Global Consolidation
    run_stage 6 "Global Consolidation" \
        python3 consolidate_global.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 7: Briefs & Entities
    run_stage 7 "Briefs & Entities Extraction" \
        python3 write_briefs_and_entities.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 8: Jaccard Linking & LLM Relationship Synthesis
    run_stage 8 "Jaccard Linking & Relation Synthesis" \
        python3 link_entities_jaccard.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    # Stage 9: Final Consolidation & Wiki Export
    run_stage 9 "Final Consolidation & Hybrid Wiki Export" \
        python3 final_consolidation.py \
            --wiki_path "$wiki_path" \
            $THINKING_FLAG \
            --resume "${EXTRA_ARGS[@]}"

    echo ""
    echo "=============================================================================="
    echo ">>> DOMAIN PIPELINE COMPLETED: [${domain^^}]"
    echo ">>> Wiki Knowledge Graph & Pages saved to: $wiki_path"
    echo "=============================================================================="
}

TOTAL_START=$(date +%s)

if [ "$TARGET_DOMAIN" = "all" ] || [ "$TARGET_DOMAIN" = "ALL" ]; then
    echo "Running complete pipeline for all 7 domains: ${ALL_DOMAINS[*]}"
    for dom in "${ALL_DOMAINS[@]}"; do
        run_domain_pipeline "$dom"
    done
else
    # Check if target domain is valid
    MATCHED=false
    for dom in "${ALL_DOMAINS[@]}"; do
        if [ "$dom" = "$TARGET_DOMAIN" ]; then
            MATCHED=true
            break
        fi
    done
    if [ "$MATCHED" = false ]; then
        echo "Warning: '$TARGET_DOMAIN' is not in standard list (${ALL_DOMAINS[*]}). Running anyway..."
    fi
    run_domain_pipeline "$TARGET_DOMAIN"
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================================="
echo "PIPELINE EXECUTION FINISHED SUCCESSFULLY!"
echo "Total Execution Time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60)) minutes)"
echo "=============================================================================="
