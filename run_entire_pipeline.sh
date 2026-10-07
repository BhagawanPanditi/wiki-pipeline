#!/usr/bin/env bash
# run_entire_pipeline.sh
# End-to-End Execution of the Complete Knowledge Graph Pipeline
# Supports skipping embedding or running all stages in sequence.
#
# Usage:
#   ./run_entire_pipeline.sh <domain> [--skip-extraction] [--skip-embed]
#
# Examples:
#   ./run_entire_pipeline.sh biology --skip-embed
#   ./run_entire_pipeline.sh biology --skip-extraction --skip-embed
#   ./run_entire_pipeline.sh biology

set -euo pipefail

DOMAIN="${1:-biology}"
shift || true

SKIP_EXTRACTION=false
SKIP_EMBED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --skip-embed)
            SKIP_EMBED=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_entire_pipeline.sh <domain> [--skip-extraction] [--skip-embed]"
            exit 1
            ;;
    esac
done

WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo " STARTING END-TO-END WIKI PIPELINE"
echo " Domain:           ${DOMAIN}"
echo " Skip Extraction:  ${SKIP_EXTRACTION}"
echo " Skip Embedding:   ${SKIP_EMBED}"
echo "========================================================================"

# Run pre-flight health check on all configured vLLM servers
python3 "${SCRIPT_DIR}/check_servers.py"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: EXTRACTION & REFINEMENT (Optional if already extracted)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_EXTRACTION" = false ]; then
    echo ""
    echo "========================================================================"
    echo " [PHASE 1/5] RAW CHUNK EXTRACTION, VERIFICATION & PASS 2 REFINEMENT"
    echo "========================================================================"

    echo ""
    echo ">>> [1.1] Pass 1 Concept Extraction..."
    python3 "${SCRIPT_DIR}/concept_extraction_pass_1.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}"

    echo ""
    echo ">>> [1.2] Extraction Verification..."
    python3 "${SCRIPT_DIR}/verify_concept_extraction.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}"

    echo ""
    echo ">>> [1.3] Pass 2 Concept Refinement..."
    python3 "${SCRIPT_DIR}/concept_refinement_pass_2.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}"

    echo ""
    echo ">>> [1.4] Document-Level Consolidation..."
    python3 "${SCRIPT_DIR}/doc_concept_consolidation.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}"
else
    echo ""
    echo ">>> Skipping Phase 1 (Extraction & Document Consolidation)..."
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: GLOBAL GRAPH DEDUPLICATION, STUBS & BRIEFS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [PHASE 2/5] GLOBAL GRAPH DEDUPLICATION, STUBS & BRIEFS"
echo "========================================================================"

echo ""
echo ">>> [2.1] Global Concept Consolidation (Exact + Fuzzy LLM Merge)..."
python3 "${SCRIPT_DIR}/global_concept_consolidation.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo ">>> [2.2] Stub Resolution (Absorbing minor stubs & major synthesis)..."
python3 "${SCRIPT_DIR}/stub_resolution.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo ">>> [2.3] Thin Concept Folding (<30 words folded into host candidates)..."
python3 "${SCRIPT_DIR}/thin_concept_handler.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo ">>> [2.4] Concept Brief Generation (Factual summaries for all concepts)..."
python3 "${SCRIPT_DIR}/add_concept_briefs.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo ">>> [2.5] Statistical Entity Overlap Linker..."
python3 "${SCRIPT_DIR}/link_entity_overlap.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}" \
    --min_shared_entities 2

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: EMBEDDING-BASED ISLAND BRIDGING (Optional)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_EMBED" = false ]; then
    echo ""
    echo "========================================================================"
    echo " [PHASE 3/5] EMBEDDING-BASED ISLAND BRIDGING"
    echo "========================================================================"
    python3 "${SCRIPT_DIR}/embed_and_bridge_islands.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}"
else
    echo ""
    echo ">>> [PHASE 3/5] Skipping Embedding Island Bridging (--skip-embed requested)..."
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: AUDIT NOISE & BIDIRECTIONAL RELATIONS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [PHASE 4/5] AUDIT NOISE ISLANDS & SYNTHESIZE BIDIRECTIONAL RELATIONS"
echo "========================================================================"

echo ""
echo ">>> [4.1] Auditing and Pruning Scraping Noise Islands..."
python3 "${SCRIPT_DIR}/audit_noise_islands.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}" \
    --max_island_size 5

echo ""
echo ">>> [4.2] Synthesizing Bidirectional Relations..."
python3 "${SCRIPT_DIR}/synthesize_bidirectional_relations.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: TWO-PASS FINAL CONSOLIDATION, HIERARCHY & MARKDOWN EXPORT
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [PHASE 5/5] TWO-PASS CONSOLIDATION, RELATION REMAPPING & MARKDOWN EXPORT"
echo "========================================================================"

echo ""
echo ">>> [5.1] Final Alphabetical Consolidation, Reciprocal Links & Remapping..."
python3 "${SCRIPT_DIR}/final_consolidate_and_link.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo ">>> [5.2] Final Topology & Island Inspection..."
python3 "${SCRIPT_DIR}/analyze_islands.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo "========================================================================"
echo " PIPELINE RUN COMPLETE FOR DOMAIN: ${DOMAIN}!"
echo " Output Markdown files: ${WIKI_BASE_DIR}/${DOMAIN}/concepts_md/"
echo " Output Concept graph:  ${WIKI_BASE_DIR}/${DOMAIN}/global_concepts_refined.json"
echo "========================================================================"
