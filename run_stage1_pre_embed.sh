#!/usr/bin/env bash
# run_stage1_pre_embed.sh
# Runs all pipeline stages before the embedding step using your 3 active vLLM A40 servers.
# Usage: ./run_stage1_pre_embed.sh <domain> (default: biology)

set -euo pipefail

DOMAIN="${1:-biology}"
WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo " STAGE 1 (PRE-EMBED): Running Global Deduplication, Stubs, Thin Folding & Briefs"
echo " Domain: ${DOMAIN} (Utilizing all configured vLLM servers)"
echo "========================================================================"

# Pre-flight health check
python3 "${SCRIPT_DIR}/check_servers.py"

# Step 1: Global Deduplication (Lemmatization + Boilerplate removal + LLM Fuzzy Merge)
echo ""
echo ">>> [1/5] Running Global Deduplication..."
python3 "${SCRIPT_DIR}/global_concept_consolidation.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# Step 2: Stub Resolution
echo ""
echo ">>> [2/5] Running Stub Resolution..."
python3 "${SCRIPT_DIR}/stub_resolution.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# Step 3: Thin Concept Folding (<30 words)
echo ""
echo ">>> [3/5] Running Thin Concept Folding..."
python3 "${SCRIPT_DIR}/thin_concept_handler.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# Step 4: Concept Brief Generation (2-3 sentences)
echo ""
echo ">>> [4/5] Generating Concept Briefs..."
python3 "${SCRIPT_DIR}/add_concept_briefs.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# Step 5: Entity Overlap Linker
echo ""
echo ">>> [5/5] Connecting Islands via Shared Entities..."
python3 "${SCRIPT_DIR}/link_entity_overlap.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}" \
    --min_shared_entities 2

echo ""
echo "========================================================================"
echo " STAGE 1 COMPLETE!"
echo " Next step:"
echo " 1. Kill 1 vLLM sbatch job."
echo " 2. Submit sbatch run_embedding.sbatch (or run embed_and_bridge_islands.py)."
echo " 3. Bring vLLM back up, then run ./run_stage2_post_embed.sh ${DOMAIN}"
echo "========================================================================"
