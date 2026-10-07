#!/usr/bin/env bash
# run_stage2_post_embed.sh
# Runs bidirectional link synthesis, markdown wiki export, and graph topology analysis.
# Usage: ./run_stage2_post_embed.sh <domain> (default: biology)

set -euo pipefail

DOMAIN="${1:-biology}"
WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo " STAGE 2 (POST-EMBED): Bidirectional Relations, Markdown Export & Analysis"
echo " Domain: ${DOMAIN}"
echo "========================================================================"

# Pre-flight health check
python3 "${SCRIPT_DIR}/check_servers.py"

# Step 1: Audit and Prune Scraping Noise / Out-of-Domain Islands (Size 1 to 5)
echo ""
echo ">>> [1/4] Auditing and Pruning Noise Islands via LLM..."
python3 "${SCRIPT_DIR}/audit_noise_islands.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}" \
    --max_island_size 5

# Step 2: Synthesize Bidirectional Relations
echo ""
echo ">>> [2/4] Synthesizing Bidirectional Relations..."
python3 "${SCRIPT_DIR}/synthesize_bidirectional_relations.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

# Step 3: Export Markdown Wiki files with [[target_slug|Canonical Title]] and cited paragraphs
echo ""
echo ">>> [3/4] Exporting Final Markdown Concepts to concepts_md/..."
python3 "${SCRIPT_DIR}/export_markdown_wiki.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}" \
    --output_dir "${WIKI_BASE_DIR}/${DOMAIN}/concepts_md"

# Step 4: Analyze Final Island & Component Distribution
echo ""
echo ">>> [4/4] Inspecting Final Island & Component Distribution..."
python3 "${SCRIPT_DIR}/analyze_islands.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "${WIKI_BASE_DIR}"

echo ""
echo "========================================================================"
echo " STAGE 2 COMPLETE! All Markdown files ready in: ${WIKI_BASE_DIR}/${DOMAIN}/concepts_md/"
echo "========================================================================"
