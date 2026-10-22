#!/usr/bin/env bash
# ==============================================================================
# Pipeline Runner for ROBOTICS (Part 1: Extraction & Refinement)
# ==============================================================================
# Dedicated to DGX 1
# ==============================================================================

set -eo pipefail

DOMAIN="robotics"
SERVERS=("http://cn12-dgx:8001/v1" "http://cn14-dgx:8000/v1" "http://cn15-dgx:8002/v1" "http://cn41-l40:8002/v1" "http://cn41-l40:8003/v1" "http://cn43-l40:8001/v1" "http://cn45-l40:8000/v1")
CONCURRENCY=16
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"

echo "========================================================================"
echo "  STARTING ROBOTICS PIPELINE: Extraction -> Refine -> Consolidate"
echo "  Server:        ${SERVER_URL}"
echo "  Concurrency:   ${CONCURRENCY} req/server"
echo "  Wiki Base Dir: ${WIKI_BASE_DIR}"
echo "  Time:          $(date)"
echo "========================================================================"

# 1. Pass 1 Extraction (1,771 chunks)
echo ""
echo ">>> [1/5] Running Pass 1 Extraction for robotics..."
python3 "$SCRIPT_DIR/extract_concepts.py" \
  --domain "$DOMAIN" \
  --wiki_base_dir "$WIKI_BASE_DIR" \
  --concurrency_per_server "$CONCURRENCY" \
  --max_tokens 16000 \
  --no_failover \
  --resume

# 2. Phase 1 Verification
echo ""
echo ">>> [2/5] Running Phase 1 Verification for robotics..."
python3 "$SCRIPT_DIR/verify_extraction.py" \
  --domain "$DOMAIN" \
  --wiki_base_dir "$WIKI_BASE_DIR" \
  --phase 1 \
  --concurrency_per_server "$CONCURRENCY" \
  --max_tokens 16000 \
  --no_failover \
  --resume

# 3. Pass 2 Refinement
echo ""
echo ">>> [3/5] Running Pass 2 Refinement for robotics..."
python3 "$SCRIPT_DIR/refine_concepts.py" \
  --domain "$DOMAIN" \
  --wiki_base_dir "$WIKI_BASE_DIR" \
  --concurrency_per_server "$CONCURRENCY" \
  --max_tokens 32000 \
  --stage_budget_minutes 300 \
  --no_failover \
  --resume

# 4. Phase 2 Verification
echo ""
echo ">>> [4/5] Running Phase 2 Verification for robotics..."
python3 "$SCRIPT_DIR/verify_extraction.py" \
  --domain "$DOMAIN" \
  --wiki_base_dir "$WIKI_BASE_DIR" \
  --phase 2 \
  --concurrency_per_server "$CONCURRENCY" \
  --max_tokens 16000 \
  --no_failover \
  --resume

# 5. Intra-Document Consolidation
echo ""
echo ">>> [5/5] Running Intra-Document Consolidation for robotics..."
python3 "$SCRIPT_DIR/consolidate_documents.py" \
  --domain "$DOMAIN" \
  --wiki_base_dir "$WIKI_BASE_DIR" \
  --concurrency_per_server "$CONCURRENCY" \
  --max_tokens 16000 \
  --no_failover \
  --stage_budget_minutes 300

echo ""
echo "========================================================================"
echo "  ROBOTICS EXTRACTION & CONSOLIDATION COMPLETE!"
echo "  Finished at: $(date)"
echo "  Now run: ./run_enrichment.sh robotics --servers ${SERVER_URL}"
echo "========================================================================"
