# Two-Node Dedicated DGX Execution Guide

This document details how to execute the entire extraction, verification, refinement, and enrichment pipeline across two independent DGX machines:
- **Node 1 (DGX 1)**: Dedicated to **`robotics`**
- **Node 2 (DGX 2)**: Dedicated to **`stackoverflow`**

---

## 1. Architecture & Entity Handling Guarantee

1. **Extraction / Verification / Pass 2 Refinement / Document Consolidation**:
   - Focus exclusively on:
     - Core thematic concept identification
     - Dense factual descriptions retaining all numbers, parameters, benchmarks, and experimental bounds
     - Rich 1-sentence outbound relations:
       ```json
       {
         "target_concept": "Target Title",
         "relation": "1 sentence describing how this concept affects or relates to Target Concept."
       }
       ```
   - **`entities` extraction is completely removed** from all extraction and verification prompts/schemas (set to `[]` to prevent token bloat, format confusion, and noisy entity lists).

2. **Dedicated Entity Extraction (`resynthesize_entities.py`)**:
   - High-precision named entities ($\le 10$ per concept) are extracted **exclusively** during this stage, right after global concept consolidation.

3. **Single Master Script for All Enrichment Stages (`run_enrichment.sh`)**:
   - Once Stage 1 (`consolidate_documents.py`) finishes, running `./run_enrichment.sh <domain>` automatically executes all subsequent stages sequentially:
     - Stage 1: `consolidate_global_exact.py` (Exact global consolidation & relation fusion)
     - Stage 2: `resynthesize_entities.py` (Extracts clean $\le 10$ key entities per concept)
     - Stage 3: `resolve_stubs.py` (Synthesizes major stubs & absorbs minor stubs)
     - Stage 4: `fold_thin_concepts.py` (Folds ultra-thin concepts into host concepts)
     - Stage 5: `write_briefs.py` (Generates dense 2-3 sentence briefs)
     - Stage 6: `link_entities.py` (Entity co-occurrence graph linking)
     - Stage 7: `audit_islands.py` (Audits disconnected graph components)
     - Stage 8: `add_reverse_links.py` (Establishes reciprocal graph links)
     - Stage 9: `final_consolidation.py` (Deep verification, reciprocal link descriptions & Markdown wiki export)
     - Stage 10: `island_report.py` (Knowledge graph topology metrics)

---

## 2. Server Configuration

Ensure your vLLM servers are running on each node.
For example, assuming:
- **DGX 1 URL**: `http://cn17-dgx:8001/v1` (or whichever DGX 1 endpoint you use)
- **DGX 2 URL**: `http://cn18-dgx:8001/v1` (or whichever DGX 2 endpoint you use)

*(You can pass `--servers <URL>` directly on the CLI to dedicate each DGX node to its respective domain).*

---

## 3. Terminal 1: Run `robotics` (on DGX 1)

```bash
cd /home/compiling-ganesh/24m0829/long_document/wiki_pipeline

# ── Part 1: Pass 1 Extraction (1,771 chunks) ──
python3 extract_concepts.py \
  --domain robotics \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 2: Pass 1 Verification ──
python3 verify_extraction.py \
  --domain robotics \
  --phase 1 \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 3: Pass 2 Refinement ──
python3 refine_concepts.py \
  --domain robotics \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 4: Pass 2 Verification ──
python3 verify_extraction.py \
  --domain robotics \
  --phase 2 \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 5: Intra-Document Consolidation ──
python3 consolidate_documents.py \
  --domain robotics \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --stage_budget_minutes 300

# ── Part 6: Full Graph Enrichment & Markdown Wiki Export (Stages 1-10) ──
./run_enrichment.sh robotics \
  --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency 16
```

---

## 4. Terminal 2: Run `stackoverflow` (on DGX 2)

```bash
cd /home/compiling-ganesh/24m0829/long_document/wiki_pipeline

# ── Part 1: Pass 1 Extraction (11,049 chunks) ──
python3 extract_concepts.py \
  --domain stackoverflow \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 2: Pass 1 Verification ──
python3 verify_extraction.py \
  --domain stackoverflow \
  --phase 1 \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 3: Pass 2 Refinement ──
python3 refine_concepts.py \
  --domain stackoverflow \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 4: Pass 2 Verification ──
python3 verify_extraction.py \
  --domain stackoverflow \
  --phase 2 \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --resume

# ── Part 5: Intra-Document Consolidation ──
python3 consolidate_documents.py \
  --domain stackoverflow \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency_per_server 16 \
  --no_failover \
  --stage_budget_minutes 300

# ── Part 6: Full Graph Enrichment & Markdown Wiki Export (Stages 1-10) ──
./run_enrichment.sh stackoverflow \
  --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1 \
  --concurrency 16
```

---

## 5. Alternative: Single Command End-to-End

If you want `run_pipeline.sh` to execute everything seamlessly in one go (Parts 1 through 6):
```bash
# For robotics on DGX 1
./run_pipeline.sh --domain robotics --servers http://<DGX_1_IP_OR_HOSTNAME>:8001/v1

# For stackoverflow on DGX 2
./run_pipeline.sh --domain stackoverflow --servers http://<DGX_2_IP_OR_HOSTNAME>:8001/v1
```
