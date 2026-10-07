#!/usr/bin/env python3
"""
Two-Pass Global Knowledge Graph Consolidation & Hierarchy Resolution
---------------------------------------------------------------------
Step 1 (Alphabet Macro Scan):
  - Partitions concepts strictly by starting character (A-Z, 0-9).
  - Sends all indexed concept titles: "[ID] Title" without aliases or artificial sub-splits.
  - LLM outputs:
      * "merge_pairs": [[id1, id2], ...]
      * "link_pairs": [[id1, id2], ...]
      * "prune_ids": [id1, id2, ...]
  - Distributed across ALL configured vLLM servers in parallel with retry loops & error tracking.
  - Raw responses logged to `macro_scan_log.jsonl`.

Step 2 (Deep Verification with Briefs):
  - Reviews candidate pairs/prune items with their full briefs and top entities.
  - Decides canonical action: MERGE, LINK, PRUNE, or KEEP_SEPARATE.
  - Highly resilient action parser (handles dict, list, alias keys).
  - Distributed across ALL servers with retries & error handling.
  - Decisions logged to `final_consolidation_log.jsonl`.

Step 3 (Graph Update, Bidirectional Remapping & Markdown Export):
  - MERGE:
      * Absorbs duplicate concept into primary canonical concept.
      * Merges descriptions, aliases, entities, and source_chunks.
      * Remaps ALL incoming relations across the entire graph pointing to the absorbed concept
        so they point to the primary concept instead.
      * Combines outbound relations: if the primary concept already has an outbound link to
        the same target, concatenates the relation descriptions instead of duplicating.
  - LINK:
      * If an outbound relation already exists between the pair in either direction, keeps it untouched.
      * If missing, adds the synthesized reciprocal bidirectional link.
  - PRUNE:
      * Removes verified scraping junk and strips any references pointing to it.
  - Re-normalizes all outbound links to surviving canonical titles.
  - Overwrites `global_concepts_refined.json` AND re-exports Markdown files to `concepts_md/`.
  - Runs island analysis.
"""

import os
import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from tqdm import tqdm

import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from pipeline_config import load_global_config
DEFAULT_SERVERS, DEFAULT_CONCURRENCY_PER_SERVER, GLOBAL_MODEL = load_global_config()

# ── Schemas ──────────────────────────────────────────────────────────────────
class MacroCandidateCluster(BaseModel):
    category: str = Field(description="'MERGE', 'LINK', or 'NOISE'")
    titles: List[str] = Field(description="Exact titles belonging to this group/pair")

class VerifiedAction(BaseModel):
    action: str = Field(description="'MERGE', 'LINK', 'PRUNE', or 'KEEP_SEPARATE'")
    primary_title: Optional[str] = Field(default=None)
    absorbed_titles: List[str] = Field(default_factory=list)
    link_description_ab: Optional[str] = Field(default=None)
    link_description_ba: Optional[str] = Field(default=None)

# ── Prompts ──────────────────────────────────────────────────────────────────
PROMPT_MACRO_SCAN = """\
You are an expert technical knowledge graph consolidator auditing concepts for domain: "{domain}".
You are provided with an indexed list of concept titles beginning with letter/group: "{letter}".

Indexed Titles:
{indexed_titles}

TASK:
Examine the list and identify:
1. "merge_pairs": Pairs of IDs that represent the SAME concept or near-duplicates:
   - American vs British spelling variants (e.g. diarrheal vs diarrhoeal).
   - Formatting / hyphenation / spacing splits (e.g. cholodnywent vs cholodny went, proteinprotein vs protein protein).
   - High phrasing overlap or minor word variations (e.g. study authors vs study authorship, additive color mixing vs additive color mixing principles).
   - Generic methodology/demographics splits (e.g. study participant demographics vs study population demographics).
   Format as pairs of IDs: [[id_a, id_b], ...]

2. "link_pairs": Pairs of IDs that are distinct topics or serialized parts, but MUST have direct reciprocal links:
   - Serialized volumes or numbers (e.g. Vol 1 vs Vol 2, UVA I vs UVA II, PCV 1 vs PCV 2, Study 1 vs Study 2).
   - Parent vs sub-category topics (e.g. Tooth Extraction vs Tooth Extraction Indications).
   Format as pairs of IDs: [[id_a, id_b], ...]

3. "prune_ids": List of single IDs that are confirmed web scraping junk, site disclaimers, cookie notices, navigation bars, or completely out-of-domain fragments.
   Format as a list of IDs: [id_1, id_2, ...]

If none found for any category, return an empty list for that key.

Return ONLY valid JSON matching this schema:
{{
  "merge_pairs": [[1, 2], [10, 15]],
  "link_pairs": [[5, 6]],
  "prune_ids": [99]
}}
"""

PROMPT_DEEP_VERIFICATION = """\
You are an expert technical knowledge graph consolidator verifying candidate concept groups for domain: "{domain}".
You are provided with candidate concept titles, their factual briefs, and top entities.

CANDIDATES FOR REVIEW:
{candidates_json}

INITIAL SUGGESTION: {suggested_category}

TASK:
Verify whether this candidate group should be:
1. "MERGE": Truly duplicate, alternative spellings, formatting splits, or heavily overlapping phrasing. Specify `primary_title` (cleanest canonical title) and `absorbed_titles` (other titles to merge into it).
2. "LINK": Distinct topics or serialized parts (e.g. Vol 1 vs Vol 2, tooth extraction vs tooth extraction indications) that each deserve their own page, but MUST have reciprocal links. Provide 1 concise sentence describing the relation in each direction.
3. "PRUNE": Confirmed web scraping boilerplate, site disclaimers, cookie notices, or completely out-of-domain fragments.
4. "KEEP_SEPARATE": False alarm; they are independent concepts that do not require merging or linking.

Return valid JSON with the exact key "actions":
{{
  "actions": [
    {{
      "action": "MERGE" | "LINK" | "PRUNE" | "KEEP_SEPARATE",
      "primary_title": "Canonical Title",
      "absorbed_titles": ["Other Title 1"],
      "link_description_ab": "Relation from primary to other",
      "link_description_ba": "Relation from other to primary"
    }}
  ]
}}
"""

def clean_json_string(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()

def deduplicate_sentences(text: str) -> str:
    """Splits text into sentences and deduplicates them preserving order."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    unique = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_norm = s_clean.lower()
        if s_norm not in seen:
            seen.add(s_norm)
            unique.append(s_clean)
    return " ".join(unique)

# ── Step 1: Macro Scan by Alphabet ───────────────────────────────────────────
async def run_macro_scan(
    concepts: List[Dict[str, Any]],
    domain_name: str,
    args,
    macro_log_path: Path
) -> List[MacroCandidateCluster]:
    letter_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in concepts:
        title = c["title"].strip()
        first_char = title[0].upper() if title else "#"
        group_key = first_char if first_char.isalpha() else "0-9"
        letter_groups.setdefault(group_key, []).append(c)

    batches = []
    # If a letter group has more than 250 concepts (e.g. C with 577, A with 370), split into sub-batches
    # so the model never times out or exceeds context limits
    for letter in sorted(letter_groups.keys()):
        group = sorted(letter_groups[letter], key=lambda c: c["title"].lower().strip())
        if len(group) <= 250:
            batches.append((f"Letter '{letter}'", group))
        else:
            chunk_size = 200
            for part_idx, i in enumerate(range(0, len(group), chunk_size), 1):
                sub_part = group[i:i + chunk_size]
                batches.append((f"Letter '{letter}' (Part {part_idx})", sub_part))

    print(f"\nStep 1 (Macro Scan): Partitioned {len(concepts)} concepts into {len(batches)} alphabetical batches (A-Z & 0-9)...")
    print(f"Distributing across {len(args.servers)} LLM servers: {args.servers} (Concurrency per server: {args.concurrency_per_server})")
    print(f"Logging all raw LLM macro scan responses to: {macro_log_path}")

    queue = asyncio.Queue()
    for b in batches:
        await queue.put(b)

    effective_concurrency = args.concurrency_per_server
    for _ in range(len(args.servers) * effective_concurrency):
        await queue.put(None)

    all_clusters: List[MacroCandidateCluster] = []
    lock = asyncio.Lock()
    pbar = tqdm(total=len(batches), desc="Macro Scanning by Alphabet")

    macro_log_file = open(macro_log_path, "w", encoding="utf-8")

    async def worker(client: AsyncOpenAI, server_url: str):
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            letter, batch_concepts = item

            # 1-indexed title list with IDs
            id_to_title = {idx + 1: c["title"] for idx, c in enumerate(batch_concepts)}
            title_to_id = {c["title"].lower().strip(): idx + 1 for idx, c in enumerate(batch_concepts)}
            indexed_lines = [f"[{idx + 1}] {c['title']}" for idx, c in enumerate(batch_concepts)]
            indexed_titles_str = "\n".join(indexed_lines)

            prompt = PROMPT_MACRO_SCAN.format(
                domain=domain_name,
                letter=letter,
                indexed_titles=indexed_titles_str
            )

            for attempt in range(10):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=4096,
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content.strip()
                    cleaned_raw = clean_json_string(raw)
                    data = json.loads(cleaned_raw)

                    batch_clusters = []

                    # 1. Parse merge_pairs
                    merge_pairs = data.get("merge_pairs") or data.get("merge_clusters") or []
                    for pair in merge_pairs:
                        if isinstance(pair, list) and len(pair) >= 2:
                            titles = []
                            for p_elem in pair:
                                try:
                                    pid = int(p_elem)
                                    if pid in id_to_title:
                                        titles.append(id_to_title[pid])
                                except (ValueError, TypeError):
                                    if isinstance(p_elem, str) and p_elem.lower().strip() in title_to_id:
                                        titles.append(id_to_title[title_to_id[p_elem.lower().strip()]])
                            if len(titles) >= 2:
                                batch_clusters.append(MacroCandidateCluster(category="MERGE", titles=titles))

                    # 2. Parse link_pairs
                    link_pairs = data.get("link_pairs") or data.get("links") or []
                    for pair in link_pairs:
                        if isinstance(pair, list) and len(pair) >= 2:
                            titles = []
                            for p_elem in pair:
                                try:
                                    pid = int(p_elem)
                                    if pid in id_to_title:
                                        titles.append(id_to_title[pid])
                                except (ValueError, TypeError):
                                    if isinstance(p_elem, str) and p_elem.lower().strip() in title_to_id:
                                        titles.append(id_to_title[title_to_id[p_elem.lower().strip()]])
                            if len(titles) >= 2:
                                batch_clusters.append(MacroCandidateCluster(category="LINK", titles=titles))

                    # 3. Parse prune_ids
                    prune_ids = data.get("prune_ids") or data.get("noise_ids") or data.get("prune") or []
                    for p_elem in prune_ids:
                        try:
                            pid = int(p_elem)
                            if pid in id_to_title:
                                batch_clusters.append(MacroCandidateCluster(category="NOISE", titles=[id_to_title[pid]]))
                        except (ValueError, TypeError):
                            if isinstance(p_elem, str) and p_elem.lower().strip() in title_to_id:
                                batch_clusters.append(MacroCandidateCluster(category="NOISE", titles=[id_to_title[title_to_id[p_elem.lower().strip()]]]))

                    # Fallback for generic "clusters" format
                    if not batch_clusters and "clusters" in data and isinstance(data["clusters"], list):
                        for cl in data["clusters"]:
                            cat = str(cl.get("category", "MERGE")).upper()
                            ids = cl.get("ids", [])
                            titles = [id_to_title[int(i)] for i in ids if str(i).isdigit() and int(i) in id_to_title]
                            if len(titles) >= (1 if cat == "NOISE" else 2):
                                batch_clusters.append(MacroCandidateCluster(category=cat, titles=titles))

                    async with lock:
                        if batch_clusters:
                            all_clusters.extend(batch_clusters)
                        macro_log_file.write(json.dumps({
                            "letter": letter,
                            "server": server_url,
                            "count": len(batch_concepts),
                            "raw_response": raw,
                            "extracted_clusters": [c.model_dump() for c in batch_clusters]
                        }) + "\n")
                        macro_log_file.flush()
                    break
                except Exception as e:
                    if attempt == 9:
                        async with lock:
                            macro_log_file.write(json.dumps({
                                "letter": letter,
                                "server": server_url,
                                "count": len(batch_concepts),
                                "error": str(e)
                            }) + "\n")
                            macro_log_file.flush()
                    await asyncio.sleep(min(1.5 * (attempt + 1), 10.0))

            pbar.update(1)
            queue.task_done()

    # Launch distributed workers across all servers
    tasks, http_clients = [], []
    for url in args.servers:
        hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=300.0)
        http_clients.append(hc)
        client = AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0)
        for _ in range(effective_concurrency):
            tasks.append(asyncio.create_task(worker(client, url)))

    await asyncio.gather(*tasks)
    pbar.close()
    macro_log_file.close()
    for hc in http_clients:
        await hc.aclose()

    print(f"Step 1 Complete: Found {len(all_clusters)} candidate clusters for deep brief verification.")
    return all_clusters

# ── Helper for Resilient Action Parsing ──────────────────────────────────────
def normalize_action_dict(d: Dict[str, Any]) -> Optional[VerifiedAction]:
    if not isinstance(d, dict):
        return None
    act = str(d.get("action", "")).upper().strip()
    if not act:
        return None

    prim = d.get("primary_title") or d.get("canonical_title") or d.get("concept_a") or d.get("title_1")
    abs_t = d.get("absorbed_titles") or d.get("duplicate_titles") or d.get("linked_titles") or []
    if isinstance(abs_t, str):
        abs_t = [abs_t]
    if not abs_t and (d.get("absorbed_title") or d.get("concept_b") or d.get("title_2")):
        other_t = d.get("absorbed_title") or d.get("concept_b") or d.get("title_2")
        if isinstance(other_t, str):
            abs_t = [other_t]

    link_ab = d.get("link_description_ab") or d.get("description_a_to_b") or d.get("link_ab") or d.get("relation")
    link_ba = d.get("link_description_ba") or d.get("description_b_to_a") or d.get("link_ba") or d.get("relation")

    return VerifiedAction(
        action=act,
        primary_title=prim,
        absorbed_titles=abs_t,
        link_description_ab=link_ab,
        link_description_ba=link_ba
    )

# ── Step 2: Deep Verification with Briefs ────────────────────────────────────
async def run_deep_verification(
    clusters: List[MacroCandidateCluster],
    concepts: List[Dict[str, Any]],
    domain_name: str,
    args,
    log_file_path: Path
) -> List[VerifiedAction]:
    concept_map = {c["title"].lower().strip(): c for c in concepts}

    # Deduplicate candidate groups across clusters
    seen_groups = set()
    valid_clusters = []
    for cl in clusters:
        existing_in_graph = [concept_map[t.lower().strip()] for t in cl.titles if t.lower().strip() in concept_map]
        if len(existing_in_graph) >= (1 if cl.category == "NOISE" else 2):
            group_key = tuple(sorted([c["title"].lower().strip() for c in existing_in_graph]))
            if group_key not in seen_groups:
                seen_groups.add(group_key)
                valid_clusters.append((cl.category, existing_in_graph))

    print(f"\nStep 2 (Deep Verification): Verifying {len(valid_clusters)} unique candidate groups using factual briefs...")
    print(f"Distributing across {len(args.servers)} LLM servers: {args.servers} (Concurrency per server: {args.concurrency_per_server})")
    print(f"Logging LLM verification queries and decisions to: {log_file_path}")

    queue = asyncio.Queue()
    for item in valid_clusters:
        await queue.put(item)

    effective_concurrency = args.concurrency_per_server
    for _ in range(len(args.servers) * effective_concurrency):
        await queue.put(None)

    verified_actions: List[VerifiedAction] = []
    lock = asyncio.Lock()
    pbar = tqdm(total=len(valid_clusters), desc="Deep Verifying with Briefs")

    log_f = open(log_file_path, "w", encoding="utf-8")

    async def worker(client: AsyncOpenAI, server_url: str):
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            cat, concept_group = item
            candidates_payload = [
                {
                    "title": c["title"],
                    "brief": c.get("brief", ""),
                    "entities": c.get("entities", [])[:5]
                }
                for c in concept_group
            ]

            prompt = PROMPT_DEEP_VERIFICATION.format(
                domain=domain_name,
                suggested_category=cat,
                candidates_json=json.dumps(candidates_payload, indent=2)
            )

            for attempt in range(10):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000,
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content.strip()
                    cleaned_raw = clean_json_string(raw)

                    parsed_actions = []
                    data = json.loads(cleaned_raw)

                    action_dicts = []
                    if isinstance(data, dict):
                        if "actions" in data and isinstance(data["actions"], list):
                            action_dicts = data["actions"]
                        elif "action" in data:
                            action_dicts = [data]
                        else:
                            for k in ["results", "clusters", "resolutions"]:
                                if k in data and isinstance(data[k], list):
                                    action_dicts = data[k]
                                    break
                            if not action_dicts:
                                for v in data.values():
                                    if isinstance(v, list):
                                        action_dicts = v
                                        break
                    elif isinstance(data, list):
                        action_dicts = data

                    group_titles = [c["title"] for c in concept_group]
                    for act_d in action_dicts:
                        norm = normalize_action_dict(act_d)
                        if norm:
                            norm._candidate_titles = group_titles
                            # If LINK and absorbed_titles is empty, auto-populate from group_titles
                            if norm.action == "LINK" and not norm.absorbed_titles:
                                norm.absorbed_titles = [t for t in group_titles if t.lower().strip() != (norm.primary_title or "").lower().strip()]
                            parsed_actions.append(norm)

                    if parsed_actions:
                        async with lock:
                            verified_actions.extend(parsed_actions)
                            log_entry = {
                                "candidates": [c["title"] for c in concept_group],
                                "server": server_url,
                                "suggested": cat,
                                "raw_response": raw,
                                "actions": [a.model_dump() for a in parsed_actions]
                            }
                            log_f.write(json.dumps(log_entry) + "\n")
                            log_f.flush()
                        break
                except Exception as e:
                    if attempt == 9:
                        async with lock:
                            log_f.write(json.dumps({
                                "candidates": [c["title"] for c in concept_group],
                                "server": server_url,
                                "error": str(e)
                            }) + "\n")
                            log_f.flush()
                    await asyncio.sleep(min(1.5 * (attempt + 1), 10.0))

            pbar.update(1)
            queue.task_done()

    tasks, http_clients = [], []
    for url in args.servers:
        hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=60.0)
        http_clients.append(hc)
        client = AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0)
        for _ in range(effective_concurrency):
            tasks.append(asyncio.create_task(worker(client, url)))

    await asyncio.gather(*tasks)
    pbar.close()
    log_f.close()
    for hc in http_clients:
        await hc.aclose()

    return verified_actions

# ── Step 3: Apply Resolutions & Update Graph ─────────────────────────────────
def apply_all_resolutions(concepts: List[Dict[str, Any]], actions: List[VerifiedAction]) -> List[Dict[str, Any]]:
    title_map = {c["title"].lower().strip(): c for c in concepts}
    merge_remap: Dict[str, str] = {}  # absorbed_lower_title -> primary_canonical_title
    pruned_titles = set()
    merged_count = 0
    linked_count = 0

    # 1. Process Actions
    for act in actions:
        if act.action == "PRUNE":
            for t in ([act.primary_title] if act.primary_title else []) + act.absorbed_titles:
                if t and t.lower().strip() in title_map:
                    pruned_titles.add(t.lower().strip())

        elif act.action == "MERGE" and act.primary_title and act.absorbed_titles:
            p_key = act.primary_title.lower().strip()
            # If the stated primary title isn't directly in graph, find which of the group is in graph
            primary_c = title_map.get(p_key)
            if not primary_c:
                # Fallback to the first valid absorbed title
                for candidate in act.absorbed_titles:
                    c_key = candidate.lower().strip()
                    if title_map.get(c_key):
                        primary_c = title_map[c_key]
                        p_key = c_key
                        break

            if not primary_c:
                continue

            for abs_t in act.absorbed_titles:
                abs_key = abs_t.lower().strip()
                if abs_key == p_key or abs_key not in title_map:
                    continue

                abs_c = title_map[abs_key]
                if abs_c is None:
                    continue

                merge_remap[abs_key] = primary_c["title"]

                # A. Merge description paragraphs
                if abs_c.get("description"):
                    if abs_c["description"] not in primary_c.get("description", ""):
                        primary_c["description"] = f"{primary_c.get('description', '')}\n\n{abs_c['description']}".strip()

                # B. Merge aliases & entities
                primary_c.setdefault("aliases", []).append(abs_c["title"])
                for a in abs_c.get("aliases", []):
                    if a not in primary_c["aliases"]:
                        primary_c["aliases"].append(a)

                for e in abs_c.get("entities", []):
                    if e not in primary_c.get("entities", []):
                        primary_c.setdefault("entities", []).append(e)

                # C. Merge source chunks
                sc_set = set(primary_c.get("source_chunks", []))
                sc_set.update(abs_c.get("source_chunks", []))
                primary_c["source_chunks"] = sorted(list(sc_set))

                # D. Merge outbound relations with target description concatenation
                existing_rels = {r.get("target_concept", "").lower().strip(): r for r in primary_c.get("outbound_relations", [])}
                for r in abs_c.get("outbound_relations", []):
                    tgt_key = r.get("target_concept", "").lower().strip()
                    if tgt_key == p_key or tgt_key == abs_key:
                        continue
                    if tgt_key in existing_rels:
                        # Concatenate description if distinct
                        existing_r = existing_rels[tgt_key]
                        new_desc = r.get("relation", "").strip()
                        combined = f"{existing_r.get('relation', '').strip()} {new_desc}".strip()
                        existing_r["relation"] = deduplicate_sentences(combined)
                        # Union source chunks
                        r_sc = set(existing_r.get("source_chunks", []))
                        r_sc.update(r.get("source_chunks", []))
                        existing_r["source_chunks"] = sorted(list(r_sc))
                    else:
                        primary_c.setdefault("outbound_relations", []).append(r)
                        existing_rels[tgt_key] = r

                title_map[abs_key] = None
                merged_count += 1

        elif act.action == "LINK" and act.primary_title:
            c1_key = act.primary_title.lower().strip()
            c1 = title_map.get(c1_key)
            if not c1:
                continue

            target_keys = [t.lower().strip() for t in act.absorbed_titles if t and t.lower().strip() != c1_key]
            
            # If absorbed_titles was empty, find the other concept from candidates if available
            if not target_keys and getattr(act, "_candidate_titles", None):
                target_keys = [t.lower().strip() for t in act._candidate_titles if t and t.lower().strip() != c1_key]

            for c2_key in target_keys:
                c2 = title_map.get(c2_key)
                if not c2 or c2_key == c1_key:
                    continue

                rel_ab = act.link_description_ab or f"Directly related in series/sub-topic to {c2['title']}."
                rel_ba = act.link_description_ba or f"Directly related in series/sub-topic to {c1['title']}."

                # A -> B: if already exists, leave untouched; else add
                existing_c1 = {r.get("target_concept", "").lower().strip() for r in c1.get("outbound_relations", [])}
                if c2_key not in existing_c1:
                    c1.setdefault("outbound_relations", []).append({
                        "target_concept": c2["title"],
                        "relation": rel_ab,
                        "source_chunks": []
                    })
                    linked_count += 1

                # B -> A: if already exists, leave untouched; else add
                existing_c2 = {r.get("target_concept", "").lower().strip() for r in c2.get("outbound_relations", [])}
                if c1_key not in existing_c2:
                    c2.setdefault("outbound_relations", []).append({
                        "target_concept": c1["title"],
                        "relation": rel_ba,
                        "source_chunks": []
                    })
                    linked_count += 1

    # 2. Filter out absorbed and pruned concepts
    surviving_concepts = [
        c for c in concepts
        if title_map.get(c["title"].lower().strip()) is not None and c["title"].lower().strip() not in pruned_titles
    ]

    # Map of lowercased title -> canonical title for surviving concepts
    surviving_titles = {c["title"].lower().strip(): c["title"] for c in surviving_concepts}

    # 3. Graph-wide Inbound & Outbound Relation Remapping:
    # Remap any reference pointing to an absorbed concept to its canonical host,
    # concatenate relations if the concept now has duplicate outbound targets,
    # and strip pruned/self-referential links.
    for c in surviving_concepts:
        my_title_key = c["title"].lower().strip()
        merged_rels: Dict[str, Dict[str, Any]] = {}

        for r in c.get("outbound_relations", []):
            tgt_raw = r.get("target_concept", "").lower().strip()

            # Remap if target was merged
            if tgt_raw in merge_remap:
                tgt_raw = merge_remap[tgt_raw].lower().strip()

            # Target must survive and not be self-reference or pruned
            if tgt_raw in surviving_titles and tgt_raw != my_title_key and tgt_raw not in pruned_titles:
                canonical_tgt = surviving_titles[tgt_raw]
                if tgt_raw in merged_rels:
                    # Concatenate description
                    existing_r = merged_rels[tgt_raw]
                    new_desc = r.get("relation", "").strip()
                    combined = f"{existing_r.get('relation', '').strip()} {new_desc}".strip()
                    existing_r["relation"] = deduplicate_sentences(combined)
                    r_sc = set(existing_r.get("source_chunks", []))
                    r_sc.update(r.get("source_chunks", []))
                    existing_r["source_chunks"] = sorted(list(r_sc))
                else:
                    r["target_concept"] = canonical_tgt
                    merged_rels[tgt_raw] = r

        c["outbound_relations"] = list(merged_rels.values())

    print(f"\n==================================================")
    print(f"CONSOLIDATION SUMMARY")
    print(f"==================================================")
    print(f"Initial Concepts:        {len(concepts)}")
    print(f"Concepts Merged:         -{merged_count}")
    print(f"Noise Concepts Pruned:   -{len(pruned_titles)}")
    print(f"Final Clean Concepts:    {len(surviving_concepts)}")
    print(f"Reciprocal Links Added:  +{linked_count}")
    print(f"==================================================")

    return surviving_concepts

async def main_pipeline(domain_dir: Path, domain_name: str, args):
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        fallback = domain_dir / "global_concepts_stubs_resolved.json"
        if fallback.exists():
            refined_file = fallback
        else:
            raise FileNotFoundError(f"Missing file: {refined_file}")

    print(f"Loading concepts from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        concepts = json.load(f).get("concepts", [])

    print(f"Loaded {len(concepts)} concepts for two-pass consolidation.")

    log_file_path = domain_dir / "final_consolidation_log.jsonl"
    if not log_file_path.exists():
        fallback_log = domain_dir / "final_consolidation_log.json"
        if fallback_log.exists():
            log_file_path = fallback_log

    if args.from_log:
        if not log_file_path.exists():
            raise FileNotFoundError(f"Requested --from_log but log file does not exist at {log_file_path}")
        print(f"\n[FAST PATH] Loading existing LLM decisions from {log_file_path}...")
        actions = []
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    entry = json.loads(line_str)
                    candidates = entry.get("candidates", [])
                    for act_d in entry.get("actions", []):
                        norm = normalize_action_dict(act_d)
                        if norm:
                            norm._candidate_titles = candidates
                            if norm.action == "LINK" and not norm.absorbed_titles:
                                norm.absorbed_titles = [t for t in candidates if t.lower().strip() != (norm.primary_title or "").lower().strip()]
                            actions.append(norm)
                except Exception as e:
                    pass
        print(f"Loaded {len(actions)} verified actions from log!")
    else:
        # 1. Macro Scan by Alphabet with IDs
        macro_log_path = domain_dir / "macro_scan_log.jsonl"
        clusters = await run_macro_scan(concepts, domain_name, args, macro_log_path)

        # 2. Deep Verification with Briefs & Logging
        actions = await run_deep_verification(clusters, concepts, domain_name, args, log_file_path)

    # 3. Apply Actions (Merge remapping, relation concatenation, reciprocal linking, pruning)
    final_concepts = apply_all_resolutions(concepts, actions)

    # 4. Save JSON
    save_file = domain_dir / "global_concepts_refined.json"
    print(f"\nSaving updated graph to {save_file}...")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump({"concepts": final_concepts}, f, indent=2)

    # 5. Re-export clean Markdown wiki in concepts_md/
    from export_markdown_wiki import export_wiki_markdown
    concepts_md_dir = domain_dir / "concepts_md"
    print(f"\nExporting clean Markdown files to {concepts_md_dir}/...")
    export_wiki_markdown(domain_dir, concepts_md_dir)

    # 6. Analyze Final Islands
    from analyze_islands import analyze_kg_islands
    analyze_kg_islands(domain_dir)

def main():
    parser = argparse.ArgumentParser(description="Two-Pass Global Knowledge Graph Consolidation & Hierarchy Resolution")
    parser.add_argument("--domain", type=str, required=True, help="Target domain (e.g. biology)")
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=GLOBAL_MODEL)
    parser.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency_per_server", type=int, default=DEFAULT_CONCURRENCY_PER_SERVER)
    parser.add_argument("--from_log", action="store_true", help="Apply existing decisions from final_consolidation_log.json(l) without making LLM calls")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    asyncio.run(main_pipeline(domain_dir, args.domain, args))

if __name__ == "__main__":
    main()
