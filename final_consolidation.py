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
  - Every LLM call logged to `log/<stage>.llm.jsonl`; macro-scan batches and
    deep-verification decisions to `log/<stage>.decisions.jsonl`.

Step 2 (Deep Verification with Briefs):
  - Reviews candidate pairs/prune items with their full briefs and top entities.
  - Decides canonical action: MERGE, LINK, PRUNE, or KEEP_SEPARATE.
  - Highly resilient action parser (handles dict, list, alias keys).
  - Distributed across ALL servers with retries & error handling.

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

import asyncio
import functools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# This stage keeps its own per-server client construction (its two passes need
# different request timeouts) but uses the shared partitioning, retry ladder,
# budget and debug log.
from utils.atomic_io import atomic_write_json
from utils.cli import build_parser, resolve_settings
from utils.event_log import EventLog
from utils.layout import log_path, run_dir_for
from utils.errors import RECOVERABLE, StageBudget, classify_error
from utils.retry import backoff_delay
from utils.json_repair import clean_json_string as _extract_json
from utils.llm_log import CURRENT_CALL, LLMCallLog, wrap_client
from utils.prompts import domain_scope

# This stage's payloads are sometimes JSON arrays rather than objects.
clean_json_string = functools.partial(_extract_json, allow_array=True)

# The two passes have very different generation budgets, so each resolves its
# own profile: the macro scan asks for 4096 tokens, deep verification 1000.
MACRO_STAGE = "final-macro-scan"
DEEP_STAGE = "final-deep-verify"

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
{scope}.
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
{scope}.
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
) -> List[MacroCandidateCluster]:
    """First pass of final consolidation: scan the concepts alphabetically for duplicates, links and noise.

    Concepts are grouped by first letter (digits and symbols share one group), and
    each group is cut into batches. The pool sends each batch to the LLM, which
    returns merge pairs, link pairs and noise titles. Each batch's answer is written
    to the decision log.

    Args:
        concepts: All concepts in the refined graph.
        domain_name: Optional domain hint, or ``None``.
        args: Parsed CLI namespace.

    Returns:
        The candidate clusters, for the deep verification pass.

    Raises:
        RuntimeError: If no LLM server is configured, or if batches fail after retries."""
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

    settings = resolve_settings(args, MACRO_STAGE)
    # This stage builds its own clients (its two passes need different request
    # timeouts), so it wires the call log by hand rather than via ServerPool.
    llm_log = LLMCallLog.from_args(args, MACRO_STAGE)
    budget = StageBudget(settings.budget_minutes)
    debug = EventLog(log_path(run_dir_for(args), MACRO_STAGE, "debug"),
                        label=MACRO_STAGE)
    if batches and not settings.servers:
        raise RuntimeError("Final consolidation requires at least one configured LLM server.")
    if batches and settings.concurrency_per_server < 1:
        raise ValueError("concurrency_per_server must be at least 1.")

    print(f"\nStep 1 (Macro Scan): Partitioned {len(concepts)} concepts into {len(batches)} alphabetical batches (A-Z & 0-9)...")
    print(f"Distributing across {len(settings.servers)} LLM servers: {settings.servers} (Concurrency per server: {settings.concurrency_per_server})")

    # One queue per server with a deterministic partition (item i -> server
    # i % N). A single shared queue let the tail of the run race onto whichever
    # worker was free, so the last few batches could all land on one server.
    effective_concurrency = settings.concurrency_per_server
    server_count = len(settings.servers)
    queues = [asyncio.Queue() for _ in range(server_count)]
    for _i, b in enumerate(batches):
        await queues[_i % server_count].put(b)
    for _q in queues:
        for _ in range(effective_concurrency):
            await _q.put(None)
    print(f"Spreading work across {server_count} server(s) at {effective_concurrency} "
          f"concurrent request(s) each (item i -> server i % N).")
    print(f"Per-item attempts: {settings.attempts} | Stage budget: "
          f"{settings.budget_minutes:g} min | Debug log: {debug.path}")

    all_clusters: List[MacroCandidateCluster] = []
    failed_batches: List[str] = []
    lock = asyncio.Lock()
    pbar = tqdm(total=len(batches), desc="Macro Scanning by Alphabet")

    macro_decisions = EventLog(log_path(run_dir_for(args), MACRO_STAGE, "decisions"),
                               label=MACRO_STAGE)

    async def worker(client, server_url: str, work_queue: asyncio.Queue):
        """Consume one server's queue of macro-scan batches until the queue is empty.

        Args:
            client: ``AsyncOpenAI`` client bound to this server.
            server_url: Base URL of this server, for logging.
            work_queue: This server's own queue of batches (the ``i % N`` partition).

        Returns:
            None. Results accumulate in the enclosing scope."""
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break

            letter, batch_concepts = item
            # Publish item identity so the wrapped client can attribute each
            # request. ServerPool normally does this; this stage drives its own
            # queues. Each worker is a separate task, so the value cannot leak.
            call_ctx = {"item": f"letter:{letter}", "server": server_url,
                        "size": len(batch_concepts)}
            CURRENT_CALL.set(call_ctx)

            # 1-indexed title list with IDs
            id_to_title = {idx + 1: c["title"] for idx, c in enumerate(batch_concepts)}
            title_to_id = {c["title"].lower().strip(): idx + 1 for idx, c in enumerate(batch_concepts)}
            indexed_lines = [f"[{idx + 1}] {c['title']}" for idx, c in enumerate(batch_concepts)]
            indexed_titles_str = "\n".join(indexed_lines)

            prompt = PROMPT_MACRO_SCAN.format(
                scope=domain_scope(
                    "You are an expert technical knowledge graph consolidator "
                    "auditing concepts", domain_name),
                letter=letter,
                indexed_titles=indexed_titles_str
            )

            completed = False
            budget.check(MACRO_STAGE)
            for attempt in range(settings.attempts):
                CURRENT_CALL.set({**call_ctx, "attempt": attempt + 1})
                try:
                    resp = await client.chat.completions.create(
                        model=settings.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=settings.max_tokens,
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
                        macro_decisions.record("macro_scan", letter=letter, server=server_url,
                                               batch_size=len(batch_concepts),
                                               clusters=[c.model_dump() for c in batch_clusters])
                    completed = True
                    break
                except Exception as e:
                    # The transport may have succeeded while this stage rejected
                    # the body; record that so --llm_log errors is not silent
                    # about the most common failure mode here.
                    llm_log.rejection(server=server_url, error=e,
                                      classification=classify_error(e))
                    debug.error(f"{server_url}", e, attempt=attempt + 1,
                                attempts=settings.attempts,
                                classification=classify_error(e), server=server_url)
                    if attempt == settings.attempts - 1:
                        async with lock:
                            macro_decisions.record("macro_scan_failed", letter=letter, server=server_url,
                                       batch_size=len(batch_concepts), error=str(e))
                    if attempt < settings.attempts - 1:
                        await asyncio.sleep(budget.cap_sleep(backoff_delay(classify_error(e), attempt + 1)))

            if not completed:
                async with lock:
                    failed_batches.append(letter)

            pbar.update(1)
            work_queue.task_done()

    # Launch distributed workers across all servers
    tasks, http_clients = [], []
    for server_index, url in enumerate(settings.servers):
        hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=settings.profile.request_timeout)
        http_clients.append(hc)
        client = wrap_client(
            AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0),
            url, llm_log)
        for _ in range(effective_concurrency):
            tasks.append(asyncio.create_task(worker(client, url, queues[server_index])))

    await asyncio.gather(*tasks)
    pbar.close()
    macro_decisions.close()
    llm_log.close()
    debug.close()
    for hc in http_clients:
        await hc.aclose()

    if failed_batches:
        raise RuntimeError(
            f"Final macro scan failed for {len(failed_batches)} batch(es) after retries: {failed_batches[:10]}"
        )

    print(f"Step 1 Complete: Found {len(all_clusters)} candidate clusters for deep brief verification.")
    return all_clusters

# ── Helper for Resilient Action Parsing ──────────────────────────────────────
def normalize_action_dict(d: Dict[str, Any]) -> Optional[VerifiedAction]:
    """Turn one action from the LLM into a VerifiedAction, tolerating key-name variants.

    Models name the same field several ways (``primary_title`` or
    ``canonical_title``, ``absorbed_titles`` or ``duplicate_titles``), so every known
    spelling is accepted.

    Args:
        d: One action object decoded from the LLM's JSON.

    Returns:
        A VerifiedAction, or ``None`` when ``d`` is not an object or names no action."""
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
) -> List[VerifiedAction]:
    """Second pass of final consolidation: verify each candidate group with its briefs.

    Candidate groups from the macro scan are de-duplicated, and any group larger than
    250 concepts is split into parts. Each group is sent with its briefs and
    entities, and the LLM answers with MERGE, LINK, PRUNE or KEEP_SEPARATE actions.
    Each answer is written to the decision log.

    Args:
        clusters: Candidate clusters from :func:`run_macro_scan`.
        concepts: All concepts, used to look up each candidate's brief.
        domain_name: Optional domain hint, or ``None``.
        args: Parsed CLI namespace.

    Returns:
        The verified actions, ready for :func:`apply_all_resolutions`.

    Raises:
        RuntimeError: If no LLM server is configured, or if groups fail after retries."""
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

    settings = resolve_settings(args, DEEP_STAGE)
    llm_log = LLMCallLog.from_args(args, DEEP_STAGE)
    budget = StageBudget(settings.budget_minutes)
    debug = EventLog(log_path(run_dir_for(args), DEEP_STAGE, "debug"),
                        label=DEEP_STAGE)
    if valid_clusters and not settings.servers:
        raise RuntimeError("Deep final verification requires at least one configured LLM server.")
    if valid_clusters and settings.concurrency_per_server < 1:
        raise ValueError("concurrency_per_server must be at least 1.")

    print(f"\nStep 2 (Deep Verification): Verifying {len(valid_clusters)} unique candidate groups using factual briefs...")
    print(f"Distributing across {len(settings.servers)} LLM servers: {settings.servers} (Concurrency per server: {settings.concurrency_per_server})")

    # One queue per server with a deterministic partition (item i -> server
    # i % N). A single shared queue let the tail of the run race onto whichever
    # worker was free, so the last few batches could all land on one server.
    effective_concurrency = settings.concurrency_per_server
    server_count = len(settings.servers)
    queues = [asyncio.Queue() for _ in range(server_count)]
    for _i, item in enumerate(valid_clusters):
        await queues[_i % server_count].put(item)
    for _q in queues:
        for _ in range(effective_concurrency):
            await _q.put(None)
    print(f"Spreading work across {server_count} server(s) at {effective_concurrency} "
          f"concurrent request(s) each (item i -> server i % N).")
    print(f"Per-item attempts: {settings.attempts} | Stage budget: "
          f"{settings.budget_minutes:g} min | Debug log: {debug.path}")

    verified_actions: List[VerifiedAction] = []
    failed_groups: List[List[str]] = []
    lock = asyncio.Lock()
    pbar = tqdm(total=len(valid_clusters), desc="Deep Verifying with Briefs")

    deep_decisions = EventLog(log_path(run_dir_for(args), DEEP_STAGE, "decisions"),
                              label=DEEP_STAGE)

    async def worker(client, server_url: str, work_queue: asyncio.Queue):
        """Consume one server's queue of candidate groups until the queue is empty.

        Args:
            client: ``AsyncOpenAI`` client bound to this server.
            server_url: Base URL of this server, for logging.
            work_queue: This server's own queue of groups (the ``i % N`` partition).

        Returns:
            None. Verified actions accumulate in the enclosing scope."""
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break

            cat, concept_group = item
            call_ctx = {"item": f"category:{cat}", "server": server_url,
                        "size": len(concept_group)}
            CURRENT_CALL.set(call_ctx)
            candidates_payload = [
                {
                    "title": c["title"],
                    "brief": c.get("brief", ""),
                    "entities": c.get("entities", [])[:5]
                }
                for c in concept_group
            ]

            prompt = PROMPT_DEEP_VERIFICATION.format(
                scope=domain_scope(
                    "You are an expert technical knowledge graph consolidator "
                    "verifying candidate concept groups", domain_name),
                suggested_category=cat,
                candidates_json=json.dumps(candidates_payload, indent=2)
            )

            completed = False
            budget.check(DEEP_STAGE)
            for attempt in range(settings.attempts):
                CURRENT_CALL.set({**call_ctx, "attempt": attempt + 1})
                try:
                    resp = await client.chat.completions.create(
                        model=settings.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=settings.max_tokens,
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
                            deep_decisions.record("deep_verification", **log_entry)
                        completed = True
                        break
                    # The reply parsed but held nothing usable. Record it as a rejection, then
                    # retry after the recoverable-class backoff, like any other unusable reply.
                    unusable = ValueError("reply contained no usable actions")
                    llm_log.rejection(server=server_url, error=unusable, classification=RECOVERABLE)
                    if attempt < settings.attempts - 1:
                        await asyncio.sleep(budget.cap_sleep(backoff_delay(RECOVERABLE, attempt + 1)))
                except Exception as e:
                    # The transport may have succeeded while this stage rejected
                    # the body; record that so --llm_log errors is not silent
                    # about the most common failure mode here.
                    llm_log.rejection(server=server_url, error=e,
                                      classification=classify_error(e))
                    debug.error(f"{server_url}", e, attempt=attempt + 1,
                                attempts=settings.attempts,
                                classification=classify_error(e), server=server_url)
                    if attempt == settings.attempts - 1:
                        async with lock:
                            deep_decisions.record("deep_verification_failed",
                                           candidates=[c["title"] for c in concept_group],
                                           server=server_url, error=str(e))
                    if attempt < settings.attempts - 1:
                        await asyncio.sleep(budget.cap_sleep(backoff_delay(classify_error(e), attempt + 1)))

            if not completed:
                async with lock:
                    failed_groups.append([c["title"] for c in concept_group])

            pbar.update(1)
            work_queue.task_done()

    tasks, http_clients = [], []
    for server_index, url in enumerate(settings.servers):
        hc = httpx.AsyncClient(limits=httpx.Limits(max_connections=effective_concurrency + 10), timeout=settings.profile.request_timeout)
        http_clients.append(hc)
        client = wrap_client(
            AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc, max_retries=0),
            url, llm_log)
        for _ in range(effective_concurrency):
            tasks.append(asyncio.create_task(worker(client, url, queues[server_index])))

    await asyncio.gather(*tasks)
    pbar.close()
    deep_decisions.close()
    llm_log.close()
    debug.close()
    for hc in http_clients:
        await hc.aclose()

    if failed_groups:
        raise RuntimeError(
            f"Deep final verification failed for {len(failed_groups)} group(s) after retries. "
            f"Examples: {failed_groups[:5]}"
        )

    return verified_actions

# ── Step 3: Apply Resolutions & Update Graph ─────────────────────────────────
def apply_all_resolutions(concepts: List[Dict[str, Any]], actions: List[VerifiedAction]) -> List[Dict[str, Any]]:
    """Apply the verified actions to the concept list and return the survivors.

    MERGE folds each absorbed concept into its primary, carrying over aliases,
    entities and citations. PRUNE drops noise. LINK adds a reciprocal relation, using
    the description the LLM wrote for each direction.

    Args:
        concepts: The concepts before resolution.
        actions: Verified actions from the deep-verification pass.

    Returns:
        The surviving concepts, with merges, prunes and links applied."""
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

    print("\n==================================================")
    print("CONSOLIDATION SUMMARY")
    print("==================================================")
    print(f"Initial Concepts:        {len(concepts)}")
    print(f"Concepts Merged:         -{merged_count}")
    print(f"Noise Concepts Pruned:   -{len(pruned_titles)}")
    print(f"Final Clean Concepts:    {len(surviving_concepts)}")
    print(f"Reciprocal Links Added:  +{linked_count}")
    print("==================================================")

    return surviving_concepts

async def main_pipeline(domain_dir: Path, domain_name: Optional[str], args):
    """Run final consolidation end to end on one run directory.

    Loads the refined graph. Then either re-applies the decisions in
    ``log/final-deep-verify.decisions.jsonl`` (``--from_log``, which makes no LLM
    calls) or runs the macro scan and the deep verification. It applies the
    resolutions and overwrites ``global_concepts_refined.json``.

    Args:
        domain_dir: Run directory.
        domain_name: Optional domain hint, or ``None``.
        args: Parsed CLI namespace.

    Returns:
        None.

    Raises:
        FileNotFoundError: If no refined graph exists, or ``--from_log`` was given but
            no decision log exists."""
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

    # Deep-verification decisions live in log/. Runs made before that change kept
    # them at the top level, so those names are still accepted for --from_log.
    log_file_path = log_path(domain_dir, DEEP_STAGE, "decisions")
    if not log_file_path.exists():
        for legacy in (domain_dir / "final_consolidation_log.jsonl",
                       domain_dir / "final_consolidation_log.json"):
            if legacy.exists():
                log_file_path = legacy
                break

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
                except Exception:
                    pass
        print(f"Loaded {len(actions)} verified actions from log!")
    else:
        # 1. Macro Scan by Alphabet with IDs
        clusters = await run_macro_scan(concepts, domain_name, args)

        # 2. Deep Verification with Briefs & Logging
        actions = await run_deep_verification(clusters, concepts, domain_name, args)

    # 3. Apply Actions (Merge remapping, relation concatenation, reciprocal linking, pruning)
    final_concepts = apply_all_resolutions(concepts, actions)

    # 4. Save JSON
    save_file = domain_dir / "global_concepts_refined.json"
    print(f"\nSaving updated graph to {save_file}...")
    atomic_write_json(save_file, {"concepts": final_concepts})

    # 5. Re-export clean Markdown wiki in concepts_md/
    from export_markdown import export_wiki_markdown
    concepts_md_dir = domain_dir / "concepts_md"
    print(f"\nExporting clean Markdown files to {concepts_md_dir}/...")
    export_wiki_markdown(domain_dir, concepts_md_dir)

    # 6. Analyze Final Islands
    from island_report import analyze_kg_islands
    analyze_kg_islands(domain_dir)

def main():
    """Parse the command line and run final consolidation.

    Returns:
        None."""
    parser = build_parser(
        stage=MACRO_STAGE,
        description="Two-Pass Global Knowledge Graph Consolidation & Hierarchy Resolution",
        include_max_tokens=False)
    parser.add_argument("--from_log", action="store_true", help="Apply the deep-verification decisions in log/final-deep-verify.decisions.jsonl "
                             "without making any LLM calls")

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    # domain_name feeds prompts only, so pass the optional hint (None when
    # --domain was omitted); args.domain stays the output folder name.
    asyncio.run(main_pipeline(domain_dir, args.domain_label, args))

if __name__ == "__main__":
    main()
