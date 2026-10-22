#!/usr/bin/env python3
"""
Document-Level Knowledge Graph Consolidation Pass
-------------------------------------------------
Merges duplicate concepts within each document across its multiple chunks.
Uses XML prompts, sandboxed <think> tags, non-thinking mode support,
and outputs thin concept statistics (word-count distribution).
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import BadRequestError
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.atomic_io import atomic_write_text
from utils.checkpoint import load_bool_log, save_log_async
from utils.cli import build_parser, resolve_settings
from utils.corpus import chunking_enabled
from utils.errors import StageBudget
from utils.llm_pool import pool_for_stage

STAGE = "consolidate-documents"
MAX_MERGE_BATCH_CHARS = 30000
MAX_MERGE_BATCH_CONCEPTS = 80

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class MergeCluster(BaseModel):
    primary_title: str = Field(description="The canonical primary title for this concept.")
    merged_titles: List[str] = Field(
        default_factory=list,
        description="Exact titles of other concepts in the input list that represent the same entity and should be merged into primary_title."
    )

class DocumentConsolidationSchema(BaseModel):
    merge_clusters: List[MergeCluster] = Field(default_factory=list)

# ── XML System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_CONSOLIDATE = """\
You are an expert technical knowledge graph consolidator.
You are given a list of candidate concepts extracted across multiple chunks of the SAME document.
Because these concepts were extracted independently per chunk, many represent the EXACT SAME real-world entity, process, software tool, gene, condition, or technical subject under slightly different titles or acronyms.

YOUR MISSION:
Identify groups of concepts that refer to the exact same core concept/entity within this document and should be merged together.

CRITICAL RULES:
1. MERGE CRITERIA: Merge concepts ONLY if they refer to the exact same canonical entity, tool, process, acronym, or subject.
2. DO NOT MERGE DISTINCT CONCEPTS: Do NOT merge distinct sub-components, related but separate concepts, or parent/child hierarchy terms.
3. PRIMARY TITLE SELECTION: Pick the most standard, canonical, explicit title as `primary_title`.
4. SCOPE: ONLY include titles that exist in the provided candidate concepts list.

THINKING PROCESS (SANDBOXED):
- If you need to reason or compare concepts, you MUST do so strictly inside `<think>...</think>` tags at the very start of your response.
- Once `</think>` is closed, output ONLY the final `<merge_clusters>...</merge_clusters>` XML block.
- Absolutely NO commentary or notes outside `<think>...</think>` or outside `<merge_clusters>` tags.

OUTPUT FORMAT:
<think>
[Consolidation analysis and deduplication plan]
</think>
<merge_clusters>
<cluster>
<primary_title>Exact Chosen Canonical Title</primary_title>
<merged_titles>
- Alternative Title 1
- Alternative Title 2
</merged_titles>
</cluster>
</merge_clusters>

If no concepts need to be merged, output:
<merge_clusters>
</merge_clusters>
"""

# ── XML Parsing Helper ────────────────────────────────────────────────────────

def parse_xml_merge_clusters(text: str) -> List[MergeCluster]:
    """Parse <merge_clusters> blocks into MergeCluster objects, stripping thinking tags."""
    if not text:
        return []

    # 1. Strip thinking tags
    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

    clusters: List[MergeCluster] = []
    # Match each <cluster> block
    cluster_blocks = re.findall(r"<cluster>(.*?)</cluster>", clean_text, re.DOTALL | re.IGNORECASE)
    for block in cluster_blocks:
        title_m = re.search(r"<primary_title>(.*?)</primary_title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m or not title_m.group(1).strip():
            continue
        primary_title = title_m.group(1).strip()

        merged_titles = []
        merged_m = re.search(r"<merged_titles>(.*?)</merged_titles>", block, re.DOTALL | re.IGNORECASE)
        if merged_m:
            lines = merged_m.group(1).strip().splitlines()
            for line in lines:
                cleaned = re.sub(r"^[-*•\d\.\s]+", "", line).strip()
                if cleaned and cleaned.lower() != primary_title.lower():
                    merged_titles.append(cleaned)

        clusters.append(MergeCluster(primary_title=primary_title, merged_titles=merged_titles))

    # JSON fallback in case model emitted JSON
    if not clusters and "{" in clean_text:
        try:
            m = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                if "merge_clusters" in payload:
                    schema = DocumentConsolidationSchema.model_validate(payload)
                    return schema.merge_clusters
        except Exception:
            pass

    return clusters

# ── Word Count Thin Concept Statistics Helper ─────────────────────────────────

def print_concept_word_distribution(concepts: List[Dict[str, Any]], title_label: str = "CONCEPTS"):
    """Compute and print word count distribution statistics for concept descriptions."""
    if not concepts:
        print(f"\n[{title_label}] No concepts found.")
        return

    word_counts = []
    for c in concepts:
        desc = c.get("description", "")
        # Count words
        words = len(re.findall(r"\b\w+\b", desc))
        word_counts.append(words)

    bins = [
        ("0 - 10 words", lambda w: 0 <= w <= 10),
        ("11 - 20 words", lambda w: 11 <= w <= 20),
        ("21 - 30 words", lambda w: 21 <= w <= 30),
        ("31 - 50 words", lambda w: 31 <= w <= 50),
        ("51 - 100 words", lambda w: 51 <= w <= 100),
        ("101 - 200 words", lambda w: 101 <= w <= 200),
        ("201+ words", lambda w: w > 200),
    ]

    bin_counts = {label: 0 for label, _ in bins}
    for count in word_counts:
        for label, fn in bins:
            if fn(count):
                bin_counts[label] += 1
                break

    total = len(word_counts)
    avg_words = sum(word_counts) / total if total else 0
    thin_30 = sum(1 for w in word_counts if w <= 30)

    print(f"\n============================================================")
    print(f"WORD-COUNT DISTRIBUTION [{title_label}] (Total Concepts: {total})")
    print(f"Average Words per Concept: {avg_words:.1f} | Thin Concepts (<= 30 words): {thin_30} ({thin_30 / total * 100:.1f}%)")
    print(f"------------------------------------------------------------")
    print(f"{'Word Count Range':<20} {'Count':<10} {'Percentage':<12}")
    print(f"------------------------------------------------------------")
    for label, _ in bins:
        count = bin_counts[label]
        pct = (count / total * 100) if total else 0.0
        bar = "█" * int(pct / 4)
        print(f"{label:<20} {count:<10} {pct:>5.1f}%   {bar}")
    print(f"============================================================\n")

# ── Batching Utilities ────────────────────────────────────────────────────────

def make_merge_batches(concepts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group concept title-stubs into token-bounded batches for LLM evaluation."""
    ordered = sorted(concepts, key=lambda c: str(c.get("title") or "").casefold())
    batches: List[List[Dict[str, Any]]] = []
    batch: List[Dict[str, Any]] = []
    size = 2
    for concept in ordered:
        item_size = len(json.dumps(concept, ensure_ascii=False, separators=(",", ":"))) + 1
        if item_size + 2 > MAX_MERGE_BATCH_CHARS:
            continue
        if batch and (len(batch) >= MAX_MERGE_BATCH_CONCEPTS or size + item_size > MAX_MERGE_BATCH_CHARS):
            batches.append(batch)
            batch = []
            size = 2
        batch.append(concept)
        size += item_size
    if batch:
        batches.append(batch)
    return batches

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiDocumentConsolidatorPipeline:
    def __init__(self, args):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")

        self.base_wiki_dir = Path(wiki_path)
        self.staging_dir = self.base_wiki_dir / "staging"
        self.consolidated_dir = self.base_wiki_dir / "consolidated"
        self.log_file = self.base_wiki_dir / "doc_consolidation_log.json"

        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory missing at {self.staging_dir}. Run extraction first.")

        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.log_data = load_bool_log(self.log_file, corrupt_message="Log corrupted. Initializing fresh.")
        self.settings = resolve_settings(args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.flush_count = 0
        self.log_lock = asyncio.Lock()

    def prepare_document_items(self) -> List[Dict[str, Any]]:
        """Collect all canonical documents and their raw chunk concepts."""
        print(f"Scanning staging directory: {self.staging_dir}...")
        doc_dirs = sorted([d for d in self.staging_dir.iterdir() if d.is_dir() and d.name.startswith("canon_")])
        doc_items = []

        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            chunk_dirs = sorted([c for c in doc_dir.iterdir() if c.is_dir() and c.name.startswith("chunk_")])
            raw_concepts = []

            for chunk_dir in chunk_dirs:
                concepts_file = chunk_dir / "concepts.json"
                chunk_id = chunk_dir.name

                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for concept in data.get("concepts", []):
                                if isinstance(concept, dict) and concept.get("title"):
                                    c_copy = dict(concept)
                                    c_copy["source_chunk"] = f"{doc_id}::{chunk_id}"
                                    raw_concepts.append(c_copy)
                    except Exception:
                        continue

            if raw_concepts:
                doc_items.append({
                    "doc_id": doc_id,
                    "raw_concepts": raw_concepts,
                    "output_dir": self.consolidated_dir / doc_id,
                })

        return doc_items

    async def get_merge_clusters(self, client: Any, item: Dict[str, Any], server_index: int) -> DocumentConsolidationSchema:
        doc_id = item["doc_id"]
        raw_concepts = item["raw_concepts"]

        lightweight_concepts = []
        for c in raw_concepts:
            desc = c.get("description", "")
            words = desc.split()[:30]
            lightweight_concepts.append({
                "title": c.get("title"),
                "aliases": c.get("aliases", [])[:8],
                "description_hint": " ".join(words)[:400],
            })

        batches = make_merge_batches(lightweight_concepts)
        semaphore = asyncio.Semaphore(16)
        batch_pbar = None
        if len(batches) > 1:
            batch_pbar = tqdm(total=len(batches), desc=f"  [doc {doc_id[:12]}] {len(raw_concepts)} concepts", leave=False)

        def format_batch_for_prompt(batch_concepts: List[Dict[str, Any]]) -> str:
            lines = []
            for i, c in enumerate(batch_concepts, 1):
                title = c.get("title", "")
                aliases = ", ".join(c.get("aliases", []))
                hint = c.get("description_hint", "").strip()
                item_lines = [f"{i}. Title: \"{title}\""]
                if aliases:
                    item_lines.append(f"   Aliases: {aliases}")
                if hint:
                    item_lines.append(f"   Context: {hint}")
                lines.append("\n".join(item_lines))
            return "\n\n".join(lines)

        async def merge_batch(batch: List[Dict[str, Any]]) -> List[MergeCluster]:
            async with semaphore:
                batch_text = format_batch_for_prompt(batch)
                user_prompt = (
                    f"DOCUMENT ID: {doc_id}\nTOTAL EXTRACTED CONCEPTS ACROSS CHUNKS: "
                    f"{len(lightweight_concepts)}\nCONCEPTS IN THIS BATCH: {len(batch)}\n\n"
                    "CANDIDATE CONCEPTS LIST (only merge titles present in this list):\n\n"
                    f"{batch_text}\n"
                )

                extra_body: Dict[str, Any] = {}
                model_kwargs: Dict[str, Any] = {}
                if getattr(self.args, "disable_thinking", False):
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                    model_kwargs = {
                        "temperature": 0.7,
                        "top_p": 0.80,
                        "presence_penalty": 1.5,
                        "extra_body": extra_body,
                    }
                else:
                    model_kwargs = {
                        "temperature": 0.1,
                    }

                try:
                    response = await client.chat.completions.create(
                        model=self.settings.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_CONSOLIDATE},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=self.settings.max_tokens or 16000,
                        **model_kwargs
                    )
                    choice = response.choices[0]
                    raw_text = choice.message.content or ""
                    clusters = parse_xml_merge_clusters(raw_text)
                    if batch_pbar is not None:
                        batch_pbar.update(1)
                    return clusters
                except Exception as exc:
                    too_long = ("hit max_tokens" in str(exc).lower() or
                                isinstance(exc, BadRequestError) and
                                ("maximum context length" in str(exc).lower() or "input_tokens" in str(exc).lower()))
                    if not too_long or len(batch) <= 1:
                        raise
                    if batch_pbar is not None:
                        batch_pbar.total += 1
                        batch_pbar.refresh()

            mid = len(batch) // 2
            left, right = await asyncio.gather(merge_batch(batch[:mid]), merge_batch(batch[mid:]))
            return left + right

        batch_results = await asyncio.gather(*(merge_batch(batch) for batch in batches))
        if batch_pbar is not None:
            batch_pbar.close()

        return DocumentConsolidationSchema(
            merge_clusters=[cluster for clusters in batch_results for cluster in clusters]
        )

    def execute_programmatic_merge(self, raw_concepts: List[Dict[str, Any]], delta: DocumentConsolidationSchema) -> List[Dict[str, Any]]:
        """Group identical concepts and fuse their aliases, descriptions, and chunks."""
        title_map: Dict[str, str] = {}
        for cluster in delta.merge_clusters:
            primary = cluster.primary_title.strip()
            title_map[primary.lower()] = primary
            for alt in cluster.merged_titles:
                title_map[alt.strip().lower()] = primary

        grouped_concepts: Dict[str, List[Dict[str, Any]]] = {}
        for c in raw_concepts:
            raw_title = c.get("title", "").strip()
            if not raw_title:
                continue
            primary_title = title_map.get(raw_title.lower(), raw_title)
            grouped_concepts.setdefault(primary_title, []).append(c)

        consolidated_concepts = []
        for primary_title, concept_group in grouped_concepts.items():
            aliases_set = set()
            source_chunks_set = set()

            for c in concept_group:
                for alias in c.get("aliases", []):
                    if alias.strip().lower() != primary_title.lower():
                        aliases_set.add(alias.strip())
                raw_title = c.get("title", "").strip()
                if raw_title.lower() != primary_title.lower():
                    aliases_set.add(raw_title)

                src_chunk = c.get("source_chunk")
                if src_chunk:
                    source_chunks_set.add(src_chunk)

            description_sentences = []
            seen_sentences = set()
            for c in concept_group:
                desc = c.get("description", "").strip()
                if desc:
                    for s in re.split(r'(?<=[.!?])\s+', desc):
                        s_clean = s.strip()
                        s_key = s_clean.lower()
                        if s_clean and s_key not in seen_sentences:
                            seen_sentences.add(s_key)
                            description_sentences.append(s_clean)
            merged_description = " ".join(description_sentences)

            consolidated_concepts.append({
                "title": primary_title,
                "aliases": sorted(list(aliases_set)),
                "description": merged_description,
                "source_chunks": sorted(list(source_chunks_set)),
            })

        return consolidated_concepts

    async def persist_consolidation(self, item: Dict[str, Any], delta: DocumentConsolidationSchema) -> None:
        doc_id = item["doc_id"]
        output_dir: Path = item["output_dir"]
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

        consolidated = self.execute_programmatic_merge(item["raw_concepts"], delta)
        out_payload = json.dumps({"doc_id": doc_id, "concepts": consolidated}, indent=2)
        await asyncio.to_thread(atomic_write_text, output_dir / "concepts.json", out_payload)

        async with self.log_lock:
            self.log_data[doc_id] = True
            self.flush_count += 1
            if self.flush_count >= 10:
                await save_log_async(self.log_file, self.log_data)
                self.flush_count = 0

    async def pass_through(self, pending: List[Dict[str, Any]]) -> None:
        no_merges = DocumentConsolidationSchema()
        for item in pending:
            await self.persist_consolidation(item, no_merges)
        if pending:
            await save_log_async(self.log_file, self.log_data)
            self.flush_count = 0

    async def run(self):
        doc_items = self.prepare_document_items()
        all_raw_concepts = [c for item in doc_items for c in item["raw_concepts"]]
        total_concepts_before = len(all_raw_concepts)

        pending = [d for d in doc_items if not self.log_data.get(d["doc_id"], False)]

        print(f"Total documents: {len(doc_items)} | Pending consolidation: {len(pending)}")
        print(f"Total raw concepts before consolidation: {total_concepts_before}")

        # Display Initial Word Distribution before consolidation
        print_concept_word_distribution(all_raw_concepts, title_label="STAGE 5: BEFORE DOCUMENT CONSOLIDATION")

        settings = self.settings
        if not chunking_enabled(self.base_wiki_dir):
            print("Chunking disabled: passing documents through without LLM calls.")
            await self.pass_through(pending)
        elif pending:
            print(f"Distributing {len(pending)} document(s) across {len(settings.servers)} server(s).")
            print(f"Per-item attempts: {settings.attempts} | Stage budget: {settings.budget_minutes:g} min")

            for item in pending:
                self.log_data.setdefault(item["doc_id"], False)

            pool = pool_for_stage(self.args, settings, STAGE)
            async with pool:
                await pool.map(
                    pending,
                    self.get_merge_clusters,
                    attempts=settings.attempts,
                    desc="Consolidating Documents",
                    key_fn=lambda it: it["doc_id"],
                    budget=self.budget,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=self.persist_consolidation,
                )
        else:
            print("All documents are already consolidated!")

        await save_log_async(self.log_file, self.log_data)

        # Collect Consolidated Concepts and Display Final Word Distribution
        all_consolidated_concepts = []
        for doc_dir in self.consolidated_dir.iterdir():
            if doc_dir.is_dir():
                concepts_file = doc_dir / "concepts.json"
                if concepts_file.exists():
                    try:
                        with open(concepts_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_consolidated_concepts.extend(data.get("concepts", []))
                    except Exception:
                        pass

        total_concepts_after = len(all_consolidated_concepts)
        print_concept_word_distribution(all_consolidated_concepts, title_label="STAGE 5: AFTER DOCUMENT CONSOLIDATION")

        print("============================================================")
        print(f"DOCUMENT CONSOLIDATION COMPLETE")
        print(f"Documents Processed:         {len(doc_items)}")
        print(f"Concepts Before Merge:       {total_concepts_before}")
        print(f"Concepts After Merge:        {total_concepts_after} (fused {total_concepts_before - total_concepts_after} duplicates)")
        print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Consolidate concepts across chunks within each document")
    args = parser.parse_args()

    pipeline = WikiDocumentConsolidatorPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
