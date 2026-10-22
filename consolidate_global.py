#!/usr/bin/env python3
"""
Global Concept Consolidation Pass (LLM-Verified Exact, Lemma & Alias Merging)
------------------------------------------------------------------------------
Fuses concepts across all documents in the corpus using LLM verification:
- Candidate pairs with identical titles, lemma plurals, punctuation differences,
  or title-to-alias alignments are identified.
- Context provided to LLM: Title, Aliases, and first ~40-50 words of Description.
- LLM confirms whether both concepts refer to the exact same canonical entity.
- Disjoint Set / UnionFind applies confirmed merges.
- Exports global_concepts.json and concepts/<title>.json with provenance citations.
- Outputs thin concept statistics (word-count distribution) before and after.
"""

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel
from tqdm import tqdm

from utils.atomic_io import atomic_write_json, sanitize_filename
from utils.cli import build_parser, resolve_settings
from utils.llm_pool import pool_for_stage

STAGE = "consolidate-global"

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
    'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't",
    'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn',
    "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
    'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most',
    'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
    'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's',
    'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so',
    'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

STRUCTURAL_BOILERPLATE_REGEXES = [
    r'^(table of contents|contents|outline of .*|overview of .*)$',
    r'^(references|see also|further reading|bibliography|notes|external links)$',
    r'^(history of .*|introduction to .*|timeline of .*)$',
    r'^(glossary of .*|index of .*|list of .*)$',
    r'^(main article|primary topic|disambiguation page)$',
]

def is_boilerplate_concept(title: str) -> bool:
    t = title.strip().lower()
    return any(re.match(pattern, t) for pattern in STRUCTURAL_BOILERPLATE_REGEXES)

def canonical_lemma_key(title: str) -> str:
    """Normalizes titles to catch typographical plurals, punctuation differences, and spacing."""
    t = re.sub(r'[-_]+', ' ', title.strip().lower())
    t = re.sub(r'[^\w\s]', '', t)
    tokens = t.split()
    normalized = []
    for tok in tokens:
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("es") and len(tok) > 4 and tok not in {"species", "series", "genes", "diabetes"}:
            if tok.endswith("sses") or tok.endswith("shes") or tok.endswith("ches") or tok.endswith("xes") or tok.endswith("zes"):
                tok = tok[:-2]
            elif tok.endswith("ses") and not tok.endswith("sses"):
                tok = tok[:-1]
            else:
                tok = tok[:-1]
        elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss") and tok not in {"this", "gas", "bias", "lens", "virus", "status", "mucus", "genus"}:
            tok = tok[:-1]
        normalized.append(tok)
    return " ".join(normalized)

def get_context_hint(desc: str, max_words: int = 40) -> str:
    """Extracts first ~40 words from description as rich context for LLM comparison."""
    if not desc:
        return "No description available."
    words = desc.strip().split()
    snippet = " ".join(words[:max_words])
    if len(words) > max_words:
        snippet += "..."
    return snippet

class UnionFind:
    """Disjoint Set with iterative path compression and cycle prevention."""
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        key = canonical_lemma_key(item)
        if not key:
            return ""

        curr = key
        path = []
        visited = set()

        while curr in self.parent and self.parent[curr] != curr:
            if curr in visited:
                self.parent[curr] = curr
                break
            visited.add(curr)
            path.append(curr)
            curr = self.parent[curr]

        root = curr
        for node in path:
            self.parent[node] = root
        if root not in self.parent:
            self.parent[root] = root
        return root

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 and root2 and root1 != root2:
            self.parent[root1] = root2

# ── XML System Prompt for LLM Concept Merge Verification ──────────────────────

SYSTEM_PROMPT_MERGE_VERIFY = """\
You are an expert technical knowledge graph consolidator evaluating candidate concepts across documents.
You are given candidate pairs of concepts that share identical titles, plural/spelling variants, or title-to-alias alignments.
Each concept is presented with its Canonical Title, Aliases, and a ~40-word Context Snippet from its description.

Your task is to determine whether both concepts refer to the EXACT SAME real-world entity, process, tool, gene, condition, or technical subject.

CRITICAL RULES:
1. MERGE (is_same_entity: true): Only if both concepts denote the identical subject (e.g., 'Amyotrophic Lateral Sclerosis' and 'Lou Gehrig's Disease', or 'Photosynthesis' and 'Photosynthetic Process').
2. KEEP SEPARATE (is_same_entity: false): Do NOT merge if one is an acronym that is polysemous, or if one is a sub-component/subtype of the other (e.g. 'T Cell' vs 'Cell', 'Cornea' vs 'Retina').
3. When in doubt, prefer FALSE. Keeping concepts separate preserves graph integrity.

THINKING PROCESS (SANDBOXED):
- If you need to reason or compare pairs, put all thinking strictly inside `<think>...</think>` at the very start of your response.
- Once `</think>` is closed, output ONLY the final `<verdicts>...</verdicts>` XML block.
- Absolutely NO commentary or notes outside `<think>...</think>` or outside `<verdicts>` tags.

OUTPUT FORMAT:
<think>
[Comparison notes]
</think>
<verdicts>
<verdict>
<pair_index>1</pair_index>
<is_same_entity>true</is_same_entity>
<chosen_canonical_title>Exact Canonical Title</chosen_canonical_title>
<reason>Direct synonymous reference.</reason>
</verdict>
</verdicts>
"""

class MergeVerdict(BaseModel):
    pair_index: int = 1
    is_same_entity: bool = False
    chosen_canonical_title: str = ""
    reason: str = ""

def parse_xml_merge_verdicts(text: str) -> List[MergeVerdict]:
    """Parse <verdicts> blocks into MergeVerdict objects, stripping thinking tags."""
    if not text:
        return []

    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

    verdicts: List[MergeVerdict] = []
    blocks = re.findall(r"<verdict>(.*?)</verdict>", clean_text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        idx_m = re.search(r"<pair_index>(\d+)</pair_index>", block, re.IGNORECASE)
        idx = int(idx_m.group(1)) if idx_m else len(verdicts) + 1

        same_m = re.search(r"<is_same_entity>(.*?)</is_same_entity>", block, re.IGNORECASE)
        is_same = False
        if same_m:
            is_same = same_m.group(1).strip().lower() in ("true", "1", "yes")

        canon_m = re.search(r"<chosen_canonical_title>(.*?)</chosen_canonical_title>", block, re.DOTALL | re.IGNORECASE)
        canon = canon_m.group(1).strip() if canon_m else ""

        reason_m = re.search(r"<reason>(.*?)</reason>", block, re.DOTALL | re.IGNORECASE)
        reason = reason_m.group(1).strip() if reason_m else ""

        verdicts.append(MergeVerdict(
            pair_index=idx,
            is_same_entity=is_same,
            chosen_canonical_title=canon,
            reason=reason
        ))

    # JSON fallback
    if not verdicts and "{" in clean_text:
        try:
            m = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                for item in payload.get("pair_verdicts", []):
                    verdicts.append(MergeVerdict(
                        pair_index=item.get("pair_index", len(verdicts) + 1),
                        is_same_entity=bool(item.get("is_same_entity", False)),
                        chosen_canonical_title=item.get("chosen_canonical_title", ""),
                        reason=item.get("reason", "")
                    ))
        except Exception:
            pass

    return verdicts

# ── Word Count Thin Concept Statistics Helper ─────────────────────────────────

def print_concept_word_distribution(concepts: List[Dict[str, Any]], title_label: str = "GLOBAL CONCEPTS"):
    """Compute and print word count distribution statistics for concept descriptions."""
    if not concepts:
        print(f"\n[{title_label}] No concepts found.")
        return

    word_counts = []
    for c in concepts:
        desc = c.get("description", "")
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

# ── Merge Verification Runner ─────────────────────────────────────────────────

def make_candidate_pairs_prompt(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    blocks = []
    for i, (c1, c2) in enumerate(pairs, 1):
        c1_context = get_context_hint(c1.get("description", ""), max_words=40)
        c2_context = get_context_hint(c2.get("description", ""), max_words=40)
        c1_aliases = ", ".join(c1.get("aliases", [])) or "None"
        c2_aliases = ", ".join(c2.get("aliases", [])) or "None"

        block = (
            f"PAIR #{i}:\n"
            f"  Concept 1: \"{c1.get('title')}\"\n"
            f"    Aliases: {c1_aliases}\n"
            f"    Context (~40 words): {c1_context}\n"
            f"  Concept 2: \"{c2.get('title')}\"\n"
            f"    Aliases: {c2_aliases}\n"
            f"    Context (~40 words): {c2_context}"
        )
        blocks.append(block)
    return "VERIFY THE FOLLOWING CONCEPT PAIRS (Decide if they denote the EXACT SAME entity):\n\n" + "\n\n".join(blocks)

async def verify_merge_batch(client: Any, batch: List[Tuple[Dict[str, Any], Dict[str, Any]]], settings: Any, args: Any) -> List[MergeVerdict]:
    user_prompt = make_candidate_pairs_prompt(batch)

    extra_body: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    if getattr(args, "disable_thinking", False):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        model_kwargs = {
            "temperature": 0.7,
            "top_p": 0.80,
            "presence_penalty": 1.5,
            "extra_body": extra_body,
        }
    else:
        model_kwargs = {
            "temperature": 0.0,
        }

    response = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MERGE_VERIFY},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=settings.max_tokens or 8192,
        **model_kwargs
    )
    raw_text = response.choices[0].message.content or ""
    return parse_xml_merge_verdicts(raw_text)

# ── Main Pipeline Execution ───────────────────────────────────────────────────

def global_consolidate(domain_dir: Path, args: Any):
    consolidated_dir = domain_dir / "consolidated"
    concepts_dir = domain_dir / "concepts"
    log_file = domain_dir / "global_consolidation.json"

    if not consolidated_dir.exists():
        raise FileNotFoundError(f"No consolidated directory found at {consolidated_dir}. Run document consolidation first.")

    # 1. Load All Consolidated Concepts from per-document folders
    all_raw_concepts = []
    doc_files = sorted(list(consolidated_dir.glob("canon_*/concepts.json")))
    if not doc_files:
        raise RuntimeError(f"No consolidated concept files found under {consolidated_dir}")

    print(f"Loading concepts from {len(doc_files)} canonical document folders...")
    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("concepts", []):
                t = c.get("title", "").strip()
                if not t or is_boilerplate_concept(t):
                    continue
                all_raw_concepts.append(c)

    raw_concepts_count = len(all_raw_concepts)
    print(f"Total concepts loaded (after dropping scraping boilerplate): {raw_concepts_count}")

    # Display initial word-count distribution before global merge
    print_concept_word_distribution(all_raw_concepts, title_label="STAGE 6: BEFORE GLOBAL CONSOLIDATION")

    # 2. Gather Candidate Merge Pairs (Exact Title, Lemma/Plural Variants, and Title-to-Alias alignments)
    # Group by canonical lemma key to find candidate exact and lemma variants across documents
    lemma_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        t = c.get("title", "").strip()
        if not t:
            continue
        key = canonical_lemma_key(t)
        lemma_groups.setdefault(key, []).append(c)

    candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen_pair_keys: Set[Tuple[str, str]] = set()

    # (A) Candidate pairs from exact/lemma match across different documents/instances
    for key, group in lemma_groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, min(i + 5, len(group))):  # pair across distinct concepts in group
                    c1, c2 = group[i], group[j]
                    pk = tuple(sorted([c1.get("title", ""), c2.get("title", "")]))
                    if pk[0].lower() != pk[1].lower() and pk not in seen_pair_keys:
                        seen_pair_keys.add(pk)
                        candidate_pairs.append((c1, c2))

    # (B) Candidate pairs from Title-to-Alias alignments
    alias_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        for alias in c.get("aliases", []):
            clean_al = canonical_lemma_key(alias)
            if len(clean_al) > 3 and clean_al not in STOP_WORDS:
                alias_map.setdefault(clean_al, []).append(c)

    for c in all_raw_concepts:
        norm_title = canonical_lemma_key(c.get("title", ""))
        if norm_title in alias_map:
            for match in alias_map[norm_title]:
                if canonical_lemma_key(match.get("title", "")) != norm_title:
                    pk = tuple(sorted([c.get("title", ""), match.get("title", "")]))
                    if pk not in seen_pair_keys:
                        seen_pair_keys.add(pk)
                        candidate_pairs.append((c, match))

    print(f"Identified {len(candidate_pairs)} candidate merge pair(s) (exact/lemma/alias) with contextual descriptions.")

    # 3. LLM Verification of Candidate Pairs
    uf = UnionFind()
    for c in all_raw_concepts:
        t = c.get("title", "").strip()
        if t:
            uf.find(t)

    # Identical strings always union immediately
    for c in all_raw_concepts:
        t = c.get("title", "").strip()
        if t:
            uf.union(t, t)

    approved_merges = 0
    if candidate_pairs and not getattr(args, "skip_alias_verification", False):
        settings = resolve_settings(args, STAGE)
        batch_size = 10
        batches = [candidate_pairs[i:i + batch_size] for i in range(0, len(candidate_pairs), batch_size)]

        async def run_verification():
            nonlocal approved_merges
            pool = pool_for_stage(args, settings, STAGE)

            async def handler(client, batch, server_index):
                return await verify_merge_batch(client, batch, settings, args)

            async def on_batch_result(batch, verdicts):
                nonlocal approved_merges
                if verdicts:
                    for verdict in verdicts:
                        idx = verdict.pair_index - 1
                        if 0 <= idx < len(batch):
                            c1, c2 = batch[idx]
                            if verdict.is_same_entity:
                                uf.union(c1["title"], c2["title"])
                                approved_merges += 1

            async with pool:
                await pool.map(
                    batches,
                    handler,
                    attempts=settings.attempts,
                    desc="LLM Verifying Candidate Merge Pairs",
                    budget=None,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=on_batch_result,
                )

        asyncio.run(run_verification())
        print(f"LLM confirmed and approved {approved_merges} concept merges.")
    else:
        print("Skipping LLM candidate pair verification (or no candidates found).")

    # 4. Final Attribute, Provenance & Description Fusion
    final_groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_raw_concepts:
        title = c.get("title", "").strip()
        if not title:
            continue
        final_groups.setdefault(uf.find(title), []).append(c)

    global_concepts = []
    for root, concepts in tqdm(final_groups.items(), desc="Fusing global concept attributes"):
        title_counter = Counter(c.get("title", "").strip() for c in concepts)
        chosen_title = title_counter.most_common(1)[0][0]

        merged_aliases_set = set()
        global_source_chunks = set()
        seen_sentences = set()
        description_sentences = []

        for c in concepts:
            merged_aliases_set.add(c.get("title", "").strip())
            for a in c.get("aliases", []):
                if a.strip():
                    merged_aliases_set.add(a.strip())

            for sc in c.get("source_chunks", []):
                if sc.strip():
                    global_source_chunks.add(sc.strip())

        desc_by_chunks: Dict[Tuple[str, ...], List[str]] = {}
        for c in concepts:
            desc = c.get("description", "").strip()
            chunks_key = tuple(sorted(c.get("source_chunks", [])))
            if desc:
                for sent in re.split(r'(?<=[.!?])\s+', desc):
                    clean_sent = sent.strip()
                    if clean_sent and clean_sent.lower() not in seen_sentences:
                        seen_sentences.add(clean_sent.lower())
                        desc_by_chunks.setdefault(chunks_key, []).append(clean_sent)
                        description_sentences.append(clean_sent)

        cited_paragraphs = []
        for chunks_tuple, sents in desc_by_chunks.items():
            if not sents:
                continue
            para_text = " ".join(sents)
            if chunks_tuple:
                cites = " ".join(f"[{ch}]" for ch in chunks_tuple)
                para_text = f"{para_text} {cites}"
            cited_paragraphs.append(para_text)

        final_aliases = sorted([a for a in merged_aliases_set if a.lower() != chosen_title.lower()])

        global_concepts.append({
            "title": chosen_title,
            "aliases": final_aliases,
            "description": "\n\n".join(cited_paragraphs) if cited_paragraphs else " ".join(description_sentences),
            "source_chunks": sorted(list(global_source_chunks)),
        })

    # Save summary and global concepts
    log_data = {
        "summary": {
            "canonical_docs_evaluated": len(doc_files),
            "raw_input_concepts": raw_concepts_count,
            "candidate_pairs_checked": len(candidate_pairs),
            "merges_approved_by_llm": approved_merges,
            "final_global_concepts": len(global_concepts),
        }
    }
    atomic_write_json(log_file, log_data)

    concepts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving global_concepts.json to {domain_dir}...")
    atomic_write_json(domain_dir / "global_concepts.json", {"concepts": global_concepts})

    print("Saving individual concept files to concepts/...")
    for gc in global_concepts:
        safe_name = sanitize_filename(gc["title"]) or "unnamed_concept"
        file_path = concepts_dir / f"{safe_name}.json"
        counter = 1
        while file_path.exists():
            file_path = concepts_dir / f"{safe_name}_{counter}.json"
            counter += 1
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(gc, f, indent=2)

    # Print final word count distribution across global concepts
    print_concept_word_distribution(global_concepts, title_label="STAGE 6: AFTER GLOBAL CONSOLIDATION")

    print("\n============================================================")
    print("GLOBAL CONSOLIDATION COMPLETE")
    print("============================================================")
    print(f"Total Canonical Docs Merged:  {len(doc_files)}")
    print(f"Raw Input Concepts Loaded:    {raw_concepts_count}")
    print(f"Candidate Pairs Audited:      {len(candidate_pairs)}")
    print(f"Approved LLM Merges:          {approved_merges}")
    print(f"Final Global Concepts:        {len(global_concepts)} (fused {raw_concepts_count - len(global_concepts)} duplicates)")
    print("============================================================\n")

def main():
    parser = build_parser(stage=STAGE, description="Consolidate concepts across documents using LLM-verified matches")
    parser.add_argument("--skip_alias_verification", action="store_true",
                        help="Skip LLM verification of candidate pairs.")
    args = parser.parse_args()

    wiki_path = getattr(args, "wiki_path", None)
    if not wiki_path:
        raise ValueError("Missing required --wiki_path argument.")

    domain_dir = Path(wiki_path)
    global_consolidate(domain_dir, args)

if __name__ == "__main__":
    main()
