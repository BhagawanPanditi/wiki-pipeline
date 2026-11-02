#!/usr/bin/env python3
"""Option A: map-reduce decomposition for oversized concepts (Stage 6b helper).
---------------------------------------------------------------------------
Used when a concept's paragraphs do not fit in a single prompt.

  MAP     Pack paragraph excerpts into token-bounded batches; each batch
          proposes candidate subtopic titles (small output).
  REDUCE  Dedupe candidate titles, then one small LLM call consolidates them
          into N final child titles (N scales with concept size).
  ASSIGN  Each paragraph is assigned to exactly one bucket (0 = pillar/overview,
          1..N = children). Paragraphs are NEVER rewritten, so doc_id provenance
          is preserved by construction and nothing can be dropped.

Integration:
    class ConceptDecompositionPipeline(OversizedDecomposerMixin): ...
"""

import asyncio
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from openai import BadRequestError

from utils.lemma import canonical_title_key
from utils.llm_params import call_llm

# ── Tunables ──────────────────────────────────────────────────────────────────

DEFAULT_MAX_MODEL_LEN = 98304
PROMPT_OVERHEAD_TOKENS = 1500     # system prompt + framing
SAFETY_MARGIN = 2000              # headroom for tokenizer mismatch
TOKEN_SAFETY_FACTOR = 1.15        # tiktoken(cl100k) vs the served model's tokenizer

PROPOSAL_BATCH_TOKENS = 24000     # keep map prompts modest: faster, more parallel
PROPOSAL_SNIPPET_WORDS = 350      # per-paragraph cap in the MAP phase
ASSIGN_BATCH_TOKENS = 16000
ASSIGN_SNIPPET_WORDS = 150        # assignment only needs the gist
MAX_ASSIGN_PER_BATCH = 120

WORDS_PER_CHILD = 2500            # target children ~ total_words / this
MAX_CHILDREN = 10
MAX_CANDIDATES_IN_REDUCE = 80

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROPOSE = """\
You are a knowledge architect. You are shown numbered excerpts that all belong to ONE broad
concept in a technical wiki. Propose the distinct SUBTOPICS these excerpts naturally fall into.

RULES:
- Titles: 1-6 words, Title Case, self-contained and specific (e.g. "String Namespace Methods",
  NOT just "Methods"). A title must never equal the parent concept title.
- Subtopics must be mutually distinct, not overlapping rewordings of each other.
- Output ONLY this XML, no commentary:
<titles>
<title>Subtopic Title A</title>
<title>Subtopic Title B</title>
</titles>
"""

SYSTEM_REDUCE = """\
You are a knowledge architect. You are given candidate subtopic titles (with how many excerpt
batches proposed each) for ONE broad concept. Many are near-duplicates or rewordings.

TASK: Produce the FINAL list of distinct child titles (at most the requested number).
- Merge near-duplicates into one clear title; prefer the most standard, specific wording.
- Together the final titles should cover the candidates' full scope with minimal overlap.
- Titles: 1-6 words, Title Case, self-contained. Never equal to the parent title.
- Output ONLY this XML, no commentary:
<titles>
<title>Final Title A</title>
<title>Final Title B</title>
</titles>
"""

SYSTEM_ASSIGN = """\
You are a knowledge architect. You are given a numbered MENU of subtopics for one broad concept,
then numbered excerpts. Assign EVERY excerpt to exactly one menu number.

- 0 = OVERVIEW: general definition, scope, introduction, or high-level role of the parent concept.
- 1..N = the single best-fitting subtopic.
- Output ONLY one line per excerpt as "<excerpt number> -> <menu number>" inside the XML tag:
<assignments>
12 -> 3
13 -> 0
</assignments>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

_ENC = None


def count_tokens(text: str) -> int:
    """Conservative token estimate: tiktoken if available, char heuristic otherwise."""
    global _ENC
    try:
        if _ENC is None:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        n = len(_ENC.encode(text, disallowed_special=()))
    except Exception:
        n = int(len(text) / 3.2) + 1
    return int(n * TOKEN_SAFETY_FACTOR) + 1


def truncate_words(text: str, max_words: int) -> str:
    text = text.strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def is_context_overflow(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, BadRequestError) and (
        "maximum context length" in msg or "input_tokens" in msg
    )


def pack_by_tokens(items: List[Dict[str, Any]], budget: int,
                   max_items: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    size = 0
    for it in items:
        t = count_tokens(it["text"]) + 20  # per-item framing
        if cur and (size + t > budget or (max_items and len(cur) >= max_items)):
            batches.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += t
    if cur:
        batches.append(cur)
    return batches


def parse_titles(text: str) -> List[str]:
    out = []
    for t in re.findall(r"<title>(.*?)</title>", text or "", re.DOTALL | re.IGNORECASE):
        t = re.sub(r"\s+", " ", t).strip().strip('"').strip()
        if t:
            out.append(t)
    return out


def parse_assignments(text: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, c in re.findall(r"(\d+)\s*(?:->|=>|=|:)\s*(\d+)", text or ""):
        out[int(i)] = int(c)
    return out


_WORD_RE = re.compile(r"[a-z0-9_+.]{3,}")


def lexical_fallback(text: str, child_titles: List[str]) -> int:
    """Deterministic bucket for paragraphs the LLM failed to assign (0 = pillar)."""
    words = set(_WORD_RE.findall(text.lower()))
    best, best_score = 0, 0
    for i, t in enumerate(child_titles, 1):
        score = len(words & set(_WORD_RE.findall(t.lower())))
        if score > best_score:
            best, best_score = i, score
    return best


# ── Mixin ─────────────────────────────────────────────────────────────────────

class OversizedDecomposerMixin:
    """Requires self.args and self.settings (as in ConceptDecompositionPipeline)."""

    # -- budgets --------------------------------------------------------------

    def _max_model_len(self) -> int:
        v = getattr(self.args, "max_model_len", None)
        if not v:
            try:
                from config import _read_config
                v = _read_config().get("max_model_len")
            except Exception:
                v = None
        return int(v or DEFAULT_MAX_MODEL_LEN)

    def single_shot_max_out(self) -> int:
        return min(self.settings.max_tokens or 4096, 4096)

    def single_shot_input_budget(self) -> int:
        """Max paragraph tokens that safely fit in the one-shot decompose prompt."""
        return (self._max_model_len() - self.single_shot_max_out()
                - PROMPT_OVERHEAD_TOKENS - SAFETY_MARGIN)

    def _batch_budget(self, max_out: int, cap: int) -> int:
        return min(cap, self._max_model_len() - max_out
                   - PROMPT_OVERHEAD_TOKENS - SAFETY_MARGIN)

    # -- LLM plumbing ---------------------------------------------------------

    async def _ask(self, client: Any, system: str, user: str, max_tokens: int) -> str:
        return await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            args=self.args,
            task="judge",
            max_growth=0,
        )

    async def _call_split(self, batch: List[Any], fn) -> List[Any]:
        """Run fn(batch); on context/length overflow, halve the batch and retry."""
        try:
            return await fn(batch)
        except Exception as exc:
            too_long = is_context_overflow(exc) or "hit max_tokens" in str(exc).lower()
            if not too_long or len(batch) <= 1:
                raise
            mid = len(batch) // 2
            left, right = await asyncio.gather(
                self._call_split(batch[:mid], fn),
                self._call_split(batch[mid:], fn),
            )
            return left + right

    # -- MAP ------------------------------------------------------------------

    async def _propose_titles(self, client, title: str, views: List[Dict[str, Any]],
                              target_n: int, sem: asyncio.Semaphore) -> List[str]:
        max_out = 600
        budget = self._batch_budget(max_out, PROPOSAL_BATCH_TOKENS)
        items = [{**v, "text": truncate_words(v["text"], PROPOSAL_SNIPPET_WORDS)} for v in views]
        batches = pack_by_tokens(items, budget)
        per_batch = min(target_n, 5)

        async def fn(batch):
            body = "\n\n".join(f"[{v['idx']}] (doc {v['doc_id']})\n{v['text']}" for v in batch)
            user = (f'PARENT CONCEPT: "{title}"\n'
                    f"PROPOSE UP TO {per_batch} SUBTOPICS FOR THE EXCERPTS BELOW.\n\n{body}\n")
            async with sem:
                raw = await self._ask(client, SYSTEM_PROPOSE, user, max_out)
            return parse_titles(raw)

        results = await asyncio.gather(*(self._call_split(b, fn) for b in batches))
        return [t for r in results for t in r]

    # -- REDUCE ---------------------------------------------------------------

    async def _reduce_titles(self, client, title: str, candidates: List[str],
                             target_n: int) -> List[str]:
        parent_key = canonical_title_key(title)
        counts: Counter = Counter()
        display: Dict[str, str] = {}
        for t in candidates:
            k = canonical_title_key(t)
            if not k or k == parent_key:
                continue
            counts[k] += 1
            display.setdefault(k, t)

        ranked = [display[k] for k, _ in counts.most_common(MAX_CANDIDATES_IN_REDUCE)]
        if len(ranked) <= target_n:
            return ranked

        listing = "\n".join(f"- {display[k]} (proposed {n}x)"
                            for k, n in counts.most_common(MAX_CANDIDATES_IN_REDUCE))
        user = (f'PARENT CONCEPT: "{title}"\nMAX FINAL TITLES: {target_n}\n\n'
                f"CANDIDATE TITLES:\n{listing}\n")
        try:
            raw = await self._ask(client, SYSTEM_REDUCE, user, 600)
            final, seen = [], {parent_key}
            for t in parse_titles(raw):
                k = canonical_title_key(t)
                if k and k not in seen:
                    seen.add(k)
                    final.append(t)
            if final:
                return final[:target_n]
        except Exception as exc:
            print(f"[WARN] title reduce failed for '{title}': {exc}; using top-{target_n} by frequency.")
        return ranked[:target_n]

    # -- ASSIGN ---------------------------------------------------------------

    async def _assign_paragraphs(self, client, title: str, views: List[Dict[str, Any]],
                                 child_titles: List[str],
                                 sem: asyncio.Semaphore) -> Dict[int, int]:
        max_out = 3000
        budget = self._batch_budget(max_out, ASSIGN_BATCH_TOKENS)
        n = len(child_titles)
        menu = "0. OVERVIEW (general definition / scope / role of the parent concept)\n" + \
               "\n".join(f"{i}. {t}" for i, t in enumerate(child_titles, 1))
        items = [{**v, "text": truncate_words(v["text"], ASSIGN_SNIPPET_WORDS)} for v in views]
        batches = pack_by_tokens(items, budget, max_items=MAX_ASSIGN_PER_BATCH)

        async def fn(batch):
            body = "\n\n".join(f"[{v['idx']}]\n{v['text']}" for v in batch)
            user = f'PARENT CONCEPT: "{title}"\n\nMENU:\n{menu}\n\nEXCERPTS:\n\n{body}\n'
            async with sem:
                raw = await self._ask(client, SYSTEM_ASSIGN, user, max_out)
            parsed = parse_assignments(raw)
            valid = {v["idx"] for v in batch}
            return [(i, c) for i, c in parsed.items() if i in valid and 0 <= c <= n]

        results = await asyncio.gather(*(self._call_split(b, fn) for b in batches))
        return {i: c for r in results for i, c in r}

    # -- Orchestrator ---------------------------------------------------------

    async def decompose_oversized(self, client: Any, item: Dict[str, Any],
                                  paras: List[Dict[str, Any]]) -> Dict[str, Any]:
        title = item.get("title", "")
        not_decomposed = {"title": title, "decomposed": False,
                          "pillar_paragraphs": [], "children": []}

        views = [{"idx": i, "doc_id": p.get("doc_id", "unknown"),
                  "text": (p.get("text") or "").strip()}
                 for i, p in enumerate(paras)]
        views = [v for v in views if v["text"]]
        if len(views) < 2:
            return not_decomposed

        sem = asyncio.Semaphore(max(1, getattr(self.args, "batch_concurrency", 4)))
        total_words = sum(len(v["text"].split()) for v in views)
        target_n = max(2, min(MAX_CHILDREN, math.ceil(total_words / WORDS_PER_CHILD)))
        print(f"[MAP-REDUCE] '{title}': {len(views)} paragraphs, {total_words} words "
              f"-> targeting up to {target_n} children.")

        # 1. MAP
        candidates = await self._propose_titles(client, title, views, target_n, sem)
        # 2. REDUCE
        child_titles = await self._reduce_titles(client, title, candidates, target_n)
        if len(child_titles) < 2:
            print(f"[MAP-REDUCE] '{title}': <2 usable child titles; leaving undecomposed.")
            return not_decomposed

        # 3. ASSIGN
        assignment = await self._assign_paragraphs(client, title, views, child_titles, sem)

        # 4. BUILD (original paragraph dicts are carried over verbatim)
        groups: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(child_titles) + 1)}
        fallback_used = 0
        for v in views:
            bucket = assignment.get(v["idx"])
            if bucket is None:
                bucket = lexical_fallback(v["text"], child_titles)
                fallback_used += 1
            groups[bucket].append(dict(paras[v["idx"]]))
        if fallback_used:
            print(f"[MAP-REDUCE] '{title}': {fallback_used} paragraph(s) assigned by lexical fallback.")

        # The pillar must have at least one paragraph: borrow from the largest child.
        if not groups[0]:
            donor = max((i for i in groups if i > 0 and len(groups[i]) >= 2),
                        key=lambda i: len(groups[i]), default=None)
            if donor is None:
                return not_decomposed
            groups[0].append(groups[donor].pop(0))

        children = [{"title": child_titles[i - 1], "paragraphs": groups[i]}
                    for i in range(1, len(child_titles) + 1) if groups[i]]
        if not children:
            return not_decomposed

        # Provenance sanity check: every input paragraph appears exactly once.
        out_count = len(groups[0]) + sum(len(c["paragraphs"]) for c in children)
        assert out_count == len(views), f"paragraph count mismatch for '{title}'"

        print(f"[MAP-REDUCE] '{title}': pillar={len(groups[0])} paras, "
              + ", ".join(f"{c['title']}={len(c['paragraphs'])}" for c in children))
        return {"title": title, "decomposed": True,
                "pillar_paragraphs": groups[0], "children": children,
                "strategy": "map_reduce"}
