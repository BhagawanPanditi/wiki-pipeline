#!/usr/bin/env python3
"""Stage 10: Post-Pipeline Fact Retention & Concept Quality Audit.
------------------------------------------------------------------
Audits the final generated Markdown concepts against the original source documents.
Targets the top N smallest documents under 20,000 tokens (~80,000 chars) for a given domain:
1. Gathers all paragraphs citing each document across `concepts_md/*.md`.
2. Evaluates factual retention, completeness vs. loss, precision/hallucination,
   and overall concept quality against the raw document text using the LLM pool.
3. Generates a comprehensive JSON benchmark audit report and summary metrics.

Usage:
    python3 audit_wiki_quality.py --wiki_path /path/to/wiki/biology --corpus_path /path/to/data/biology.jsonl --limit 40
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from config import _read_config
from utils.atomic_io import atomic_write_json
from utils.cli import build_parser, resolve_settings
from utils.corpus import load_corpus
from utils.errors import StageBudget
from utils.llm_params import call_llm
from utils.llm_pool import pool_for_stage

STAGE = "audit-quality"

SYSTEM_PROMPT_AUDIT = """\
You are an expert scientific auditor evaluating information extraction and factual retention in a Knowledge Wiki.
Your task is to compare the EXTRACTED CONCEPT PARAGRAPHS citing a specific document against the ORIGINAL SOURCE DOCUMENT.

EVALUATION CRITERIA:
1. Fact Retention: Does the conceptual extraction capture the primary definitions, metrics, parameters, mechanisms, and findings from the source document?
2. Completeness vs Loss: What critical facts or core concepts were omitted?
3. Precision & Non-Hallucination: Are all statements in the extracted concepts faithful to the source document?
4. Coherence: Are the concepts cohesive and meaningful units of knowledge?

OUTPUT FORMAT:
Output your detailed analysis followed strictly by a JSON block:
```json
{
  "doc_id": "string",
  "fact_retention_score": <number 0-100>,
  "concept_quality_score": <number 0-100>,
  "key_retained_facts": ["string"],
  "omitted_facts": ["string"],
  "hallucinations": ["string"],
  "summary_verdict": "<1-3 sentences summarizing retention quality and precision>"
}
```
"""


def extract_paragraphs_for_doc(md_dir: Path, target_doc_id: str) -> List[Dict[str, str]]:
    """Scan all markdown concept files and return all paragraphs citing target_doc_id."""
    extracted = []
    doc_pattern = re.compile(rf"\[\s*{re.escape(target_doc_id)}(?:::[^\]]+)?\s*\]", re.IGNORECASE)

    if not md_dir.is_dir():
        return extracted

    for md_file in sorted(md_dir.glob("*.md")):
        if md_file.name in ("concept_index.md", "entity_index.md"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if target_doc_id not in content:
            continue

        blocks = re.split(r"\n\s*\n", content)
        for b in blocks:
            b_str = b.strip()
            if not b_str:
                continue
            if b_str.startswith("# ") or b_str.startswith("## Related") or b_str.startswith("**Aliases**") or b_str.startswith("**Covers**"):
                continue
            if doc_pattern.search(b_str):
                extracted.append({
                    "concept_file": md_file.name,
                    "text": b_str
                })
    return extracted


class WikiQualityAuditor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if not wiki_path:
            raise ValueError("Missing required --wiki_path argument.")
        self.wiki_path = Path(wiki_path)
        self.domain = self.wiki_path.name
        self.md_dir = self.wiki_path / "concepts_md"
        self.report_file = self.wiki_path / "wiki_quality_audit_report.json"

        self.cfg = _read_config()
        self.settings = resolve_settings(args, "verify-pass1")  # reuse verification profile

    def select_audit_documents(self) -> List[Dict[str, Any]]:
        """Select top N smallest documents that are under 20,000 tokens (~80,000 chars)."""
        corpus_path = getattr(self.args, "corpus_path", None)
        if not corpus_path:
            corpus_path = self.wiki_path.parent.parent / "data" / f"{self.domain}.jsonl"
            if not Path(corpus_path).exists():
                corpus_path = Path("data") / f"{self.domain}.jsonl"

        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        print(f"Loading corpus from {corpus_file}...")
        corpus = load_corpus(corpus_file)
        print(f"Loaded {len(corpus)} documents from corpus.")

        # Estimate tokens as approx chars / 4 (20k tokens ~ 80,000 chars)
        max_chars = getattr(self.args, "max_doc_chars", 80000)
        eligible = []
        for doc in corpus:
            text = doc.get("text", "")
            doc_len = len(text)
            if doc_len > 0 and doc_len <= max_chars:
                eligible.append({
                    "doc_id": doc["id"],
                    "text": text,
                    "chars": doc_len,
                    "est_tokens": doc_len // 4,
                })

        # Sort ascending by length (smallest first)
        eligible.sort(key=lambda d: d["chars"])
        limit = getattr(self.args, "limit", 40)
        selected = eligible[:limit]
        print(f"Selected {len(selected)} smallest documents (all <= 20,000 tokens, length: {selected[0]['chars']} to {selected[-1]['chars']} chars).")
        return selected

    async def audit_doc(self, client: Any, item: Dict[str, Any], server_index: int) -> Dict[str, Any]:
        doc_id = item["doc_id"]
        orig_text = item["text"]
        paras = extract_paragraphs_for_doc(self.md_dir, doc_id)

        paras_text = "\n\n".join([f"[{p['concept_file']}]:\n{p['text']}" for p in paras]) if paras else "[No citations found in Concepts]"

        prompt = f"""\
======================================================================
ORIGINAL SOURCE DOCUMENT ({doc_id}) [Length: {item['chars']} chars, ~{item['est_tokens']} tokens]
======================================================================
{orig_text}

======================================================================
EXTRACTED CONCEPT PARAGRAPHS CITING THIS DOCUMENT ({len(paras)} paragraphs)
======================================================================
{paras_text}

Analyze factual retention, completeness, and extraction quality according to the system instructions.
"""
        max_model_len = getattr(self.settings, "max_model_len", 65536)
        response_text = await call_llm(
            client=client,
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AUDIT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.settings.max_tokens or 4096,
            args=self.args,
            cfg=self.cfg,
            task="judge",
            max_growth=0,
            max_model_len=max_model_len,
        )

        m = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        parsed = json.loads(m.group(1)) if m else {}

        return {
            "doc_id": doc_id,
            "doc_chars": item["chars"],
            "est_tokens": item["est_tokens"],
            "cited_paragraphs_count": len(paras),
            "cited_concepts_count": len(set(p["concept_file"] for p in paras)),
            "fact_retention_score": parsed.get("fact_retention_score", 0),
            "concept_quality_score": parsed.get("concept_quality_score", 0),
            "key_retained_facts": parsed.get("key_retained_facts", []),
            "omitted_facts": parsed.get("omitted_facts", []),
            "hallucinations": parsed.get("hallucinations", []),
            "summary_verdict": parsed.get("summary_verdict", ""),
            "full_critique": response_text,
        }

    async def run(self):
        items = self.select_audit_documents()
        if not items:
            print("No eligible documents found under 20k tokens.")
            return

        pool = pool_for_stage(self.args, self.settings, STAGE)
        budget = StageBudget(self.settings.budget_minutes)

        results = []

        async def on_result(item, res):
            results.append(res)
            print(f"  [Audited] {res['doc_id']}: Fact Retention = {res['fact_retention_score']}%, Quality = {res['concept_quality_score']}% ({res['cited_paragraphs_count']} paras)")

        async with pool:
            res_obj = await pool.map(
                items,
                self.audit_doc,
                attempts=self.settings.attempts,
                desc=f"Auditing Wiki Quality [{self.domain}]",
                key_fn=lambda x: x["doc_id"],
                budget=budget,
                attempt_timeout=self.settings.profile.attempt_timeout,
                on_result=on_result,
            )
            res_obj.raise_if_failed(STAGE)

        # Aggregate summary metrics
        total = len(results)
        avg_retention = sum(r["fact_retention_score"] for r in results) / total if total else 0.0
        avg_quality = sum(r["concept_quality_score"] for r in results) / total if total else 0.0
        total_paras = sum(r["cited_paragraphs_count"] for r in results)

        summary = {
            "domain": self.domain,
            "total_documents_audited": total,
            "average_fact_retention_score": round(avg_retention, 2),
            "average_concept_quality_score": round(avg_quality, 2),
            "total_cited_paragraphs_evaluated": total_paras,
            "documents": results,
        }

        atomic_write_json(self.report_file, summary)
        print("\n" + "=" * 65)
        print(f"WIKI QUALITY & FACT RETENTION BENCHMARK [{self.domain.upper()}]")
        print(f"Documents Audited:            {total}")
        print(f"Average Fact Retention Score: {avg_retention:.1f}%")
        print(f"Average Concept Quality:      {avg_quality:.1f}%")
        print(f"Report Saved To:              {self.report_file}")
        print("=" * 65 + "\n")


def main():
    parser = build_parser(
        stage="verify-pass1",
        description="Benchmark and audit fact retention and concept quality against source documents",
        include_max_tokens=True,
        include_resume=False,
    )
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Path to corpus JSONL (e.g. data/biology.jsonl)")
    parser.add_argument("--limit", type=int, default=40,
                        help="Number of smallest documents under 20k tokens to audit (default: 40)")
    parser.add_argument("--max_doc_chars", type=int, default=80000,
                        help="Maximum document character count to consider (~20,000 tokens, default: 80,000)")
    args = parser.parse_args()

    auditor = WikiQualityAuditor(args)
    asyncio.run(auditor.run())


if __name__ == "__main__":
    main()
