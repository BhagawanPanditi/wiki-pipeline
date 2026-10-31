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
from utils.atomic_io import atomic_write_json, atomic_write_text
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

        self.generate_audit_markdown(summary)

    def generate_audit_markdown(self, summary: Dict[str, Any]) -> None:
        """Generate comprehensive audit.md in wiki directory combining all pipeline metrics."""
        audit_md_path = self.wiki_path / "audit.md"
        print(f"Compiling complete pipeline audit into {audit_md_path}...")

        lines = [
            f"# Complete Knowledge Wiki Quality & Pipeline Audit Report: {self.domain.upper()}\n",
            "> Comprehensive post-pipeline benchmark combining extraction audits, stage deltas, topological island health, link connectivity, word-count distributions, and document fact retention.\n",
            "## 1. Executive Summary Benchmark\n",
            "| Metric | Result | Benchmark Status |",
            "| :--- | :--- | :--- |",
            f"| **Evaluated Documents (Smallest < 20k tokens)** | `{summary.get('total_documents_audited', 0)}` | Completed |",
            f"| **Average Factual Retention Score** | `{summary.get('average_fact_retention_score', 0):.1f}%` | {'High Fidelity (>=85%)' if summary.get('average_fact_retention_score', 0) >= 85 else 'Acceptable'} |",
            f"| **Average Concept Quality Score** | `{summary.get('average_concept_quality_score', 0):.1f}%` | {'High Coherence (>=85%)' if summary.get('average_concept_quality_score', 0) >= 85 else 'Acceptable'} |",
            f"| **Total Cited Concept Paragraphs Audited** | `{summary.get('total_cited_paragraphs_evaluated', 0)}` | Verified |",
            "\n---\n",
        ]

        # 2. Verification Pass 1 & Pass 2 Delta
        summary_json_path = self.wiki_path / "verification_summary.json"
        if summary_json_path.exists():
            try:
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                before = v_data.get("before", {})
                after = v_data.get("after", {})
                improvement = v_data.get("improvement", {})

                lines.extend([
                    "## 2. Extraction Verification Audit & Delta (Pass 1 vs. Pass 2)\n",
                    f"- **Total Chunks Verified**: {v_data.get('total_chunks_evaluated', before.get('total_chunks', 0))}",
                    "\n| Verification Metric | Pass 1 (Raw Extraction) | Pass 2 (Refined Extraction) | Absolute Delta | Relative Change |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                    f"| **Average Coverage Score** | {before.get('average_coverage', 0):.3f} / 1.000 | {after.get('average_coverage', 0):.3f} / 1.000 | `{improvement.get('average_coverage_delta', 0):+.3f}` | - |",
                    f"| **Pass Rate (Sufficient Chunks)** | {before.get('pass_rate', 0):.1f}% ({before.get('sufficient_count', 0)}) | {after.get('pass_rate', 0):.1f}% ({after.get('sufficient_count', 0)}) | `{improvement.get('pass_rate_delta', 0):+.1f}p` | `{improvement.get('sufficient_delta', 0):+d} chunks` |",
                    f"| **Missed Concepts Identified** | {before.get('missed_concepts', 0)} | {after.get('missed_concepts', 0)} | `{improvement.get('missed_concepts_delta', 0):+d}` | `{improvement.get('missed_concepts_pct_change', 0):+.1f}%` |",
                    f"| **Missed Facts Identified** | {before.get('missed_facts', 0)} | {after.get('missed_facts', 0)} | `{improvement.get('missed_facts_delta', 0):+d}` | `{improvement.get('missed_facts_pct_change', 0):+.1f}%` |",
                    "\n",
                ])
            except Exception as e:
                lines.append(f"\n*(Notice: Could not parse verification_summary.json: {e})*\n")

        # 3. Graph Topology & Island Report
        islands_json_path = self.wiki_path / "islands_report.json"
        if islands_json_path.exists():
            try:
                with open(islands_json_path, "r", encoding="utf-8") as f:
                    isl = json.load(f)
                reach = isl.get("directed_top_k_reachability", {})
                lines.extend([
                    "## 3. Knowledge Graph Topology & Island Audit\n",
                    "| Graph Topological Metric | Value |",
                    "| :--- | :--- |",
                    f"| **Total Concepts in Final Wiki** | `{isl.get('total_concepts', 0)}` |",
                    f"| **Giant Component Coverage** | **{isl.get('giant_component_coverage_pct', 0):.1f}%** ({isl.get('giant_component_size', 0)}/{isl.get('total_concepts', 0)} concepts) |",
                    f"| **Total Disconnected Islands** | `{isl.get('total_islands', 0)}` |",
                    f"| **Completely Isolated Singletons** | `{isl.get('isolated_concepts_count', 0)}` |",
                    f"| **Primary Inbound Connected Concepts** | `{reach.get('final_primary_inbound_connected', 0)}/{isl.get('total_concepts', 0)}` |",
                    f"| **Rescued Source-Only Peripheral Links** | `{reach.get('rescued_backlinks_injected', 0)}` |",
                    f"| **Bidirectionally Connected Concepts** | `{reach.get('final_primary_bidirectional', 0)}` |",
                    "\n",
                ])
            except Exception as e:
                lines.append(f"\n*(Notice: Could not parse islands_report.json: {e})*\n")

        # 4. Word Count Distribution & Granularity Stats
        global_concepts_path = self.wiki_path / "global_concepts.json"
        if global_concepts_path.exists():
            try:
                with open(global_concepts_path, "r", encoding="utf-8") as f:
                    gc_data = json.load(f)
                concepts = gc_data.get("concepts", [])
                word_counts = [len(re.findall(r"\b\w+\b", c.get("description", ""))) for c in concepts]
                link_counts = [len(c.get("outbound_relations", [])) for c in concepts]

                total_c = len(word_counts)
                if total_c > 0:
                    word_counts.sort()
                    avg_wc = sum(word_counts) / total_c
                    median_wc = word_counts[total_c // 2]
                    p10 = word_counts[int(total_c * 0.10)]
                    p90 = word_counts[int(total_c * 0.90)]

                    bins = [
                        ("0 - 10 words (Stubs)", 0, 10),
                        ("11 - 20 words (Stubs)", 11, 20),
                        ("21 - 30 words (Thin)", 21, 30),
                        ("31 - 50 words (Concise)", 31, 50),
                        ("51 - 100 words (Standard)", 51, 100),
                        ("101 - 200 words (Multi-sentence)", 101, 200),
                        ("201 - 350 words (Comprehensive)", 201, 350),
                        ("351 - 500 words (Dense Merge)", 351, 500),
                        ("501 - 750 words (Broad Pillar)", 501, 750),
                        ("751 - 1000 words (Composite Topic)", 751, 1000),
                        ("1001+ words (Massive Topic)", 1001, 9999999),
                    ]

                    lines.extend([
                        "## 4. Concept Description Word Count Distribution\n",
                        f"- **Total Concepts Analyzed**: {total_c}",
                        f"- **Average Word Count**: `{avg_wc:.1f} words`",
                        f"- **Median Word Count**: `{median_wc} words` (P10: `{p10}`, P90: `{p90}`, Min: `{word_counts[0]}`, Max: `{word_counts[-1]}`)",
                        "\n| Word Count Range | Concept Count | Percentage | Profile |",
                        "| :--- | :--- | :--- | :--- |",
                    ])

                    for label, low, high in bins:
                        cnt = sum(1 for w in word_counts if low <= w <= high)
                        pct = (cnt / total_c * 100)
                        bar = "█" * int(pct / 4)
                        lines.append(f"| **{label}** | {cnt} | {pct:>5.1f}% | `{bar}` |")
                    lines.append("\n")

                    # Link Distribution
                    avg_links = sum(link_counts) / total_c if total_c else 0.0
                    lines.extend([
                        "## 5. Related Link Distribution & Connectivity\n",
                        f"- **Average Outbound Links per Concept**: `{avg_links:.2f}`",
                        f"- **Max Outbound Links on Hub Concept**: `{max(link_counts) if link_counts else 0}`",
                        f"- **Zero-Outbound Concepts**: `{sum(1 for l in link_counts if l == 0)}`",
                        "\n| Link Range | Concept Count | Percentage |",
                        "| :--- | :--- | :--- |",
                    ])
                    link_bins = [
                        ("0 links", 0, 0),
                        ("1 - 5 links", 1, 5),
                        ("6 - 10 links", 6, 10),
                        ("11 - 15 links (Top Window)", 11, 15),
                        ("16 - 25 links", 16, 25),
                        ("26+ links (Hubs)", 26, 99999),
                    ]
                    for label, low, high in link_bins:
                        cnt = sum(1 for l in link_counts if low <= l <= high)
                        pct = (cnt / total_c * 100)
                        lines.append(f"| **{label}** | {cnt} | {pct:>5.1f}% |")
                    lines.append("\n")

            except Exception as e:
                lines.append(f"\n*(Notice: Could not analyze global_concepts.json: {e})*\n")

        # 6. Detailed Document Fact Retention Breakdown
        docs = summary.get("documents", [])
        if docs:
            lines.extend([
                "## 6. Document-Level Fact Retention & Quality Sample Breakdown\n",
                "| Document ID | Chars (~Tokens) | Cited Paras | Concepts | Fact Retention | Concept Quality | Summary Verdict |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for d in docs:
                v = d.get("summary_verdict", "").replace("\n", " ")
                if len(v) > 90:
                    v = v[:87] + "..."
                lines.append(
                    f"| `{d['doc_id']}` | {d['doc_chars']} (~{d['est_tokens']}) | {d['cited_paragraphs_count']} | {d['cited_concepts_count']} | **{d['fact_retention_score']}%** | **{d['concept_quality_score']}%** | {v} |"
                )
            lines.append("\n")

        atomic_write_text(audit_md_path, "\n".join(lines))
        print(f"Successfully wrote unified markdown audit to: {audit_md_path}")


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
