#!/usr/bin/env python3
"""
Refinement Pass 2: Second-Pass Knowledge Graph Concept Extraction & Patching
- Resolves omissions, adds missing concepts, appends missing facts into descriptions.
- Supports adding/removing outbound relations, adding aliases, and adding brand new concepts.
"""

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from config import _read_config
from utils.checkpoint import load_bool_log, save_log_async, save_log_sync
from utils.chunks import load_cached_chunks
from utils.cli import build_parser, resolve_settings
from utils.errors import StageBudget
from utils.llm_pool import pool_for_stage

STAGE = "refine-concepts"

# ── Pydantic Data Schemas for Second Pass ─────────────────────────────────────

class ConceptUpdate(BaseModel):
    title: str = Field(description="Exact title of the EXISTING concept being updated.")
    additional_aliases: List[str] = Field(
        default_factory=list, 
        description="New acronyms or alternative canonical names to add."
    )
    additional_description: str = Field(
        default="",
        description="Dense factual sentences to append to existing description paragraph."
    )

class Concept(BaseModel):
    title: str = Field(description="Exact canonical title (1-6 words, singular).")
    aliases: List[str] = Field(default_factory=list, description="Explicit acronyms or alternative names.")
    description: str = Field(description="Single dense factual paragraph with all raw metrics, parameters, and details.")

class Pass2ExtractionSchema(BaseModel):
    new_concepts: List[Concept] = Field(
        default_factory=list, 
        description="Brand new substantive concepts missed entirely in Pass 1."
    )
    concept_updates: List[ConceptUpdate] = Field(
        default_factory=list, 
        description="Updates and factual patches for concepts extracted in Pass 1."
    )

# ── Second-Pass System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT_PASS2 = """\
You are an expert technical knowledge auditor performing a SECOND-PASS EXTRACTION & REFINEMENT on an LLM Knowledge Graph.

YOUR MISSION:
Perform an exhaustive comparison between the RAW CHUNK text and the EXISTING EXTRACTED CONCEPTS (from Pass 1).
Identify any missing concepts, omitted technical metrics, parameters, or factual details, and generate XML patch updates.

CONTEXT BOUNDARIES:
- CURRENT CHUNK is your source for facts. Use CONTEXT BEFORE / CONTEXT AFTER ONLY to resolve pronouns (e.g. "this system", "they") or acronyms to canonical names.
- Never extract facts that appear exclusively in CONTEXT BEFORE or CONTEXT AFTER.

CRITICAL RULES:
1. ONLY SOLID CONCEPTS:
   - Extract major, solid, well-developed subjects. Minor details and properties must be absorbed into the comprehensive description of their primary concept.
2. ZERO INFORMATION LOSS (SQUEEZE THE CHUNK DRY):
   - No useful piece of substantive informational content, technical metric, parameter bound, procedural detail, condition, or finding from CURRENT CHUNK should be left uncaptured.
   - Squeeze every useful piece of information into a concept description (either updating an existing concept via <update_concept> or adding a missed major concept via <new_concept>).
3. FACTS ARE COMPLETE SENTENCES:
   - ALL technical facts, numbers, bounds, parameters, tool mentions, and findings MUST be complete sentences inside concept descriptions.
4. SELF-AUDIT FACTUAL GAPS:
   - Perform an objective comparison between the raw chunk and existing concepts. If a metric, parameter, or concept was missed, output a patch.

THINKING PROCESS (SANDBOXED):
- If you need to analyze the text, plan missing concepts, or deliberate on updates, you MUST do so strictly inside `<think>...</think>` tags at the very beginning of your response.
- Once `</think>` is closed, output ONLY the final XML update blocks.
- Absolutely NO commentary or notes outside `<think>...</think>`.

OUTPUT FORMAT:
Output brand-new concepts using <new_concept>...</new_concept> blocks:
<new_concept>
<title>Exact Canonical Title</title>
<aliases>Acronym or Alt Name 1; Alt Name 2</aliases>
<description>A single comprehensive, fact-dense paragraph detailing what the concept is along with all raw quantitative metrics and findings from this chunk.</description>
</new_concept>

Output factual updates to existing Pass 1 concepts using <update_concept>...</update_concept> blocks:
<update_concept>
<title>Exact Canonical Title of Existing Concept</title>
<additional_aliases>New Alt Name 1; New Alt Name 2</additional_aliases>
<additional_description>Dense factual sentences containing newly extracted facts/metrics to append to the existing description.</additional_description>
</update_concept>

If the existing extraction is already complete and no updates or new concepts are needed, output nothing.
"""

# ── XML Parsing Helper for Refinement ────────────────────────────────────────

def parse_xml_refinement(text: str) -> Pass2ExtractionSchema:
    if not text:
        return Pass2ExtractionSchema()

    # 1. Strip thinking tags
    clean_text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"<thought>.*?(?:</thought>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)

    new_concepts: List[Concept] = []
    # Match <new_concept> or <concept>
    new_blocks = re.findall(
        r"<(?:new_concept|concept)>((?:(?!<(?:new_concept|concept)>).)*?)</(?:new_concept|concept)>",
        clean_text, re.DOTALL | re.IGNORECASE
    )
    for block in new_blocks:
        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m or not title_m.group(1).strip():
            continue
        title = title_m.group(1).strip()

        desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE)
        description = desc_m.group(1).strip() if desc_m else ""

        aliases = []
        aliases_m = re.search(r"<aliases>(.*?)</aliases>", block, re.DOTALL | re.IGNORECASE)
        if aliases_m:
            raw_aliases = aliases_m.group(1).strip()
            aliases = [a.strip() for a in re.split(r"[;,]", raw_aliases) if a.strip()]

        new_concepts.append(Concept(title=title, aliases=aliases, description=description))

    concept_updates: List[ConceptUpdate] = []
    # Match <update_concept>
    update_blocks = re.findall(
        r"<update_concept>((?:(?!<update_concept>).)*?)</update_concept>",
        clean_text, re.DOTALL | re.IGNORECASE
    )
    for block in update_blocks:
        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        if not title_m or not title_m.group(1).strip():
            continue
        title = title_m.group(1).strip()

        desc_m = re.search(r"<(?:additional_description|description)>(.*?)</(?:additional_description|description)>", block, re.DOTALL | re.IGNORECASE)
        additional_desc = desc_m.group(1).strip() if desc_m else ""

        aliases = []
        aliases_m = re.search(r"<(?:additional_aliases|aliases)>(.*?)</(?:additional_aliases|aliases)>", block, re.DOTALL | re.IGNORECASE)
        if aliases_m:
            raw_aliases = aliases_m.group(1).strip()
            aliases = [a.strip() for a in re.split(r"[;,]", raw_aliases) if a.strip()]

        concept_updates.append(ConceptUpdate(
            title=title,
            additional_aliases=aliases,
            additional_description=additional_desc
        ))

    # Fallback to JSON repair if no XML tags found
    if not new_concepts and not concept_updates and "{" in clean_text:
        try:
            cleaned_json = clean_json_string(clean_text)
            parsed = json.loads(cleaned_json)
            if isinstance(parsed, dict):
                nc_list = []
                for c in parsed.get("new_concepts", []):
                    if isinstance(c, dict) and c.get("title"):
                        nc_list.append(Concept(
                            title=c["title"],
                            aliases=c.get("aliases", []),
                            description=c.get("description", "")
                        ))
                cu_list = []
                for u in parsed.get("concept_updates", []):
                    if isinstance(u, dict) and u.get("title"):
                        sentences = u.get("additional_description_sentences", [])
                        add_desc = " ".join(sentences) if isinstance(sentences, list) else str(u.get("additional_description", ""))
                        cu_list.append(ConceptUpdate(
                            title=u["title"],
                            additional_aliases=u.get("additional_aliases", []),
                            additional_description=add_desc
                        ))
                return Pass2ExtractionSchema(new_concepts=nc_list, concept_updates=cu_list)
        except Exception:
            pass

    return Pass2ExtractionSchema(new_concepts=new_concepts, concept_updates=concept_updates)

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiSecondPassPipeline:
    def __init__(self, args):
        """Prepare Pass 2 refinement. Pass 1 output must already exist.

        Args:
            args: Parsed CLI namespace.

        Raises:
            FileNotFoundError: If the staging folder from Pass 1 is missing."""
        self.args = args
        wiki_path = getattr(args, "wiki_path", None)
        if wiki_path:
            self.base_wiki_dir = Path(wiki_path)
            self.domain = self.base_wiki_dir.name
        else:
            self.domain = getattr(args, "domain", "default")
            wiki_base = getattr(args, "wiki_base_dir", None) or "wiki"
            self.base_wiki_dir = Path(wiki_base) / self.domain

        self.cfg = _read_config()
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.pass2_log_file = self.base_wiki_dir / "pass2_log.json"

        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory missing at {self.staging_dir}. Run Pass 1 extraction first.")

        self.pass2_log = load_bool_log(
            self.pass2_log_file,
            corrupt_message="Pass 2 log file corrupted. Re-initializing.")

    def cleanup_incomplete_states(self):
        """Purges pass2_log entries marked False and restores backups upon resume."""
        print("Sanitizing incomplete Pass 2 extraction states...")
        purged = 0
        for key, completed in list(self.pass2_log.items()):
            if not completed:
                doc_id, chunk_id = key.split("::")
                chunk_dir = self.staging_dir / doc_id / chunk_id

                pass1_backup = chunk_dir / "concepts_pass1_backup.json"
                concepts_file = chunk_dir / "concepts.json"

                # If backup exists, restore it to fix any corrupted merges
                if pass1_backup.exists():
                    shutil.copy(pass1_backup, concepts_file)

                del self.pass2_log[key]
                purged += 1

        save_log_sync(self.pass2_log_file, self.pass2_log)
        print(f"Purged {purged} incomplete Pass 2 chunk entries.")

    def prepare_pass2_items(self) -> List[Dict[str, Any]]:
        """Prepares items containing raw chunks, Pass 1 concepts, and audit verification if present."""
        print("Loading cached chunks index...")
        chunks_map = load_cached_chunks(self.cache_dir)

        items = []
        missing_or_invalid = []
        for key, chunk_data in chunks_map.items():
            doc_id, chunk_id = key.split("::")
            chunk_dir = self.staging_dir / doc_id / chunk_id
            concepts_file = chunk_dir / "concepts.json"

            if not concepts_file.exists():
                missing_or_invalid.append(key)
                continue

            # Load Pass 1 concepts
            try:
                with open(concepts_file, "r", encoding="utf-8") as f:
                    pass1_concepts = json.load(f)
                if not isinstance(pass1_concepts.get("concepts"), list):
                    raise ValueError("concepts.json does not contain a concepts list")
            except Exception as exc:
                missing_or_invalid.append(f"{key} ({type(exc).__name__}: {exc})")
                continue

            items.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "key": key,
                "current_chunk": chunk_data["current_chunk"],
                "context_before": chunk_data.get("context_before", ""),
                "context_after": chunk_data.get("context_after", ""),
                "pass1_concepts": pass1_concepts.get("concepts", []) if isinstance(pass1_concepts, dict) else (pass1_concepts if isinstance(pass1_concepts, list) else []),
                "chunk_dir": chunk_dir
            })

        if missing_or_invalid:
            raise RuntimeError(
                f"{len(missing_or_invalid)} cached chunk(s) are missing valid Pass 1 concepts. "
                f"Examples: {missing_or_invalid[:10]}. Complete/resume Pass 1 before Pass 2."
            )

        print(f"Loaded {len(items)} chunks ready for Second Pass refinement.")
        return items

    async def extract_pass2_refinement(self, client, item: Dict[str, Any],
                                       server_index: int) -> Pass2ExtractionSchema:
        """Queries the LLM for second-pass deltas and missing facts."""
        key = item["key"]
        pass1_lines = []
        concepts_list = item["pass1_concepts"]
        if isinstance(concepts_list, dict):
            concepts_list = concepts_list.get("concepts", [])
        if isinstance(concepts_list, list):
            for i, c in enumerate(concepts_list, 1):
                if not isinstance(c, dict):
                    continue
                title = c.get("title", "")
                aliases = ", ".join(c.get("aliases", [])) if isinstance(c.get("aliases"), list) else ""
                desc = c.get("description", "").strip() if isinstance(c.get("description"), str) else ""
                block = [f"CONCEPT {i}: {title}"]
                if aliases:
                    block.append(f"  Aliases: {aliases}")
                if desc:
                    block.append(f"  Description: {desc}")
                pass1_lines.append("\n".join(block))

        pass1_text = "\n\n".join(pass1_lines) if pass1_lines else "None (empty extraction)"

        user_prompt = f"""\
CONTEXT BEFORE:
{item['context_before']}

CURRENT CHUNK:
{item['current_chunk']}

CONTEXT AFTER:
{item['context_after']}

EXISTING EXTRACTED CONCEPTS (PASS 1):
{pass1_text}

Compare the CURRENT CHUNK against EXISTING EXTRACTED CONCEPTS.
If any primary concepts were missed, output them as <new_concept>.
If any quantitative metrics, bounds, or key facts were missed from existing concepts, output <update_concept> with the exact factual sentences to append.
If the existing extraction is already factually complete, output nothing.
"""
        raw_text = None
        finish_reason = None
        try:
            cfg_enable = self.cfg.get("enable_thinking") if hasattr(self, "cfg") else None
            disable_cli = getattr(self.args, "disable_thinking", False)
            enable_cli = getattr(self.args, "enable_thinking", False)

            if disable_cli or (cfg_enable is False and not enable_cli):
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                }
                temperature = 0.7
                top_p = 0.80
                presence_penalty = 1.5
            elif enable_cli or (cfg_enable is True):
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": True},
                }
                temperature = 0.1
                top_p = 0.95
                presence_penalty = 0.0
            else:
                extra_body = None
                temperature = 0.1
                top_p = 0.95
                presence_penalty = 0.0

            response = await client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_PASS2},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=self.settings.max_tokens,
                extra_body=extra_body,
            )

            choice = response.choices[0]
            raw_text = choice.message.content or ""
            return parse_xml_refinement(raw_text)

        except Exception:
            raise

    async def persist_refinement(self, item: Dict[str, Any],
                                 delta: Pass2ExtractionSchema) -> None:
        """SUCCESS: back up Pass 1, then merge into concepts.json."""
        key = item["key"]
        chunk_dir = item["chunk_dir"]

        # 1. Backup Pass 1 concepts.json if backup does not exist
        pass1_backup = chunk_dir / "concepts_pass1_backup.json"
        concepts_file = chunk_dir / "concepts.json"
        if not pass1_backup.exists() and concepts_file.exists():
            await asyncio.to_thread(shutil.copy, concepts_file, pass1_backup)

        # 2. Apply updates & merged output to concepts.json
        merged_data = self.apply_merges(item["pass1_concepts"], delta)
        merged_payload = json.dumps(merged_data, indent=2)
        await asyncio.to_thread(concepts_file.write_text, merged_payload, encoding="utf-8")

        self.pass2_log[key] = True
        self.flush_count += 1
        if self.flush_count >= 50:
            await save_log_async(self.pass2_log_file, self.pass2_log.copy())
            self.flush_count = 0

    def apply_merges(
        self, pass1_data: Any, pass2_delta: Pass2ExtractionSchema
    ) -> Dict[str, Any]:
        """Merges Pass 2 updates cleanly into existing Pass 1 concepts JSON."""
        if isinstance(pass1_data, list):
            concepts_list = pass1_data
        elif isinstance(pass1_data, dict):
            concepts_list = pass1_data.get("concepts", [])
        else:
            concepts_list = []
        concept_map = {c["title"].strip().lower(): c for c in concepts_list if isinstance(c, dict) and "title" in c}

        # 1. Apply concept updates
        for update in pass2_delta.concept_updates:
            target_key = update.title.strip().lower()
            if target_key in concept_map:
                c = concept_map[target_key]

                # Append aliases (deduplicated)
                existing_aliases = set(c.get("aliases", []))
                for alias in update.additional_aliases:
                    if alias not in existing_aliases:
                        c.setdefault("aliases", []).append(alias)
                        existing_aliases.add(alias)

                # Append factual sentences to description
                if update.additional_description.strip():
                    existing_desc = c.get("description", "").strip()
                    c["description"] = f"{existing_desc} {update.additional_description.strip()}".strip()
            else:
                if update.additional_description.strip():
                    new_item = {
                        "title": update.title,
                        "aliases": update.additional_aliases,
                        "description": update.additional_description.strip(),
                    }
                    concepts_list.append(new_item)
                    concept_map[target_key] = new_item

        # 2. Add brand-new concepts
        for new_c in pass2_delta.new_concepts:
            new_key = new_c.title.strip().lower()
            if new_key not in concept_map:
                c_dict = {
                    "title": new_c.title,
                    "aliases": new_c.aliases,
                    "description": new_c.description,
                }
                concepts_list.append(c_dict)
                concept_map[new_key] = c_dict
            else:
                c = concept_map[new_key]
                existing_aliases = set(c.get("aliases", []))
                for a in new_c.aliases:
                    if a not in existing_aliases:
                        c.setdefault("aliases", []).append(a)
                        existing_aliases.add(a)
                if new_c.description.strip():
                    existing_desc = c.get("description", "").strip()
                    c["description"] = f"{existing_desc} {new_c.description.strip()}".strip()

        return {"concepts": concepts_list}

    async def run(self):
        """Refine every chunk that verification flagged, merging the LLM's additions into its Pass 1 concepts.

        Before a chunk's concepts are changed, its Pass 1 version is backed up, so the
        original is never lost. Each refinement is checkpointed as it completes.

        Returns:
            None. Updates each chunk's ``concepts.json`` and writes ``pass2_log.json``.

        Raises:
            RuntimeError: If any chunk is still incomplete after retries."""
        self.settings = resolve_settings(self.args, STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.flush_count = 0

        # Trigger cleanup if the resume flag is passed
        if hasattr(self.args, 'resume') and self.args.resume:
            self.cleanup_incomplete_states()

        items = self.prepare_pass2_items()
        pending = [
            i for i in items 
            if not self.pass2_log.get(i["key"], False)
        ]

        settings = self.settings
        total_pending = len(pending)
        print(f"Total chunks: {len(items)} | Pending Pass 2 refinement: {total_pending}")
        print(f"Spreading work across {len(settings.servers)} server(s) at "
              f"{settings.concurrency_per_server} concurrent request(s) each "
              f"(item i -> server i % N).")
        print(f"Per-item attempts: {settings.attempts} | Stage budget: "
              f"{settings.budget_minutes:g} min")

        if not pending:
            print("All chunks are already refined in Pass 2!")
            return

        # Mark every pending chunk in-progress
        for item in pending:
            self.pass2_log.setdefault(item["key"], False)

        pool = pool_for_stage(self.args, settings, STAGE)
        try:
            async with pool:
                result = await pool.map(
                    pending,
                    self.extract_pass2_refinement,
                    attempts=settings.attempts,
                    desc=f"Pass 2 Refinement [{self.domain}]",
                    key_fn=lambda i: i["key"],
                    budget=self.budget,
                    debug=None,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=self.persist_refinement,
                )
        finally:
            await save_log_async(self.pass2_log_file, self.pass2_log)

        print("Requests served per server:")
        for url, count in sorted(result.per_server.items(), key=lambda kv: -kv[1]):
            print(f"  {url:<40} {count}")

        failed_items = [i["key"] for i in items if not self.pass2_log.get(i["key"], False)]
        if failed_items:
            print(f"[ERROR] {len(failed_items)} chunk(s) remain incomplete after Pass 2 retries. Examples: {failed_items[:10]}")
            raise RuntimeError(
                f"Pass 2 left {len(failed_items)} chunk(s) incomplete; rerun with --resume after checking server health."
            )

        print(f"\nPass 2 refinement successfully finished for domain: {self.domain}! All {len(items)} chunks are complete.")

# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def main():
    """Parse the command line and run Pass 2 refinement."""
    parser = build_parser(
        stage=STAGE,
        description="Second-Pass Concept Extraction & Patching Pipeline",
        include_resume=True)
    args = parser.parse_args()
    pipeline = WikiSecondPassPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()