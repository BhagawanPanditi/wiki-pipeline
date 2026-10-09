#!/usr/bin/env python3
"""
Refinement Pass 2: Second-Pass Knowledge Graph Concept Extraction & Patching
- Resolves omissions, adds missing concepts, appends missing facts into descriptions.
- Supports adding/removing outbound relations, adding aliases, and adding brand new concepts.
"""

import asyncio
import functools
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.checkpoint import load_bool_log, save_log_async, save_log_sync
from utils.chunks import load_cached_chunks
from utils.cli import build_parser, resolve_settings
from utils.attempts import AttemptHistory
from utils.event_log import EventLog
from utils.layout import log_path
from utils.errors import StageBudget
from utils.json_repair import clean_json_string as _extract_json
from utils.llm_pool import pool_for_stage

# Pass 2 historically also accepted an unfenced outermost JSON object.
clean_json_string = functools.partial(_extract_json, allow_bare_object=True)

STAGE = "refine-concepts"

# ── Pydantic Data Schemas for Second Pass ─────────────────────────────────────

class OutboundRelation(BaseModel):
    target_concept: str = Field(description="Exact canonical title of the target concept.")
    relation: str = Field(description="1 sentence describing how this concept affects/relates outward to Target Concept.")

class ConceptUpdate(BaseModel):
    title: str = Field(description="Exact title of the EXISTING concept being updated.")
    additional_aliases: List[str] = Field(
        default_factory=list, 
        description="New acronyms or alternative canonical names to add."
    )
    additional_description_sentences: List[str] = Field(
        default_factory=list,
        description="Dense factual sentences containing newly extracted facts/metrics to append to the existing description paragraph."
    )
    additional_entities: List[str] = Field(
        default_factory=list, 
        description="New minor named terms, tool versions, biological codes, or dataset identifiers to add to entities."
    )
    additional_outbound_relations: List[OutboundRelation] = Field(
        default_factory=list, 
        description="New outbound relations to add."
    )
    relations_to_remove: List[str] = Field(
        default_factory=list,
        description="List of target_concept titles of outbound relations that are inaccurate, obsolete, or should be removed."
    )

class Concept(BaseModel):
    title: str = Field(description="Exact canonical title (1-6 words, singular).")
    aliases: List[str] = Field(default_factory=list, description="Explicit acronyms or alternative names.")
    description: str = Field(description="Single dense factual paragraph with all raw metrics, parameters, and details.")
    entities: List[str] = Field(default_factory=list, description="Flat list of minor named entities, tools, biological terms, dates, parameters.")
    outbound_relations: List[OutboundRelation] = Field(default_factory=list)

class Pass2ExtractionSchema(BaseModel):
    new_concepts: List[Concept] = Field(
        default_factory=list, 
        description="Brand new substantive concepts missed entirely in Pass 1."
    )
    concept_updates: List[ConceptUpdate] = Field(
        default_factory=list, 
        description="Updates and factual patches for concepts extracted in Pass 1."
    )

# ── Helper Utilities ─────────────────────────────────────────────────────────

# ── Second-Pass System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT_PASS2 = """\
You are an expert technical knowledge auditor performing a SECOND-PASS EXTRACTION & REFINEMENT on an LLM Knowledge Graph.

YOUR MISSION:
Perform an exhaustive comparison between the RAW CHUNK text and the EXISTING EXTRACTED CONCEPTS (from Pass 1).
Identify any missing concepts, omitted technical metrics, parameters, or relationships, and generate patch updates.

CRITICAL RULES:
1. NO CONCEPT LIMITS: Do NOT artificially restrict concept counts. Extract EVERY substantive concept or sub-topic that has factual explanations in the chunk.
2. FACTS ARE SENTENCES, NOT ENTITIES:
   - Entities (`entities`) are ONLY flat named terms (e.g., "CRISPR-Cas9", "ResNet-50", "GSE12345").
   - ALL technical facts, numbers, bounds, statistical parameters, and procedural details MUST be written as complete sentences inside concept descriptions (`description` or `additional_description_sentences`). Do NOT treat entities as a place to store facts!
3. AUDIT FEEDBACK HINTS:
   - If AUDIT FEEDBACK is provided below, prioritize extracting all missing concepts and missing facts listed in that audit.
4. OUTBOUND RELATIONS REFINEMENT:
   - If an existing relation between Concept A and Concept B is imprecise or redundant, specify its `target_concept` in `relations_to_remove` so it can be pruned.
   - Add new high-value outbound relations in `additional_outbound_relations`.

OUTPUT FORMAT:
Return ONLY valid JSON matching this exact structure:
{
  "new_concepts": [
    {
      "title": "Exact Title of Missed Concept",
      "aliases": ["Acronym or Alt Name"],
      "description": "Dense factual paragraph with all quantitative metrics and findings from this chunk.",
      "entities": ["Tool Name", "Dataset Code"],
      "outbound_relations": [
        {
          "target_concept": "Target Title",
          "relation": "1 sentence describing relationship."
        }
      ]
    }
  ],
  "concept_updates": [
    {
      "title": "Exact Title of Existing Concept to Update",
      "additional_aliases": ["New Acronym"],
      "additional_description_sentences": [
        "First sentence containing specific metric or finding missed in Pass 1.",
        "Second sentence describing experimental parameter or condition."
      ],
      "additional_entities": ["New Named Entity"],
      "additional_outbound_relations": [
        {
          "target_concept": "Target Title",
          "relation": "1 sentence describing newly identified outbound relation."
        }
      ],
      "relations_to_remove": [
        "Title of Target Concept whose relation should be deleted"
      ]
    }
  ]
}
"""

# ── Pipeline Implementation ──────────────────────────────────────────────────

class WikiSecondPassPipeline:
    def __init__(self, args):
        """Prepare Pass 2 refinement. Pass 1 output must already exist.

        Args:
            args: Parsed CLI namespace.

        Raises:
            FileNotFoundError: If the staging folder from Pass 1 is missing."""
        self.args = args
        self.domain = args.domain
        self.base_wiki_dir = Path(args.wiki_base_dir) / self.domain
        self.staging_dir = self.base_wiki_dir / "staging"
        self.cache_dir = self.base_wiki_dir / "cache"
        self.pass2_log_file = self.base_wiki_dir / "pass2_log.json"

        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory missing at {self.staging_dir}. Run Pass 1 extraction first.")

        self.pass2_log = load_bool_log(
            self.pass2_log_file,
            corrupt_message="Pass 2 log file corrupted. Re-initializing.")

    def cleanup_incomplete_states(self):
        """Purges pass2_log entries marked False and cleans up incomplete staging files upon resume."""
        print("Sanitizing incomplete Pass 2 extraction states...")
        purged = 0
        for key, completed in list(self.pass2_log.items()):
            if not completed:
                doc_id, chunk_id = key.split("::")
                chunk_dir = self.staging_dir / doc_id / chunk_id

                pass1_backup = chunk_dir / "concepts_pass1_backup.json"
                concepts_file = chunk_dir / "concepts.json"
                pass2_file = chunk_dir / "pass2_refinement.json"

                # If backup exists, always restore it to fix any corrupted merges
                if pass1_backup.exists():
                    shutil.copy(pass1_backup, concepts_file)

                # Delete the partial LLM output if it exists
                if pass2_file.exists():
                    try:
                        pass2_file.unlink()
                    except Exception:
                        pass

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

            # Check if audit verification log exists for extra guidance
            verification_file = chunk_dir / "verification.json"
            audit_feedback = None
            if not getattr(self.args, "ignore_audit_feedback", False) and verification_file.exists():
                try:
                    with open(verification_file, "r", encoding="utf-8") as f:
                        vf_data = json.load(f)
                        if not vf_data.get("is_sufficient", True):
                            audit_feedback = {
                                "missing_concepts": vf_data.get("missing_concepts", []),
                                "missing_facts": vf_data.get("missing_facts", []),
                                "critique": vf_data.get("critique", "")
                            }
                except Exception:
                    pass

            items.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "key": key,
                "current_chunk": chunk_data["current_chunk"],
                "context_before": chunk_data.get("context_before", ""),
                "context_after": chunk_data.get("context_after", ""),
                "pass1_concepts": pass1_concepts,
                "audit_feedback": audit_feedback,
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
        """Queries the LLM for second-pass deltas and missing facts.

        Raises on any unusable response so ServerPool can classify it and apply
        the retry ladder. Unlike Pass 1, token-limit truncation is raised as a
        plain ValueError here, so it stays retryable on another server - that is
        the behaviour this stage has always had. Manual client rotation per
        attempt is gone because the pool re-dispatches to a healthy server.
        """
        key = item["key"]
        audit_str = ""
        if item["audit_feedback"]:
            audit_str = f"""
AUDIT FEEDBACK (PREVIOUS VERIFICATION FOUND OMISSIONS):
- Missing Concepts: {json.dumps(item['audit_feedback']['missing_concepts'])}
- Missing Facts/Metrics: {json.dumps(item['audit_feedback']['missing_facts'])}
- Auditor Critique: {item['audit_feedback']['critique']}
"""

        user_prompt = f"""\
CONTEXT BEFORE:
{item['context_before']}

CURRENT CHUNK:
{item['current_chunk']}

CONTEXT AFTER:
{item['context_after']}

EXISTING EXTRACTED CONCEPTS (PASS 1):
{json.dumps(item['pass1_concepts'], indent=2)}
{audit_str}
"""
        raw_text = None
        finish_reason = None
        try:
            response = await client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_PASS2},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=self.settings.max_tokens,
                response_format={"type": "json_object"}
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            raw_text = choice.message.content

            if finish_reason == "length":
                raise ValueError(f"Response truncated: Hit max_tokens ({self.settings.max_tokens}).")

            cleaned_json = clean_json_string(raw_text)
            return Pass2ExtractionSchema.model_validate_json(cleaned_json)

        except Exception as exc:
            self.attempts.record(key, server=str(client.base_url), finish_reason=finish_reason,
                                     raw_text=raw_text, error_type=type(exc).__name__,
                                     error_message=str(exc))
            raise

    async def persist_refinement(self, item: Dict[str, Any],
                                 delta: Pass2ExtractionSchema) -> None:
        """SUCCESS: stream the delta, back up Pass 1, then merge into concepts.json."""
        key = item["key"]
        chunk_dir = item["chunk_dir"]

        # 1. Save Pass 2 raw delta refinement JSON
        pass2_file = chunk_dir / "pass2_refinement.json"
        await asyncio.to_thread(
            pass2_file.write_text, delta.model_dump_json(indent=2), encoding="utf-8"
        )

        # 2. Backup Pass 1 concepts.json if backup does not exist
        pass1_backup = chunk_dir / "concepts_pass1_backup.json"
        concepts_file = chunk_dir / "concepts.json"
        if not pass1_backup.exists() and concepts_file.exists():
            await asyncio.to_thread(shutil.copy, concepts_file, pass1_backup)

        # 3. Apply updates & merged output to concepts.json
        merged_data = self.apply_merges(item["pass1_concepts"], delta)
        self.decisions.record(
            "concept_refined", chunk=key,
            concepts_updated=[{"title": u.title, "aliases_added": list(u.additional_aliases),
                               "sentences_added": len(u.additional_description_sentences)}
                              for u in delta.concept_updates])
        merged_payload = json.dumps(merged_data, indent=2)
        await asyncio.to_thread(concepts_file.write_text, merged_payload, encoding="utf-8")

        self.pass2_log[key] = True
        self.flush_count += 1
        if self.flush_count >= 50:
            await save_log_async(self.pass2_log_file, self.pass2_log.copy())
            self.flush_count = 0

    async def write_failure_debug(self, item: Dict[str, Any],
                                  failure: Dict[str, Any]) -> None:
        """FAILURE: save the Pass 2 debug log with every attempt, untruncated."""
        key = item["key"]
        debug_data = {
            "key": key,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempts": self.attempts.get(key),
            "final_error": failure.get("error"),
            "final_classification": failure.get("classification"),
            "last_server": failure.get("server"),
        }
        payload = json.dumps(debug_data, indent=2)
        await asyncio.to_thread((item["chunk_dir"] / "pass2_debug.json").write_text,
                                payload, encoding="utf-8")

    def apply_merges(
        self, pass1_data: Dict[str, Any], pass2_delta: Pass2ExtractionSchema
    ) -> Dict[str, Any]:
        """Merges Pass 2 updates cleanly into existing Pass 1 concepts JSON."""
        concepts_list = pass1_data.get("concepts", [])
        concept_map = {c["title"].strip().lower(): c for c in concepts_list}

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
                if update.additional_description_sentences:
                    new_text = " ".join(update.additional_description_sentences)
                    c["description"] = f"{c.get('description', '').strip()} {new_text.strip()}".strip()

                # Append entities (deduplicated)
                existing_entities = set(c.get("entities", []))
                for ent in update.additional_entities:
                    if ent not in existing_entities:
                        c.setdefault("entities", []).append(ent)
                        existing_entities.add(ent)

                # Prune specified relations
                if update.relations_to_remove:
                    remove_targets = {r.strip().lower() for r in update.relations_to_remove}
                    c["outbound_relations"] = [
                        rel for rel in c.get("outbound_relations", [])
                        if rel.get("target_concept", "").strip().lower() not in remove_targets
                    ]

                # Append new outbound relations
                existing_rel_targets = {
                    rel.get("target_concept", "").strip().lower() 
                    for rel in c.get("outbound_relations", [])
                }
                for new_rel in update.additional_outbound_relations:
                    rel_dict = new_rel.model_dump()
                    if rel_dict["target_concept"].strip().lower() not in existing_rel_targets:
                        c.setdefault("outbound_relations", []).append(rel_dict)
                        existing_rel_targets.add(rel_dict["target_concept"].strip().lower())

        # 2. Add brand-new concepts
        for new_c in pass2_delta.new_concepts:
            new_key = new_c.title.strip().lower()
            if new_key not in concept_map:
                c_dict = new_c.model_dump()
                concepts_list.append(c_dict)
                concept_map[new_key] = c_dict

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
        self.decisions = EventLog(log_path(self.base_wiki_dir, STAGE, "decisions"), label=STAGE)
        self.debug = EventLog(log_path(self.base_wiki_dir, STAGE, "debug"),
                                 label=STAGE)
        self.budget = StageBudget(self.settings.budget_minutes)
        self.attempts = AttemptHistory()
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
              f"{settings.budget_minutes:g} min | Debug log: {self.debug.path}")

        if not pending:
            print("All chunks are already refined in Pass 2!")
            return

        # Mark every pending chunk in-progress, as the old worker did under a lock,
        # so --resume can purge anything left half-written.
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
                    debug=self.debug,
                    attempt_timeout=settings.profile.attempt_timeout,
                    on_result=self.persist_refinement,
                )
        finally:
            await save_log_async(self.pass2_log_file, self.pass2_log)
            self.debug.close()

        for failure in result.failures:
            await self.write_failure_debug(pending[failure["index"]], failure)

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
    """Parse the command line and run Pass 2 refinement.

    Returns:
        None."""
    parser = build_parser(
        stage=STAGE,
        description="Second-Pass Concept Extraction & Patching Pipeline",
        include_resume=True)
    parser.add_argument("--ignore_audit_feedback", action="store_true",
                        help="Do not read existing Pass 1 verification files for refinement hints.")
    args = parser.parse_args()
    pipeline = WikiSecondPassPipeline(args)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()