#!/usr/bin/env python3
"""
Compute Concept Embeddings and Semantic Cross-Links (qwen-embed)
----------------------------------------------------------------
Can run under an sbatch job with conda env: qwen-embed
1. Loads concepts from `global_concepts_refined.json`.
2. Computes embeddings for each concept using Qwen embedding model.
3. Finds nearest neighbors for isolated components/islands.
4. Adds cross-links to bridge islands into the giant component.
5. Saves updated `global_concepts_refined.json` and embeddings cache.
"""

import json
import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Set, Any
import numpy as np
from tqdm import tqdm
from pipeline_config import load_embedding_model
from atomic_io import atomic_write_json

DEFAULT_EMBEDDING_MODEL = load_embedding_model()


def embedding_signature(model_name: str, texts: List[str]) -> str:
    """Fingerprint the exact text/model inputs so cached vectors cannot go stale."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def run_embedding_island_bridging(
    domain_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3,
    sim_threshold: float = 0.78
):
    refined_file = domain_dir / "global_concepts_refined.json"
    if not refined_file.exists():
        raise FileNotFoundError(f"Missing refined graph file: {refined_file}")

    print(f"Loading concepts from {refined_file}...")
    with open(refined_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        concepts = data.get("concepts", [])

    print(f"Loaded {len(concepts)} concepts. Preparing texts for embedding...")

    # Form text representation for embedding: Title + Brief + Entities
    texts = []
    for c in concepts:
        title = c["title"]
        brief = c.get("brief", "")
        entities_str = ", ".join(c.get("entities", [])[:5])
        rep = f"Concept: {title}\nSummary: {brief}\nEntities: {entities_str}"
        texts.append(rep)

    if not texts:
        raise RuntimeError("Cannot embed an empty concept graph.")

    emb_cache_file = domain_dir / "concept_embeddings.npy"
    emb_manifest_file = domain_dir / "concept_embeddings.meta.json"
    input_signature = embedding_signature(model_name, texts)
    embeddings = None

    # Reuse only a complete cache matching the current model and exact concept texts.
    if emb_cache_file.exists() and emb_manifest_file.exists():
        try:
            with emb_manifest_file.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            candidate = np.load(emb_cache_file, allow_pickle=False)
            if (
                manifest.get("signature") == input_signature
                and candidate.ndim == 2
                and candidate.shape[0] == len(texts)
                and np.isfinite(candidate).all()
            ):
                embeddings = candidate
                print(f"Reusing validated embedding cache: {emb_cache_file}")
        except Exception as exc:
            print(f"Ignoring incomplete/invalid embedding cache ({type(exc).__name__}: {exc}).")

    if embeddings is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model_name}...")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        except Exception as e:
            print(f"SentenceTransformer not available or failed: {e}")
            print("Falling back to HuggingFace transformers pipeline...")
            import torch
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
            model.eval()

            embed_list = []
            batch_size = 16
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                batch_texts = texts[i:i + batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model(**encoded)
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    embed_list.append(emb.cpu().numpy())
            embeddings = np.vstack(embed_list)

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
            raise RuntimeError(f"Embedding output has invalid shape or values: {embeddings.shape}")

        # Atomic cache writes keep a preempted/failed run from leaving a usable-looking partial file.
        temp_embedding_file = emb_cache_file.with_name(emb_cache_file.name + ".tmp")
        with temp_embedding_file.open("wb") as f:
            np.save(f, embeddings)
        os.replace(temp_embedding_file, emb_cache_file)

        temp_manifest_file = emb_manifest_file.with_suffix(".json.tmp")
        with temp_manifest_file.open("w", encoding="utf-8") as f:
            json.dump({"model": model_name, "signature": input_signature, "count": len(texts)}, f, indent=2)
        os.replace(temp_manifest_file, emb_manifest_file)
        print(f"Saved validated embeddings to {emb_cache_file}")

    # Compute pairwise similarities for isolated or low-degree concepts
    print("Computing cosine similarities to bridge graph islands...")
    title_to_idx = {c["title"].lower().strip(): idx for idx, c in enumerate(concepts)}

    added_edges = 0
    for i, c in enumerate(tqdm(concepts, desc="Bridging low-connectivity nodes")):
        # If node has <= 1 outbound relation, link to its top nearest neighbors
        if len(c.get("outbound_relations", [])) <= 1:
            query_vec = embeddings[i]
            sims = np.dot(embeddings, query_vec)
            sims[i] = -1.0  # exclude self

            top_indices = np.argsort(sims)[::-1][:top_k]
            existing_targets = {r.get("target_concept", "").lower().strip() for r in c.get("outbound_relations", [])}

            for neighbor_idx in top_indices:
                score = float(sims[neighbor_idx])
                if score >= sim_threshold:
                    target_concept = concepts[neighbor_idx]
                    tgt_title = target_concept["title"]
                    if tgt_title.lower().strip() not in existing_targets:
                        existing_targets.add(tgt_title.lower().strip())
                        c.setdefault("outbound_relations", []).append({
                            "target_concept": tgt_title,
                            "relation": f"High semantic affinity and domain relevance (similarity score {score:.2f}).",
                            "source_chunks": []
                        })
                        added_edges += 1

    print(f"Added {added_edges} semantic similarity cross-links!")

    print(f"Saving updated graph to {refined_file}...")
    atomic_write_json(refined_file, {"concepts": concepts})

    print("Embedding bridging complete!")


def main():
    parser = argparse.ArgumentParser(description="Embed concepts and bridge islands")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--wiki_base_dir", type=str, default="/home/compiling-ganesh/24m0829/long_document/wiki")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--sim_threshold", type=float, default=0.78)

    args = parser.parse_args()
    domain_dir = Path(args.wiki_base_dir) / args.domain
    run_embedding_island_bridging(domain_dir, model_name=args.model, sim_threshold=args.sim_threshold)


if __name__ == "__main__":
    main()
