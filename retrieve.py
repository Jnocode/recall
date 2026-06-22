# recall. — Retrieval Layer
# Two-path retrieval: keyword SQL JOIN + pure vector search
# Feynman-approved: no LLM, no hybrid scoring

import re
import numpy as np
from typing import Optional

from store import Memory, SQLiteStore, extract_keywords as extract_entities
from embed import embed

TOP_K = 10

# ─── Domain vocabulary (safety net, kept per Feynman) ─────────────────────────
DOMAIN_VOCAB = {
    "deploy": ["docker", "docker-compose", "container", "deployment"],
    "deployment": ["docker", "docker-compose", "deploy"],
    "docker": ["docker", "docker-compose", "container", "deployment"],
    "docker-compose": ["docker-compose", "docker", "deploy", "deployment"],
    "hosting": ["ec2", "ecs", "fargate", "cloud", "infrastructure"],
    "ec2": ["ec2", "ecs", "fargate", "migration", "infrastructure"],
    "ecs": ["ecs", "fargate", "ec2", "migration"],
    "fargate": ["fargate", "ecs", "ec2", "migration"],
    "migrate": ["migration", "ec2", "ecs", "fargate"],
    "migration": ["migrate", "ec2", "ecs", "fargate"],
    "infrastructure": ["ec2", "ecs", "fargate", "hosting"],
    "platform": ["ecs", "fargate", "hosting", "infrastructure"],
    "database": ["postgresql", "postgres", "sql", "asyncpg"],
    "postgresql": ["postgresql", "postgres", "database", "sql", "asyncpg"],
    "code": ["type", "hint", "pr", "review", "quality"],
    "review": ["review", "pr", "code", "quality", "type"],
    "type": ["type", "hint", "annotation"],
    "pr": ["pr", "pull", "request", "review", "merge"],
    "api": ["api", "router", "service", "model", "schema", "endpoint", "fastapi"],
    "endpoint": ["endpoint", "api", "router", "fastapi"],
    "fastapi": ["fastapi", "api", "router", "service", "model"],
}


def expand_query(query: str, max_terms: int = 10) -> str:
    words = query.lower().split()
    expanded = set(words)
    for w in words:
        clean = w.rstrip("?.,!;:'\"s")
        if clean in DOMAIN_VOCAB and len(expanded) < max_terms:
            expanded.update(DOMAIN_VOCAB[clean])
    return " ".join(expanded)


# ─── ANN ──────────────────────────────────────────────────────────────────────

def ann_search(store: SQLiteStore, query_embedding: list[float], k: int = 20) -> list[str]:
    if not store.vec_available or not store.count():
        return []
    import sqlite3, sqlite_vec
    conn = sqlite3.connect(store.db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    vec_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    rows = conn.execute(
        "SELECT id FROM vec_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (vec_bytes, k)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ─── Two-path retrieval ───────────────────────────────────────────────────────

def retrieve_relevant(
    query: str,
    store: SQLiteStore,
    k: int = TOP_K,
    tag_filter: Optional[str] = None,
    weights: Optional[dict] = None,
    hops: int = 2,
) -> list[Memory]:
    if weights is None:
        weights = {"path_a": 0.4, "path_b": 0.6}

    query_embedding = embed(expand_query(query, max_terms=20))

    # Path B: Direct vector search
    vec_ids = set(ann_search(store, query_embedding, k=k * 3))

    # Path A: Multi-hop keyword SQL JOIN (SAG-style)
    # Step 1: Extract keywords from query directly
    query_keywords = list(set(expand_query(query, max_terms=20).split()[:10]))
    query_kw_ids = set(store.search_by_keywords(query_keywords, limit=k * 3))

    # Step 2: Also get keywords from vector-matched memories
    seed_ids = list(vec_ids | query_kw_ids)
    kw_ids = set(seed_ids)

    for hop in range(hops):
        related_keywords = store.get_related_keywords(seed_ids, limit=15) if seed_ids else []
        new_ids = set(store.search_by_keywords(related_keywords, limit=k * 3))
        added = new_ids - kw_ids
        kw_ids.update(new_ids)
        seed_ids = list(added)
        if not added or hop >= hops - 1:
            break

    # Weighted fusion
    combined_ids = {}
    for mid in vec_ids:
        combined_ids[mid] = weights["path_b"]
    for mid in kw_ids:
        current = combined_ids.get(mid, 0.0)
        combined_ids[mid] = max(current, weights["path_a"])
    for mid in vec_ids & kw_ids:
        combined_ids[mid] = weights["path_a"] + weights["path_b"]

    if tag_filter:
        combined_ids = {mid: w for mid, w in combined_ids.items()
                        if (mem := store.get(mid)) and mem.tag == tag_filter}

    if not combined_ids:
        all_mems = store.get_all()
        if tag_filter:
            all_mems = [m for m in all_mems if m.tag == tag_filter]
    else:
        all_mems = []
        for mid in combined_ids:
            mem = store.get(mid)
            if mem:
                all_mems.append(mem)

    if not all_mems:
        return []

    # Rank by cosine similarity × path weight
    scored = []
    for mem in all_mems:
        if mem.embedding:
            a, b = np.array(query_embedding), np.array(mem.embedding)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            weight = combined_ids.get(mem.id, 0.5)
            scored.append((sim * weight, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:k]]


def pure_vector_search(query: str, store: SQLiteStore, k: int = TOP_K) -> list[Memory]:
    """Pure vector search, no keyword expansion. For baseline comparison."""
    query_embedding = embed(query)
    all_mems = store.get_all()
    scored = []
    for mem in all_mems:
        if mem.embedding:
            a, b = np.array(query_embedding), np.array(mem.embedding)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            scored.append((sim, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:k]]
