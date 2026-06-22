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


def expand_query(query: str) -> str:
    words = query.lower().split()
    expanded = set(words)
    for w in words:
        clean = w.rstrip("?.,!;:'\"s")
        if clean in DOMAIN_VOCAB:
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
) -> list[Memory]:
    # Embed query (with domain vocab expansion as safety net)
    query_embedding = embed(expand_query(query))

    # ── Path B: Direct vector search ──────────────────────────────────────────
    vec_ids = set(ann_search(store, query_embedding, k=k * 2))

    # ── Path A: Keyword SQL JOIN ──────────────────────────────────────────────
    # Collect keywords from vector-matched memories
    related_keywords = store.get_related_keywords(list(vec_ids), limit=10) if vec_ids else []

    # SQL JOIN: find all memories that share these keywords
    kw_ids = set(store.search_by_keywords(related_keywords, limit=k * 2))

    # ── Combine both paths ────────────────────────────────────────────────────
    combined_ids = vec_ids | kw_ids
    if tag_filter:
        combined_ids = {mid for mid in combined_ids
                        if (mem := store.get(mid)) and mem.tag == tag_filter}

    if not combined_ids:
        # Fallback: brute force all memories
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

    # Rank by cosine similarity
    scored = []
    for mem in all_mems:
        if mem.embedding:
            a, b = np.array(query_embedding), np.array(mem.embedding)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            scored.append((sim, mem))

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
