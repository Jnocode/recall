"""recall. — Development benchmarks

Pure vector search baseline (moved from production code per audit).
Not part of the public API. For ablation studies and performance regression.
"""

import numpy as np
from recall.store import Memory, SQLiteStore
from recall.embed import embed

TOP_K = 10


def pure_vector_search(query: str, store: SQLiteStore, k: int = TOP_K) -> list[Memory]:
    """Pure vector search, no keyword expansion. For baseline comparison."""
    query_embedding = embed(query)
    if query_embedding is None:
        return []
    all_mems = store.get_all()
    scored = []
    for mem in all_mems:
        if mem.embedding:
            a, b = np.array(query_embedding), np.array(mem.embedding)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            scored.append((sim, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:k]]
