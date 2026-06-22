"""Keyword vector match experiment."""
import sys, os, re, json, sqlite3, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
from embed import embed

def extract_keywords(text):
    keywords = set()
    keywords.update(re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+[-_][a-zA-Z0-9][-a-zA-Z0-9]*\b', text))
    keywords.update(re.findall(r'\b[A-Z][a-zA-Z0-9+#_-]{2,}\b', text))
    keywords.update(re.findall(r'\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b', text))
    stop = {"the","with","from","that","this","have","been","user","prefers","must","over","all","for","and","not"}
    for w in re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()):
        if w not in stop:
            keywords.add(w)
    return list(keywords)

db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_p0.db")
if os.path.exists(db):
    os.remove(db)

import sqlite_vec
conn = sqlite3.connect(db)
conn.enable_load_extension(True)
sqlite_vec.load(conn)

conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, embedding BLOB, timestamp TEXT, session_id TEXT, tag TEXT)")
conn.execute("CREATE TABLE keywords (keyword TEXT, memory_id TEXT, PRIMARY KEY (keyword, memory_id))")
conn.execute("CREATE INDEX idx_kw ON keywords(keyword)")
conn.execute("CREATE VIRTUAL TABLE vec_memories USING vec0(id TEXT PRIMARY KEY, embedding float[768] distance_metric=cosine)")
conn.commit()

seed = [
    ("User prefers docker-compose over Dockerfile for local dev", 2),
    ("User uses PostgreSQL with asyncpg for production", 5),
    ("Production database pool exhausted at 2AM", 10),
    ("FastAPI project structure uses routers/services/models", 15),
    ("EC2 migration to ECS Fargate for cost savings", 30),
    ("User requires type hints on all PRs before merge", 8),
]

print("Seeding with keyword index...")
for i, (content, days_ago) in enumerate(seed):
    mid = f"mem_{i:02d}"
    emb = embed(content)
    ts = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
    conn.execute("INSERT INTO memories VALUES (?,?,?,?,?,?)", (mid, content, json.dumps(emb), ts, "seed", "episodic"))
    conn.execute("INSERT INTO vec_memories VALUES (?,?)", (mid, np.array(emb, dtype=np.float32).tobytes()))
    for kw in extract_keywords(content):
        conn.execute("INSERT INTO keywords VALUES (?,?)", (kw.lower(), mid))
conn.commit()
print(f"Memories: {len(seed)}  Keywords: {conn.execute('SELECT COUNT(*) FROM keywords').fetchone()[0]}")
print("Top keywords:")
for kw, cnt in conn.execute("SELECT keyword, COUNT(*) as cnt FROM keywords GROUP BY keyword ORDER BY cnt DESC LIMIT 15").fetchall():
    print(f"  {kw:20s} ({cnt} mems)")

queries = [
    "How should I deploy the app?",
    "What database should I use?",
    "Where should I put my API endpoints?",
    "What are the code review rules?",
    "What platform for hosting our services?",
]

print(f"\n{'='*55}")
print("  Keyword Vector Match + Two-Path Retrieval")
print(f"{'='*55}")

for q in queries:
    q_emb = embed(q)
    q_vec = np.array(q_emb, dtype=np.float32).tobytes()
    
    # Path B: Direct vector search
    vec_hits = conn.execute("SELECT id FROM vec_memories WHERE embedding MATCH ? ORDER BY distance LIMIT 5", (q_vec,)).fetchall()
    vec_ids = {r[0] for r in vec_hits}
    
    # Collect keywords from vector-matched memories
    related_kws = set()
    if vec_ids:
        ph = ",".join("?" for _ in vec_ids)
        for r in conn.execute(f"SELECT DISTINCT keyword FROM keywords WHERE memory_id IN ({ph})", list(vec_ids)).fetchall():
            related_kws.add(r[0])
    
    # Path A: SQL JOIN using those keywords
    expanded_ids = set()
    if related_kws:
        ph2 = ",".join("?" for _ in related_kws)
        for r in conn.execute(f"SELECT DISTINCT memory_id FROM keywords WHERE keyword IN ({ph2})", list(related_kws)).fetchall():
            expanded_ids.add(r[0])
    
    combined = vec_ids | expanded_ids
    print(f"\n  Q: {q[:35]}")
    print(f"  Related keywords: {list(related_kws)[:8]}")
    print(f"  Path B (vector): {len(vec_ids)}  Path A (SQL): {len(expanded_ids)}  Combined: {len(combined)}")

conn.close()
print(f"\nDone")
