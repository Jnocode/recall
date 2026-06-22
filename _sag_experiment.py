"""SAG-inspired two-path retrieval experiment."""
import sys, os, json, urllib.request, numpy as np, re, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
from embed import embed

DEEPSEEK_KEY = None
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        if "DEEPSEEK" in line and "=" in line:
            DEEPSEEK_KEY = line.strip().split("=", 1)[1].strip().strip('"')
            break

def extract_entities(text):
    prompt = f"Extract technologies, tools, concepts from as JSON array.\nText: \"{text}\"\nReturn: [\"entity1\", \"entity2\"]"
    try:
        body = json.dumps({"model":"deepseek-v4-flash","messages":[{"role":"user","content":prompt}],"max_tokens":200,"temperature":0.0}).encode()
        resp = urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/v1/chat/completions",data=body,headers={"Authorization":f"Bearer {DEEPSEEK_KEY}","Content-Type":"application/json"}),timeout=15)
        content = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n",1)[1].rsplit("\n",1)[0]
        return json.loads(content)
    except:
        words = re.findall(r'[A-Z][a-zA-Z0-9+#_-]{2,}|[a-z]+[-_][a-zA-Z0-9]+|[a-zA-Z]{4,}', text)
        return list(set(w.lower() for w in words if w.lower() not in {"the","and","for","with","from","that","this","have","been","user","prefers","must","over","all"}))

# Build DB
db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_p0.db")
if os.path.exists(db):
    os.remove(db)

conn = sqlite3.connect(db)
conn.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(conn)

conn.execute("CREATE TABLE events (id TEXT PRIMARY KEY, event_text TEXT NOT NULL, embedding BLOB, timestamp TEXT, session_id TEXT, tag TEXT)")
conn.execute("CREATE TABLE event_entities (event_id TEXT, entity TEXT, PRIMARY KEY (event_id, entity))")
conn.execute("CREATE VIRTUAL TABLE vec_events USING vec0(id TEXT PRIMARY KEY, embedding float[768] distance_metric=cosine)")
conn.commit()

seed = [
    ("User prefers docker-compose over Dockerfile for local dev", 2),
    ("User uses PostgreSQL with asyncpg for production", 5),
    ("Production database connection pool exhausted at 2AM", 10),
    ("FastAPI project structure uses routers/services/models", 15),
    ("EC2 migration to ECS Fargate for cost savings", 30),
    ("User requires type hints on all PRs before merge", 8),
    ("Team decided to use Prometheus for metrics collection", 35),
    ("API versioning strategy: URL path versioning v1, v2", 85),
    ("Redis cache invalidation bug on data update", 48),
    ("Security policy: all secrets must use Vault", 78),
]

print("Seeding...")
for i, (content, days_ago) in enumerate(seed):
    eid = f"evt_{i:02d}"
    emb = embed(content)
    ts = datetime.utcnow() - timedelta(days=days_ago)
    entities = extract_entities(content)
    conn.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?)", (eid, content, json.dumps(emb), ts.isoformat(), "seed", "episodic"))
    for ent in set(entities):
        conn.execute("INSERT OR REPLACE INTO event_entities VALUES (?,?)", (eid, ent))
    conn.execute("INSERT OR REPLACE INTO vec_events VALUES (?,?)", (eid, np.array(emb, dtype=np.float32).tobytes()))
conn.commit()

ec = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
eec = conn.execute("SELECT COUNT(*) FROM event_entities").fetchone()[0]
print(f"Events: {ec}  Entity links: {eec}")

# Test
queries = [
    ("How should I deploy the app?", ["docker-compose", "docker"]),
    ("What database should I use?", ["postgresql", "asyncpg"]),
    ("What are the code review rules?", ["type", "hint"]),
    ("What platform for hosting?", ["ecs", "fargate"]),
]

print(f"\n{'='*55}")
print("  SAG Two-Path Retrieval")
print(f"{'='*55}")

for q, expected in queries:
    q_ents = extract_entities(q)
    placeholders = ",".join("?" for _ in q_ents)
    path_a = conn.execute(f"SELECT DISTINCT e.id, e.event_text FROM events e JOIN event_entities ee ON e.id=ee.event_id WHERE ee.entity IN ({placeholders})", q_ents).fetchall() if q_ents else []
    
    q_emb = embed(q)
    path_b = conn.execute("SELECT id FROM vec_events WHERE embedding MATCH ? ORDER BY distance LIMIT 10", (np.array(q_emb, dtype=np.float32).tobytes(),)).fetchall()
    
    correct = {f"evt_{i:02d}" for i,(c,_) in enumerate(seed) if any(e.lower() in c.lower() for e in expected)}
    a_hit = bool({r[0] for r in path_a} & correct)
    b_hit = bool({r[0] for r in path_b} & correct)
    
    print(f"\n  Q: {q[:35]}")
    print(f"  Entities: {q_ents}")
    print(f"  Path A (SQL): {len(path_a)} candidates {'✅' if a_hit else '❌'}")
    print(f"  Path B (Vec): {len(path_b)} candidates {'✅' if b_hit else '❌'}")

conn.close()
print(f"\n✅ Done")
