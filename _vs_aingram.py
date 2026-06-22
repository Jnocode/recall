"""recall. vs AIngram — 對比測試"""
import sys, os, json, urllib.request, numpy as np, re, time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Embedding ────────────────────────────────────────────────────────────────
def nomic_embed(text):
    body = json.dumps({"model":"nomic-embed-text-v1.5","input":[text],"encoding_format":"float"}).encode()
    resp = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:1234/v1/embeddings",data=body,headers={"Content-Type":"application/json"}),timeout=15)
    return json.loads(resp.read())["data"][0]["embedding"]

# ─── Same 30 seed memories (from recall eval) ─────────────────────────────────
SEED = [
    "User asked to migrate CI from Jenkins to GitHub Actions",
    "Frontend team decided to use React 19 for the new dashboard",
    "Production incident: database connection pool exhaustion at 2AM",
    "Sprint planning: Q3 focus on reducing API latency under 100ms",
    "Architecture decision: use Kafka for event streaming instead of RabbitMQ",
    "User prefers SQLAlchemy 2.0 async over raw SQL for new projects",
    "Code review comment: all API endpoints must have rate limiting",
    "Docker image size reduced from 1.2GB to 380MB using multi-stage build",
    "Team agreed on trunk-based development with short-lived feature branches",
    "User complained about slow pytest suite, suggested splitting into parallel runs",
    "Security audit found: JWT tokens not being rotated, fix required",
    "Database migration: partition orders table by month",
    "User rejected PR because it lacked type hints",
    "Infrastructure decision: switch from EC2 to ECS Fargate",
    "Project requirement: support WebSocket connections",
    "User mentioned they prefer Tailwind CSS over styled-components",
    "Bug fix: Redis cache invalidation not triggering",
    "Technical debt: upgrade Python 3.9 to 3.12 across all services",
    "New hire onboarding doc: setup guide for local development",
    "Vendor evaluation: chose DigitalOcean over Linode",
    "User stated they hate YAML configuration files, prefer TOML",
    "Legacy system: mainframe batch job scheduler needs replacement",
    "Team retrospective: improve code review turnaround time",
    "API versioning strategy decided: URL path versioning v1, v2",
    "User requested Grafana dashboard for real-time monitoring",
    "Security policy: all secrets must use Vault, not env vars",
    "Load testing revealed: N+1 query problem in user listing",
    "Team decided to use Prometheus for metrics collection",
    "User rejected proposal to use MongoDB, prefers Postgres",
    "Deployment pipeline now supports rollback to previous version",
]

EVAL_Q = [
    ("Which deployment method does user prefer?", {0,1,11,20}),
    ("What database issues encountered?", {2,12,16,26}),
    ("What frontend technology does user prefer?", {9,15}),
    ("What CI/CD decisions?", {0,13}),
    ("What security requirements?", {11,25}),
    ("How should APIs be structured?", {10,23,4}),
    ("What performance problems?", {2,3,19,26}),
    ("What infrastructure changes?", {14,20,22}),
    ("What Python decisions?", {9,17,1,6}),
    ("What monitoring tools?", {24}),
    ("Config file format preference?", {21}),
    ("Docker decisions?", {0,8,14}),
    ("DB schema changes?", {12}),
    ("Code quality standards?", {6,13,22}),
    ("Development workflow?", {5,13}),
    ("Recent incidents?", {2}),
    ("Migration projects?", {0,14,20,17}),
    ("User tool preferences?", {1,4,15,21,9}),
    ("Architecture decisions?", {4,14,23,22}),
    ("Testing improvements?", {10}),
]

def recall_test():
    """Test recall. with two-path retrieval."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from store import Memory, SQLiteStore
    from retrieve import retrieve_relevant, extract_entities
    
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_recall.db")
    if os.path.exists(db): os.remove(db)
    store = SQLiteStore(db, vec_dim=768)
    
    for i, content in enumerate(SEED):
        mem = Memory(content=content, entities=extract_entities(content),
                     timestamp=datetime.utcnow()-timedelta(days=i*3),
                     session_id="eval", tag="episodic", embedding=nomic_embed(content))
        mem.id = f"seed_{i:02d}"
        store.add(mem)
    
    total_r = 0
    t0 = time.time()
    for q, truth in EVAL_Q:
        mems = retrieve_relevant(q, store, k=5)
        hits = {int(m.id.split("_")[1]) for m in mems if m.id.startswith("seed_")}
        r = len(hits & truth) / len(truth) if truth else 0
        total_r += r
    elapsed = time.time() - t0
    del store
    import gc; gc.collect()
    os.remove(db)
    return total_r / len(EVAL_Q), elapsed / len(EVAL_Q)

def aingram_test():
    """Test AIngram with its default retrieval."""
    from aingram import MemoryStore
    
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_aingram.db")
    if os.path.exists(db): os.remove(db)
    
    store = MemoryStore(db)
    
    for i, content in enumerate(SEED):
        store.remember(content)
    
    total_r = 0
    t0 = time.time()
    for q, truth in EVAL_Q:
        # AIngram recall returns entries with scores
        results = store.recall(q, limit=5)
        hits = set()
        for r in results:
            content = r.entry.content
            if isinstance(content, str) and content.startswith("{"):
                content = json.loads(content).get("text", content)
            elif isinstance(content, dict):
                content = content.get("text", str(content))
            try:
                idx = SEED.index(content)
                hits.add(idx)
            except ValueError:
                pass
        r = len(hits & truth) / len(truth) if truth else 0
        total_r += r
    elapsed = time.time() - t0
    store.close()
    del store
    import gc; gc.collect()
    os.remove(db)
    return total_r / len(EVAL_Q), elapsed / len(EVAL_Q)

print("="*60)
print("  recall. vs AIngram — 30 Memories × 20 Questions")
print("="*60)
print(f"\n  Seeding both with {len(SEED)} identical memories...")
print(f"  Query set: {len(EVAL_Q)} multi-hop questions")
print()

# Warmup
print("  Warming up Nomic embed...")
nomic_embed("warmup")
print()

r_r, t_r = recall_test()
print(f"  📍 recall. (two-path):")
print(f"     Recall@5: {r_r:.3f}   Avg latency: {t_r*1000:.0f}ms")

print()
a_r, t_a = aingram_test()
print(f"  📍 AIngram (FTS5+vector+graph):")
print(f"     Recall@5: {a_r:.3f}   Avg latency: {t_a*1000:.0f}ms")

print(f"\n{'='*60}")
if a_r > r_r:
    print(f"  🏆 AIngram wins by {(a_r-r_r)/r_r*100:.1f}%")
elif r_r > a_r:
    print(f"  🏆 recall. wins by {(r_r-a_r)/a_r*100:.1f}%")
else:
    print(f"  🤝 Tie")
print(f"{'='*60}")
