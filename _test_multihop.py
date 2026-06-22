"""Multi-hop recall test"""
import sys, os, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Memory, SQLiteStore
from retrieve import extract_entities, retrieve_relevant
from embed import embed
from datetime import datetime, timedelta

# 30 memories in 3 chains of 10, each chain connected by shared keywords
CHAINS = {
    "deploy": [
        "User prefers docker-compose for local dev",
        "Docker Compose manages multi-container apps",
        "Production uses Kubernetes cluster",
        "Kubernetes deployment uses Helm charts",
        "Helm charts managed in Git repo",
        "Git repository uses GitHub Actions for CI",
        "GitHub Actions deploys to ECS Fargate",
        "ECS Fargate runs Docker containers",
        "Docker images built with multi-stage",
        "Multi-stage builds reduce image size",
    ],
    "database": [
        "Database uses PostgreSQL for production",
        "PostgreSQL connected via asyncpg driver",
        "asyncpg requires connection pooling",
        "Connection pool size set to 25 connections",
        "Connection pool exhaustion caused incident",
        "Incident required database restart at 2AM",
        "Database restart caused cache invalidation",
        "Cache invalidation handled by Redis pub/sub",
        "Redis pub/sub notifies all services",
        "Services must reconnect after Redis failover",
    ],
    "monitor": [
        "Applications monitored via Prometheus",
        "Prometheus scrapes metrics every 15s",
        "Metrics visualized in Grafana dashboards",
        "Grafana alerts configured for p99 latency",
        "Latency under 100ms is target for Q3",
        "Q3 performance improvements are top priority",
        "Performance testing done with k6",
        "k6 scripts run in CI pipeline",
        "CI pipeline also runs linting and tests",
        "Linting uses Ruff replacing Flake8",
    ],
}

# Memory IDs: deploy_00..09, database_00..09, monitor_00..09
MEM_IDS = {}
for chain, mems in CHAINS.items():
    for i in range(len(mems)):
        MEM_IDS[(chain, i)] = f"m_{chain}_{i:02d}"

EVAL = [
    # 2-hop (keywords: docker-compose → kubernetes → helm)
    ("Container orchestration from local dev to production", {"deploy": {0,1,2,3}}),
    # 3-hop (docker → github-actions → ecs)
    ("CI/CD pipeline from commit to deployment", {"deploy": {5,6,7,8}}),
    # 2-hop (postgresql → asyncpg → pooling)
    ("Database connection architecture", {"database": {0,1,2,3}}),
    # 3-hop (pooling → incident → cache → redis)
    ("Database incident chain of events", {"database": {4,5,6,7}}),
    # 2-hop (prometheus → grafana → latency)
    ("Application monitoring setup", {"monitor": {0,1,2,3}}),
    # 3-hop (latency → q3 → k6 → ci)
    ("Performance improvement workflow", {"monitor": {4,5,6,7}}),
]

def truth_to_ids(truth):
    ids = set()
    for chain, indices in truth.items():
        for i in indices:
            ids.add(MEM_IDS[(chain, i)])
    return ids

db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mh.db")
if os.path.exists(db): os.remove(db)
store = SQLiteStore(db, vec_dim=768)

for chain, mems in CHAINS.items():
    for i, content in enumerate(mems):
        emb = embed(content)
        mem = Memory(content=content, entities=extract_entities(content),
                     timestamp=datetime.utcnow()-timedelta(days=i*3),
                     session_id=chain, tag="episodic", embedding=emb)
        mem.id = MEM_IDS[(chain, i)]
        store.add(mem)

print(f"Created: {store.count()} memories")
print(f"Eval: {len(EVAL)} multi-hop questions\n")

for h in [1, 2, 3]:
    tr = 0
    for q, truth in EVAL:
        mems = retrieve_relevant(q, store, k=5, hops=h)
        found = {m.id for m in mems}
        correct = truth_to_ids(truth)
        hit = found & correct
        r = len(hit) / len(correct) if correct else 0
        tr += r
    
    n = len(EVAL)
    print(f"  hops={h}: R@5={tr/n:.3f}")

# Show what each hop adds
print(f"\nHop detail for Q1:")
q = EVAL[0][0]
for h in [1, 2]:
    mems = retrieve_relevant(q, store, k=5, hops=h)
    print(f"  hops={h}:")
    for m in mems[:5]:
        print(f"    {m.id}: {m.content[:50]}")

del store
import gc; gc.collect()
if os.path.exists(db): os.remove(db)
