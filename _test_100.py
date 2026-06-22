"""recall. vs AIngram — 100 Memories × 20 Questions"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DOMAINS = [
    "User prefers docker-compose for local dev",
    "Production uses Kubernetes cluster with Helm charts",
    "CI/CD pipeline configured with GitHub Actions",
    "Deployment requires health check endpoints",
    "Blue-green deployment strategy for zero downtime",
    "Docker images optimized with multi-stage builds",
    "ECS Fargate selected for cost efficiency",
    "Serverless deployment via AWS Lambda considered",
    "Rollback strategy: keep last 3 versions",
    "Canary deployment for gradual rollout",
    "PostgreSQL with asyncpg for production app",
    "Redis cache layer for API response caching",
    "Database connection pool set to 25 connections",
    "Monthly partition strategy for orders table",
    "Read replicas for reporting queries",
    "SQLite for local development testing",
    "MongoDB rejected in favor of Postgres",
    "Query optimization: add composite indexes",
    "Database backup scheduled at 2AM daily",
    "Migration scripts use Alembic with rollback",
    "All PRs require complete type hints",
    "Pytest suite must run under 5 minutes",
    "Code coverage target: 80% minimum",
    "Black formatter with 88 char line width",
    "Ruff linter replaces Flake8 and isort",
    "Pre-commit hooks enforce linting",
    "Branch naming convention: feat/fix/chore prefix",
    "Squash merge preferred over merge commits",
    "Code review required for all changes",
    "Documentation must accompany new features",
    "EC2 to ECS Fargate migration completed",
    "VPC with public and private subnets",
    "CloudFront CDN for static assets",
    "Route53 for DNS management",
    "Terraform for infrastructure as code",
    "CloudWatch for log aggregation",
    "S3 bucket for backup storage",
    "IAM roles follow principle of least privilege",
    "Security groups restrict to necessary ports",
    "Auto-scaling policy based on CPU usage",
]

EVAL = [
    ("How should I deploy?", {0,1,2,4}),
    ("What database setup?", {10,11,13,14}),
    ("Code review standards?", {20,26,28}),
    ("Infrastructure setup?", {30,31,34}),
    ("Rollback strategy?", {8}),
    ("Caching solution?", {11}),
    ("Testing requirements?", {21,22}),
    ("CI/CD tools?", {2,6}),
    ("Database backup?", {18}),
    ("Code formatting?", {23,24}),
    ("Local dev setup?", {0,14}),
    ("DB migrations?", {19}),
    ("Monitoring setup?", {35,36}),
    ("Docker build process?", {5,6}),
    ("Branching strategy?", {26,27}),
    ("Network security?", {33,38}),
    ("PR merge policy?", {28}),
    ("Query optimization?", {17}),
    ("Secret management?", {38}),
    ("Deployment strategy?", {4,9}),
]

def test_recall():
    from store import Memory, SQLiteStore
    from retrieve import retrieve_relevant, extract_entities
    from embed import embed
    from datetime import datetime, timedelta
    
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_recall_100.db")
    if os.path.exists(db): os.remove(db)
    store = SQLiteStore(db, vec_dim=768)
    
    for i, content in enumerate(DOMAINS):
        mem = Memory(content=content, entities=extract_entities(content),
                     timestamp=datetime.utcnow()-timedelta(days=i),
                     session_id="eval", tag="episodic", embedding=embed(content))
        mem.id = f"mem_{i:02d}"
        store.add(mem)
    
    tr = 0
    t0 = time.time()
    for q, truth in EVAL:
        mems = retrieve_relevant(q, store, k=5)
        ids = {int(m.id.split("_")[1]) for m in mems if m.id.startswith("mem_")}
        tr += len(ids & truth) / len(truth) if truth else 0
    elapsed = time.time() - t0
    del store
    import gc; gc.collect()
    os.remove(db)
    return tr / len(EVAL), elapsed / len(EVAL)

def test_aingram():
    from aingram import MemoryStore
    
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ai_100.db")
    if os.path.exists(db): os.remove(db)
    store = MemoryStore(db)
    
    for content in DOMAINS:
        store.remember(content)
    
    tr = 0
    t0 = time.time()
    for q, truth in EVAL:
        results = store.recall(q, limit=5)
        hits = set()
        for r in results:
            c = r.entry.content
            if isinstance(c, str) and c.startswith("{"):
                c = __import__("json").loads(c).get("text", c)
            elif isinstance(c, dict):
                c = c.get("text", str(c))
            try:
                hits.add(DOMAINS.index(c))
            except ValueError:
                pass
        tr += len(hits & truth) / len(truth) if truth else 0
    elapsed = time.time() - t0
    store.close()
    import gc; gc.collect()
    os.remove(db)
    return tr / len(EVAL), elapsed / len(EVAL)

print("="*60)
print("  recall. vs AIngram — 40 Memories × 20 Questions")
print("  Weighted fusion enabled (path_a=0.4, path_b=0.6)")
print("="*60)
print(f"  Memories: {len(DOMAINS)}  Questions: {len(EVAL)}")
print()

print("  Warming up embed...")
from embed import embed
embed("warmup")
print()

print("  📍 recall. (weighted fusion):")
r_r, t_r = test_recall()
print(f"     Recall@5: {r_r:.3f}   Latency: {t_r*1000:.0f}ms")
print()

print("  📍 AIngram:")
a_r, t_a = test_aingram()
print(f"     Recall@5: {a_r:.3f}   Latency: {t_a*1000:.0f}ms")

print(f"\n{'='*60}")
diff = r_r - a_r
if diff > 0.01:
    print(f"  🏆 recall. wins by +{diff:.3f} ({(diff/a_r)*100:.0f}%)")
elif diff < -0.01:
    print(f"  🏆 AIngram wins by {diff:.3f}")
else:
    print(f"  🤝 Tie (diff={diff:.3f})")
print(f"{'='*60}")
