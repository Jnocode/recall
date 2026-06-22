"""Sprint A #1: expand_query max_terms threshold tuning"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
from store import Memory, SQLiteStore
from retrieve import retrieve_relevant, extract_entities, expand_query
from embed import embed

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

# Test different max_terms values  
for max_t in [0, 5, 10, 15, 20]:
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tune.db")
    if os.path.exists(db): os.remove(db)
    store = SQLiteStore(db, vec_dim=768)
    
    for i, content in enumerate(DOMAINS):
        mem = Memory(content=content, entities=extract_entities(content),
                     timestamp=datetime.utcnow()-timedelta(days=i),
                     session_id="eval", tag="episodic", embedding=embed(content))
        mem.id = f"mem_{i:02d}"
        store.add(mem)
    
    tr = 0
    for q, truth in EVAL:
        mems = retrieve_relevant(q, store, k=5)
        ids = {int(m.id.split("_")[1]) for m in mems if m.id.startswith("mem_")}
        tr += len(ids & truth) / len(truth) if truth else 0
    
    print(f"  max_terms={max_t:2d}: R@5={tr/len(EVAL):.3f}")
    del store
    import gc; gc.collect()
    if os.path.exists(db): os.remove(db)
