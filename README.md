# recall. 🧠

**Better contextual retrieval for AI agents.**
Hybrid scoring (semantic + recency + entity). Currently ≈ pure vector — entity signal improvement in progress.

```python
from recall import retrieve_relevant
store.add("User prefers docker-compose over Dockerfile for local dev")
results = retrieve_relevant("How should I deploy?", store)
```

## Status

- [x] P0: Hybrid scoring prototype
- [x] P0.5: store + retrieve + cli modules
- [x] P1: sqlite-vec ANN, eval pipeline, entity extraction
- [ ] P2: Improve entity signal, real agent integration

Hybrid ≈ Pure vector on 20-question eval (R=0.385 vs R=0.389).

## Quick start

```bash
pip install sentence-transformers typer
python3 cli.py add "User prefers docker-compose for local dev"
python3 cli.py query "How to deploy?"
```

## How it works

Three scoring signals:

```
score = 0.5 × semantic_similarity 
      + 0.3 × recency 
      + 0.2 × entity_overlap
```

## CLI

```bash
recall add "content"          # Store a memory
recall query "question"       # Hybrid retrieval
recall pure "question"        # Pure vector baseline
recall stats                  # Store statistics
recall delete <id>            # Remove a memory
```

## Design decisions

| Decision | Rationale |
|----------|-----------|
| No LLM re-rank | Cold start makes rerank useless |
| No hypergraph | SQL JOIN is sufficient |
| SQLite first | Zero-deployment; PostgreSQL later |
| sqlite-vec | ANN search, 384-dim cosine distance |

## License

Apache 2.0
