# recall. — CLI (typer)

import typer
from datetime import datetime
from .store import Memory, SQLiteStore
from .retrieve import retrieve_relevant
from .config import DEFAULT_DB_PATH

app = typer.Typer()
store: SQLiteStore = None  # initialized lazily


def get_store(db_path: str = DEFAULT_DB_PATH) -> SQLiteStore:
    global store
    if store is None:
        store = SQLiteStore(db_path)
    return store


@app.command()
def add(
    content: str,
    session: str = "default",
    tag: str = "episodic",
):
    """Add a memory to the store."""
    s = get_store()
    mem = Memory(content=content, session_id=session, tag=tag)
    mem_id = s.add(mem)
    typer.echo(f"✅ [{mem_id[:8]}] {content[:60]}...")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="What to search for"),
    k: int = 5,
):
    """Query memories using hybrid scoring."""
    s = get_store()
    results = retrieve_relevant(query_text, s, k=k)
    typer.echo(f"\n🔍 Query: {query_text}")
    typer.echo(f"{'─'*50}")
    for i, mem in enumerate(results, 1):
        ts = mem.timestamp.strftime("%m-%d")
        typer.echo(f"  {i}. [{ts}] [{mem.tag}] {mem.content[:80]}")
    if not results:
        typer.echo("  (no results)")


@app.command()
def stats():
    """Show store statistics."""
    s = get_store()
    total = s.count()
    episodic = s.count(tag="episodic")
    semantic = s.count(tag="semantic")
    typer.echo(f"📊 recall. stats")
    typer.echo(f"{'─'*30}")
    typer.echo(f"  Total:     {total}")
    typer.echo(f"  Episodic:  {episodic}")
    typer.echo(f"  Semantic:  {semantic}")
    if total > 0:
        latest = s.get_all(limit=1)
        typer.echo(f"  Latest:    {latest[0].content[:50]}...")


@app.command()
def delete(
    memory_id: str = typer.Argument(..., help="Memory ID to delete"),
):
    """Delete a memory by ID."""
    s = get_store()
    if s.delete(memory_id):
        typer.echo(f"🗑️  Deleted {memory_id}")
    else:
        typer.echo(f"❌ Not found: {memory_id}")


@app.command()
def clear():
    """Clear ALL memories."""
    typer.confirm("Delete all memories?", abort=True)
    s = get_store()
    s.clear()
    typer.echo("🗑️  All memories cleared.")


if __name__ == "__main__":
    app()
