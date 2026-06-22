"""Extract Honcho memories → recall SQLite store"""
import sys, os, asyncio, asyncpg, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Memory, SQLiteStore
from retrieve import extract_entities
from datetime import datetime

async def extract_honcho():
    conn = await asyncpg.connect(user='honcho', password='honcho', host='localhost', port=5433, database='honcho')
    
    # Get all messages with content
    rows = await conn.fetch("""
        SELECT content, session_name, peer_name, created_at, public_id 
        FROM messages 
        WHERE content IS NOT NULL AND length(content) > 10
        ORDER BY created_at DESC
    """)
    print(f"Total messages: {len(rows)}")
    await conn.close()
    return rows

def to_recall(rows):
    """Convert messages to recall's format and store in SQLite."""
    db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_p0.db")
    # Don't delete - append to existing
    store = SQLiteStore(db, vec_dim=768)
    
    existing = store.count()
    print(f"Existing memories: {existing}")
    
    added = 0
    for r in rows:
        content = str(r['content'])[:200]  # truncate long messages
        session = str(r['session_name']) if r['session_name'] else "unknown"
        ts = r['created_at']
        
        # Skip very short or system messages
        if len(content) < 15:
            continue
        
        # Check duplicate by content hash
        import hashlib
        cid = hashlib.md5(content.encode()).hexdigest()[:12]
        if store.get(cid):
            continue
        
        # Get embedding via Nomic
        body = json.dumps({"model":"nomic-embed-text-v1.5","input":[content],"encoding_format":"float"}).encode()
        try:
            resp = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:1234/v1/embeddings", data=body, headers={"Content-Type":"application/json"}), timeout=15)
            emb = json.loads(resp.read())["data"][0]["embedding"]
        except:
            continue
        
        mem = Memory(content=content, entities=extract_entities(content),
                     timestamp=ts, session_id=session, tag="episodic", embedding=emb)
        mem.id = cid
        store.add(mem)
        added += 1
        if added % 50 == 0:
            print(f"  Added {added}...")
    
    print(f"\nAdded: {added}")
    print(f"Total: {store.count()}")

if __name__ == "__main__":
    print("Extracting from Honcho...")
    rows = asyncio.run(extract_honcho())
    print(f"Converting to recall format...")
    to_recall(rows)
