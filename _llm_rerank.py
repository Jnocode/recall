"""recall. — LLM Re-rank Experiment

Hypothesis: LLM re-ranking of top-20 vector candidates can push recall@5 > 0.6.

Method:
1. Pure vector search → top-20 candidates
2. LLM scores each candidate: "How relevant is this memory to the query?" (0-10)
3. Re-rank by LLM score → top-5

Requires: DeepSeek API key in ~/.hermes/.env or DEEPSEEK_API_KEY env var
"""

import os, sys, json, urllib.request, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from store import SQLiteStore, Memory
from embed import embed

# ─── Config ───────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_p0.db")
TOP_K_VECTOR = 20  # initial candidates from vector search
TOP_K_FINAL = 5    # final after re-rank

# DeepSeek API
DEEPSEEK_KEY = None
env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "DEEPSEEK" in line and "=" in line:
                DEEPSEEK_KEY = line.strip().split("=", 1)[1].strip().strip('"')
                break


def llm_rerank(query: str, candidates: list[list]) -> list[list]:
    """Use local LLM (via LM Studio) to re-rank candidate memories."""
    if not candidates:
        return candidates

    # Build compact prompt
    mem_lines = []
    for i, (mem, score, mem_id) in enumerate(candidates):
        mem_lines.append(f"[{i}] {mem.content[:100]}")
    
    prompt = f"""Question: {query}

Memories:
{chr(10).join(mem_lines)}

Rate each memory 0-10. Return ONLY JSON array: [scores]
0=irrelevant 10=directly answers the question
"""

    body = json.dumps({
        "model": "qwen3.6",
        "messages": [
            {"role": "system", "content": "You are a relevance judge. Output ONLY a JSON array of scores."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }).encode()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:1234/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON array from response
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        scores = json.loads(content)

        if len(scores) != len(candidates):
            print(f"  ⚠️  LLM returned {len(scores)} scores, expected {len(candidates)}")
            return candidates

        # Apply LLM scores, weight: 0.7 LLM + 0.3 vector similarity
        for i, (mem, vec_score, mem_id) in enumerate(candidates):
            llm_s = max(0, min(10, scores[i])) / 10.0  # normalize to 0-1
            combined = 0.7 * llm_s + 0.3 * vec_score
            candidates[i] = (mem, combined, mem_id)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    except Exception as e:
        print(f"  ⚠️  LLM rerank failed: {e}")
        return candidates


# ─── Eval questions ───────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    ("Which deployment method does user prefer?", {"seed_00","seed_01","seed_11","seed_20"}),
    ("What database issues encountered?", {"seed_02","seed_12","seed_16","seed_26"}),
    ("What frontend technology does user prefer?", {"seed_09","seed_15"}),
    ("What CI/CD decisions?", {"seed_00","seed_13"}),
    ("What security requirements?", {"seed_11","seed_25"}),
    ("How should APIs be structured?", {"seed_10","seed_23","seed_04"}),
    ("What performance problems?", {"seed_02","seed_03","seed_19","seed_26"}),
    ("What infrastructure changes?", {"seed_14","seed_20","seed_22"}),
    ("What Python decisions?", {"seed_09","seed_17","seed_01","seed_06"}),
    ("What monitoring tools?", {"seed_24"}),
    ("Config file format preference?", {"seed_21"}),
    ("Docker decisions?", {"seed_00","seed_08","seed_14"}),
    ("DB schema changes?", {"seed_12"}),
    ("Code quality standards?", {"seed_06","seed_13","seed_22"}),
    ("Development workflow?", {"seed_05","seed_13"}),
    ("Recent incidents?", {"seed_02"}),
    ("Migration projects?", {"seed_00","seed_14","seed_20","seed_17"}),
    ("User tool preferences?", {"seed_01","seed_04","seed_15","seed_21","seed_09"}),
    ("Architecture decisions?", {"seed_04","seed_14","seed_23","seed_22"}),
    ("Testing improvements?", {"seed_10"}),
]


if __name__ == "__main__":
    store = SQLiteStore(DB_PATH)
    count = store.count()
    print(f"📊 LLM Re-rank Experiment")
    print(f"   Memories: {count}")
    print(f"   Vector top-K: {TOP_K_VECTOR} → LLM re-rank → top-{TOP_K_FINAL}")
    print(f"   LLM: DeepSeek {'✅' if DEEPSEEK_KEY else '❌ (no key)'}")
    print()

    vector_recall = 0
    llm_recall = 0
    llm_calls = 0

    for qi, (q, truth) in enumerate(EVAL_QUESTIONS):
        # Step 1: Vector search → top-K
        q_emb = embed(q)
        all_mems = store.get_all()
        scored = []
        for mem in all_mems:
            if mem.embedding:
                a, b = np.array(q_emb), np.array(mem.embedding)
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
                scored.append((mem, sim, mem.id))
        scored.sort(key=lambda x: x[1], reverse=True)
        top20 = scored[:TOP_K_VECTOR]

        # Vector recall (top-5)
        vec_top5_ids = {mem.id for mem, _, _ in scored[:TOP_K_FINAL]}
        vec_hit = vec_top5_ids & truth
        vr = len(vec_hit) / len(truth) if truth else 0
        vector_recall += vr

        # Step 2: LLM re-rank top-20 → top-5
        reranked = llm_rerank(q, top20)
        llm_calls += 1
        llm_top5_ids = {mem.id for mem, _, _ in reranked[:TOP_K_FINAL]}
        llm_hit = llm_top5_ids & truth
        lr = len(llm_hit) / len(truth) if truth else 0
        llm_recall += lr

        status = "✅" if lr > vr else ("❌" if lr < vr else "=")
        print(f"  {status} Q{qi+1:2d}: vector R={vr:.3f}  LLM R={lr:.3f}  (truth={len(truth)})")

    n = len(EVAL_QUESTIONS)
    print(f"\n{'='*60}")
    print(f"  Vector recall@5: {vector_recall/n:.3f}")
    print(f"  LLM re-rank@5:   {llm_recall/n:.3f}")
    print(f"  Improvement:     {(llm_recall - vector_recall)/n:.3f}")
    print(f"  LLM calls:       {llm_calls}")
    if vector_recall > 0:
        print(f"  Ratio:           {llm_recall/vector_recall:.2f}x")
    print(f"{'='*60}")
