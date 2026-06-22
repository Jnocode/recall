def expand_query(query: str, max_terms: int = 15) -> str:
    """Expand query with domain vocabulary.
    
    Controlled by max_terms parameter to prevent over-expansion.
    Higher max_terms = more aggressive domain vocab usage.
    """
    words = query.lower().split()
    expanded = set(words)
    for w in words:
        clean = w.rstrip("?.,!;:'\"s")
        if clean in DOMAIN_VOCAB and len(expanded) < max_terms:
            expanded.update(DOMAIN_VOCAB[clean])
    return " ".join(expanded)
