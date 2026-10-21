def build_context(hits, max_chars=1200):
    return "\n\n".join(f"[{h['url']}]\n{h['content'][:max_chars]}" for h in hits)