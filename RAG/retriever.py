import re

from RAG.chunker import chunk_text
from RAG.embeddings import embed
from RAG.loader import load_documents
from RAG.vector_store import VectorStore

_store = None
PROJECT_OVERVIEW = "research_agent_overview.md"


def is_project_specific_query(query):
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", str(query).casefold()).split())
    project_context = (
        r"\bautonomous research agent\b",
        r"\bthis (?:project|repository|repo|codebase|app)\b",
        r"\bour (?:project|repository|repo|codebase|app)\b",
        r"\bproject (?:architecture|workflow|implementation|structure|files)\b",
        r"\b(?:repository|repo|codebase) (?:architecture|workflow|implementation|structure)\b",
    )
    return any(re.search(pattern, normalized) for pattern in project_context)


def _get_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def index_size():
    return _get_store().size()


def build_index():
    """Re-read every file in data/ and rebuild the local knowledge base."""
    store = _get_store()
    store.clear()
    chunks, texts = [], []
    for doc in load_documents():
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append({"name": doc["name"], "chunk": i, "text": piece})
            texts.append(piece)
    if texts:
        store.add(embed(texts), chunks)
    store.save()
    return len(texts)


def retrieve(query, k=3, min_score=0.25):
    store = _get_store()
    if store.size() == 0:
        return []
    excluded_names = (
        set()
        if is_project_specific_query(query)
        else {PROJECT_OVERVIEW}
    )
    hits = store.search(
        embed([query])[0],
        k,
        exclude_names=excluded_names,
    )
    return [
        {
            "title": h["name"],
            "url": f"local://{h['name']}#chunk{h['chunk']}",
            "content": h["text"],
            "score": round(h["score"], 3),
        }
        for h in hits
        if h["score"] >= min_score
    ]