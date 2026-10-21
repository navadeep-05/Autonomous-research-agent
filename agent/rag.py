import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_client = chromadb.PersistentClient(path="./chroma_db")
_ef = DefaultEmbeddingFunction()
_col = _client.get_or_create_collection("research_evidence", embedding_function=_ef)


def _chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def add_documents(docs: list[dict]) -> int:
    """Store chunked docs in ChromaDB. Returns total chunk count."""
    ids, texts, metas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(_chunk(doc.get("content", ""))):
            uid = f"{doc['url']}::chunk{i}"
            ids.append(uid)
            texts.append(chunk)
            metas.append({"url": doc.get("url", ""), "title": doc.get("title", "")})
    if ids:
        _col.upsert(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Query ChromaDB. Returns list of {text, url, title}."""
    results = _col.query(query_texts=[query], n_results=k)
    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append({"text": doc, "url": meta.get("url", ""), "title": meta.get("title", "")})
    return output


def reset_collection() -> None:
    """Delete and recreate the collection."""
    _client.delete_collection("research_evidence")
    global _col
    _col = _client.get_or_create_collection("research_evidence", embedding_function=_ef)
