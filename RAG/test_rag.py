from RAG.context import build_context
from RAG.retriever import build_index, retrieve

print("chunks indexed:", build_index())
hits = retrieve("main topic of this document", k=3, min_score=0.0)
for h in hits:
    print(h["score"], h["url"])
print(build_context(hits)[:300])