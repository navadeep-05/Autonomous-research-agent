import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(ROOT, "kb_store")
VEC_PATH = os.path.join(STORE_DIR, "vectors.npy")
META_PATH = os.path.join(STORE_DIR, "chunks.json")


class VectorStore:
    def __init__(self):
        self.vectors = None
        self.chunks = []
        self.load()

    def size(self):
        return len(self.chunks)

    def clear(self):
        self.vectors, self.chunks = None, []

    def add(self, vectors, chunks):
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vec, k=4):
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vec  # cosine: vectors are normalised
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def save(self):
        os.makedirs(STORE_DIR, exist_ok=True)
        if self.vectors is not None:
            np.save(VEC_PATH, self.vectors)
        elif os.path.exists(VEC_PATH):
            os.remove(VEC_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self):
        if os.path.exists(VEC_PATH) and os.path.exists(META_PATH):
            self.vectors = np.load(VEC_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                self.chunks = json.load(f)