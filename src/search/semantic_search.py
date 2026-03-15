from __future__ import annotations

from src.embeddings.embedder import MultimodalEmbedder
from src.search.reranker import CrossEncoderReranker
from src.search.vector_store import ChromaVideoStore


class SemanticVideoSearch:
    def __init__(self, store: ChromaVideoStore, embedder: MultimodalEmbedder, reranker: CrossEncoderReranker | None = None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5, initial_k: int = 20) -> list[dict]:
        query_vector = self.embedder.encode([query])[0].tolist()
        retrieved = self.store.query(query_vector, top_k=initial_k)
        if self.reranker:
            retrieved = self.reranker.rerank(query, retrieved, top_k=top_k)
        else:
            retrieved = retrieved[:top_k]
        return retrieved
