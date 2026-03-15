from __future__ import annotations

from src.embeddings.vector_store import SimpleVectorStore


def run_similarity_search(store: SimpleVectorStore, query_embedding, top_k: int = 5):
    return store.search(query_embedding, top_k=top_k)
