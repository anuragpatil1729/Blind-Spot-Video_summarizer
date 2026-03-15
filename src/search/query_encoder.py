from __future__ import annotations

from src.embeddings.embedder import TextEmbedder


def encode_query(query: str, embedder: TextEmbedder):
    return embedder.encode([query])[0]
