from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if self.model is not None:
            return np.array(self.model.encode(texts, normalize_embeddings=True))
        return np.vstack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.fallback_dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.fallback_dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
