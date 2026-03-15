from __future__ import annotations

from typing import Iterable

import numpy as np


class MultimodalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = [str(t) for t in texts]
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32)
