from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from src.utils.file_utils import load_json, save_json


@dataclass
class VectorItem:
    text: str
    frame_path: str
    timestamp_sec: float
    embedding: list[float]


class SimpleVectorStore:
    def __init__(self, index_file: str):
        self.index_file = index_file
        self.items: List[VectorItem] = []

    def add_many(self, texts: list[str], frame_paths: list[str], timestamps: list[float], embeddings: np.ndarray):
        for i, text in enumerate(texts):
            self.items.append(
                VectorItem(
                    text=text,
                    frame_path=frame_paths[i],
                    timestamp_sec=float(timestamps[i]),
                    embedding=embeddings[i].astype(float).tolist(),
                )
            )

    def save(self):
        save_json(self.index_file, [asdict(item) for item in self.items])

    def load(self):
        rows = load_json(self.index_file, default=[])
        self.items = [VectorItem(**row) for row in rows]

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if not self.items:
            return []
        q = query_embedding.astype(np.float32)
        scores = []
        for item in self.items:
            v = np.array(item.embedding, dtype=np.float32)
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
            score = float(np.dot(q, v) / denom)
            scores.append((score, item))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": round(score, 4),
                "text": item.text,
                "frame_path": item.frame_path,
                "timestamp_sec": item.timestamp_sec,
            }
            for score, item in scores[:top_k]
        ]
