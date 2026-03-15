from __future__ import annotations

from pathlib import Path

import chromadb


class ChromaVideoStore:
    def __init__(self, persist_dir: str = "data/index/chroma", collection_name: str = "video_moments"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        data = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = data.get("ids", [[]])[0]
        docs = data.get("documents", [[]])[0]
        metas = data.get("metadatas", [[]])[0]
        dists = data.get("distances", [[]])[0]
        rows = []
        for i in range(len(ids)):
            rows.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": float(dists[i]) if i < len(dists) else 0.0,
                }
            )
        return rows
