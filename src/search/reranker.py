from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["document"]) for c in candidates]
        scores = self.model.predict(pairs)
        reranked = []
        for row, score in zip(candidates, scores):
            enriched = {**row, "rerank_score": float(score)}
            reranked.append(enriched)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
