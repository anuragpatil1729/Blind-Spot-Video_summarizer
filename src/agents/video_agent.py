from __future__ import annotations

import requests

from src.search.semantic_search import SemanticVideoSearch


class VideoAgent:
    def __init__(self, search_engine: SemanticVideoSearch, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.search_engine = search_engine
        self.base_url = base_url.rstrip("/")
        self.model = model

    def answer(self, user_question: str, evidence_k: int = 5) -> dict:
        evidence = self.search_engine.search(user_question, top_k=evidence_k, initial_k=20)
        evidence_text = "\n".join(
            f"- t={row['metadata'].get('timestamp_sec', 0)}s | {row['document']}" for row in evidence
        )
        prompt = (
            "You are an AI Video Agent. Use the retrieved evidence to answer what happens in the video. "
            "Include key moments with timestamps. If uncertain, say so.\n"
            f"Question: {user_question}\nEvidence:\n{evidence_text}"
        )
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return {"answer": str(response.json().get("response", "")).strip(), "evidence": evidence}
