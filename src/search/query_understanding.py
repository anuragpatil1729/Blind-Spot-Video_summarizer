from __future__ import annotations

import requests


class OllamaQueryRewriter:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            "Rewrite the video search query into 3 concise variants for semantic retrieval. "
            "Return as newline-separated list without numbering. Query: "
            f"{query}"
        )
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        text = str(response.json().get("response", "")).strip()
        variants = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        deduped = []
        for v in [query] + variants:
            if v and v not in deduped:
                deduped.append(v)
        return deduped[:4]
