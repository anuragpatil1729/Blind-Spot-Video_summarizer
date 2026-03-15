from __future__ import annotations

import base64
from pathlib import Path

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def caption_image(self, image_path: str, prompt: str) -> str:
        p = Path(image_path)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "images": [base64.b64encode(p.read_bytes()).decode("utf-8")],
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
