from __future__ import annotations

import base64
from pathlib import Path

import requests


class LLaVACaptioner:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def caption_image(self, image_path: str, prompt: str | None = None) -> str:
        prompt_text = prompt or (
            "Describe this frame for semantic video retrieval. "
            "Mention subjects, actions, setting, objects, mood, and notable details."
        )
        image_bytes = Path(image_path).read_bytes()
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "images": [base64.b64encode(image_bytes).decode("utf-8")],
            "stream": False,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()

    def caption_batch(self, frame_records: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for row in frame_records:
            caption = self.caption_image(row["frame_path"])
            enriched.append({**row, "caption": caption})
        return enriched
