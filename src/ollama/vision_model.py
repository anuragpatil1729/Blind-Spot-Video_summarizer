from __future__ import annotations

import base64
from pathlib import Path

import requests


class VisionCaptioner:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.model = "llava"
        self.prompt = (
            "Describe this image in detail. Focus on people, objects, actions and environment."
        )

    def generate_caption(self, frame_path: str) -> str:
        image_path = Path(frame_path)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [image_b64],
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
