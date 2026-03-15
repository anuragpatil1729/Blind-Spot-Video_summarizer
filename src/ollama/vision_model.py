from __future__ import annotations

import base64
from pathlib import Path
from time import sleep

import requests


class VisionCaptioner:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llava",
        prompt: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt = prompt or (
            "Analyze this frame for general video understanding (not only driving). "
            "Return 3 short sections:\n"
            "1) Scene: where this appears to happen.\n"
            "2) Details: visible objects, people/animals, actions, and notable text.\n"
            "3) Tags: 8-15 comma-separated keywords covering all important entities and events."
        )

    def generate_caption(self, frame_path: str) -> str:
        image_path = Path(frame_path)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 220,
            },
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()
                if text:
                    return text
                raise ValueError("Empty caption response from Ollama")
            except Exception as exc:  # retry external API call
                last_error = exc
                sleep(0.8 * (attempt + 1))

        raise RuntimeError(f"Caption generation failed for {frame_path}: {last_error}")
