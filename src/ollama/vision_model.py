from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.ollama.ollama_client import OllamaClient


class VisionCaptioner:
    def __init__(self, use_ollama: bool = False, base_url: str = "http://localhost:11434", model: str = "llava"):
        self.use_ollama = use_ollama
        self.client = OllamaClient(base_url=base_url, model=model) if use_ollama else None

    def caption_frame(self, frame_path: str) -> str:
        if self.client is not None:
            try:
                prompt = "Describe this driving scene briefly and mention notable hazards."
                response = self.client.caption_image(frame_path, prompt)
                if response:
                    return response
            except Exception:
                pass
        return self._fallback_caption(frame_path)

    @staticmethod
    def _fallback_caption(frame_path: str) -> str:
        image_path = Path(frame_path)
        if not image_path.exists():
            return "Frame unavailable."

        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        mean_color = image.mean(axis=(0, 1))
        brightness = float(image.mean())

        gray = image.mean(axis=2)
        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        texture = float((grad_x + grad_y) / 2)

        lighting = "bright" if brightness > 130 else "dim"
        complexity = "busy" if texture > 20 else "calm"
        return (
            f"Road frame {w}x{h}, {lighting} lighting, {complexity} scene, "
            f"avg RGB=({mean_color[0]:.0f},{mean_color[1]:.0f},{mean_color[2]:.0f})."
        )
