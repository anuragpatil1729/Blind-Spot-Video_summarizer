from __future__ import annotations

from src.ollama.vision_model import VisionCaptioner


def generate_captions(frames: list[dict], captioner: VisionCaptioner):
    captions = []
    for item in frames:
        caption = captioner.generate_caption(item["frame_path"])
        captions.append({**item, "caption": caption})
    return captions
