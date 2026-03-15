from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.embeddings.embedder import TextEmbedder
from src.embeddings.vector_store import SimpleVectorStore
from src.ollama.vision_model import VisionCaptioner
from src.pipeline.caption_pipeline import generate_captions
from src.pipeline.frame_sampler import sample_frames
from src.search.similarity_search import run_similarity_search
from src.utils.file_utils import save_json


class PipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with Path(config_path).open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.embedder = TextEmbedder()

    def build(
        self,
        *,
        video_path: str | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        use_ollama: bool | None = None,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
    ) -> dict[str, Any]:
        paths = self.config["paths"]
        sampling = self.config["sampling"]
        ollama_cfg = self.config["ollama"]

        effective_video_path = video_path or paths["video_path"]
        effective_fps = float(fps if fps is not None else sampling["fps"])
        effective_max_frames = int(max_frames if max_frames is not None else sampling["max_frames"])
        effective_use_ollama = bool(use_ollama if use_ollama is not None else ollama_cfg["enabled"])

        frames = sample_frames(
            effective_video_path,
            paths["frames_dir"],
            fps=effective_fps,
            max_frames=effective_max_frames,
        )

        captioner = VisionCaptioner(
            use_ollama=effective_use_ollama,
            base_url=ollama_base_url or ollama_cfg["base_url"],
            model=ollama_model or ollama_cfg["model"],
        )
        captions = generate_captions(frames, captioner)
        save_json(paths["captions_file"], captions)

        texts = [c["caption"] for c in captions]
        frame_paths = [c["frame_path"] for c in captions]
        timestamps = [c["timestamp_sec"] for c in captions]

        embeddings = self.embedder.encode(texts)
        store = SimpleVectorStore(paths["index_file"])
        store.add_many(texts, frame_paths, timestamps, embeddings)
        store.save()
        return {
            "frames": len(frames),
            "captions": len(captions),
            "index": paths["index_file"],
            "video_path": effective_video_path,
            "fps": effective_fps,
            "max_frames": effective_max_frames,
            "ollama_enabled": effective_use_ollama,
        }

    def query(self, text: str, top_k: int | None = None):
        paths = self.config["paths"]
        search_cfg = self.config["search"]
        store = SimpleVectorStore(paths["index_file"])
        store.load()

        if not store.items:
            return []

        q = self.embedder.encode([text])[0]
        return run_similarity_search(store, q, top_k=top_k or int(search_cfg["top_k"]))
