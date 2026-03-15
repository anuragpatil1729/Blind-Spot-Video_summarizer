from __future__ import annotations

from pathlib import Path
import re
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

        frames = sample_frames(
            effective_video_path,
            paths["frames_dir"],
            fps=effective_fps,
            max_frames=effective_max_frames,
        )

        if not bool(use_ollama if use_ollama is not None else ollama_cfg.get("enabled", True)):
            raise RuntimeError("Ollama captioning is disabled. Enable it to build a searchable index.")

        captioner = VisionCaptioner(
            base_url=ollama_base_url or ollama_cfg["base_url"],
            model=ollama_model or ollama_cfg.get("model", "llava"),
        )
        captions = generate_captions(frames, captioner)
        save_json(paths["captions_file"], captions)

        raw_captions = [c["caption"] for c in captions]
        texts = [_build_search_text(c) for c in raw_captions]
        frame_paths = [c["frame_path"] for c in captions]
        timestamps = [c["timestamp_sec"] for c in captions]

        embeddings = self.embedder.encode(texts)
        store = SimpleVectorStore(paths["index_file"])
        store.add_many(texts, frame_paths, timestamps, embeddings, captions=raw_captions)
        store.save()
        return {
            "frames": len(frames),
            "captions": len(captions),
            "index": paths["index_file"],
            "video_path": effective_video_path,
            "fps": effective_fps,
            "max_frames": effective_max_frames,
            "ollama_enabled": True,
            "ollama_model": ollama_model or ollama_cfg.get("model", "llava"),
        }

    def query(
        self,
        text: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ):
        paths = self.config["paths"]
        search_cfg = self.config["search"]
        store = SimpleVectorStore(paths["index_file"])
        store.load()

        if not store.items:
            return []

        q = self.embedder.encode([text])[0]
        results = run_similarity_search(store, q, top_k=top_k or int(search_cfg["top_k"]))
        effective_min_score = float(min_score if min_score is not None else search_cfg.get("min_score", 0.0))
        filtered_results = [row for row in results if float(row["score"]) >= effective_min_score]
        if filtered_results:
            return filtered_results

        if effective_min_score <= 0 or not results:
            return results

        relaxed_min_score = min(0.05, effective_min_score)
        relaxed_results = [row for row in results if float(row["score"]) >= relaxed_min_score]
        if relaxed_results:
            return relaxed_results

        return results[: min(3, len(results))]


_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "this", "that", "there", "where", "about", "image", "frame", "video", "appears", "show", "shows", "visible", "likely",
}


def _build_search_text(caption: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-']+", caption.lower())
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= 24:
            break

    if not keywords:
        return caption
    return f"{caption}\nkeywords: {', '.join(keywords)}"
