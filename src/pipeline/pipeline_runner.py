from __future__ import annotations

from pathlib import Path

import yaml

from src.embeddings.embedder import TextEmbedder
from src.embeddings.vector_store import SimpleVectorStore
from src.ollama.vision_model import VisionCaptioner
from src.pipeline.caption_pipeline import generate_captions
from src.pipeline.frame_sampler import sample_frames
from src.search.query_encoder import encode_query
from src.search.similarity_search import run_similarity_search
from src.utils.file_utils import save_json


class PipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with Path(config_path).open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def build(self):
        paths = self.config["paths"]
        sampling = self.config["sampling"]
        ollama_cfg = self.config["ollama"]

        frames = sample_frames(
            paths["video_path"],
            paths["frames_dir"],
            fps=float(sampling["fps"]),
            max_frames=int(sampling["max_frames"]),
        )

        captioner = VisionCaptioner(
            use_ollama=bool(ollama_cfg["enabled"]),
            base_url=ollama_cfg["base_url"],
            model=ollama_cfg["model"],
        )
        captions = generate_captions(frames, captioner)
        save_json(paths["captions_file"], captions)

        texts = [c["caption"] for c in captions]
        frame_paths = [c["frame_path"] for c in captions]
        timestamps = [c["timestamp_sec"] for c in captions]

        embedder = TextEmbedder()
        embeddings = embedder.encode(texts)
        store = SimpleVectorStore(paths["index_file"])
        store.add_many(texts, frame_paths, timestamps, embeddings)
        store.save()
        return {"frames": len(frames), "captions": len(captions), "index": paths["index_file"]}

    def query(self, text: str, top_k: int | None = None):
        paths = self.config["paths"]
        search_cfg = self.config["search"]
        store = SimpleVectorStore(paths["index_file"])
        store.load()

        embedder = TextEmbedder()
        q = encode_query(text, embedder)
        return run_similarity_search(store, q, top_k=top_k or int(search_cfg["top_k"]))
