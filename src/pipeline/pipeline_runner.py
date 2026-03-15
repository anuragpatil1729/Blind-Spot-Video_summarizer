from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.agents.video_agent import VideoAgent
from src.audio.whisper_transcriber import WhisperTranscriber
from src.embeddings.embedder import MultimodalEmbedder
from src.ocr.ocr_extractor import OCRExtractor
from src.search.query_understanding import OllamaQueryRewriter
from src.search.reranker import CrossEncoderReranker
from src.search.semantic_search import SemanticVideoSearch
from src.search.vector_store import ChromaVideoStore
from src.video_processing.frame_extractor import extract_clip, extract_scene_frames
from src.video_processing.scene_detection import detect_scenes
from src.vision.llava_captioner import LLaVACaptioner


class VideoIntelligencePipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with Path(config_path).open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.embedder = MultimodalEmbedder(self.config["embeddings"]["model"])
        self.store = ChromaVideoStore(
            persist_dir=self.config["paths"]["chroma_dir"],
            collection_name=self.config["paths"].get("chroma_collection", "video_moments"),
        )
        self.search_engine = SemanticVideoSearch(self.store, self.embedder, CrossEncoderReranker())
        self.query_rewriter = OllamaQueryRewriter(
            base_url=self.config["ollama"]["base_url"],
            model=self.config["ollama"].get("query_model", "llama3"),
        )
        self.agent = VideoAgent(
            self.search_engine,
            base_url=self.config["ollama"]["base_url"],
            model=self.config["ollama"].get("agent_model", "llama3"),
        )

    def index_video(self, video_path: str | None = None) -> dict[str, Any]:
        video = video_path or self.config["paths"]["video_path"]
        scenes = detect_scenes(video, threshold=float(self.config["scene_detection"].get("threshold", 27.0)))
        frame_records = extract_scene_frames(
            video,
            scenes,
            output_dir=self.config["paths"]["frames_dir"],
            max_scenes=int(self.config["performance"].get("max_scenes", 300)),
        )

        captioner = LLaVACaptioner(self.config["ollama"]["base_url"], self.config["ollama"]["vision_model"])
        frame_records = captioner.caption_batch(frame_records)

        ocr = OCRExtractor(self.config["ocr"].get("languages", ["en"]))
        frame_records = ocr.extract_batch(frame_records)

        transcriber = WhisperTranscriber(self.config["audio"].get("whisper_model", "base"))
        transcript_segments = transcriber.transcribe(video)

        indexed_rows = []
        for row in frame_records:
            ts = float(row["timestamp_sec"])
            transcript = _transcript_at_timestamp(transcript_segments, ts)
            row["transcript"] = transcript
            row["search_text"] = _build_search_text(row["caption"], row["ocr_text"], transcript)
            indexed_rows.append(row)

        embeddings = self.embedder.encode([r["search_text"] for r in indexed_rows], batch_size=32)
        self.store.reset()
        self.store.upsert(
            ids=[r["scene_id"] for r in indexed_rows],
            embeddings=[e.tolist() for e in embeddings],
            documents=[r["search_text"] for r in indexed_rows],
            metadatas=[
                {
                    "scene_id": r["scene_id"],
                    "timestamp_sec": float(r["timestamp_sec"]),
                    "scene_start": float(r["scene_start"]),
                    "scene_end": float(r["scene_end"]),
                    "frame_path": r["frame_path"],
                    "caption": r["caption"],
                    "ocr_text": r["ocr_text"][:1800],
                    "transcript": r["transcript"][:1800],
                }
                for r in indexed_rows
            ],
        )

        timeline = [
            {"timestamp": _fmt_ts(r["timestamp_sec"]), "event": r["caption"]}
            for r in sorted(indexed_rows, key=lambda x: float(x["timestamp_sec"]))
        ]
        output = {
            "video_path": video,
            "scene_count": len(scenes),
            "indexed_moments": len(indexed_rows),
            "timeline": timeline,
            "transcript_segments": [s.__dict__ for s in transcript_segments],
        }

        out_file = Path(self.config["paths"]["artifacts_dir"]) / "index_manifest.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        expanded_queries = self.query_rewriter.rewrite(query)
        merged: dict[str, dict] = {}
        for q in expanded_queries:
            for row in self.search_engine.search(q, top_k=top_k, initial_k=20):
                key = row["id"]
                old = merged.get(key)
                if old is None or row.get("rerank_score", -1e9) > old.get("rerank_score", -1e9):
                    merged[key] = row
        final = sorted(merged.values(), key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return final[:top_k]

    def extract_result_clip(self, video_path: str, timestamp_sec: float) -> str:
        clip_path = Path(self.config["paths"]["clips_dir"]) / f"clip_{int(timestamp_sec*1000):08d}.mp4"
        return extract_clip(video_path, timestamp_sec, str(clip_path))


def _transcript_at_timestamp(segments, timestamp: float) -> str:
    hits = [s.text for s in segments if s.start <= timestamp <= s.end]
    if hits:
        return " ".join(hits)
    nearest = sorted(segments, key=lambda s: abs(((s.start + s.end) / 2.0) - timestamp))[:2]
    return " ".join(seg.text for seg in nearest)


def _build_search_text(caption: str, ocr_text: str, transcript: str) -> str:
    return f"caption: {caption}\nocr: {ocr_text or 'none'}\ntranscript: {transcript or 'none'}"


def _fmt_ts(seconds: float) -> str:
    sec = int(seconds)
    return f"{sec//60:02d}:{sec%60:02d}"


class PipelineRunner(VideoIntelligencePipeline):
    def build(self, **kwargs):
        video_path = kwargs.get("video_path")
        return self.index_video(video_path)

    def query(self, text: str, top_k: int | None = None, min_score: float | None = None):
        _ = min_score
        return self.search(text, top_k=top_k or 5)
