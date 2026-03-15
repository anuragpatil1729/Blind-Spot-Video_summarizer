from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SceneSegment:
    scene_id: str
    scene_start: float
    scene_end: float
    representative_ts: float
    representative_frame: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return max(float(out or 0.0), 0.0)


def _fallback_segments(video_path: str, chunk_seconds: float = 6.0) -> list[SceneSegment]:
    duration = _probe_duration(video_path)
    if duration <= 0:
        return [SceneSegment(scene_id="scene_0000", scene_start=0.0, scene_end=1.0, representative_ts=0.0)]

    segments: list[SceneSegment] = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        midpoint = start + ((end - start) / 2.0)
        segments.append(
            SceneSegment(
                scene_id=f"scene_{idx:04d}",
                scene_start=round(start, 3),
                scene_end=round(end, 3),
                representative_ts=round(midpoint, 3),
            )
        )
        idx += 1
        start = end
    return segments


def detect_scenes(video_path: str, threshold: float = 27.0) -> list[SceneSegment]:
    """Detect scenes with PySceneDetect; fallback to fixed chunks."""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()

        segments: list[SceneSegment] = []
        for idx, (start_t, end_t) in enumerate(scene_list):
            start = start_t.get_seconds()
            end = end_t.get_seconds()
            midpoint = start + ((end - start) / 2.0)
            segments.append(
                SceneSegment(
                    scene_id=f"scene_{idx:04d}",
                    scene_start=round(start, 3),
                    scene_end=round(end, 3),
                    representative_ts=round(midpoint, 3),
                )
            )
        return segments or _fallback_segments(video_path)
    except Exception:
        return _fallback_segments(video_path)


def assign_representative_frames(segments: list[SceneSegment], frame_records: list[dict]) -> list[SceneSegment]:
    if not frame_records:
        return segments
    for segment in segments:
        nearest = min(frame_records, key=lambda row: abs(float(row["timestamp_sec"]) - segment.representative_ts))
        segment.representative_frame = str(Path(nearest["frame_path"]))
    return segments
