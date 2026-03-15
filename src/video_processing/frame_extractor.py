from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.file_utils import ensure_dir


def extract_frame_at_timestamp(video_path: str, timestamp_sec: float, output_path: str) -> str:
    out = Path(output_path)
    ensure_dir(out.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(timestamp_sec, 0.0):.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out)


def extract_scene_frames(video_path: str, scenes: list, output_dir: str, max_scenes: int | None = None) -> list[dict]:
    out_dir = ensure_dir(output_dir)
    frames: list[dict] = []
    selected = scenes[:max_scenes] if max_scenes else scenes
    for scene in selected:
        frame_path = out_dir / f"{scene.scene_id}.jpg"
        extracted = extract_frame_at_timestamp(video_path, scene.representative_ts, str(frame_path))
        scene.representative_frame = extracted
        frames.append(
            {
                "scene_id": scene.scene_id,
                "frame_path": extracted,
                "timestamp_sec": scene.representative_ts,
                "scene_start": scene.scene_start,
                "scene_end": scene.scene_end,
            }
        )
    return frames


def extract_clip(video_path: str, center_timestamp: float, output_path: str, pre_sec: float = 6.0, post_sec: float = 8.0) -> str:
    start = max(0.0, center_timestamp - pre_sec)
    duration = pre_sec + post_sec
    out = Path(output_path)
    ensure_dir(out.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out)
