from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.utils.file_utils import ensure_dir


def _write_placeholder(output_dir: Path):
    frame_path = output_dir / "frame_0000.jpg"
    image = Image.fromarray(np.full((360, 640, 3), 40, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "Placeholder frame: video decoder unavailable", fill=(255, 255, 255))
    image.save(frame_path)
    return [{"frame_path": str(frame_path), "timestamp_sec": 0.0}]


def _extract_with_ffmpeg(video_path: str, out_dir: Path, fps: float, max_frames: int):
    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(int(max_frames)),
        output_pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    probe = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    data = json.loads(subprocess.check_output(probe, text=True))
    rate = data["streams"][0]["r_frame_rate"]
    num, den = rate.split("/")
    native_fps = float(num) / float(den)
    step_sec = 1.0 / fps if fps > 0 else 0.0

    frames = []
    for i, frame_path in enumerate(sorted(out_dir.glob("frame_*.jpg"))[:max_frames]):
        _ = native_fps
        frames.append({"frame_path": str(frame_path), "timestamp_sec": round(i * step_sec, 2)})
    return frames


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0, max_frames: int = 100):
    out_dir = ensure_dir(output_dir)
    fps = max(float(fps), 0.01)

    try:
        return _extract_with_ffmpeg(video_path, out_dir, fps, max_frames)
    except Exception:
        return _write_placeholder(out_dir)
