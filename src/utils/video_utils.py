from __future__ import annotations

import json
import tempfile
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


def build_combined_clip(
    video_path: str,
    timestamps: list[float],
    output_path: str,
    clip_duration_sec: float = 2.0,
) -> str | None:
    if not timestamps:
        return None

    source = Path(video_path)
    if not source.exists():
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    unique_timestamps: list[float] = []
    for ts in sorted(float(t) for t in timestamps):
        if not unique_timestamps or abs(ts - unique_timestamps[-1]) > 0.01:
            unique_timestamps.append(max(ts, 0.0))

    if not unique_timestamps:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="clip_concat_"))
    segment_files: list[Path] = []

    try:
        for idx, ts in enumerate(unique_timestamps, 1):
            segment_path = tmp_dir / f"segment_{idx:03d}.mp4"
            segment_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(source),
                "-t",
                f"{clip_duration_sec:.2f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-an",
                str(segment_path),
            ]
            subprocess.run(segment_cmd, check=True, capture_output=True, text=True)
            segment_files.append(segment_path)

        concat_list = tmp_dir / "concat_list.txt"
        concat_lines = [f"file '{segment.resolve()}'" for segment in segment_files]
        concat_list.write_text("\n".join(concat_lines), encoding="utf-8")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output),
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        return str(output)
    except Exception:
        return None
