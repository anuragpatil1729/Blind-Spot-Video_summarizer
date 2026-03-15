from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.file_utils import ensure_dir


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
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return max(float(out or 0.0), 0.0)
    except Exception:
        return 0.0


def _clamp_timestamp(timestamp_sec: float, video_duration_sec: float) -> float:
    ts = max(timestamp_sec, 0.0)
    if video_duration_sec <= 0:
        return ts
    # keep the seek target slightly inside the file to avoid empty-frame ffmpeg failures.
    return min(ts, max(video_duration_sec - 0.1, 0.0))


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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # Retry a little earlier in case the requested timestamp is too close to EOF.
        retry_ts = max(timestamp_sec - 0.25, 0.0)
        retry_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{retry_ts:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        try:
            subprocess.run(retry_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"Failed to extract frame at {timestamp_sec:.3f}s from {video_path}. ffmpeg stderr: {stderr}"
            ) from exc
    return str(out)


def extract_scene_frames(video_path: str, scenes: list, output_dir: str, max_scenes: int | None = None) -> list[dict]:
    out_dir = ensure_dir(output_dir)
    video_duration_sec = _probe_duration(video_path)
    frames: list[dict] = []
    selected = scenes[:max_scenes] if max_scenes else scenes
    for scene in selected:
        clamped_ts = _clamp_timestamp(scene.representative_ts, video_duration_sec)
        frame_path = out_dir / f"{scene.scene_id}.jpg"
        extracted = extract_frame_at_timestamp(video_path, clamped_ts, str(frame_path))
        scene.representative_frame = extracted
        scene.representative_ts = clamped_ts
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
