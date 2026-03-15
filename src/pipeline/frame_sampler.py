from __future__ import annotations

from src.utils.video_utils import extract_frames


def sample_frames(video_path: str, frames_dir: str, fps: float, max_frames: int):
    return extract_frames(video_path, frames_dir, fps=fps, max_frames=max_frames)
