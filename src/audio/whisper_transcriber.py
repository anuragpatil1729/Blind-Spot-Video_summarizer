from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, video_path: str) -> list[TranscriptSegment]:
        model = self._load()
        result = model.transcribe(video_path, verbose=False)
        segments = result.get("segments", [])
        return [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in segments
            if str(seg.get("text", "")).strip()
        ]
