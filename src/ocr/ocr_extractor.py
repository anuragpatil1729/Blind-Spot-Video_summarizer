from __future__ import annotations


class OCRExtractor:
    def __init__(self, langs: list[str] | None = None):
        self.langs = langs or ["en"]
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=False)
        return self._reader

    def extract_text(self, image_path: str) -> str:
        reader = self._load()
        results = reader.readtext(image_path, detail=0)
        cleaned = [str(x).strip() for x in results if str(x).strip()]
        return " | ".join(cleaned)

    def extract_batch(self, frame_records: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for row in frame_records:
            ocr_text = self.extract_text(row["frame_path"])
            enriched.append({**row, "ocr_text": ocr_text})
        return enriched
