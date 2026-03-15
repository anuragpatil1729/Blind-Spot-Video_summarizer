# Blind Spot Video Summarizer

A local, privacy-friendly video understanding tool for any footage.

It samples frames from a video, captions each frame with an Ollama vision model, builds a lightweight semantic index, and lets you search moments using natural language in both CLI and Streamlit UI.

---

## Features

- **Frame extraction** with `ffmpeg` (automatic placeholder fallback if decoder is unavailable).
- **Scene caption generation** with Ollama multimodal captioning (default model: `llava`).
- **Semantic retrieval** over captions using text embeddings.
- **Interactive GUI** with build controls, upload support, and result previews.
- **Embedding fallback** to hash-based vectors if `sentence-transformers` cannot be loaded.

---

## Project Structure

```text
configs/               Runtime configuration
scripts/               CLI entry points
src/
  embeddings/          Text embedding + vector store
  ollama/              Ollama API client + vision captioner
  pipeline/            Build pipeline logic
  search/              Query encoding + similarity search
  utils/               Shared file/video utilities
ui/streamlit_app.py    Streamlit GUI
```

---

## Requirements

- Python 3.10+
- `ffmpeg` + `ffprobe` on PATH (recommended)
- Running Ollama server at `http://localhost:11434`
- Ollama vision model installed (default: `llava`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If you prefer, use the included setup script:

```bash
./setup.sh
```

---

## Quick Start (CLI)

> The current pipeline requires Ollama to be running. Start it first:

```bash
ollama serve
```

In another terminal, pull the default model once:

```bash
ollama pull llava
```

1. Put a video at `data/videos/input.mp4` (or pass another path in UI).
2. Build index and run a test query:

```bash
python -m scripts.build_index --query "person holding an object" --min-score 0.20
```

Expected output includes:

- Build stats (frames, captions, index path)
- Top semantic matches with score + timestamp

---

## Quick Start (GUI)

Run:

```bash
streamlit run ui/streamlit_app.py
```

Then in the app:

1. Optionally upload a video.
2. Adjust FPS / max frames / Ollama settings.
3. Click **Build / Rebuild Index**.
4. Enter a natural-language query and click **Search**.

---

## Configuration

Default runtime settings are in `configs/config.yaml`:

- `paths.video_path`: input video path
- `sampling.fps`: extracted frames per second
- `sampling.max_frames`: cap on extracted frames
- `ollama.enabled`: enable multimodal captioning
- `search.top_k`: default result count
- `search.min_score`: minimum cosine similarity threshold (filters weak matches)

---

## Troubleshooting

### 1) "It still doesn’t work" / no useful frames

- Verify the input video path exists and is readable.
- If `ffmpeg` fails, the app generates a placeholder frame so pipeline still runs.
- Try a short MP4 first to validate end-to-end behavior.

### 2) Ollama captions not appearing

- Confirm Ollama server is running (`http://localhost:11434` by default).
- Ensure the selected vision model is installed (e.g., `llava`).
- If you changed Ollama host/port, update `configs/config.yaml` (`ollama.base_url`).

### 3) Embedding model download issues

- The system automatically falls back to hash embeddings if `sentence-transformers` is unavailable.
- You can still build and search, with lower semantic quality.

---

## Notes

- This repository is designed to run locally without cloud dependencies.
- Search quality improves with clear footage and meaningful caption generation.
