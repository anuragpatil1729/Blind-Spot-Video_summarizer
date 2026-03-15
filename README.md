# Blind-Spot-Video_summarizer → Production Multimodal Video Intelligence

This repository is now a **multimodal RAG system for video moment retrieval**:

> "Google Search for Video Moments"

You can ask:
- "show the moment where a nun walks on the road"
- "when does the car appear"
- "find the scary scene"

And the system returns:
- timestamp(s)
- representative frame(s)
- semantic explanation
- playable extracted clip(s)

---

## Architecture

```text
VIDEO
 ↓
Scene Detection (PySceneDetect)
 ↓
Representative Frame Extraction (ffmpeg)
 ↓
Vision Captioning (LLaVA via Ollama)
 ↓
Audio Transcription (Whisper)
 ↓
OCR Text Extraction (EasyOCR)
 ↓
Embedding Generation (all-MiniLM-L6-v2)
 ↓
Vector Database (ChromaDB)
 ↓
Semantic Search + Query Rewriting + Cross-Encoder Rerank
 ↓
Agentic Answering (Llama3 via Ollama)
```

---

## Project Structure

```text
src/
  video_processing/
    scene_detection.py
    frame_extractor.py
  vision/
    llava_captioner.py
  audio/
    whisper_transcriber.py
  ocr/
    ocr_extractor.py
  embeddings/
    embedder.py
  search/
    vector_store.py
    semantic_search.py
    reranker.py
    query_understanding.py
  agents/
    video_agent.py
  pipeline/
    pipeline_runner.py
  ui/
    streamlit_app.py
ui/
  streamlit_app.py  # launcher wrapper
scripts/
  build_index.py
configs/
  config.yaml
```

---

## Setup

### 1) System dependencies

- Python 3.10+
- `ffmpeg` and `ffprobe`
- Ollama running locally

Install Python deps:

```bash
pip install -r requirements.txt
```

### 2) Start Ollama and pull models

```bash
ollama serve
ollama pull llava
ollama pull llama3
```

---

## Example dataset

Default video path:

```text
data/videos/input.mp4
```

Replace this file with your own source video (up to ~30 minutes recommended).

---

## Run CLI demo

```bash
python -m scripts.build_index --video data/videos/input.mp4 --query "when does the car appear" --top-k 5
```

Outputs ranked moments with timestamps + captions.

---

## Run Streamlit app

```bash
streamlit run ui/streamlit_app.py
```

UI features:
- Video upload
- Full indexing pipeline trigger
- Semantic search
- Timeline explorer
- Frame previews
- One-click clip extraction/playback
- Agentic question answering ("What happens in this video?")

---

## Production notes

- **Scene detection first** reduces redundant frames and increases semantic precision.
- **Batch embeddings** are used for indexing throughput.
- **Two-stage retrieval**: vector recall + cross-encoder rerank for stronger precision.
- **Multimodal context fusion**: caption + OCR + aligned transcript at each scene timestamp.
- **Agent mode** grounds answers in top retrieved evidence.

---

## Troubleshooting

1. **Ollama timeouts**
   - Ensure `ollama serve` is active.
   - Confirm models are pulled (`llava`, `llama3`).

2. **Whisper/EasyOCR install size**
   - These are heavy dependencies; use a virtual environment.

3. **No scenes detected**
   - Pipeline auto-falls back to fixed time chunks.

4. **Slow indexing on long videos**
   - Lower scene count via `performance.max_scenes` in `configs/config.yaml`.

---

## Demo workflow

1. Upload (or point to) a video.
2. Click **Process & Index**.
3. Query naturally: "nun walking on roadside".
4. Inspect top matches and play extracted clip.
5. Ask the AI Video Agent: "What happens in this video?"
