# Blind Spot Video Summarizer

Simple local pipeline that:
1. Extracts frames from `data/videos/input.mp4`
2. Creates frame captions (Ollama optional; fallback heuristic enabled)
3. Builds a text embedding index
4. Supports semantic search over scene descriptions

## Setup

```bash
./setup.sh
```

## Run

```bash
python -m scripts.build_index --query "car in adjacent lane"
```

Or launch UI:

```bash
streamlit run ui/streamlit_app.py
```
