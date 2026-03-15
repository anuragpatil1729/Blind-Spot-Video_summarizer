from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline_runner import PipelineRunner


def score_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"

st.set_page_config(page_title="Blind Spot Video Summarizer", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .status-card {
        border: 1px solid #2f2f2f;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        background: rgba(50, 50, 50, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 Blind Spot Video Summarizer")
st.caption("Build a searchable scene index from driving footage and quickly inspect relevant moments.")

runner = PipelineRunner()

if "build_result" not in st.session_state:
    st.session_state.build_result = None

paths = runner.config["paths"]
default_video_path = paths["video_path"]

with st.sidebar:
    st.header("⚙️ Build settings")
    uploaded = st.file_uploader("Optional: upload a video", type=["mp4", "mov", "avi", "mkv"])
    video_path = st.text_input("Video path", value=default_video_path)
    fps = st.slider("Frame sampling FPS", min_value=0.1, max_value=3.0, value=float(runner.config["sampling"]["fps"]), step=0.1)
    max_frames = st.slider("Maximum frames", min_value=5, max_value=300, value=int(runner.config["sampling"]["max_frames"]), step=5)

    st.subheader("Ollama (optional)")
    use_ollama = st.toggle("Enable Ollama vision model", value=bool(runner.config["ollama"]["enabled"]))
    ollama_base_url = st.text_input("Ollama base URL", value=str(runner.config["ollama"]["base_url"]))
    ollama_model = st.text_input("Ollama model", value=str(runner.config["ollama"]["model"]))

if uploaded is not None:
    upload_path = Path("data/videos") / uploaded.name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded.read())
    video_path = str(upload_path)
    st.success(f"Uploaded video saved to: {video_path}")

left, right = st.columns([1, 1])
with left:
    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Processing frames, generating captions, and building index..."):
            try:
                st.session_state.build_result = runner.build(
                    video_path=video_path,
                    fps=fps,
                    max_frames=max_frames,
                    use_ollama=use_ollama,
                    ollama_base_url=ollama_base_url,
                    ollama_model=ollama_model,
                )
                st.success("Index built successfully.")
            except Exception as exc:
                st.session_state.build_result = None
                st.error(f"Build failed: {exc}")

with right:
    build_result = st.session_state.build_result
    if build_result:
        st.markdown('<div class="status-card">✅ Build complete and ready for search.</div>', unsafe_allow_html=True)
        metrics = st.columns(3)
        metrics[0].metric("Frames", build_result["frames"])
        metrics[1].metric("Captions", build_result["captions"])
        metrics[2].metric("FPS", f"{build_result['fps']:.1f}")
    else:
        st.markdown('<div class="status-card">ℹ️ Build index to enable semantic search.</div>', unsafe_allow_html=True)

st.divider()

search_col, options_col = st.columns([3, 1])
with search_col:
    query = st.text_input("Search scene description", "car near right lane")
with options_col:
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=int(runner.config["search"]["top_k"]))
    min_score = st.slider(
        "Min similarity",
        min_value=0.0,
        max_value=1.0,
        value=float(runner.config["search"].get("min_score", 0.0)),
        step=0.01,
    )

if st.button("🔎 Search", use_container_width=True):
    try:
        results = runner.query(query, top_k=int(top_k), min_score=float(min_score))
        if not results:
            st.warning("No results found. Build the index first or use a different query.")
        else:
            st.subheader("Search results")
            for i, row in enumerate(results, 1):
                with st.container(border=True):
                    score = float(row["score"])
                    st.markdown(
                        f"**#{i} • t={row['timestamp_sec']}s • "
                        f"score={score:.3f} ({score_label(score)} match)**"
                    )
                    r1, r2 = st.columns([2, 1])
                    with r1:
                        st.write(row["text"])
                    with r2:
                        st.image(row["frame_path"], use_container_width=True)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
