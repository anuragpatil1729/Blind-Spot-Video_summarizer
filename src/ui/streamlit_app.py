from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.pipeline.pipeline_runner import VideoIntelligencePipeline

st.set_page_config(page_title="Video Moment Search", layout="wide")
st.title("🎬 Google Search for Video Moments")
st.caption("Multimodal RAG over scenes, captions, OCR, and speech.")

pipeline = VideoIntelligencePipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = None
if "results" not in st.session_state:
    st.session_state.results = []

with st.sidebar:
    st.header("Index Video")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])
    default_video = pipeline.config["paths"]["video_path"]
    video_path = st.text_input("Video path", value=default_video)
    if uploaded:
        local_path = Path("data/videos") / uploaded.name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(uploaded.read())
        video_path = str(local_path)
        st.success(f"Saved upload to {video_path}")
    if st.button("🚀 Process & Index", use_container_width=True):
        with st.spinner("Running scene detection, LLaVA, Whisper, OCR, embeddings, and indexing..."):
            st.session_state.indexed = pipeline.index_video(video_path)

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("Search moments", placeholder="show the moment where a nun walks on the road")
with col2:
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

if st.button("🔎 Search", use_container_width=True) and query:
    st.session_state.results = pipeline.search(query, top_k=top_k)

if st.session_state.indexed:
    st.subheader("AI Timeline")
    timeline = st.session_state.indexed.get("timeline", [])
    for item in timeline[:50]:
        st.write(f"**{item['timestamp']}** — {item['event']}")

if st.session_state.results:
    st.subheader("Results")
    active_video = st.session_state.indexed.get("video_path") if st.session_state.indexed else video_path
    for i, row in enumerate(st.session_state.results, start=1):
        meta = row["metadata"]
        with st.container(border=True):
            st.markdown(f"**#{i}** @ `{meta['timestamp_sec']:.2f}s`  ")
            st.write(meta.get("caption", ""))
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(meta.get("frame_path"), use_container_width=True)
            with cols[1]:
                if st.button(f"Play clip #{i}", key=f"clip-{i}"):
                    clip = pipeline.extract_result_clip(active_video, float(meta["timestamp_sec"]))
                    st.video(clip)
                st.code(row["document"][:1200])

st.subheader("AI Video Agent")
agent_q = st.text_input("Ask agent", placeholder="What happens in this video?")
if st.button("🤖 Analyze Video") and agent_q:
    response = pipeline.agent.answer(agent_q)
    st.markdown(response["answer"])
