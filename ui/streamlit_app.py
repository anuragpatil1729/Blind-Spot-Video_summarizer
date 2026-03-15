from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline_runner import PipelineRunner

st.set_page_config(page_title="Blind Spot Video Summarizer", layout="wide")
st.title("Blind Spot Video Summarizer")

runner = PipelineRunner()

if st.button("Build Index"):
    with st.spinner("Processing video..."):
        st.json(runner.build())

query = st.text_input("Search scene description", "car near right lane")
if st.button("Search"):
    results = runner.query(query)
    for row in results:
        st.markdown(f"**t={row['timestamp_sec']}s | score={row['score']}**")
        st.write(row["text"])
        st.image(row["frame_path"], width=350)
