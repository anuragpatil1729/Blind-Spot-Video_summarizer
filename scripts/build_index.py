from __future__ import annotations

import argparse

from src.pipeline.pipeline_runner import VideoIntelligencePipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a video and run a semantic query.")
    parser.add_argument("--video", default=None)
    parser.add_argument("--query", default="when does the car appear")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = VideoIntelligencePipeline()
    manifest = pipeline.index_video(args.video)
    print(f"Indexed {manifest['indexed_moments']} moments across {manifest['scene_count']} scenes")

    results = pipeline.search(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query}\n")
    for i, row in enumerate(results, start=1):
        meta = row["metadata"]
        print(f"#{i} t={meta['timestamp_sec']:.2f}s | rerank={row.get('rerank_score', 0.0):.4f}")
        print(f" caption: {meta.get('caption', '')}")
        print(f" frame: {meta.get('frame_path', '')}\n")
