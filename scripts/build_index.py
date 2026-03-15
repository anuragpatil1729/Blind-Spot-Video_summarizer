import argparse
from textwrap import shorten

from src.pipeline.pipeline_runner import PipelineRunner


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _print_results(results: list[dict]):
    if not results:
        print("No matching scenes found. Try a different query or lower --min-score.")
        return

    print("\nTop matching scenes")
    print("-" * 90)
    for i, row in enumerate(results, start=1):
        score = float(row["score"])
        summary = shorten(str(row["text"]), width=90, placeholder="...")
        print(
            f"#{i:<2} t={row['timestamp_sec']:>6}s | "
            f"score={score:.3f} ({_score_label(score)} match)\n"
            f"    {summary}\n"
            f"    frame: {row['frame_path']}"
        )
        print("-" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="car on the side")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    runner = PipelineRunner()
    build_result = runner.build()
    print("Build complete")
    print(
        f"Frames: {build_result['frames']} | "
        f"Captions: {build_result['captions']} | "
        f"FPS: {build_result['fps']} | "
        f"Video: {build_result['video_path']}"
    )
    print(f"\nQuery: {args.query}")
    results = runner.query(args.query, min_score=args.min_score)
    _print_results(results)
