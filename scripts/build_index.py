import argparse

from src.pipeline.pipeline_runner import PipelineRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="car on the side")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    runner = PipelineRunner()
    build_result = runner.build()
    print("Build:", build_result)
    print("Query:", runner.query(args.query, min_score=args.min_score))
