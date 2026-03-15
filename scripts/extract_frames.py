from src.pipeline.pipeline_runner import PipelineRunner


if __name__ == "__main__":
    runner = PipelineRunner()
    result = runner.build()
    print(result)
