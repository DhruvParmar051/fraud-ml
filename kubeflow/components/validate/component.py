"""Kubeflow component: validate raw PaySim data with Great Expectations."""

from kfp import dsl

# See pipeline.py for the image rationale (custom image with src/ + deps).
PIPELINE_IMAGE = "ghcr.io/dhruvparmar051/fraud-ml-pipeline:latest"


@dsl.component(base_image=PIPELINE_IMAGE)
def validate_data(input_path: str) -> str:
    """Run the GE suite as a subprocess; pass the input path through on success."""
    import subprocess

    subprocess.run(["python", "-m", "src.data.validate", "--input", input_path], check=True)
    return input_path
