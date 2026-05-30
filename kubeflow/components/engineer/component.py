"""Kubeflow component: PySpark feature engineering."""

from kfp import dsl

PIPELINE_IMAGE = "ghcr.io/dhruvparmar051/fraud-ml-pipeline:latest"


@dsl.component(base_image=PIPELINE_IMAGE)
def engineer_features(input_path: str, output_dir: str) -> str:
    """Run the Spark ETL; return the produced Parquet path."""
    import subprocess

    subprocess.run(
        ["python", "-m", "src.data.etl", "--input", input_path, "--output", output_dir],
        check=True,
    )
    return f"{output_dir.rstrip('/')}/features.parquet"
