"""Kubeflow component: train the XGBoost model and log to MLflow."""

from kfp import dsl

PIPELINE_IMAGE = "ghcr.io/dhruvparmar051/fraud-ml-pipeline:latest"


@dsl.component(base_image=PIPELINE_IMAGE)
def train_model(features_path: str) -> str:
    """Run the training pipeline (logs + registers `fraud-detector` in MLflow)."""
    import subprocess

    subprocess.run(["python", "-m", "src.training.train"], check=True)
    return "models:/fraud-detector/Production"
