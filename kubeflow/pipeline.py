"""Kubeflow training pipeline.

Compile:
    python -m kubeflow.pipeline       # writes kubeflow/fraud_pipeline.yaml

Deploy:
    The pipeline references ghcr.io/<owner>/fraud-ml-pipeline:latest — a custom
    image that bundles `src/` together with PySpark + MLflow + xgboost + Feast,
    so each step can run as a self-contained container. Compile-time this is
    just a string; only the running cluster needs to pull it. minikube isn't
    installed locally, so this is validated by KFP's compiler (which produces a
    valid pipeline YAML), not by an actual cluster run.
"""

from pathlib import Path

from kfp import compiler, dsl

from kubeflow.components.engineer.component import engineer_features
from kubeflow.components.evaluate.component import evaluate_and_register
from kubeflow.components.train.component import train_model
from kubeflow.components.validate.component import validate_data


@dsl.pipeline(
    name="fraud-detection-training",
    description="Validate -> engineer -> train -> evaluate, registering fraud-detector.",
)
def fraud_pipeline(
    input_path: str = "data/raw/paysim.csv",
    output_dir: str = "data/processed/",
) -> None:
    """Wire the training DAG."""
    v = validate_data(input_path=input_path)
    f = engineer_features(input_path=v.output, output_dir=output_dir)
    t = train_model(features_path=f.output)
    evaluate_and_register(model_uri=t.output)


if __name__ == "__main__":
    out = Path(__file__).with_name("fraud_pipeline.yaml")
    compiler.Compiler().compile(pipeline_func=fraud_pipeline, package_path=str(out))
    print(f"wrote {out}")
