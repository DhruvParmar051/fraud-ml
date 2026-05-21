FROM python:3.11-slim

# mlflow = the tracking server; boto3 = lets it write artifacts to MinIO's S3 API
RUN pip install --no-cache-dir "mlflow>=2.11,<3" boto3

EXPOSE 5000

# The actual `mlflow server ...` command is supplied by docker-compose.yml