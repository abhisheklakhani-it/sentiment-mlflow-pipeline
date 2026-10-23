# Official MLflow image + the PostgreSQL driver (psycopg 3) it needs for a postgres backend.
FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir "psycopg[binary]==3.3.6"
