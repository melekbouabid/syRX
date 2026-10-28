# MLflow tracking server + Model Registry, with the PostgreSQL driver the official image
# lacks. Same MLflow version as vision/requirements.txt so client and server agree.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install "mlflow==3.17.2" "psycopg2-binary>=2.9"
RUN useradd --create-home --uid 1000 mlflow && mkdir /mlartifacts && chown mlflow /mlartifacts
USER mlflow
EXPOSE 5000
