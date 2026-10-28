# Tujya API: FastAPI + ONNX Runtime (CPU). No model, data or secret is baked in: the
# models (/models), config (/config) and data (/data) are mounted by docker-compose.yml.
#   docker build -t tujya-backend backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# LightGBM (bonus toxicity tool) needs the OpenMP runtime, absent from the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

# The app's project root is the parent of /app, i.e. "/": the default paths of
# app/config.py become /models, /config/..., /data/... which are the compose mounts.
RUN useradd --create-home --uid 1000 tujya
USER tujya

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
