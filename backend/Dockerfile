# Tujya API: FastAPI + ONNX Runtime (CPU). No model, data or secret is baked in: the
# models, config and data are mounted by docker-compose.yml.
# The build context is the repository root, because the API imports the module packages
# (src/rag, src/agent, ...); Dockerfile.dockerignore next to this file is the allow-list.
#   docker build -f backend/Dockerfile -t tujya-backend .      # from the repository root
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Same layout as the repository under /tujya: the backend folder is /tujya/backend, so
# the default paths of app/config.py are /tujya/models, /tujya/config/..., /tujya/data/...,
# which are the compose mounts. PYTHONPATH plays the role of the editable install
# (pyproject.toml): package `app` and the repository root.
WORKDIR /tujya/backend
ENV PYTHONPATH=/tujya:/tujya/backend
# LightGBM (bonus toxicity tool) needs the OpenMP runtime, absent from the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY module2_ai_engineering /tujya/module2_ai_engineering
COPY bonus /tujya/bonus
COPY backend/app ./app

RUN useradd --create-home --uid 1000 tujya
USER tujya

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
