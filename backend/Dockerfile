# Tujya API: FastAPI + ONNX Runtime (CPU). No model, data or secret is baked in: the
# models, config and data are mounted by docker-compose.yml.
# The build context is the repository root, because the API imports the module packages
# (src/rag, src/agent, ...); Dockerfile.dockerignore next to this file is the allow-list.
#   docker build -f backend/Dockerfile -t tujya-backend .      # from the repository root
#
# Two stages: `build` installs the Python dependencies into a virtual environment, `runtime`
# copies only that environment and the code, so pip, its cache and the build layers stay out
# of the final image.

FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# LightGBM (bonus toxicity tool) needs the OpenMP runtime, absent from the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# Same layout as the repository under /tujya: the backend folder is /tujya/backend, so
# the default paths of app/config.py are /tujya/models, /tujya/config/..., /tujya/data/...,
# which are the compose mounts. PYTHONPATH plays the role of the editable install
# (pyproject.toml): package `app` and the repository root.
WORKDIR /tujya/backend
ENV PYTHONPATH=/tujya:/tujya/backend

COPY src /tujya/src
COPY backend/app ./app

# /tujya/cache/fastembed: fastembed models (BM25, reranker), a compose volume owned by the user.
RUN useradd --create-home --uid 1000 tujya \
    && mkdir -p /tujya/cache/fastembed && chown tujya /tujya/cache/fastembed
USER tujya

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
