# SynergyRx data pipeline. The build context is the repository root (the pipeline reads the
# shared config/ folder); Dockerfile.dockerignore next to this file keeps everything else out.
#   docker build -f data_pipeline/Dockerfile -t synergyrx-data .              # from the repository root
#   docker build -f data_pipeline/Dockerfile -t synergyrx-data --build-arg INSTALL_TDC=1 .   # adds PyTDC (large)
# Python 3.12 is used here because PyTDC does not install on 3.13+.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libxrender / libxext are runtime libraries RDKit's wheel links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Same layout as the repository: /app is the project root (data/ and config/), the pipeline
# runs from /app/data_pipeline where its config.yaml says `project_root: ..`.
WORKDIR /app/data_pipeline
COPY data_pipeline/pyproject.toml data_pipeline/README.md ./
COPY data_pipeline/src ./src
ARG INSTALL_TDC=0
RUN if [ "$INSTALL_TDC" = "1" ]; then pip install ".[tdc]"; else pip install .; fi

COPY data_pipeline/config.yaml ./
COPY data_pipeline/docs ./docs
COPY config /app/config

# Run unprivileged. Everything the pipeline writes goes under /app/data, which is meant
# to be mounted from the host.
RUN useradd --create-home --uid 1000 synergyrx \
    && mkdir -p /app/data \
    && chown -R synergyrx:synergyrx /app
USER synergyrx
VOLUME ["/app/data"]

# Secrets are passed at run time (--env-file .env); they are never baked into the image.
ENTRYPOINT ["synergyrx"]
CMD ["--help"]
