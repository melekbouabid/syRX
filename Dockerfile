# SynergyRx data pipeline.
#   docker build -t synergyrx-data .
#   docker build -t synergyrx-data --build-arg INSTALL_TDC=1 .   # adds PyTDC (large)
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

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
ARG INSTALL_TDC=0
RUN if [ "$INSTALL_TDC" = "1" ]; then pip install ".[tdc]"; else pip install .; fi

COPY config.yaml ./
COPY config ./config
COPY docs ./docs

# Run unprivileged. Everything the pipeline writes goes under /app/data, which is meant
# to be mounted from the host.
RUN useradd --create-home --uid 1000 synergyrx \
    && mkdir -p data \
    && chown -R synergyrx:synergyrx /app
USER synergyrx
VOLUME ["/app/data"]

# Secrets are passed at run time (--env-file .env); they are never baked into the image.
ENTRYPOINT ["synergyrx"]
CMD ["--help"]
