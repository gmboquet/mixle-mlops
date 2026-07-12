# syntax=docker/dockerfile:1
#
# mixle-mlops platform gateway image (NOT the old single-model scoring server).
# Multi-stage, slim, non-root. Builds a wheel of the package, installs it (which pulls the pinned
# mixle core straight from PyPI per pyproject.toml's `mixle>=0.6.1`), and runs `mixle-serve`
# (uvicorn mixle_mlops.gateway.app:app on 0.0.0.0:8000).
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t mixle-mlops:latest .
#
# mixle core installs from PyPI by default (the pinned dependency above, resolved automatically by
# `pip install <wheel>`). To test against an unreleased mixle branch/commit instead, override MIXLE_GIT:
#   --build-arg MIXLE_GIT="git+https://github.com/gmboquet/mixle@<branch-or-sha>"

############################
# Stage 1 — builder
############################
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# git is needed to pip-install mixle from a git ref.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build a wheel of the platform package itself.
COPY pyproject.toml README.md ./
COPY mixle_mlops ./mixle_mlops
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels .

# Install the platform wheel (which pulls mixle from PyPI) into a self-contained venv we can copy out.
# MIXLE_GIT is empty by default; set it only to override mixle with an unreleased git ref for testing.
ARG MIXLE_GIT=""
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip \
    && if [ -n "$MIXLE_GIT" ]; then pip install "mixle @ ${MIXLE_GIT}"; fi \
    && pip install /wheels/*.whl

############################
# Stage 2 — runtime
############################
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MIXLE_DATA_DIR=/data \
    MIXLE_HOST=0.0.0.0 \
    MIXLE_PORT=8000

# Non-root user; /data is the writable volume for sqlite + the model registry.
RUN useradd --create-home --uid 10001 mixle \
    && mkdir -p /data \
    && chown -R mixle:mixle /data

COPY --from=builder /opt/venv /opt/venv

USER mixle
WORKDIR /home/mixle

EXPOSE 8000
VOLUME ["/data"]

# Container-level healthcheck hits the gateway's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# `mixle-serve` -> uvicorn mixle_mlops.gateway.app:app on 0.0.0.0:8000.
CMD ["mixle-serve"]
