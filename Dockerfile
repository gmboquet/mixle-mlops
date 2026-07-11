# mixle model server image (standalone mixle-mlops package).
#
#   docker build -t <registry>/mixle-mlops:latest .
#
# Installs mixle (the core model library) + this serving package. The image bundles no model --
# the model is loaded at runtime from the registry volume (seed it with `mixle-seed`).

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mixle_mlops ./mixle_mlops

# mixle is published on PyPI (see pyproject.toml's `mixle>=0.6.1`); `pip install .` resolves it there,
# no separate git install needed.
RUN pip install --no-cache-dir .

ENV MIXLE_REGISTRY_ROOT=/models \
    MIXLE_MODEL_NAME=model \
    MIXLE_MODEL_ALIAS=production \
    MIXLE_ACTIVITY_LOG=/dev/stdout

EXPOSE 8000

# Console script from pyproject [project.scripts] -> mixle_mlops.cli:serve -> uvicorn on
# mixle_mlops.gateway.app:app (the full platform gateway). NOTE: this Dockerfile is kept only for the
# vestigial single-model mixle_mlops/app.py server whose env vars are set above; that server is not what
# `mixle-serve` runs. For the actual platform gateway image, build deploy/Dockerfile instead (see
# deploy/docker-compose.yml / deploy/README.md).
CMD ["mixle-serve"]
