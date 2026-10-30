# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --default-timeout=120 --retries 5 --no-cache-dir hatchling \
    && pip install --default-timeout=120 --retries 5 --no-cache-dir --no-build-isolation --prefix /install packaging pyyaml .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 sidecardb

COPY --from=build /install /usr/local

USER sidecardb
WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

CMD ["sidecardb", "serve"]
