# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# One step and one environment: pip builds the wheel in an isolated env and
# resolves every dependency here, where nothing is pre-installed, then lands
# all of it under /install. Splitting the wheel build into the system env
# makes pip mark transitive deps of the build tooling (e.g. packaging) as
# already satisfied and silently omit them from /install.
RUN pip install --no-cache-dir --prefix /install .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 askdb

COPY --from=build /install /usr/local

USER askdb
WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

CMD ["askdb", "serve"]
