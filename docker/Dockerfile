FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install ".[api]" \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/artifacts /app/configs /app/data \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["ml-platform"]
CMD ["--help"]
