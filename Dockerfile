FROM python:3.10-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-dev --no-install-project --index-strategy unsafe-best-match

FROM base
COPY --from=dependencies /app/.venv /app/.venv
COPY . ./
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]
