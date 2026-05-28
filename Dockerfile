# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/.cache/huggingface

# System deps: curl for the Streamlit healthcheck; build-essential for any
# packages that still ship sdists (rare with the wheels we use, but cheap).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image; pins us to a known-good release.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# --- Dependency layer (cached across rebuilds when uv.lock is unchanged) ---
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Application layer ---
COPY app.py ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY .streamlit/ ./.streamlit/

# Install the project itself (`exoprompt_inference` package).
RUN uv sync --frozen --no-dev

# HF Spaces persistent disk for the HF Hub cache (checkpoints downloaded by the
# app land here and survive restarts).
RUN mkdir -p /data/.cache/huggingface

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false"]