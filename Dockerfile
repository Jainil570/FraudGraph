# ─── FraudGraph-AI Docker Image ───────────────────────────────────────
# Multi-stage build for production FastAPI deployment
# Build:  docker build -t fraudgraph-ai .
# Run:    docker run -p 8000:8000 fraudgraph-ai
# ──────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only torch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.2.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric

# Copy source code
COPY setup.py .
COPY src/ src/
COPY api/ api/
COPY models/ models/

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
