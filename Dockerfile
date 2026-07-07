# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /build

# System libs needed by OpenCV, PaddleOCR, and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.9-slim AS runtime

LABEL maintainer="Banking MLOps Team"
LABEL description="Banking Document Processing & Information Extraction API"

WORKDIR /app

# Copy system libraries installed in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY configs/ ./configs/
COPY src/      ./src/
COPY params.yaml .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# MLflow tracking directory
RUN mkdir -p mlruns models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
