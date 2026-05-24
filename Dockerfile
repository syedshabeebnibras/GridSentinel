FROM python:3.13-slim

WORKDIR /app

# Build deps for hdbscan and friends; remove after install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[ml]"

# Pre-stage the simulator data + a trained model — this is for the demo image.
# In production you'd mount /app/data and /app/mlruns as volumes.
COPY data/ ./data/
COPY mlruns/ ./mlruns/

EXPOSE 8080
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "gridsentinel.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
