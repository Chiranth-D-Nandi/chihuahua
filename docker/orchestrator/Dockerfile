# FISSION — Orchestrator service
# Build from project root:
#   docker build -f docker/orchestrator/Dockerfile -t fission-orchestrator:latest .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-cpu.txt

# Copy only what the orchestrator needs (+ shared deps)
COPY config/          /app/config/
COPY shared/          /app/shared/
COPY orchestrator/    /app/orchestrator/
COPY defenses/b4_tciv /app/defenses/b4_tciv/
COPY defenses/__init__.py /app/defenses/__init__.py
COPY docker/orchestrator/entrypoint.py /app/entrypoint.py

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import redis,os; r=redis.Redis(host=os.getenv('REDIS_HOST','fission-redis'),port=int(os.getenv('REDIS_PORT',6379))); r.ping()" || exit 1

CMD ["python", "/app/entrypoint.py"]
