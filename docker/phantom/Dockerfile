# FISSION — PHANTOM honeypot (B5) service
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

COPY config/   /app/config/
COPY shared/   /app/shared/
COPY defenses/ /app/defenses/
COPY honeypot/ /app/honeypot/
COPY docker/phantom/entrypoint.py /app/entrypoint.py

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import redis,os; r=redis.Redis(host=os.getenv('REDIS_HOST','fission-redis'),port=int(os.getenv('REDIS_PORT',6379))); r.ping()" || exit 1

CMD ["python", "/app/entrypoint.py"]
