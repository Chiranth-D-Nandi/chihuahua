# FISSION — Dashboard (FastAPI + static UI)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-cpu.txt \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers scikit-image

COPY config/       /app/config/
COPY shared/       /app/shared/
COPY orchestrator/ /app/orchestrator/
COPY defenses/     /app/defenses/
COPY executor/     /app/executor/
COPY attacks/      /app/attacks/
COPY evaluation/   /app/evaluation/
COPY dashboard/    /app/dashboard/
COPY adversarial_dataset/ /app/adversarial_dataset/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
