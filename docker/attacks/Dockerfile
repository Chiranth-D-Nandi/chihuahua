# FISSION — Attacks / red-team runner (GPU-capable)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch is already in the base image; install the rest without re-pulling torch
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install \
        numpy scipy Pillow requests httpx redis \
        transformers sentence-transformers scikit-image \
        mlflow jinja2 matplotlib reportlab rich pytest \
        fastapi uvicorn python-multipart pydantic \
 && pip install git+https://github.com/openai/CLIP.git || true

COPY config/       /app/config/
COPY shared/       /app/shared/
COPY orchestrator/ /app/orchestrator/
COPY defenses/     /app/defenses/
COPY executor/     /app/executor/
COPY attacks/      /app/attacks/
COPY evaluation/   /app/evaluation/
COPY adversarial_dataset/ /app/adversarial_dataset/
COPY docker/attacks/entrypoint.py /app/entrypoint.py

VOLUME ["/artifacts"]

CMD ["python", "/app/entrypoint.py"]
