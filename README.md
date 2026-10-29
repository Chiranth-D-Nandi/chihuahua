# FISSION

**Cross-Modal Prompt Injection in Tool-Calling VLM Multi-Agent Systems**  
Abhyudaya M Hegde · Chiranth D Nandi · PES CCNCS Summer Internship 2026

## Deliverables (status)

| # | Abstract deliverable | Status in this repo |
|---|----------------------|---------------------|
| 1 | Web UI for model selection + automated security benchmark report | **Done** — Ollama model picker + HF Hub browser; `evaluation/` report generator (HTML/JSON/PDF); dashboard **Security Report** tab |
| 2 | Cross-architecture benchmark: LLaVA, Qwen2.5-VL, InternVL2 | **Scaffolded** — profiles + `--model` flag + `python -m evaluation.cross_model`; LLaVA numbers in `adversarial_dataset/manifest.json`. Live Qwen/InternVL2 rows require `ollama pull` + GPU host |
| 3 | Fully containerized pipeline (Docker + MLflow) | **Done** — 5 Dockerfiles under `docker/*/`, compose fixed to build from project root, MLflow client in `evaluation/mlflow_logger.py` wired into R1 + report path |
| 4 | Research paper draft | Outside this repo (Overleaf/Docs) |

Also closed from the gap report:

- **R0 / BadCLIP groundwork** — `attacks/r0_backdoor/badclip.py` builds a real triggered dataset + poisoned probe and runs B0 Neural Cleanse against it
- **Reporting layer** — was empty `evaluation/__init__.py`; now full package

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Redis + Ollama on the host
docker run -d -p 6379:6379 redis:7-alpine
ollama serve   # separate terminal
ollama pull llava

# Unit-ish smoke
python run.py --tciv-only
python -m evaluation.report_generator --manifest adversarial_dataset/manifest.json

# Dashboard
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

## Docker stack

```bash
# from project root
docker compose -f docker/docker-compose.yml --env-file docker/.env.example up --build
```

Services: Redis, MLflow (`:5000`), Ollama (`:11434`), Orchestrator, Executor, PHANTOM, Attacks, Dashboard (`:8000`).

## Cross-architecture benchmark

```bash
# After pulling models:
ollama pull llava
ollama pull qwen2.5vl:7b
ollama pull internvl2:8b

# Transfer-attack infer per model (see Session 17 note in generate_adversarial_dataset.py)
python -m evaluation.cross_model --models llava:latest qwen2.5vl:7b internvl2:8b \
  --stage infer --runs 5 --epsilons 1 2 4 8

# Or merge whatever manifests already exist:
python -m evaluation.cross_model --merge-only
```

Non-LLaVA ISR numbers are **transfer** results until per-model probes are trained.

## R0 BadCLIP + B0 validation

```bash
python -m attacks.r0_backdoor.badclip --n-images 40 --poison-rate 0.25 --epochs 200
# writes adversarial_dataset/r0_backdoor/ + attacks/r0_backdoor/poisoned_probe.pt
# and b0_detection.json
```

## MLflow

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python attacks/r1_pixel/generate_adversarial_dataset.py --stage infer --mlflow --report
# or
python -c "from evaluation.mlflow_logger import log_manifest_to_mlflow; print(log_manifest_to_mlflow('adversarial_dataset/manifest.json'))"
```

## Layout

```
attacks/          R0–R5b attack modules
defenses/         B0–B5 defense modules
orchestrator/     VLM client, prompt, parser, bus producer
executor/         B3→B4→tool dispatch
evaluation/       Report generator + MLflow client + cross-model runner
dashboard/        FastAPI + static UI
docker/           compose + 5 service Dockerfiles
config/           cfg + model_profiles.json + model_registry
```
