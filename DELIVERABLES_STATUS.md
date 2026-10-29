# FISSION — Deliverables Closure Status

*Maps each gap from `FISSION_Deliverables_Gap_Report.md` to what now exists in-tree.
Generated 2026-07-20.*

---

## 2.1 Web UI for HuggingFace model selection + automated security benchmark report

| Claim | Before | After |
|-------|--------|-------|
| Model selection UI | Ollama tags only, hardcoded `<option>`s | Dynamic `/api/models` from `model_profiles.json` + pulled Ollama list; **HuggingFace Hub browser** via `/api/models/hf/search` with curated offline fallback |
| Automated report generation | `evaluation/` empty | Full package: `evaluation/report_generator.py` (HTML/JSON/PDF), CLI `python -m evaluation.report_generator`, dashboard **Security Report** tab, `POST /api/report/generate`, `GET /api/report/latest` |

**Honest wording for the panel:** Live inference still runs through **Ollama** (correct for the containerized VLM path). HuggingFace IDs are first-class for white-box components (R1 gradient runner, B2 proxy) and are browsable in the UI. If the guide insists the abstract must say “Ollama model selection,” that is a one-line abstract edit; the HF browser still satisfies a literal reading.

**Evidence on disk:**
- `dashboard/app.py` — `/api/models`, `/api/models/hf/search`, `/api/report/*`
- `dashboard/static/index.html` — HF browser + Report tab
- `reports/fission_security_report_latest.html` — generated from real LLaVA manifest

---

## 2.2 Cross-architecture benchmark: LLaVA, Qwen2.5-VL, InternVL2

| Claim | Before | After |
|-------|--------|-------|
| Model profiles + agnostic client | Real | Unchanged (already good) |
| Completed run on Qwen / InternVL2 | **No evidence** | **Runner + status scaffolding ready**; live numbers still need GPU/Ollama host |

**What you run on the lab machine (this environment has no GPU/Ollama):**

```bash
ollama pull llava
ollama pull qwen2.5vl:7b
ollama pull internvl2:8b

python -m evaluation.cross_model \
  --models llava:latest qwen2.5vl:7b internvl2:8b \
  --stage infer --runs 10 --epsilons 1 2 4 8
```

**Evidence on disk:**
- `evaluation/cross_model.py`
- `adversarial_dataset/cross_model/BENCHMARK_STATUS.json` — LLaVA complete; Qwen/InternVL2 `pending_live_run` with exact commands
- `reports/fission_cross_architecture_latest.html`
- R1 `--model` flag + MLflow/report flags in `generate_adversarial_dataset.py`

**Caveat (unchanged, documented):** non-LLaVA ISR is a **transfer** attack until per-model probes exist (Session 17).

---

## 2.3 Fully containerized pipeline using Docker and MLflow

| Claim | Before | After |
|-------|--------|-------|
| `docker-compose.yml` design | Real 8-service stack | Kept; build contexts fixed to project root; bridge network for plain Compose (Swarm optional) |
| 5 Dockerfiles | **Missing** | **Present** under `docker/{orchestrator,executor,phantom,attacks,dashboard}/` |
| MLflow client | **Zero `import mlflow`** | `evaluation/mlflow_logger.py`; auto-logs when `MLFLOW_TRACKING_URI` set; wired into R1, cross-model merge, report endpoint, attacks container entrypoint |

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env.example up --build
```

---

## 2.4 Research paper draft

Not in this repository (expected). No change.

---

## Related: BadCLIP / R0

| Claim | Before | After |
|-------|--------|-------|
| BadCLIP / R0 | Empty stub | `attacks/r0_backdoor/badclip.py` — patch trigger, poisoned dataset, poisoned probe training, ASR/CA eval, **B0 Neural Cleanse against the poisoned probe** |
| Dashboard | — | `POST /api/benchmark/r0-backdoor` |

```bash
python -m attacks.r0_backdoor.badclip --n-images 40 --poison-rate 0.25
```

This is a **scoped** BadCLIP-style artefact (trigger + poisoned probe), not a full dual-objective CLIP finetune — enough to stop B0 from only ever seeing synthetic fixtures.

---

## Recommended demo order for the panel

1. Open dashboard → Pipeline Runner (Ollama model picker + HF browser).
2. Benchmark Suite → R1 Manifest Summary (real LLaVA ISR/SSIM/SDS).
3. Security Report tab → Generate Report → show HTML artefact.
4. Show `docker/docker-compose.yml` + five Dockerfiles + MLflow UI on `:5000`.
5. Show `BENCHMARK_STATUS.json` + cross-architecture report; if live Qwen/InternVL2 runs finished, show those rows instead of pending.
6. Run R0 card or CLI → B0 detection against poisoned probe.

---

## Files added / materially changed

```
docker/orchestrator/Dockerfile + entrypoint.py
docker/executor/Dockerfile + entrypoint.py
docker/phantom/Dockerfile + entrypoint.py
docker/attacks/Dockerfile + entrypoint.py
docker/dashboard/Dockerfile
docker/docker-compose.yml          (rewritten build contexts)
docker/.env.example
evaluation/__init__.py
evaluation/report_generator.py
evaluation/mlflow_logger.py
evaluation/cross_model.py
attacks/r0_backdoor/badclip.py
attacks/r0_backdoor/__init__.py
attacks/r1_pixel/generate_adversarial_dataset.py  (--mlflow/--report)
dashboard/app.py                   (models, HF search, report, r0, r1)
dashboard/static/index.html        (HF browser, report tab, new cards)
requirements.txt / requirements-cpu.txt
README.md
DELIVERABLES_STATUS.md
scripts/run_cross_model_benchmark.sh
adversarial_dataset/cross_model/*
reports/*
```
