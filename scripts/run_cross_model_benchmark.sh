#!/usr/bin/env bash
# FISSION — live cross-architecture benchmark (run on a machine with Ollama + GPU)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS=(llava:latest qwen2.5vl:7b internvl2:8b)
RUNS="${RUNS:-10}"
EPS="${EPS:-1 2 4 8}"
N_IMAGES="${N_IMAGES:-5}"

echo "==> Ensuring Ollama models are pulled"
for m in "${MODELS[@]}"; do
  echo "    ollama pull $m"
  ollama pull "$m" || echo "WARN: pull failed for $m — infer may fail"
done

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

echo "==> Shared adversarial image generation (once, LLaVA probe)"
python attacks/r1_pixel/generate_adversarial_dataset.py \
  --stage generate \
  --model llava:latest \
  --n-images "$N_IMAGES" \
  --epsilons $EPS \
  --attack fgsm

echo "==> Per-model inference + MLflow + report"
python -m evaluation.cross_model \
  --models "${MODELS[@]}" \
  --stage infer \
  --runs "$RUNS" \
  --epsilons $EPS \
  --n-images "$N_IMAGES" \
  --report-dir reports

echo "==> Done. Open reports/fission_cross_architecture_latest.html"
echo "    and adversarial_dataset/cross_model/BENCHMARK_STATUS.json"
