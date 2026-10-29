"""FISSION attacks container entrypoint.

Default behaviour: if ATTACK_CMD is set, run that shell command.
Otherwise run a multi-model transfer-benchmark scaffold that logs to
MLflow (using any on-disk adversarial_dataset/manifest.json) and writes
a security report under /artifacts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.attacks.entrypoint")


def _apply_env_overrides() -> None:
    if h := os.getenv("REDIS_HOST"):
        os.environ.setdefault("FISSION_REDIS_HOST", h)
    if p := os.getenv("REDIS_PORT"):
        os.environ.setdefault("FISSION_REDIS_PORT", p)
    ollama_host = os.getenv("OLLAMA_HOST", "fission-ollama")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    os.environ.setdefault(
        "FISSION_OLLAMA_HOST", f"http://{ollama_host}:{ollama_port}"
    )
    if m := os.getenv("ACTIVE_MODEL"):
        os.environ.setdefault("FISSION_LLAVA_MODEL", m)
    if uri := os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = uri


def main() -> int:
    _apply_env_overrides()
    sys.path.insert(0, "/app")

    custom = os.getenv("ATTACK_CMD")
    if custom:
        log.info("Running ATTACK_CMD: %s", custom)
        return subprocess.call(custom, shell=True)

    # Default: generate a security report from any existing manifest and
    # log summary metrics to MLflow. Full live multi-model runs are
    # triggered via ATTACK_CMD to avoid unexpected multi-hour jobs.
    log.info("No ATTACK_CMD set — running report + MLflow summary path")
    from evaluation.report_generator import generate_report_from_manifest
    from evaluation.mlflow_logger import log_manifest_to_mlflow

    manifest = os.getenv(
        "MANIFEST_PATH", "/app/adversarial_dataset/manifest.json"
    )
    out_dir = os.getenv("ARTIFACT_DIR", "/artifacts")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(manifest):
        report_path = generate_report_from_manifest(
            manifest_path=manifest,
            output_dir=out_dir,
            title="FISSION Cross-Architecture Security Benchmark",
        )
        log.info("Report written: %s", report_path)
        run_id = log_manifest_to_mlflow(
            manifest_path=manifest,
            experiment_name=os.getenv("MLFLOW_EXPERIMENT", "fission-r1"),
            run_name=os.getenv("MLFLOW_RUN_NAME", "container-default"),
        )
        log.info("MLflow run id: %s", run_id)
    else:
        log.warning(
            "No manifest at %s — nothing to report. "
            "Set ATTACK_CMD to run generate_adversarial_dataset.py.",
            manifest,
        )

    # Keep container alive if requested (compose restart policies handle
    # one-shot jobs; set KEEP_ALIVE=1 for interactive debugging).
    if os.getenv("KEEP_ALIVE", "0") == "1":
        log.info("KEEP_ALIVE=1 — sleeping forever")
        import time
        while True:
            time.sleep(3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
