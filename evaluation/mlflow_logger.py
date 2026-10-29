"""
FISSION — evaluation/mlflow_logger.py
=======================================
Thin MLflow client wrapper used by attack/benchmark scripts and the
reporting layer.

All functions degrade gracefully when MLflow is unreachable or not
installed: they log a warning and return None instead of crashing a
benchmark run. This keeps local `python attacks/...` workflows working
without a live tracking server.

Environment:
    MLFLOW_TRACKING_URI  (default: http://localhost:5000)

Usage:
    from evaluation.mlflow_logger import mlflow_run, log_manifest_to_mlflow

    with mlflow_run("fission-r1", run_name="llava-fgsm") as run:
        mlflow.log_metric("isr", 0.56)
        ...

    log_manifest_to_mlflow("adversarial_dataset/manifest.json")
"""

from __future__ import annotations

import json
import logging
import os
import contextlib
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger("fission.evaluation.mlflow")

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _import_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        log.warning(
            "mlflow is not installed — metrics will not be tracked. "
            "Install with: pip install mlflow"
        )
        return None


def _configure(tracking_uri: Optional[str] = None):
    mlflow = _import_mlflow()
    if mlflow is None:
        return None
    uri = tracking_uri or DEFAULT_TRACKING_URI
    try:
        mlflow.set_tracking_uri(uri)
        return mlflow
    except Exception as exc:
        log.warning("Could not set MLflow tracking URI %s: %s", uri, exc)
        return None


@contextlib.contextmanager
def mlflow_run(
    experiment_name: str = "fission",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
    nested: bool = False,
) -> Iterator[Any]:
    """
    Context manager that starts an MLflow run, or a no-op if MLflow is
    unavailable. Yields the active run object (or None).
    """
    mlflow = _configure(tracking_uri)
    if mlflow is None:
        yield None
        return

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        log.warning("set_experiment(%s) failed: %s — continuing without MLflow", experiment_name, exc)
        yield None
        return

    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if tags:
                try:
                    mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
                except Exception as exc:
                    log.debug("set_tags failed: %s", exc)
            yield run
    except Exception as exc:
        log.warning("mlflow.start_run failed: %s — yielding no-op", exc)
        yield None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a dict of numeric metrics to the active run (no-op if none)."""
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        if mlflow.active_run() is None:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if clean:
            mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        log.debug("log_metrics failed: %s", exc)


def log_params(params: Dict[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        if mlflow.active_run() is None:
            return
        # MLflow params must be strings / numbers; coerce everything.
        clean = {str(k): str(v)[:250] for k, v in params.items() if v is not None}
        if clean:
            mlflow.log_params(clean)
    except Exception as exc:
        log.debug("log_params failed: %s", exc)


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        if mlflow.active_run() is None:
            return
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:
        log.debug("log_artifact(%s) failed: %s", path, exc)


def _aggregate_manifest(entries: List[dict]) -> Dict[str, Any]:
    """Collapse a list of per-image manifest rows into summary metrics."""
    by_key = defaultdict(list)
    models = set()
    attacks = set()
    for e in entries:
        if e.get("isr") is None:
            continue
        key = (e.get("attack", "?"), e.get("epsilon_int", e.get("epsilon")), e.get("model", "unknown"))
        by_key[key].append(e)
        models.add(e.get("model", "unknown"))
        attacks.add(e.get("attack", "?"))

    summary_metrics: Dict[str, float] = {}
    group_rows = []
    for (attack, eps, model), group in sorted(by_key.items(), key=lambda x: (str(x[0][2]), str(x[0][0]), x[0][1] or 0)):
        mean_isr = sum(g["isr"] for g in group) / len(group)
        mean_ssim = sum(g["ssim"] for g in group if g.get("ssim") is not None) / max(
            1, sum(1 for g in group if g.get("ssim") is not None)
        )
        sds_vals = [g["sds"] for g in group if g.get("sds") is not None]
        mean_sds = sum(sds_vals) / len(sds_vals) if sds_vals else None
        sbi = mean_isr >= 0.5 and mean_ssim >= 0.95
        # Metric keys must be MLflow-safe (no spaces / slashes)
        safe_model = str(model).replace(":", "_").replace("/", "_")
        prefix = f"{safe_model}_{attack}_eps{eps}"
        summary_metrics[f"{prefix}_isr"] = float(mean_isr)
        summary_metrics[f"{prefix}_ssim"] = float(mean_ssim)
        if mean_sds is not None:
            summary_metrics[f"{prefix}_sds"] = float(mean_sds)
        summary_metrics[f"{prefix}_sbi"] = 1.0 if sbi else 0.0
        summary_metrics[f"{prefix}_n"] = float(len(group))
        group_rows.append({
            "model": model,
            "attack": attack,
            "epsilon": eps,
            "isr": round(mean_isr, 4),
            "ssim": round(mean_ssim, 4),
            "sds": round(mean_sds, 4) if mean_sds is not None else None,
            "sbi": sbi,
            "n": len(group),
        })

    # Overall means
    all_isr = [e["isr"] for e in entries if e.get("isr") is not None]
    all_ssim = [e["ssim"] for e in entries if e.get("ssim") is not None]
    if all_isr:
        summary_metrics["overall_mean_isr"] = float(sum(all_isr) / len(all_isr))
    if all_ssim:
        summary_metrics["overall_mean_ssim"] = float(sum(all_ssim) / len(all_ssim))
    summary_metrics["n_entries"] = float(len(entries))
    summary_metrics["n_models"] = float(len(models))

    return {
        "metrics": summary_metrics,
        "groups": group_rows,
        "models": sorted(models),
        "attacks": sorted(attacks),
    }


def log_manifest_to_mlflow(
    manifest_path: str,
    experiment_name: str = "fission-r1",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Read a generate_adversarial_dataset.py manifest.json, aggregate ISR/SSIM/SDS
    per (model, attack, epsilon), and log everything to one MLflow run.

    Returns the run_id string, or None if MLflow was unavailable.
    """
    if not os.path.exists(manifest_path):
        log.error("Manifest not found: %s", manifest_path)
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        log.error("Manifest must be a JSON list, got %s", type(entries))
        return None

    agg = _aggregate_manifest(entries)
    params = {
        "manifest_path": manifest_path,
        "n_entries": len(entries),
        "models": ",".join(agg["models"]),
        "attacks": ",".join(agg["attacks"]),
    }
    if extra_params:
        params.update(extra_params)

    run_name = run_name or f"manifest-{'-'.join(agg['models'][:2]) or 'empty'}"

    with mlflow_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags or {"source": "manifest", "project": "FISSION"},
        tracking_uri=tracking_uri,
    ) as run:
        if run is None:
            log.warning("MLflow unavailable — printed aggregate metrics instead")
            log.info("Aggregate: %s", json.dumps(agg["metrics"], indent=2)[:500])
            return None
        log_params(params)
        log_metrics(agg["metrics"])
        log_artifact(manifest_path, artifact_path="manifests")
        run_id = run.info.run_id
        log.info(
            "Logged %d metrics from %s to MLflow run %s",
            len(agg["metrics"]),
            manifest_path,
            run_id,
        )
        return run_id


def log_benchmark_result(
    module: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "fission-benchmarks",
    run_name: Optional[str] = None,
) -> Optional[str]:
    """Log a single dashboard/module benchmark result (R3, B4, B5, ...)."""
    with mlflow_run(
        experiment_name=experiment_name,
        run_name=run_name or module,
        tags={"module": module, "project": "FISSION"},
    ) as run:
        if run is None:
            return None
        if params:
            log_params(params)
        log_params({"module": module})
        log_metrics(metrics)
        return run.info.run_id
