"""
FISSION — evaluation package (Reporting Layer).

Lazy public API (imports resolve on first attribute access):
    from evaluation import generate_report_from_manifest, log_manifest_to_mlflow
"""

__all__ = [
    "SecurityReport",
    "generate_report_from_manifest",
    "generate_report",
    "log_manifest_to_mlflow",
    "mlflow_run",
]


def __getattr__(name: str):
    if name in {"SecurityReport", "generate_report_from_manifest", "generate_report"}:
        from evaluation import report_generator as _rg
        return getattr(_rg, name)
    if name in {"log_manifest_to_mlflow", "mlflow_run"}:
        from evaluation import mlflow_logger as _ml
        return getattr(_ml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
