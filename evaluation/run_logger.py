"""
FISSION — evaluation/run_logger.py
====================================
Structured Run Logger for FISSION Pipeline Executions.

Records each and every pipeline run (model, prompt, attack, verdict, layer breakdowns,
and defense metrics) into `logs/pipeline_runs.jsonl` and `logs/pipeline_runs.log`.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("fission.evaluation.run_logger")

FISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(FISSION_ROOT, "logs")
JSONL_PATH = os.path.join(LOGS_DIR, "pipeline_runs.jsonl")
TEXT_LOG_PATH = os.path.join(LOGS_DIR, "pipeline_runs.log")


def _ensure_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def log_pipeline_run(
    session_id: str,
    task: str,
    model: str,
    attack_type: str,
    final_status: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    tool_result: Optional[str] = None,
    blocked_at: Optional[str] = None,
    block_reason: Optional[str] = None,
    layers: Optional[List[dict]] = None,
    total_duration_ms: float = 0.0,
    metrics: Optional[dict] = None,
) -> dict:
    """
    Log a completed pipeline run into structured JSONL and formatted text log files.
    """
    _ensure_dir()
    now_utc = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": now_utc,
        "session_id": session_id,
        "model": model,
        "task": task,
        "attack_type": attack_type or "none",
        "final_status": final_status,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result,
        "blocked_at": blocked_at,
        "block_reason": block_reason,
        "total_duration_ms": total_duration_ms,
        "layers": layers or [],
        "metrics": metrics or {},
    }

    # 1. Append to JSONL file
    try:
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"Could not append run to {JSONL_PATH}: {e}")

    # 2. Append to human-readable log file
    try:
        status_symbol = "✓ PASS" if final_status == "CLEAN" else f"❌ {final_status}"
        text_line = (
            f"[{now_utc}] [{model}] [{attack_type}] "
            f"Task: \"{task[:60]}\" -> Status: {status_symbol} "
            f"({total_duration_ms:.1f}ms)"
        )
        if blocked_at:
            text_line += f" | Blocked: {blocked_at} — {block_reason}"
        if tool_name:
            text_line += f" | Tool: {tool_name}"

        with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text_line + "\n")
    except Exception as e:
        log.warning(f"Could not append run to {TEXT_LOG_PATH}: {e}")

    log.info(f"Pipeline run logged [{session_id}] model={model} status={final_status}")
    return record


def get_recent_runs(limit: int = 50) -> List[dict]:
    """Return the most recent run records from logs/pipeline_runs.jsonl."""
    if not os.path.exists(JSONL_PATH):
        return []

    runs = []
    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"Failed to read {JSONL_PATH}: {e}")

    return runs[-limit:]
