"""FISSION orchestrator container entrypoint.

Keeps the Orchestrator process alive and ready to accept tasks via the
Redis MCP bus. In the full multi-service stack the dashboard (or attacks
service) drives Orchestrator.run(); this process simply holds the
registration and can optionally drain a request stream.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.orchestrator.entrypoint")


def _apply_env_overrides() -> None:
    """Map compose-style env vars onto config.cfg before importing agents."""
    # Redis
    if h := os.getenv("REDIS_HOST"):
        os.environ.setdefault("FISSION_REDIS_HOST", h)
    if p := os.getenv("REDIS_PORT"):
        os.environ.setdefault("FISSION_REDIS_PORT", p)
    # Ollama
    ollama_host = os.getenv("OLLAMA_HOST", "fission-ollama")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    os.environ.setdefault(
        "FISSION_OLLAMA_HOST", f"http://{ollama_host}:{ollama_port}"
    )
    if m := os.getenv("DEFAULT_MODEL"):
        os.environ.setdefault("FISSION_LLAVA_MODEL", m)
    if s := os.getenv("HMAC_SECRET"):
        os.environ.setdefault("FISSION_HMAC_SECRET", s)


def main() -> int:
    _apply_env_overrides()

    # Late import so env overrides take effect
    sys.path.insert(0, "/app")
    from orchestrator.orchestrator import Orchestrator

    orch = Orchestrator()
    log.info(
        "Orchestrator ready | ollama=%s | model=%s",
        os.environ.get("FISSION_OLLAMA_HOST"),
        os.environ.get("FISSION_LLAVA_MODEL", "llava:latest"),
    )

    stop = False

    def _handle(sig, frame):
        nonlocal stop
        log.info("Received signal %s — shutting down", sig)
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    # Idle loop: Orchestrator is request-driven (dashboard / attacks push
    # work). Keeping the process alive holds the bus registration.
    while not stop:
        time.sleep(5)

    log.info("Orchestrator stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
