"""FISSION executor container entrypoint — blocks on the MCP bus."""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.executor.entrypoint")


def _apply_env_overrides() -> None:
    if h := os.getenv("REDIS_HOST"):
        os.environ.setdefault("FISSION_REDIS_HOST", h)
    if p := os.getenv("REDIS_PORT"):
        os.environ.setdefault("FISSION_REDIS_PORT", p)
    if s := os.getenv("HMAC_SECRET"):
        os.environ.setdefault("FISSION_HMAC_SECRET", s)
    if t := os.getenv("ICS_THRESHOLD"):
        os.environ.setdefault("FISSION_ICS_THRESHOLD", t)
    if t := os.getenv("ASV_THRESHOLD"):
        os.environ.setdefault("FISSION_ASV_THRESHOLD", t)


def main() -> int:
    _apply_env_overrides()
    sys.path.insert(0, "/app")

    from executor.executor import Executor

    log.info("Starting Executor listen loop...")
    Executor().listen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
