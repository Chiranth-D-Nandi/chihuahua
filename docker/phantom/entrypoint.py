"""FISSION PHANTOM honeypot container entrypoint.

Listens on the same executor stream as a decoy agent. Any tool-call
payload that arrives is scored for worm/context drift and never executed.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.phantom.entrypoint")


def _apply_env_overrides() -> None:
    if h := os.getenv("REDIS_HOST"):
        os.environ.setdefault("FISSION_REDIS_HOST", h)
    if p := os.getenv("REDIS_PORT"):
        os.environ.setdefault("FISSION_REDIS_PORT", p)


def main() -> int:
    _apply_env_overrides()
    sys.path.insert(0, "/app")

    from config.config import cfg
    from shared.redis_bus import RedisMCPBus
    from defenses.b5_honeypot.honeypot_agent import PHANTOMAgent

    bus = RedisMCPBus(host=cfg.mcp.redis_host, port=cfg.mcp.redis_port)
    # Register under a distinct role so it can shadow the executor stream
    phantom_role = os.getenv("PHANTOM_ROLE", "phantom")
    bus.register(phantom_role)

    original_intent = os.getenv(
        "PHANTOM_ORIGINAL_INTENT",
        "Summarise documents and notify the research team",
    )
    drift_threshold = float(os.getenv("CONTEXT_DRIFT_THRESHOLD", "0.30"))
    phantom = PHANTOMAgent(
        original_intent=original_intent,
        drift_threshold=drift_threshold,
    )

    stop = False

    def _handle(sig, frame):
        nonlocal stop
        log.info("Signal %s — stopping PHANTOM", sig)
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    log.info("PHANTOM honeypot listening as role=%s", phantom_role)
    while not stop:
        msg = bus.receive(phantom_role, timeout=5.0)
        if msg is None:
            continue
        if msg.msg_type not in ("tool_call_request", "honeypot_probe"):
            continue
        payload = msg.payload if isinstance(msg.payload, dict) else {}
        result = phantom.receive(payload)
        log.info(
            "PHANTOM scored payload tool=%s worm=%s drift=%.4f reason=%s",
            payload.get("tool_name"),
            result.worm_detected,
            result.drift_score,
            result.reason,
        )
        # Optionally publish detection events for the dashboard
        try:
            bus.send_raw = getattr(bus, "send", None)
            if bus.send_raw is not None:
                from shared.redis_bus import MCPMessage
                bus.send(
                    MCPMessage(
                        sender=phantom_role,
                        recipient="dashboard",
                        msg_type="honeypot_detection",
                        payload={
                            "tool_name": payload.get("tool_name"),
                            "worm_detected": result.worm_detected,
                            "drift_score": result.drift_score,
                            "reason": result.reason,
                            "apr": phantom.apr(),
                        },
                    )
                )
        except Exception as exc:
            log.debug("Could not publish honeypot event: %s", exc)

    log.info("Final APR=%.2f | detections logged", phantom.apr())
    return 0


if __name__ == "__main__":
    sys.exit(main())
