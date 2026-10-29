"""
FISSION — B5 PHANTOM Honeypot Agent
Poisoned Honeypot Agent for Multi-agent Threat Observation Network.
Receives payloads, detects worm propagation and context poisoning.
Never executes tool calls.
"""

import json
import numpy as np
from dataclasses import dataclass, field
import logging

log = logging.getLogger("fission.b5_phantom")

def _get_model():
    from shared.embedding_model import get_embedding_model
    return get_embedding_model()


@dataclass
class DetectionResult:
    payload: dict
    worm_detected: bool
    drift_score: float
    kill_chain: bool
    reason: str


class PHANTOMAgent:
    """
    Honeypot agent that appears identical to executor on the MCP bus.
    Logs all payloads, detects anomalies, never executes tool calls.
    """

    SUSPICIOUS_TOOLS = {
        "delete_all_files", "open_document", "read_file",
        "list_dir", "exec_shell", "system_command",
    }

    SUSPICIOUS_ARGS = [
        "/admin", "passwords", "credentials", "bypass",
        "token", "/etc/passwd", "auth", "secret",
    ]

    def __init__(self, original_intent: str, drift_threshold: float = 0.30):
        self.original_intent = original_intent
        self.drift_threshold = drift_threshold
        self.original_embedding = _get_model().encode(
            [original_intent], normalize_embeddings=True
        )[0]
        self.payload_log: list[dict] = []

    def _compute_drift(self, payload: dict) -> float:
        payload_text = json.dumps(payload)
        payload_emb = _get_model().encode([payload_text], normalize_embeddings=True)[0]
        similarity = float(np.dot(self.original_embedding, payload_emb))
        return 1.0 - similarity

    def _suspicious_content(self, payload: dict) -> bool:
        tool = payload.get("tool_name", "")
        args = json.dumps(payload.get("arguments", {})).lower()

        if tool in self.SUSPICIOUS_TOOLS:
            return True
        if any(s in args for s in self.SUSPICIOUS_ARGS):
            return True
        return False

    def receive(self, payload: dict) -> DetectionResult:
        """
        Receive a payload from the MCP bus.
        Log it, analyze it, never execute it.
        """
        self.payload_log.append(payload)
        log.info(f"PHANTOM received payload: {json.dumps(payload)[:120]}")

        drift = self._compute_drift(payload)
        suspicious = self._suspicious_content(payload)
        worm_detected = drift > self.drift_threshold or suspicious
        kill_chain = worm_detected

        reason = []
        if drift > self.drift_threshold:
            reason.append(f"semantic drift={drift:.4f}")
        if suspicious:
            reason.append("suspicious tool/args detected")

        result = DetectionResult(
            payload=payload,
            worm_detected=worm_detected,
            drift_score=drift,
            kill_chain=kill_chain,
            reason=", ".join(reason) if reason else "clean",
        )

        if worm_detected:
            log.warning(f"PHANTOM: WORM DETECTED — {result.reason}")
        else:
            log.info(f"PHANTOM: payload clean (drift={drift:.4f})")

        return result

    def dump_log(self):
        log.info(f"PHANTOM payload log ({len(self.payload_log)} entries):")
        for i, p in enumerate(self.payload_log):
            log.info(f"  [{i}] {json.dumps(p)[:100]}")

    def apr(self) -> float:
        """Agent Propagation Rate — % payloads that were worm attempts."""
        if not self.payload_log:
            return 0.0
        worm_count = sum(
            1 for p in self.payload_log
            if self._compute_drift(p) > self.drift_threshold or self._suspicious_content(p)
        )
        return worm_count / len(self.payload_log)
