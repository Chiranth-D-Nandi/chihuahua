"""
FISSION — defenses/b4_tciv/tciv.py
=====================================
Tool Call Integrity Verifier (B4 TCIV)

Three-layer structural verification on every MCP tool call
before it reaches the executor's dispatch function.

Layer 1 — Whitelist enforcement
    Tool name must appear in the registered tool registry
    for the calling agent's role.

Layer 2 — Schema validation
    Every required argument must be present.
    Every argument must be the correct type.
    Extra arguments not in the schema are flagged.

Layer 3 — HMAC intent binding
    The tool call is cryptographically bound to the original
    orchestrator instruction via HMAC-SHA256.
    Silent modification of any field breaks the signature.

THE GRAY AREA:
    B4 TCIV is structural. It cannot verify that a call is
    semantically correct — only that it is correctly shaped.
    A backdoor-triggered call (R0) or a semantically hijacked
    call can pass all three layers while encoding attacker intent
    in argument values. B3 Semantic Anomaly Guard runs before
    B4 to close this gap.

ABHYUDAYA — B4 TCIV owner.
"""

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.config import cfg

log = logging.getLogger("fission.b4_tciv")


# ══════════════════════════════════════════════════════════════
#  Result types
# ══════════════════════════════════════════════════════════════

class TCIVVerdict(Enum):
    PASS    = "PASS"
    BLOCKED = "BLOCKED"


@dataclass
class TCIVResult:
    verdict:      TCIVVerdict
    block_reason: Optional[str] = None
    layer_failed: Optional[str] = None   # "whitelist" | "schema" | "hmac"

    @property
    def passed(self) -> bool:
        return self.verdict == TCIVVerdict.PASS


# ══════════════════════════════════════════════════════════════
#  TCIV
# ══════════════════════════════════════════════════════════════

class TCIV:
    """
    Tool Call Integrity Verifier.

    Usage:
        tciv = TCIV()

        # Orchestrator signs the call:
        sig = tciv.sign(tool_name, arguments, intent_text, task_id)

        # Executor verifies:
        result = tciv.verify(
            tool_name    = "summarise_document",
            arguments    = {"document_id": "doc_001", "max_length": 150},
            agent_role   = "executor",
            intent_text  = "Summarise doc_001 and post to Slack",
            task_id      = "task_123",
            hmac_sig     = sig,
        )

        if result.passed:
            dispatch_tool(...)
        else:
            log.warning(result.block_reason)
    """

    def __init__(self):
        self._secret   = cfg.defense.hmac_secret.encode()
        self._registry = cfg.mcp.registered_tools

    # ── HMAC ────────────────────────────────────────────────────

    def sign(
        self,
        tool_name:   str,
        arguments:   dict,
        intent_text: str,
        task_id:     str,
    ) -> str:
        """
        Produce HMAC-SHA256 signature.
        Called by the orchestrator before putting the call on the bus.
        """
        payload = json.dumps({
            "tool":    tool_name,
            "args":    arguments,
            "intent":  intent_text,
            "task_id": task_id,
        }, sort_keys=True).encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    # ── Three-layer verification ─────────────────────────────────

    def verify(
        self,
        tool_name:   str,
        arguments:   dict,
        agent_role:  str,
        intent_text: str,
        task_id:     str,
        hmac_sig:    str,
    ) -> TCIVResult:
        """
        Run all three verification layers.
        Returns on first failure — does not continue after a block.
        """

        # ── Layer 1: Whitelist ───────────────────────────────────
        result = self._check_whitelist(tool_name, agent_role)
        if not result.passed:
            log.warning(f"[TCIV BLOCKED] Whitelist: {result.block_reason}")
            return result

        # ── Layer 2: Schema ──────────────────────────────────────
        result = self._check_schema(tool_name, arguments)
        if not result.passed:
            log.warning(f"[TCIV BLOCKED] Schema: {result.block_reason}")
            return result

        # ── Layer 3: HMAC ────────────────────────────────────────
        result = self._check_hmac(tool_name, arguments, intent_text, task_id, hmac_sig)
        if not result.passed:
            log.warning(f"[TCIV BLOCKED] HMAC: {result.block_reason}")
            return result

        log.info(f"[TCIV PASS] {tool_name}({list(arguments.keys())})")
        return TCIVResult(verdict=TCIVVerdict.PASS)

    # ── Layer implementations ────────────────────────────────────

    def _check_whitelist(self, tool_name: str, agent_role: str) -> TCIVResult:
        """
        Check 1: tool must be in the registry.
        Check 2: calling agent's role must be in the tool's allowed_roles.
        """
        if tool_name not in self._registry:
            return TCIVResult(
                verdict=TCIVVerdict.BLOCKED,
                block_reason=f"tool '{tool_name}' is not in the registered tool registry",
                layer_failed="whitelist",
            )

        allowed = self._registry[tool_name].get("allowed_roles", [])
        if agent_role not in allowed:
            return TCIVResult(
                verdict=TCIVVerdict.BLOCKED,
                block_reason=(
                    f"role '{agent_role}' is not permitted to call '{tool_name}'. "
                    f"Allowed roles: {allowed}"
                ),
                layer_failed="whitelist",
            )

        return TCIVResult(verdict=TCIVVerdict.PASS)

    def _check_schema(self, tool_name: str, arguments: dict) -> TCIVResult:
        """
        Check all required arguments are present and correctly typed.
        Also flag unexpected arguments not in the schema.
        """
        schema = self._registry[tool_name].get("schema", {})

        # Check required args are present and correctly typed
        for arg_name, expected_type in schema.items():
            if arg_name not in arguments:
                return TCIVResult(
                    verdict=TCIVVerdict.BLOCKED,
                    block_reason=f"missing required argument '{arg_name}'",
                    layer_failed="schema",
                )
            actual = arguments[arg_name]
            if not isinstance(actual, expected_type):
                return TCIVResult(
                    verdict=TCIVVerdict.BLOCKED,
                    block_reason=(
                        f"argument '{arg_name}': "
                        f"expected {expected_type.__name__}, "
                        f"got {type(actual).__name__} (value: {repr(actual)[:40]})"
                    ),
                    layer_failed="schema",
                )

        # Flag unexpected arguments (warn but do not block)
        unexpected = [k for k in arguments if k not in schema]
        if unexpected:
            log.warning(
                f"[TCIV WARN] Unexpected arguments in '{tool_name}': {unexpected} "
                f"(passing — not in schema but not blocking)"
            )

        return TCIVResult(verdict=TCIVVerdict.PASS)

    def _check_hmac(
        self,
        tool_name:   str,
        arguments:   dict,
        intent_text: str,
        task_id:     str,
        provided:    str,
    ) -> TCIVResult:
        """
        Recompute HMAC and compare with provided signature.
        A mismatch means the call was modified after signing — block it.
        """
        expected = self.sign(tool_name, arguments, intent_text, task_id)
        if not hmac.compare_digest(provided, expected):
            return TCIVResult(
                verdict=TCIVVerdict.BLOCKED,
                block_reason=(
                    "HMAC signature mismatch — tool call does not match the "
                    "original orchestrator instruction. Possible tampering in transit."
                ),
                layer_failed="hmac",
            )
        return TCIVResult(verdict=TCIVVerdict.PASS)

    # ── Metrics helpers ──────────────────────────────────────────

    def compute_fpr(
        self,
        legitimate_calls: list[dict],
    ) -> float:
        """
        Compute TCIV False Positive Rate on a list of known-legitimate calls.
        Each call dict: {tool_name, arguments, agent_role, intent_text, task_id}

        TCIV-FPR must stay <= 5% (cfg.eval.tciv_fpr_max).
        """
        if not legitimate_calls:
            return 0.0

        blocked = 0
        for call in legitimate_calls:
            sig = self.sign(
                call["tool_name"],
                call["arguments"],
                call["intent_text"],
                call["task_id"],
            )
            result = self.verify(
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                agent_role=call.get("agent_role", cfg.mcp.role_executor),
                intent_text=call["intent_text"],
                task_id=call["task_id"],
                hmac_sig=sig,
            )
            if not result.passed:
                blocked += 1

        fpr = blocked / len(legitimate_calls)
        log.info(f"TCIV-FPR: {fpr:.3f} ({blocked}/{len(legitimate_calls)} blocked)")
        return fpr

