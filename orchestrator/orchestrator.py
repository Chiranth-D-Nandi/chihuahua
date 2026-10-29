"""
FISSION — orchestrator/orchestrator.py
========================================
Main Orchestrator agent.

Drives the VLM to process user tasks, produce tool calls, sign them
with TCIV (B4), and dispatch them to the Executor via the MCP bus.

The Orchestrator is the central pipeline component that L5a (worm
propagation) drives directly — each AutonomousWorm.run() hop calls
orchestrator.run() with the current accumulated context.

ABHYUDAYA — orchestration layer.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from PIL import Image

from config.config import cfg
from orchestrator.prompt_builder import PromptBuilder
from orchestrator.response_parser import ResponseParser
from shared.redis_bus import RedisMCPBus, MCPMessage

logger = logging.getLogger("fission.orchestrator")


@dataclass
class OrchestratorResult:
    """Result of a single orchestrator.run() call."""
    task_text: str
    raw_vlm_output: str
    tool_name: Optional[str]
    arguments: dict
    status: str              # "CLEAN" | "BLOCKED" | "PARSE_FAILED"
    block_reason: Optional[str]
    tool_result: Optional[str]
    duration_ms: float
    tciv_verdict: Optional[dict] = None   # for debugging


class Orchestrator:
    """
    Main orchestrator: user task → VLM → tool call → TCIV sign → bus dispatch.

    Attributes
    ----------
        bus: MCP message bus (Redis or in-memory).
        prompt_builder: System prompt builder. L5a's worm attack sets
            prompt_builder._system_prompt_cache to the infected prompt.
        parser: VLM response parser.
        tciv: Tool Call Integrity Verifier (B4 defense).
    """

    def __init__(self, bus: RedisMCPBus | None = None):
        self.bus = bus or RedisMCPBus(
            host=cfg.mcp.redis_host,
            port=cfg.mcp.redis_port,
        )
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()
        self.task_id_counter = 0

        # Register on the bus
        self.bus.register(cfg.mcp.role_orchestrator)

        # Lazy-import B4 TCIV
        from defenses.b4_tciv.tciv import TCIV
        self.tciv = TCIV()

        # Lazy VLM client
        self._vlm_available: Optional[bool] = None
        self._ollama_host = cfg.model.ollama_host
        self._llava = cfg.model.llava

        # Context key for shared orchestrator state (used by L5a + L5b)
        self.context_key = cfg.mcp.context_buffer_key

        logger.info(f"Orchestrator initialised | bus={type(self.bus).__name__}")

    def is_vlm_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._vlm_available is None:
            try:
                r = requests.get(f"{self._ollama_host}/api/tags", timeout=5)
                r.raise_for_status()
                models = [m["name"] for m in r.json().get("models", [])]
                self._vlm_available = self._llava in models
                if not self._vlm_available:
                    logger.warning(f"Ollama up but '{self._llava}' not pulled. Available: {models}")
            except Exception as exc:
                self._vlm_available = False
                logger.warning(f"Ollama unreachable: {exc}")
        return self._vlm_available

    def run(
        self,
        task_text: str,
        image_path: str | None = None,
        timeout: float = 45.0,
    ) -> OrchestratorResult:
        """
        Process a user task end-to-end.

        Steps
        -----
          1. Build VLM prompt (system + task + image if provided)
          2. Query Ollama/LLaVA
          3. Parse VLM output into a tool call
          4. Sign with TCIV (B4)
          5. Send tool call request to Executor via bus
          6. Wait for Executor response
          7. Return structured result

        Args
        ----
            task_text: The user's task.
            image_path: Path to an image to include in the VLM query.
            timeout: Max seconds to wait for the Executor response.

        Returns
        -------
            OrchestratorResult
        """
        start = time.monotonic()
        self.task_id_counter += 1
        task_id = f"orch_task_{self.task_id_counter:04d}"

        system_prompt = self.prompt_builder.get_system_prompt()

        # ── Step 1: preprocess image ────────────────────────────────────
        image_b64 = None
        if image_path:
            try:
                image_b64 = self._preprocess_image(image_path)
            except Exception as exc:
                logger.error(f"Image preprocess failed: {exc}")
                return OrchestratorResult(
                    task_text=task_text,
                    raw_vlm_output="",
                    tool_name=None,
                    arguments={},
                    status="PARSE_FAILED",
                    block_reason=f"image_preprocess_failed:{exc}",
                    tool_result=None,
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        # ── Step 2: VLM query ─────────────────────────────────────────
        raw_output = self._call_vlm(system_prompt, task_text, image_b64)
        if raw_output is None:
            return OrchestratorResult(
                task_text=task_text,
                raw_vlm_output="",
                tool_name=None,
                arguments={},
                status="PARSE_FAILED",
                block_reason="vlm_unreachable_or_timeout",
                tool_result=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # ── Step 3: parse ─────────────────────────────────────────────
        parsed = self.parser.parse(raw_output)
        if parsed is None or parsed.tool_name is None:
            return OrchestratorResult(
                task_text=task_text,
                raw_vlm_output=raw_output,
                tool_name=None,
                arguments={},
                status="PARSE_FAILED",
                block_reason=f"parse_failed:{self.parser.last_error}",
                tool_result=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # ── Step 4: TCIV sign ──────────────────────────────────────────
        sig = self.tciv.sign(
            tool_name=parsed.tool_name,
            arguments=parsed.arguments,
            intent_text=task_text,
            task_id=task_id,
        )

        # ── Step 5: dispatch to executor ──────────────────────────────
        msg = MCPMessage(
            sender=cfg.mcp.role_orchestrator,
            recipient=cfg.mcp.role_executor,
            msg_type="tool_call_request",
            payload={
                "task_id": task_id,
                "tool_name": parsed.tool_name,
                "arguments": parsed.arguments,
                "intent_text": task_text,
                "hmac_sig": sig,
            },
        )
        self.bus.send(msg, timeout=5.0)

        # ── Step 6: wait for executor response ─────────────────────────
        response = self.bus.receive(cfg.mcp.role_orchestrator, timeout=timeout)

        if response is None:
            return OrchestratorResult(
                task_text=task_text,
                raw_vlm_output=raw_output,
                tool_name=parsed.tool_name,
                arguments=parsed.arguments,
                status="BLOCKED",
                block_reason="executor_timeout",
                tool_result=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        resp_payload = response.payload
        exec_status = resp_payload.get("status", "UNKNOWN")
        tool_result = resp_payload.get("tool_result")
        block_reason = resp_payload.get("block_reason")

        duration = (time.monotonic() - start) * 1000

        return OrchestratorResult(
            task_text=task_text,
            raw_vlm_output=raw_output,
            tool_name=parsed.tool_name,
            arguments=parsed.arguments,
            status=exec_status,
            block_reason=block_reason,
            tool_result=tool_result,
            duration_ms=duration,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _preprocess_image(self, path: str) -> str:
        """Load an image and return base64-encoded PNG."""
        pil = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_vlm(
        self,
        system_prompt: str,
        task_text: str,
        image_b64: str | None,
    ) -> Optional[str]:
        """Send a request to Ollama/LLaVA."""
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nTASK: {task_text}"}
        ]
        if image_b64:
            messages[0]["images"] = [image_b64]

        payload = {
            "model": self._llava,
            "stream": False,
            "messages": messages,
            "options": {"temperature": 0.1, "num_predict": 256},
        }

        try:
            r = requests.post(
                f"{self._ollama_host}/api/chat",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as exc:
            logger.error(f"VLM call failed: {exc}")
            return None

