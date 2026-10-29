"""
FISSION — executor/executor.py
===============================
Executor agent.

Listens on the MCP bus for tool_call_request messages from the Orchestrator,
runs defenses (B3 Semantic Guard → B4 TCIV), and dispatches to the
registered tool implementation.

ABHYUDAYA — execution layer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from config.config import cfg
from shared.redis_bus import RedisMCPBus, MCPMessage
from defenses.b4_tciv.tciv import TCIV, TCIVVerdict

logger = logging.getLogger("fission.executor")


@dataclass
class ExecutorConfig:
    """Tool implementations — these are the actual tool functions."""
    tools: dict = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = {
                "summarise_document": self._summarise_document,
                "send_slack_message": self._send_slack_message,
                "query_database": self._query_database,
            }

    @staticmethod
    def _summarise_document(**kwargs) -> str:
        doc_id = kwargs.get("document_id", "unknown")
        max_length = kwargs.get("max_length", 100)
        return (
            f"[MOCK] Document '{doc_id}' summarised in {max_length} words. "
            "Summary: The document covers Q3 financial performance with "
            "revenue of $4.2M, profit of $1.1M, and positive outlook for Q4."
        )

    @staticmethod
    def _send_slack_message(**kwargs) -> str:
        channel = kwargs.get("channel", "#general")
        message = kwargs.get("message", "")
        return f"[MOCK] Message posted to {channel}: {message}"

    @staticmethod
    def _query_database(**kwargs) -> str:
        query = kwargs.get("query", "SELECT 1")
        database = kwargs.get("database", "unknown")
        return f"[MOCK] Query on '{database}': {query} | returned 3 rows"


class Executor:
    """
    Executor agent — receives tool call requests, runs defenses, executes.

    Defense order (in handle()):
      1. B3 Semantic Guard — intent/tool semantic consistency check
      2. B4 TCIV — whitelist + schema + HMAC verification

    If both pass: dispatch to the registered tool function and return result.
    If either blocks: return BLOCKED status with block_reason.
    """

    def __init__(self, bus: RedisMCPBus | None = None):
        self.bus = bus or RedisMCPBus(
            host=cfg.mcp.redis_host,
            port=cfg.mcp.redis_port,
        )
        self.tools = ExecutorConfig().tools
        self.tciv = TCIV()

        # Lazy-import B3
        from defenses.b3_semantic.semantic_guard import SemanticGuard
        self.semantic_guard = SemanticGuard()

        # Register
        self.bus.register(cfg.mcp.role_executor)

        # Redis client for tool-result propagation (L5a measures this)
        self._redis = None
        self._tool_result_key = "fission:tool_results"

        logger.info("Executor initialised")

    def _store_tool_result(self, task_id: str, tool_name: str, arguments: dict, result: str) -> None:
        """
        Store the tool result in Redis so downstream agents (L5a measurement)
        can read the real tool-call output.

        L5a measures: did the malicious tool's output survive long enough to
        be read by a downstream agent? This is the paper's "tool-result
        propagation" mechanism, not just a text-append simulation.
        """
        try:
            if self._redis is None:
                self._redis = __import__("redis").Redis(
                    host=cfg.mcp.redis_host,
                    port=cfg.mcp.redis_port,
                    decode_responses=False,
                )

            import json
            entry = {
                "task_id": task_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
            # Push to a list: newest last
            self._redis.rpush(self._tool_result_key, json.dumps(entry))
            # Keep only last 20 results to avoid unbounded memory growth
            self._redis.ltrim(self._tool_result_key, -20, -1)
            logger.debug(f"[Executor] Tool result stored in Redis key={self._tool_result_key} task={task_id}")
        except Exception as exc:
            # Redis write failure is non-fatal — the tool still executed
            logger.warning(f"[Executor] Failed to store tool result in Redis: {exc}")

    def listen(self):
        """
        Main receive loop. Runs in a background thread (started by caller).
        """
        logger.info("Executor listening for tool_call_request messages...")
        while True:
            msg = self.bus.receive(cfg.mcp.role_executor, timeout=10.0)
            if msg is None:
                continue
            if msg.msg_type == "tool_call_request":
                result_payload = self.handle(msg)
                response = MCPMessage(
                    sender=cfg.mcp.role_executor,
                    recipient=cfg.mcp.role_orchestrator,
                    msg_type="tool_call_response",
                    payload=result_payload,
                )
                self.bus.send(response, timeout=5.0)
            else:
                logger.debug(f"Executor ignored msg type: {msg.msg_type}")

    def handle(self, msg: MCPMessage) -> dict:
        """
        Process one tool_call_request message.

        Defense pipeline:
          1. B3 Semantic Guard
          2. B4 TCIV

        Returns a dict with keys: status, tool_result, block_reason, blocked_at.
        """
        payload = msg.payload
        tool_name = payload.get("tool_name", "")
        arguments = payload.get("arguments", {})
        intent_text = payload.get("intent_text", "")
        task_id = payload.get("task_id", "")
        hmac_sig = payload.get("hmac_sig", "")

        # ── Defense Layer 1: B3 Semantic Guard ────────────────────────
        b3_result = self.semantic_guard.check(
            tool_name=tool_name,
            arguments=arguments,
            intent_text=intent_text,
        )
        if not b3_result.passed:
            logger.info(
                f"[Executor] B3 blocked: {tool_name} | "
                f"layer={b3_result.layer_failed} | reason={b3_result.block_reason}"
            )
            return {
                "status": "BLOCKED",
                "tool_result": None,
                "block_reason": f"b3:{b3_result.layer_failed}:{b3_result.block_reason}",
                "blocked_at": "b3_semantic_guard",
                "task_id": task_id,
            }

        # ── Defense Layer 2: B4 TCIV ──────────────────────────────────
        tciv_result = self.tciv.verify(
            tool_name=tool_name,
            arguments=arguments,
            agent_role="executor",
            intent_text=intent_text,
            task_id=task_id,
            hmac_sig=hmac_sig,
        )
        if not tciv_result.passed:
            logger.info(
                f"[Executor] B4 blocked: {tool_name} | "
                f"layer={tciv_result.layer_failed} | reason={tciv_result.block_reason}"
            )
            return {
                "status": "BLOCKED",
                "tool_result": None,
                "block_reason": f"b4:{tciv_result.layer_failed}:{tciv_result.block_reason}",
                "blocked_at": "b4_tciv",
                "task_id": task_id,
            }

        # ── Execute the tool ───────────────────────────────────────────
        tool_fn = self.tools.get(tool_name)
        if tool_fn is None:
            return {
                "status": "BLOCKED",
                "tool_result": None,
                "block_reason": f"unknown_tool:{tool_name}",
                "blocked_at": "executor",
                "task_id": task_id,
            }

        try:
            tool_result = tool_fn(**arguments)
            logger.info(f"[Executor] Executed: {tool_name}({arguments}) -> {tool_result[:60]}")

            # ── L5a: store tool result in Redis for propagation measurement ──
            self._store_tool_result(task_id, tool_name, arguments, tool_result)

            return {
                "status": "CLEAN",
                "tool_result": tool_result,
                "block_reason": None,
                "blocked_at": None,
                "task_id": task_id,
            }
        except Exception as exc:
            logger.error(f"[Executor] Tool execution failed: {exc}")
            return {
                "status": "BLOCKED",
                "tool_result": None,
                "block_reason": f"tool_execution_error:{exc}",
                "blocked_at": "executor",
                "task_id": task_id,
            }

