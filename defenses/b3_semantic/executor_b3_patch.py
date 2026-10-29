# executor.py — B3 patch
#
# Replaces the hardcoded "CLEAN" bypass with a real SemanticGuard.check()
# call, in the exact position it already occupies (before B4 TCIV, per
# tciv.py's own docstring: "B3 runs before B4 to close this gap").
#
# Two changes: an import addition near the top of the file, and the
# stub block inside Executor.handle().

# ── 1. Add near the top of executor.py, alongside the other imports ──

from defenses.b3_semantic.semantic_guard import SemanticGuard


# ── 2. In Executor.__init__, initialize the guard alongside self.tciv ─

# Before:
#     def __init__(self):
#         self.tciv = TCIV()
#         self.bus  = RedisMCPBus()
#         self.bus.register(cfg.mcp.role_executor)
#         log.info("Executor ready")
#
# After:
    def __init__(self):
        self.tciv = TCIV()
        self.b3_guard = SemanticGuard()
        self.bus  = RedisMCPBus()
        self.bus.register(cfg.mcp.role_executor)
        log.info("Executor ready")


# ── 3. Replace the stub block inside Executor.handle() ────────────────
#
# Before:
#
#         # ── B3 Semantic Anomaly Guard ────────────────────────────
#         # Stub — always CLEAN until Phase 3 implementation
#         # TODO: import and call B3Guard here (Chiranth + Abhyudaya, Phase 3)
#         b3_status = "CLEAN"
#         b3_reason = "B3 stub — all layers pass"
#         log.info(f"[{task_id}] B3: {b3_status} ({b3_reason})")
#
#         if b3_status == "BLOCKED":
#             duration = (time.time() - t0) * 1000
#             self._send_result(task_id, tool_name, "BLOCKED", None,
#                               "B3 Semantic Anomaly Guard", b3_reason)
#             return ExecutorResult(
#                 task_id=task_id, tool_name=tool_name,
#                 status="BLOCKED", tool_result=None,
#                 blocked_at="B3", block_reason=b3_reason,
#                 duration_ms=duration,
#             )
#
# After:

        # ── B3 Semantic Anomaly Guard ────────────────────────────
        b3_result = self.b3_guard.check(
            tool_name=tool_name,
            arguments=arguments,
            intent_text=intent_text,
        )
        log.info(
            f"[{task_id}] B3: {b3_result.verdict.value} "
            f"(ics={b3_result.ics_score}, escalated={b3_result.escalated})"
        )

        if not b3_result.passed:
            duration = (time.time() - t0) * 1000
            self._send_result(task_id, tool_name, "BLOCKED", None,
                              f"B3 Semantic Anomaly Guard ({b3_result.layer_failed})",
                              b3_result.block_reason)
            return ExecutorResult(
                task_id=task_id, tool_name=tool_name,
                status="BLOCKED", tool_result=None,
                blocked_at=f"B3 ({b3_result.layer_failed})",
                block_reason=b3_result.block_reason,
                duration_ms=duration,
            )

# Everything below this point (B4 TCIV, tool dispatch) is unchanged —
# no other part of Executor.handle() needs to move or be touched.
