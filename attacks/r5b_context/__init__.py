"""
FISSION — attacks/r5b_context/__init__.py
==========================================
L5b — Context Poisoning Attack.

Exports
-------
    ContextPoisoner         — main attack driver
    ContextHopResult        — per-hop CPR + CBD state
    ContextPoisoningResult — aggregated benchmark result
    MALICIOUS_CONTEXT_CHAIN — the poisoning payload chain
    run_context_poisoning_benchmark — CLI entry point

See context_poisoner.py for full documentation.
"""

from attacks.r5b_context.context_poisoner import (
    ContextPoisoner,
    ContextHopResult,
    ContextPoisoningResult,
    MALICIOUS_CONTEXT_CHAIN,
    run_context_poisoning_benchmark,
)

__all__ = [
    "ContextPoisoner",
    "ContextHopResult",
    "ContextPoisoningResult",
    "MALICIOUS_CONTEXT_CHAIN",
    "run_context_poisoning_benchmark",
]

