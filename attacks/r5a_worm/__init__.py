"""
FISSION — attacks/r5a_worm/__init__.py
=======================================
L5a — Autonomous Worm Propagation Attack.

This package was a stub (empty __init__.py with only "\n\n") in the
original codebase. It has been replaced by the real AutonomousWorm
implementation in worm_attack.py.

Exports
-------
    AutonomousWorm    — main attack driver
    WormHop          — per-hop state record
    WormBenchmarkResult — aggregated benchmark result
    MALICIOUS_TOOLS  — the tool pool the worm targets
    run_worm_benchmark — CLI entry point

See worm_attack.py for full documentation.
"""

from attacks.r5a_worm.worm_attack import (
    AutonomousWorm,
    WormHop,
    WormBenchmarkResult,
    MALICIOUS_TOOLS,
    run_worm_benchmark,
)

__all__ = [
    "AutonomousWorm",
    "WormHop",
    "WormBenchmarkResult",
    "MALICIOUS_TOOLS",
    "run_worm_benchmark",
]

