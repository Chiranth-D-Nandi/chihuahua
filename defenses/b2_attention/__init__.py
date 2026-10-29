"""
FISSION — defenses/b2_attention
================================
B2 Attention Poisoning Defense — real-time three-layer attention monitor
(ADR/AER threshold -> KL divergence -> Mahalanobis outlier check).
"""
from .attention_monitor import (
    AttentionMonitor,
    MonitorVerdict,
    MonitorResult,
    LayerAttentionStats,
)
from .attention_baseline import AttentionBaseline

__all__ = [
    "AttentionMonitor",
    "MonitorVerdict",
    "MonitorResult",
    "LayerAttentionStats",
    "AttentionBaseline",
]
