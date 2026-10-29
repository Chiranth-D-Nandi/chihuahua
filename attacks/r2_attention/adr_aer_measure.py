"""
FISSION — attacks/r2_attention/adr_aer_measure.py
===================================================
ADR and AER measurement utilities — compares clean vs adversarial
attention distributions from REAL captured weights.

Metrics:
  ADR — Attention Dominance Ratio = sum(top-5 weights) / sum(all weights)
        High ADR = few tokens dominate = potential poisoning
  AER — Attention Entropy Ratio = Shannon_entropy / max_entropy
        Low AER = attention collapsed = potential poisoning

CHIRANTH — R2 measurement.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

log = logging.getLogger("fission.r2_measure")


@dataclass
class AttentionResult:
    """Per-pass attention measurement result."""
    adr: float = 0.0
    aer: float = 0.0
    top5_weights: list = None
    entropy_raw: float = 0.0
    kl_from_uniform: float = 0.0
    layer_count: int = 0
    flagged: bool = False

    def __post_init__(self):
        if self.top5_weights is None:
            self.top5_weights = []


@dataclass
class ComparisonResult:
    """Clean vs adversarial comparison."""
    clean_adr: float
    adversarial_adr: float
    clean_aer: float
    adversarial_aer: float
    adr_delta: float        # adversarial - clean (positive = more dominated)
    aer_delta: float        # adversarial - clean (negative = more collapsed)
    clean_kl: float
    adversarial_kl: float
    kl_delta: float
    anomalous: bool


def compute_attention_metrics(
    weight_matrices: List[np.ndarray],
) -> AttentionResult:
    """
    Compute ADR, AER, KL from a list of attention weight matrices.

    Args:
        weight_matrices: list of numpy arrays, each (seq_q, seq_k)
                         representing one attention layer's weights

    Returns:
        AttentionResult with real computed metrics
    """
    from scipy.special import rel_entr

    if not weight_matrices:
        return AttentionResult()

    all_weights = []
    for mat in weight_matrices:
        all_weights.append(mat.flatten())

    combined = np.concatenate(all_weights)
    combined = np.abs(combined)
    total = combined.sum()

    if total < 1e-12:
        return AttentionResult(layer_count=len(weight_matrices))

    dist = combined / total
    n = len(dist)

    sorted_dist = np.sort(dist)[::-1]
    top5 = sorted_dist[:5]
    adr = float(top5.sum())

    entropy_raw = float(-np.sum(dist * np.log(dist + 1e-12)))
    max_entropy = np.log(n + 1e-12)
    aer = float(entropy_raw / max_entropy) if max_entropy > 0 else 1.0

    uniform = np.ones_like(dist) / n
    kl = float(np.sum(rel_entr(dist, uniform)))

    flagged = aer < 0.60

    return AttentionResult(
        adr=round(adr, 6),
        aer=round(aer, 6),
        top5_weights=[round(float(w), 6) for w in top5],
        entropy_raw=round(entropy_raw, 6),
        kl_from_uniform=round(kl, 6),
        layer_count=len(weight_matrices),
        flagged=flagged,
    )


def compare_attention(
    clean_result: AttentionResult,
    adversarial_result: AttentionResult,
    aer_threshold: float = 0.60,
    adr_spike_threshold: float = 0.20,
) -> ComparisonResult:
    """
    Compare clean vs adversarial attention metrics.
    Returns a ComparisonResult with deltas and anomaly flag.
    """
    adr_delta = adversarial_result.adr - clean_result.adr
    aer_delta = adversarial_result.aer - clean_result.aer
    kl_delta = adversarial_result.kl_from_uniform - clean_result.kl_from_uniform

    anomalous = (
        adversarial_result.aer < aer_threshold or
        adr_delta > adr_spike_threshold
    )

    comparison = ComparisonResult(
        clean_adr=clean_result.adr,
        adversarial_adr=adversarial_result.adr,
        clean_aer=clean_result.aer,
        adversarial_aer=adversarial_result.aer,
        adr_delta=round(adr_delta, 6),
        aer_delta=round(aer_delta, 6),
        clean_kl=clean_result.kl_from_uniform,
        adversarial_kl=adversarial_result.kl_from_uniform,
        kl_delta=round(kl_delta, 6),
        anomalous=anomalous,
    )

    return comparison


def log_metrics(comparison: ComparisonResult):
    """Log comparison metrics in a structured format."""
    log.info(
        f"ADR  clean={comparison.clean_adr:.4f}  "
        f"adv={comparison.adversarial_adr:.4f}  "
        f"delta={comparison.adr_delta:+.4f}"
    )
    log.info(
        f"AER  clean={comparison.clean_aer:.4f}  "
        f"adv={comparison.adversarial_aer:.4f}  "
        f"delta={comparison.aer_delta:+.4f}"
    )
    log.info(
        f"KL   clean={comparison.clean_kl:.4f}  "
        f"adv={comparison.adversarial_kl:.4f}  "
        f"delta={comparison.kl_delta:+.4f}"
    )
    log.info(f"Anomalous: {comparison.anomalous}")

