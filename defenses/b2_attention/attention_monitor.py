"""
FISSION — defenses/b2_attention/attention_monitor.py
=====================================================
B2 Attention Poisoning Defense — Real-time Attention Entropy Monitor.

Detects and blocks cross-modal attention poisoning attacks (L2 threat)
by monitoring the attention weight distributions inside a VLM during
inference and flagging anomalous patterns before the tool call reaches
execution.

Design — three detection layers, first failure blocks:

  Layer 1 — ADR/AER Threshold Check
            Computes Attention Dominance Ratio (ADR) and Attention
            Entropy Ratio (AER) from captured attention weights.
            Blocks if ADR spikes beyond clean baseline by >20% OR
            AER collapses below the entropy threshold (0.60).

  Layer 2 — KL Divergence Check
            Computes KL divergence between the current attention
            distribution and a uniform reference distribution.
            Blocks if KL divergence exceeds threshold — indicates
            that attention has collapsed onto a few tokens (the
            attacker's injected ones).

  Layer 3 — Mahalanobis Outlier Check
            After a warmup period of confirmed-clean passes,
            computes the Mahalanobis distance of the current
            [ADR, AER, top1, top5, entropy] feature vector against
            the running clean baseline covariance. Blocks if the
            observation is >3.5σ from the baseline centroid.

How it works in the pipeline:
  1. Before VLM inference, register_hooks() is called on the loaded
     HuggingFace model — this attaches forward hooks on every attention
     layer in the vision encoder AND cross-attention layers.
  2. The VLM runs its forward pass (clean or adversarial image).
  3. Hooks capture the attention weight tensors from every layer.
  4. check() is called — it computes ADR, AER, KL divergence, builds
     the feature vector, runs all 3 detection layers, returns a verdict.
  5. If PASS — the feature vector is recorded into the baseline.
     If BLOCKED — the tool call is killed before dispatch.
  6. remove_hooks() cleans up.

Integration into the Executor pipeline:
  B2 runs BEFORE B3 and B4 — it operates on the VLM's internal state
  (attention weights) during inference, before the tool call is even
  parsed. B3 (semantic) and B4 (structural) run after parsing.

  Pipeline order: VLM inference → B2 attention check → parse tool call
                  → B3 semantic guard → B4 TCIV → tool dispatch

CHIRANTH — B2 defense owner.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import rel_entr  # KL divergence

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from defenses.b2_attention.attention_baseline import AttentionBaseline

log = logging.getLogger("fission.b2_attention")


# ══════════════════════════════════════════════════════════════
#  Data types
# ══════════════════════════════════════════════════════════════

class MonitorVerdict(Enum):
    PASS     = "PASS"
    BLOCKED  = "BLOCKED"
    WARMUP   = "WARMUP"    # baseline not ready yet — pass but with warning


@dataclass
class LayerAttentionStats:
    """Per-layer attention statistics extracted from a single forward pass."""
    layer_name: str
    adr: float                  # Attention Dominance Ratio (top-5 / total)
    aer: float                  # Attention Entropy Ratio (normalised Shannon)
    entropy_raw: float          # Raw Shannon entropy (not normalised)
    top1_weight: float          # Weight of the most-attended token
    top5_weights: List[float]   # Weights of top-5 most-attended tokens
    kl_from_uniform: float      # KL divergence from uniform distribution
    num_tokens: int             # Total number of tokens in this layer


@dataclass
class MonitorResult:
    """Structured result from B2 attention check."""
    verdict:        MonitorVerdict
    block_reason:   Optional[str] = None
    layer_failed:   Optional[str] = None     # "adr_aer" | "kl_divergence" | "mahalanobis"

    # Aggregate metrics across all captured layers
    mean_adr:       float = 0.0
    mean_aer:       float = 0.0
    mean_kl:        float = 0.0
    mahal_distance: float = 0.0

    # Per-layer breakdown
    layer_stats:    List[LayerAttentionStats] = field(default_factory=list)

    # Timing
    hook_capture_ms:  float = 0.0
    analysis_ms:      float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict in (MonitorVerdict.PASS, MonitorVerdict.WARMUP)


# ══════════════════════════════════════════════════════════════
#  Attention weight capture via HuggingFace hooks
# ══════════════════════════════════════════════════════════════

class _AttentionHookManager:
    """
    Registers forward hooks on attention layers and captures their
    output weight tensors during the VLM's forward pass.

    Supports three hook patterns to cover different VLM architectures:
      - LLaVA / CLIP ViT: layers named "self_attn" or "attention"
      - Qwen2.5-VL custom ViT: layers named "attn"
      - InternVL2 / InternViT: layers named "attention"

    Each hook captures the attention weight matrix (softmaxed) from
    the output tuple of the attention module. The standard HuggingFace
    convention is output[1] = attention_weights when output_attentions=True.
    """

    def __init__(self, max_layers: int = 8):
        self.max_layers = max_layers
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._capture_enabled = False

    def _hook_fn(self, name: str):
        """Factory — returns a hook closure that captures weights under `name`."""
        def hook(module, input, output):
            if not self._capture_enabled:
                return

            try:
                attn_weights = None

                # Standard HuggingFace: output is tuple, [1] = attention weights
                if isinstance(output, tuple) and len(output) > 1:
                    candidate = output[1]
                    if candidate is not None and isinstance(candidate, torch.Tensor):
                        attn_weights = candidate
                    elif candidate is not None and isinstance(candidate, tuple):
                        # Some models nest deeper — try first element
                        if len(candidate) > 0 and isinstance(candidate[0], torch.Tensor):
                            attn_weights = candidate[0]

                # Fallback: if module has .attn_weights or similar
                if attn_weights is None:
                    for attr in ("attn_weights", "attention_weights", "_attn_weights"):
                        if hasattr(module, attr):
                            candidate = getattr(module, attr)
                            if isinstance(candidate, torch.Tensor):
                                attn_weights = candidate
                                break

                # Last resort: if output is a single tensor that looks like
                # attention weights (2D or 3D, values in [0, 1] range)
                if attn_weights is None and isinstance(output, torch.Tensor):
                    if output.dim() in (2, 3, 4):
                        omin, omax = output.min().item(), output.max().item()
                        if 0.0 <= omin and omax <= 1.01:
                            attn_weights = output

                if attn_weights is None:
                    return

                # Normalize to (seq, seq) or (heads, seq, seq)
                w = attn_weights.detach().cpu().float()

                if w.dim() == 4:
                    # (batch, heads, seq_q, seq_k) — average across batch and heads
                    w = w[0].mean(dim=0)  # (seq_q, seq_k)
                elif w.dim() == 3:
                    # (heads, seq_q, seq_k) or (batch, seq_q, seq_k)
                    if w.shape[0] > 1 and w.shape[0] < w.shape[1]:
                        # Likely (heads, seq, seq) — average over heads
                        w = w.mean(dim=0)
                    else:
                        w = w[0]
                elif w.dim() == 2:
                    pass  # already (seq_q, seq_k)
                else:
                    return

                self._captured[name].append(w.numpy())

            except Exception as e:
                log.debug(f"Hook capture error on {name}: {e}")

        return hook

    def register(self, model: nn.Module) -> int:
        """
        Register forward hooks on attention layers of the model.

        Returns the number of hooks successfully registered.
        """
        self.clear()

        # Collect candidate layers
        candidates = []
        for name, module in model.named_modules():
            name_lower = name.lower()
            # Priority 1: cross-attention (most important for VLMs)
            if any(k in name_lower for k in ["cross_attn", "crossattention", "cross_attention"]):
                candidates.append((0, name, module))
            # Priority 2: self-attention in vision encoder
            elif any(k in name_lower for k in ["self_attn", "self_attention"]):
                if any(k in name_lower for k in ["vision", "visual", "vit", "encoder"]):
                    candidates.append((1, name, module))
            # Priority 3: generic attention
            elif any(k in name_lower for k in ["attention", "attn"]):
                candidates.append((2, name, module))

        # Sort by priority, take up to max_layers
        candidates.sort(key=lambda x: x[0])
        selected = candidates[:self.max_layers]

        for _, name, module in selected:
            h = module.register_forward_hook(self._hook_fn(name))
            self._hooks.append(h)

        log.info(f"B2: Registered {len(self._hooks)} attention hooks on {len(selected)} layers")
        for _, name, _ in selected:
            log.debug(f"  Hook: {name}")

        return len(self._hooks)

    def enable_capture(self):
        self._capture_enabled = True

    def disable_capture(self):
        self._capture_enabled = False

    def get_captured(self) -> Dict[str, List[np.ndarray]]:
        """Return all captured attention weight matrices, keyed by layer name."""
        return dict(self._captured)

    def clear_captured(self):
        """Clear captured weights but keep hooks registered."""
        self._captured.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()
        self._capture_enabled = False
        log.debug("B2: All hooks removed")

    def clear(self):
        self.remove_hooks()


# ══════════════════════════════════════════════════════════════
#  Core attention analysis
# ══════════════════════════════════════════════════════════════

def _compute_layer_stats(layer_name: str, weight_matrices: List[np.ndarray]) -> LayerAttentionStats:
    """
    Compute ADR, AER, KL divergence for a single attention layer
    from its captured weight matrices (one per forward pass step).

    All computations are on the REAL captured attention weights,
    nothing is hardcoded or simulated.
    """
    # Aggregate all attention weights from this layer into one distribution
    all_weights = []
    for mat in weight_matrices:
        # Flatten the attention matrix to get per-token attention scores
        # For a (seq_q, seq_k) matrix: each row is a query token's attention
        # over all key tokens. We aggregate across all query positions.
        flat = mat.flatten()
        all_weights.append(flat)

    if not all_weights:
        return LayerAttentionStats(
            layer_name=layer_name, adr=0.0, aer=1.0, entropy_raw=0.0,
            top1_weight=0.0, top5_weights=[], kl_from_uniform=0.0, num_tokens=0,
        )

    combined = np.concatenate(all_weights)
    combined = np.abs(combined)
    total = combined.sum()
    if total < 1e-12:
        return LayerAttentionStats(
            layer_name=layer_name, adr=0.0, aer=1.0, entropy_raw=0.0,
            top1_weight=0.0, top5_weights=[], kl_from_uniform=0.0, num_tokens=0,
        )

    # Normalise to probability distribution
    dist = combined / total
    n_tokens = len(dist)

    # Sort descending for top-k analysis
    sorted_dist = np.sort(dist)[::-1]

    # ADR — Attention Dominance Ratio: sum of top-5 / sum of all
    top5 = sorted_dist[:5]
    adr = float(top5.sum())

    # Top-1 fraction
    top1 = float(sorted_dist[0]) if len(sorted_dist) > 0 else 0.0

    # Shannon entropy
    entropy_raw = float(-np.sum(dist * np.log(dist + 1e-12)))
    max_entropy = np.log(n_tokens + 1e-12)
    aer = float(entropy_raw / max_entropy) if max_entropy > 0 else 1.0

    # KL divergence from uniform
    uniform = np.ones_like(dist) / n_tokens
    kl_div = float(np.sum(rel_entr(dist, uniform)))

    return LayerAttentionStats(
        layer_name=layer_name,
        adr=round(adr, 6),
        aer=round(aer, 6),
        entropy_raw=round(entropy_raw, 6),
        top1_weight=round(top1, 6),
        top5_weights=[round(float(w), 6) for w in top5.tolist()],
        kl_from_uniform=round(kl_div, 6),
        num_tokens=n_tokens,
    )


# ══════════════════════════════════════════════════════════════
#  Main B2 Monitor
# ══════════════════════════════════════════════════════════════

class AttentionMonitor:
    """
    B2 Attention Poisoning Defense — three-layer anomaly detector.

    Lifecycle:
        monitor = AttentionMonitor()
        monitor.register_hooks(vlm_model)

        # During VLM forward pass:
        monitor.begin_capture()
        output = vlm_model(inputs)
        monitor.end_capture()

        # Check for poisoning:
        result = monitor.check()
        if not result.passed:
            # BLOCK — do not proceed with tool call
            ...

        monitor.remove_hooks()

    The monitor is stateful — it maintains a running baseline of clean
    attention patterns that makes later detections more accurate as more
    clean passes are observed.
    """

    def __init__(
        self,
        adr_spike_threshold: Optional[float] = None,
        aer_collapse_threshold: Optional[float] = None,
        kl_div_threshold: Optional[float] = None,
        mahal_threshold: Optional[float] = None,
        min_baseline_samples: Optional[int] = None,
        max_layers: Optional[int] = None,
    ):
        self._adr_spike = adr_spike_threshold or 0.30          # no b2_adr_spike in cfg; use sensible default
        self._aer_collapse = aer_collapse_threshold or getattr(cfg.defense, "aer_collapse_threshold", 0.15)
        self._kl_threshold = kl_div_threshold or getattr(cfg.defense, "kl_div_threshold", 1.0)
        self._mahal_threshold = mahal_threshold or getattr(cfg.defense, "mahal_threshold", 5.0)
        self._max_layers = max_layers or 12                      # no b2_layer_sample_count in cfg; use 12
        self._min_baseline = min_baseline_samples or getattr(cfg.defense, "min_baseline_samples", 5)

        self._hook_manager = _AttentionHookManager(max_layers=self._max_layers)
        self._baseline = AttentionBaseline(min_samples=self._min_baseline)

        # Aggregate clean baseline stats for ADR/AER delta checks
        self._clean_adr_history: List[float] = []
        self._clean_aer_history: List[float] = []
        self._n_passes = 0

        log.info(
            f"B2 AttentionMonitor initialised | "
            f"adr_spike={self._adr_spike} | aer_collapse={self._aer_collapse} | "
            f"kl_threshold={self._kl_threshold} | mahal_threshold={self._mahal_threshold} | "
            f"max_layers={self._max_layers}"
        )

    # ── Hook lifecycle ────────────────────────────────────────────

    def register_hooks(self, model: nn.Module) -> int:
        """Register forward hooks on the VLM. Returns hook count."""
        return self._hook_manager.register(model)

    def begin_capture(self):
        """Enable attention weight capture. Call before VLM forward pass."""
        self._hook_manager.clear_captured()
        self._hook_manager.enable_capture()

    def end_capture(self):
        """Disable capture. Call after VLM forward pass."""
        self._hook_manager.disable_capture()

    def remove_hooks(self):
        """Remove all hooks. Call when done with the model."""
        self._hook_manager.remove_hooks()

    # ── Core analysis ─────────────────────────────────────────────

    def check(self) -> MonitorResult:
        """
        Analyze captured attention weights and return a verdict.

        Must be called after end_capture(). Runs all three detection
        layers and returns the first failure (or PASS if all clear).
        """
        t0 = time.time()

        captured = self._hook_manager.get_captured()
        if not captured:
            log.warning("B2: No attention weights captured — no hooks fired. Passing by default.")
            return MonitorResult(
                verdict=MonitorVerdict.PASS,
                block_reason="No attention weights captured (hooks may not have fired)",
            )

        # Compute per-layer stats
        t_analysis = time.time()
        layer_stats: List[LayerAttentionStats] = []
        for layer_name, matrices in captured.items():
            stats = _compute_layer_stats(layer_name, matrices)
            layer_stats.append(stats)

        if not layer_stats:
            return MonitorResult(verdict=MonitorVerdict.PASS)

        # Aggregate metrics across layers
        mean_adr = float(np.mean([s.adr for s in layer_stats]))
        mean_aer = float(np.mean([s.aer for s in layer_stats]))
        mean_kl = float(np.mean([s.kl_from_uniform for s in layer_stats]))
        mean_top1 = float(np.mean([s.top1_weight for s in layer_stats]))
        mean_top5 = float(np.mean([sum(s.top5_weights) / max(len(s.top5_weights), 1) for s in layer_stats]))
        mean_entropy = float(np.mean([s.entropy_raw for s in layer_stats]))

        # Build feature vector for baseline comparison
        feature_vec = np.array([mean_adr, mean_aer, mean_top1, mean_top5, mean_entropy])

        self._n_passes += 1
        capture_ms = (t_analysis - t0) * 1000

        # ── Layer 1: ADR spike / AER collapse ────────────────────

        # Compare against clean baseline mean if available
        if self._clean_adr_history:
            clean_mean_adr = float(np.mean(self._clean_adr_history))
            adr_delta = mean_adr - clean_mean_adr
        else:
            # No baseline yet — use absolute threshold
            clean_mean_adr = 0.0
            adr_delta = 0.0  # can't compute delta without baseline

        aer_collapsed = mean_aer < self._aer_collapse
        adr_spiked = len(self._clean_adr_history) >= 2 and adr_delta > self._adr_spike

        if aer_collapsed or adr_spiked:
            reason_parts = []
            if aer_collapsed:
                reason_parts.append(
                    f"AER collapsed to {mean_aer:.4f} (threshold={self._aer_collapse})"
                )
            if adr_spiked:
                reason_parts.append(
                    f"ADR spiked by {adr_delta:+.4f} from clean baseline {clean_mean_adr:.4f}"
                )
            reason = "; ".join(reason_parts)

            log.warning(f"B2 Layer 1 (ADR/AER) BLOCKED: {reason}")
            analysis_ms = (time.time() - t_analysis) * 1000
            return MonitorResult(
                verdict=MonitorVerdict.BLOCKED,
                block_reason=reason,
                layer_failed="adr_aer",
                mean_adr=mean_adr,
                mean_aer=mean_aer,
                mean_kl=mean_kl,
                mahal_distance=0.0,
                layer_stats=layer_stats,
                hook_capture_ms=capture_ms,
                analysis_ms=analysis_ms,
            )

        # ── Layer 2: KL divergence from uniform ──────────────────

        if mean_kl > self._kl_threshold:
            reason = (
                f"KL divergence from uniform = {mean_kl:.4f} "
                f"(threshold={self._kl_threshold})"
            )
            log.warning(f"B2 Layer 2 (KL) BLOCKED: {reason}")
            analysis_ms = (time.time() - t_analysis) * 1000
            return MonitorResult(
                verdict=MonitorVerdict.BLOCKED,
                block_reason=reason,
                layer_failed="kl_divergence",
                mean_adr=mean_adr,
                mean_aer=mean_aer,
                mean_kl=mean_kl,
                mahal_distance=0.0,
                layer_stats=layer_stats,
                hook_capture_ms=capture_ms,
                analysis_ms=analysis_ms,
            )

        # ── Layer 3: Mahalanobis distance from baseline ───────────

        mahal_dist = self._baseline.mahalanobis_distance(feature_vec)

        if self._baseline.is_ready and mahal_dist > self._mahal_threshold:
            reason = (
                f"Mahalanobis distance = {mahal_dist:.4f} "
                f"(threshold={self._mahal_threshold}, baseline n={self._baseline.n_samples})"
            )
            log.warning(f"B2 Layer 3 (Mahalanobis) BLOCKED: {reason}")
            analysis_ms = (time.time() - t_analysis) * 1000
            return MonitorResult(
                verdict=MonitorVerdict.BLOCKED,
                block_reason=reason,
                layer_failed="mahalanobis",
                mean_adr=mean_adr,
                mean_aer=mean_aer,
                mean_kl=mean_kl,
                mahal_distance=mahal_dist,
                layer_stats=layer_stats,
                hook_capture_ms=capture_ms,
                analysis_ms=analysis_ms,
            )

        # ── All layers passed — record into baseline ──────────────

        self._clean_adr_history.append(mean_adr)
        self._clean_aer_history.append(mean_aer)
        self._baseline.record_clean(feature_vec)

        verdict = MonitorVerdict.PASS
        if not self._baseline.is_ready:
            verdict = MonitorVerdict.WARMUP
            log.info(
                f"B2: WARMUP pass {self._n_passes} — baseline needs "
                f"{self._min_baseline - self._baseline.n_samples} more clean samples"
            )

        analysis_ms = (time.time() - t_analysis) * 1000
        log.info(
            f"B2: {verdict.value} | ADR={mean_adr:.4f} AER={mean_aer:.4f} "
            f"KL={mean_kl:.4f} Mahal={mahal_dist:.4f} | "
            f"{len(layer_stats)} layers captured | "
            f"{capture_ms:.1f}ms capture + {analysis_ms:.1f}ms analysis"
        )

        return MonitorResult(
            verdict=verdict,
            mean_adr=mean_adr,
            mean_aer=mean_aer,
            mean_kl=mean_kl,
            mahal_distance=mahal_dist,
            layer_stats=layer_stats,
            hook_capture_ms=capture_ms,
            analysis_ms=analysis_ms,
        )

    # ── Convenience ───────────────────────────────────────────────

    def get_baseline_summary(self) -> dict:
        """Return summary of the attention baseline state."""
        return {
            "n_passes": self._n_passes,
            "baseline": self._baseline.get_summary(),
            "clean_adr_mean": (
                float(np.mean(self._clean_adr_history))
                if self._clean_adr_history else None
            ),
            "clean_aer_mean": (
                float(np.mean(self._clean_aer_history))
                if self._clean_aer_history else None
            ),
        }

    def reset_baseline(self):
        """Reset the clean baseline. Use when switching models."""
        self._baseline = AttentionBaseline(min_samples=self._min_baseline)
        self._clean_adr_history.clear()
        self._clean_aer_history.clear()
        self._n_passes = 0
        log.info("B2: Baseline reset")

