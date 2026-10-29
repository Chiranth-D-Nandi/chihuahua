"""
FISSION — defenses/b2_attention/attention_baseline.py
======================================================
Running statistical baseline of clean attention patterns.

Maintains a per-layer history of attention feature vectors (ADR, AER,
top-k weight concentration) from confirmed-clean forward passes.
Computes Mahalanobis distance for new observations against this baseline
to detect out-of-distribution attention patterns without hardcoded
thresholds — the thresholds are learned from real clean data.

Used by AttentionMonitor as Layer 3 (statistical outlier gating).

CHIRANTH — B2 defense.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("fission.b2_baseline")


@dataclass
class BaselineStats:
    """Running statistics for one feature dimension."""
    mean: float = 0.0
    var: float = 0.0
    n: int = 0

    def update(self, value: float):
        """Welford's online algorithm for streaming mean + variance."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.var += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0  # avoid division by zero
        return max(np.sqrt(self.var / (self.n - 1)), 1e-8)


class AttentionBaseline:
    """
    Maintains a running baseline of clean attention patterns.

    Feature vector per observation: [adr, aer, top1_frac, top5_frac, entropy_raw]
    These are computed from real attention weights, not hardcoded.

    Once min_samples clean observations are recorded, new observations
    are scored via Mahalanobis distance against the running mean + covariance.
    """

    FEATURE_NAMES = ["adr", "aer", "top1_frac", "top5_frac", "entropy_raw"]

    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self._features: List[np.ndarray] = []  # list of feature vectors
        self._stats = {name: BaselineStats() for name in self.FEATURE_NAMES}
        self._cov_matrix: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._mean_vec: Optional[np.ndarray] = None
        self._dirty = True  # recompute cov on next distance query

    @property
    def n_samples(self) -> int:
        return len(self._features)

    @property
    def is_ready(self) -> bool:
        return self.n_samples >= self.min_samples

    def record_clean(self, feature_vec: np.ndarray):
        """
        Record a feature vector from a confirmed-clean forward pass.

        Args:
            feature_vec: shape (5,) — [adr, aer, top1_frac, top5_frac, entropy_raw]
        """
        assert feature_vec.shape == (len(self.FEATURE_NAMES),), \
            f"Expected shape ({len(self.FEATURE_NAMES)},), got {feature_vec.shape}"

        self._features.append(feature_vec.copy())
        for i, name in enumerate(self.FEATURE_NAMES):
            self._stats[name].update(float(feature_vec[i]))
        self._dirty = True

        log.debug(
            f"Baseline updated — n={self.n_samples} | "
            f"means=[{', '.join(f'{self._stats[n].mean:.4f}' for n in self.FEATURE_NAMES)}]"
        )

    def _recompute_covariance(self):
        """Recompute covariance matrix + inverse from stored features."""
        if self.n_samples < self.min_samples:
            self._cov_inv = None
            self._mean_vec = None
            return

        X = np.array(self._features)  # (N, 5)
        self._mean_vec = X.mean(axis=0)

        # Regularised covariance to avoid singular matrices at low N
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Tikhonov regularisation — add small diagonal to ensure invertibility
        reg = 1e-4 * np.eye(cov.shape[0])
        self._cov_matrix = cov + reg

        try:
            self._cov_inv = np.linalg.inv(self._cov_matrix)
        except np.linalg.LinAlgError:
            log.warning("Covariance matrix singular even after regularisation — using pseudo-inverse")
            self._cov_inv = np.linalg.pinv(self._cov_matrix)

        self._dirty = False

    def mahalanobis_distance(self, feature_vec: np.ndarray) -> float:
        """
        Compute Mahalanobis distance of a new observation from the baseline.

        Returns:
            Mahalanobis distance as a float. Higher = more anomalous.
            Returns 0.0 if baseline is not yet ready (not enough samples).
        """
        if not self.is_ready:
            return 0.0

        if self._dirty:
            self._recompute_covariance()

        if self._cov_inv is None or self._mean_vec is None:
            return 0.0

        diff = feature_vec - self._mean_vec
        # d = sqrt( (x-μ)^T Σ^{-1} (x-μ) )
        mahal_sq = float(diff @ self._cov_inv @ diff)
        return float(np.sqrt(max(mahal_sq, 0.0)))

    def get_summary(self) -> dict:
        """Return human-readable summary of baseline state."""
        return {
            "n_samples": self.n_samples,
            "is_ready": self.is_ready,
            "means": {
                name: round(self._stats[name].mean, 4)
                for name in self.FEATURE_NAMES
            },
            "stds": {
                name: round(self._stats[name].std, 4)
                for name in self.FEATURE_NAMES
            },
        }

