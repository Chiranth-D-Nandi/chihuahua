"""
FISSION — tests/test_attention_baseline.py
=============================================
Unit tests for defenses/b2_attention/attention_baseline.py.

Unlike attention_monitor.py (which needs a real transformer model),
this module is pure statistics — Welford's online mean/variance and
Mahalanobis distance with Tikhonov-regularised covariance — with zero
model dependency. Tested directly and thoroughly against numpy as a
reference implementation.

Run with:
    python -m pytest tests/test_attention_baseline.py -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from defenses.b2_attention.attention_baseline import AttentionBaseline, BaselineStats


class TestBaselineStatsWelford(unittest.TestCase):
    """Welford's online algorithm, checked against numpy as ground truth."""

    def test_mean_matches_numpy(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = BaselineStats()
        for v in vals:
            s.update(v)
        self.assertAlmostEqual(s.mean, float(np.mean(vals)), places=9)

    def test_std_matches_numpy_sample_std(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = BaselineStats()
        for v in vals:
            s.update(v)
        self.assertAlmostEqual(s.std, float(np.std(vals, ddof=1)), places=9)

    def test_std_with_larger_random_sequence(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(loc=5.0, scale=2.0, size=200)
        s = BaselineStats()
        for v in vals:
            s.update(float(v))
        self.assertAlmostEqual(s.mean, float(np.mean(vals)), places=6)
        self.assertAlmostEqual(s.std, float(np.std(vals, ddof=1)), places=6)

    def test_single_sample_std_avoids_division_by_zero(self):
        s = BaselineStats()
        s.update(3.0)
        self.assertEqual(s.n, 1)
        self.assertEqual(s.std, 1.0, "n<2 should return the documented fallback of 1.0")

    def test_zero_samples_std_is_fallback(self):
        s = BaselineStats()
        self.assertEqual(s.std, 1.0)

    def test_identical_values_std_floored_not_zero(self):
        # All-identical input -> true variance is 0. std should be
        # floored at 1e-8, not literally 0.0, so downstream division
        # (Mahalanobis distance) can never divide by zero.
        s = BaselineStats()
        for _ in range(10):
            s.update(7.0)
        self.assertGreater(s.std, 0.0)
        self.assertAlmostEqual(s.std, 1e-8, places=12)


class TestAttentionBaselineRecording(unittest.TestCase):

    def test_n_samples_and_is_ready(self):
        baseline = AttentionBaseline(min_samples=3)
        self.assertEqual(baseline.n_samples, 0)
        self.assertFalse(baseline.is_ready)

        for i in range(3):
            baseline.record_clean(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

        self.assertEqual(baseline.n_samples, 3)
        self.assertTrue(baseline.is_ready)

    def test_wrong_shape_raises(self):
        baseline = AttentionBaseline(min_samples=3)
        with self.assertRaises(AssertionError):
            baseline.record_clean(np.array([0.1, 0.2, 0.3]))  # wrong length

    def test_record_clean_does_not_alias_input_array(self):
        # record_clean copies the array — mutating the caller's array
        # afterward must not corrupt the stored baseline data.
        baseline = AttentionBaseline(min_samples=1)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline.record_clean(vec)
        vec[:] = 0.0
        self.assertTrue(np.array_equal(baseline._features[0], [1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_get_summary_structure(self):
        baseline = AttentionBaseline(min_samples=2)
        baseline.record_clean(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        baseline.record_clean(np.array([0.15, 0.25, 0.35, 0.45, 0.55]))
        summary = baseline.get_summary()
        self.assertEqual(summary["n_samples"], 2)
        self.assertTrue(summary["is_ready"])
        self.assertEqual(set(summary["means"].keys()), set(AttentionBaseline.FEATURE_NAMES))
        self.assertEqual(set(summary["stds"].keys()), set(AttentionBaseline.FEATURE_NAMES))


class TestMahalanobisDistance(unittest.TestCase):

    def test_zero_before_ready(self):
        baseline = AttentionBaseline(min_samples=5)
        baseline.record_clean(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        # Only 1 sample recorded, min_samples=5 -> not ready yet.
        dist = baseline.mahalanobis_distance(np.array([10.0, 10.0, 10.0, 10.0, 10.0]))
        self.assertEqual(dist, 0.0)

    def test_distance_at_the_mean_is_near_zero(self):
        baseline = AttentionBaseline(min_samples=5)
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=0.5, scale=0.05, size=(30, 5))
        for row in samples:
            baseline.record_clean(row)

        mean_vec = samples.mean(axis=0)
        dist = baseline.mahalanobis_distance(mean_vec)
        self.assertLess(dist, 0.5, "distance at the exact sample mean should be small")

    def test_far_outlier_has_much_larger_distance_than_typical_point(self):
        baseline = AttentionBaseline(min_samples=5)
        rng = np.random.default_rng(1)
        samples = rng.normal(loc=0.5, scale=0.05, size=(30, 5))
        for row in samples:
            baseline.record_clean(row)

        typical_dist = baseline.mahalanobis_distance(samples[0])
        outlier = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        outlier_dist = baseline.mahalanobis_distance(outlier)
        self.assertGreater(outlier_dist, typical_dist * 10)

    def test_identical_clean_samples_do_not_crash_singular_covariance(self):
        # All recorded samples identical -> true covariance is exactly
        # singular (rank 0). Tikhonov regularisation should keep this
        # invertible; if not, the code falls back to pinv. Either way,
        # this must not raise.
        baseline = AttentionBaseline(min_samples=5)
        for _ in range(10):
            baseline.record_clean(np.array([0.3, 0.3, 0.3, 0.3, 0.3]))

        dist = baseline.mahalanobis_distance(np.array([0.3, 0.3, 0.3, 0.3, 0.3]))
        self.assertFalse(dist != dist, "distance must not be NaN")
        self.assertGreaterEqual(dist, 0.0)

        far_dist = baseline.mahalanobis_distance(np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
        self.assertFalse(far_dist != far_dist)
        self.assertGreater(far_dist, dist)

    def test_distance_is_never_negative(self):
        baseline = AttentionBaseline(min_samples=5)
        rng = np.random.default_rng(2)
        for _ in range(20):
            baseline.record_clean(rng.normal(0.5, 0.1, size=5))
        for _ in range(10):
            probe = rng.normal(0.5, 5.0, size=5)  # some wild, some tame
            dist = baseline.mahalanobis_distance(probe)
            self.assertGreaterEqual(dist, 0.0)

    def test_caching_recompute_on_dirty_flag(self):
        # After a new record_clean(), covariance should be marked dirty
        # and recomputed on next distance query — not silently stale.
        baseline = AttentionBaseline(min_samples=3)
        for _ in range(5):
            baseline.record_clean(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
        baseline.mahalanobis_distance(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))  # forces first compute
        self.assertFalse(baseline._dirty)

        baseline.record_clean(np.array([9.0, 9.0, 9.0, 9.0, 9.0]))  # wildly different sample
        self.assertTrue(baseline._dirty, "recording a new sample must mark baseline dirty again")


if __name__ == "__main__":
    unittest.main(verbosity=2)

