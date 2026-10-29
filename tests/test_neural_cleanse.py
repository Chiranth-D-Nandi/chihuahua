"""
FISSION — tests/test_neural_cleanse.py
=========================================
Unit tests for defenses/b0_scanner/neural_cleanse.py.

This module's core trigger reverse-engineering is real gradient descent
(default: 300 Adam iterations per class, on a 224x224x3 tensor through
the probe) — far too slow to run at full scale in a unit test. This file
tests in three tiers instead of trying to run the real thing at full
size:

  1. TestComputeAnomalyIndex — the MAD-based anomaly scoring is pure,
     fast, deterministic math with no torch training involved. Tested
     directly and thoroughly, since this is the actual detection logic
     the rest of the module exists to support.
  2. TestDetectDecisionLogic — detect()'s aggregation/decision behaviour
     (backdoor_detected flag, poisoned_class assignment, per-class
     result structure) is tested by monkeypatching
     _reverse_engineer_trigger to return controlled, instant mask_l1
     values instead of running real optimisation — this tests the
     *decision* logic in isolation from the (expensive, stochastic)
     optimisation it depends on.
  3. TestEndToEndSmoke — one real, unmocked run through
     _reverse_engineer_trigger, but with a tiny image size and a
     handful of iterations (not the default 300), on a small linear
     probe — just enough to catch shape/wiring bugs (e.g. a tensor
     dimension mismatch) that the mocked tests in tier 2 cannot see,
     without taking minutes to run.

Run with:
    python -m pytest tests/test_neural_cleanse.py -v
"""

import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from defenses.b0_scanner.neural_cleanse import (
    NeuralCleanseDetector,
    NeuralCleanseConfig,
)


class _TinyProbe(nn.Module):
    """Minimal linear probe standing in for CLIPProbe in tests — avoids
    loading a real CLIP ViT-L/14 model just to test detector logic that
    doesn't care what's inside the probe."""

    def __init__(self, in_features=3 * 8 * 8, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))


def _make_detector(num_classes=3, **cfg_overrides):
    probe = _TinyProbe(num_classes=num_classes)
    tool_names = [f"tool_{i}" for i in range(num_classes)]
    config = NeuralCleanseConfig(device="cpu", **cfg_overrides)
    return NeuralCleanseDetector(
        probe=probe, num_classes=num_classes, tool_names=tool_names, config=config
    )


class TestComputeAnomalyIndex(unittest.TestCase):
    """Pure MAD-based anomaly scoring — the actual detection math."""

    def setUp(self):
        self.detector = _make_detector()

    def test_all_equal_gives_zero_anomaly(self):
        # No spread at all -> every point is exactly at the median.
        result = self.detector._compute_anomaly_index([1.0, 1.0, 1.0])
        for ai in result:
            self.assertAlmostEqual(ai, 0.0, places=6)

    def test_symmetric_spread_around_median(self):
        # [1, 2, 3]: median=2, deviations=[1,0,1], MAD=median([1,0,1])=1
        # -> anomaly indices = [1/1, 0/1, 1/1] = [1, 0, 1]
        result = self.detector._compute_anomaly_index([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result[0], 1.0, places=3)
        self.assertAlmostEqual(result[1], 0.0, places=3)
        self.assertAlmostEqual(result[2], 1.0, places=3)

    def test_clear_outlier_gets_highest_anomaly_index(self):
        norms = [0.95, 1.05, 1.00, 0.02]  # index 3 is the backdoor-sized outlier
        result = self.detector._compute_anomaly_index(norms)
        outlier_idx = result.index(max(result))
        self.assertEqual(outlier_idx, 3)
        # And it should clear the default mad_threshold of 2.0 by a wide margin.
        self.assertGreater(result[3], 2.0)

    def test_degenerate_zero_mad_does_not_crash(self):
        # 3 of 4 values identical -> MAD collapses to exactly 0 before the
        # 1e-8 floor kicks in. Documents real behaviour: the single
        # differing value's anomaly index becomes extremely large (not
        # NaN/inf, not a crash) rather than a small bounded number. This
        # is mathematically correct MAD behaviour, but it means with few
        # classes (FISSION has 3) any near-duplicate pair among "clean"
        # norms makes the detector very oversensitive to the third value.
        result = self.detector._compute_anomaly_index([1.0, 1.0, 1.0, 0.5])
        for ai in result:
            self.assertFalse(
                ai != ai, "anomaly index should never be NaN"  # ai != ai is the NaN check
            )
        self.assertGreater(result[3], 1000, "degenerate MAD should blow up, not stay small")

    def test_two_points(self):
        # Smallest n where median/MAD are still well-defined.
        result = self.detector._compute_anomaly_index([1.0, 2.0])
        self.assertEqual(len(result), 2)
        for ai in result:
            self.assertFalse(ai != ai)  # no NaN

    def test_negative_and_positive_norms_handled(self):
        # mask_l1 is a sum of absolute values so should never be
        # negative in real use, but the function itself is pure math
        # over whatever list it's given — confirm it doesn't assume
        # non-negativity anywhere (e.g. via an unguarded sqrt).
        result = self.detector._compute_anomaly_index([-1.0, 0.0, 1.0])
        self.assertEqual(len(result), 3)
        for ai in result:
            self.assertFalse(ai != ai)


class TestDetectDecisionLogic(unittest.TestCase):
    """
    detect()'s aggregation/decision behaviour, isolated from the
    expensive real optimisation via a monkeypatched
    _reverse_engineer_trigger that returns instantly.
    """

    def _fake_trigger_factory(self, mask_l1_by_class):
        """Builds a drop-in replacement for _reverse_engineer_trigger
        that returns a plausible-shaped (mask, pattern) pair instantly,
        with a caller-controlled mask_l1 value per class."""
        def _fake(self_, target_class, image_shape=(3, 224, 224)):
            H, W = image_shape[1], image_shape[2]
            mask = torch.zeros(1, 1, H, W)
            pattern = torch.zeros(1, 3, H, W)
            return mask, pattern, mask_l1_by_class[target_class]
        return _fake

    def test_no_backdoor_when_norms_genuinely_tight(self):
        # Values within ~1% of each other -> should not be flagged.
        detector = _make_detector(num_classes=3)
        fake = self._fake_trigger_factory([10.0, 10.05, 9.95])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        self.assertFalse(result["backdoor_detected"])
        self.assertIsNone(result["poisoned_class"])

    def test_finding_n3_mad_oversensitive_to_mild_variation(self):
        # FINDING, not a bug in this test: with only 3 classes (FISSION's
        # actual tool count), MAD is the median of just 3 absolute
        # deviations, so it is extremely unstable. [10.0, 10.5, 9.8] is
        # a mild, clearly-benign ~5% spread with no real outlier, yet:
        #   median=10.0, deviations=[0, 0.5, 0.2], MAD=median(...)=0.2
        #   anomaly_indices = [0, 0.5/0.2, 0.2/0.2] = [0, 2.5, 1.0]
        # 2.5 clears the default mad_threshold of 2.0, so this flags a
        # completely clean model as backdoored. This test documents that
        # real, current behaviour rather than asserting it "shouldn't"
        # happen — at n=3 classes, mad_threshold=2.0 may need to be
        # higher, or a different statistic than raw 3-point MAD may be
        # needed, before this detector's false-positive rate on clean
        # models is trustworthy. Flagging for the module owner rather
        # than unilaterally changing detection sensitivity.
        detector = _make_detector(num_classes=3)
        fake = self._fake_trigger_factory([10.0, 10.5, 9.8])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        self.assertTrue(
            result["backdoor_detected"],
            "if this starts failing, the oversensitivity finding above "
            "may have been fixed — great, but please update/remove this "
            "test's docstring rather than deleting it silently."
        )

    def test_backdoor_detected_for_clear_outlier(self):
        detector = _make_detector(num_classes=3)
        # class 2 (tool_2) has a suspiciously small mask -> likely backdoor target
        fake = self._fake_trigger_factory([10.0, 10.5, 0.1])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        self.assertTrue(result["backdoor_detected"])
        self.assertEqual(result["poisoned_class"], "tool_2")
        self.assertEqual(result["poisoned_class_idx"], 2)

    def test_per_class_results_structure(self):
        detector = _make_detector(num_classes=3)
        fake = self._fake_trigger_factory([5.0, 5.0, 5.0])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        self.assertEqual(len(result["per_class"]), 3)
        for entry in result["per_class"]:
            self.assertIn("class_idx", entry)
            self.assertIn("tool_name", entry)
            self.assertIn("mask_l1_norm", entry)
            self.assertIn("anomaly_index", entry)
            self.assertIn("mask", entry)
            self.assertIn("pattern", entry)

    def test_summary_fields_present(self):
        detector = _make_detector(num_classes=3)
        fake = self._fake_trigger_factory([1.0, 1.0, 1.0])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        for key in (
            "per_class", "anomaly_indices", "backdoor_detected",
            "poisoned_class", "poisoned_class_idx", "max_anomaly_index",
            "mad_threshold", "median_mask_norm",
        ):
            self.assertIn(key, result)

    def test_custom_mad_threshold_changes_sensitivity(self):
        # Same mild outlier: default threshold=2.0 might or might not
        # flag it depending on exact spread; a very high threshold
        # should definitely NOT flag it, proving the threshold is
        # actually being respected rather than hardcoded.
        detector = _make_detector(num_classes=3, mad_threshold=1000.0)
        fake = self._fake_trigger_factory([10.0, 10.5, 5.0])
        with patch.object(
            NeuralCleanseDetector, "_reverse_engineer_trigger", fake
        ):
            result = detector.detect(image_size=(3, 8, 8))
        self.assertFalse(result["backdoor_detected"])
        self.assertEqual(result["mad_threshold"], 1000.0)


class TestSaveTriggers(unittest.TestCase):

    def test_no_save_when_no_backdoor(self):
        import shutil
        out_dir = "/tmp/fission_b0_test_nosave"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        detector = _make_detector(num_classes=3)
        fake_result = {"backdoor_detected": False}
        # Should return early without raising, without needing
        # per_class/poisoned_class_idx to be present at all, and
        # critically: without creating output_dir as a side effect.
        detector.save_triggers(fake_result, output_dir=out_dir)
        self.assertFalse(os.path.exists(out_dir))

    def test_save_when_backdoor_detected(self):
        import shutil
        out_dir = "/tmp/fission_b0_test_save"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        detector = _make_detector(num_classes=3)
        fake_result = {
            "backdoor_detected": True,
            "poisoned_class_idx": 1,
            "per_class": [
                {"tool_name": "tool_0", "mask": torch.zeros(1, 1, 4, 4), "pattern": torch.zeros(1, 3, 4, 4)},
                {"tool_name": "tool_1", "mask": torch.ones(1, 1, 4, 4),  "pattern": torch.ones(1, 3, 4, 4)},
                {"tool_name": "tool_2", "mask": torch.zeros(1, 1, 4, 4), "pattern": torch.zeros(1, 3, 4, 4)},
            ],
        }
        detector.save_triggers(fake_result, output_dir=out_dir)

        self.assertTrue(os.path.exists(os.path.join(out_dir, "mask_tool_1.pt")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "pattern_tool_1.pt")))

        saved_mask = torch.load(os.path.join(out_dir, "mask_tool_1.pt"))
        self.assertTrue(torch.equal(saved_mask, torch.ones(1, 1, 4, 4)))

        shutil.rmtree(out_dir)


class TestEndToEndSmoke(unittest.TestCase):
    """
    One real (unmocked) pass through _reverse_engineer_trigger, kept
    small enough to run in well under a second: tiny image, few
    iterations. Not a claim that a tiny/short-optimised trigger is
    representative of the real 224x224/300-iteration result — just a
    check that the actual tensor wiring (shapes, device placement,
    gradient flow, clamping) doesn't silently break.
    """

    def test_real_trigger_optimization_runs_without_crashing(self):
        detector = _make_detector(
            num_classes=3, num_iterations=5, learning_rate=0.1
        )
        mask, pattern, mask_l1 = detector._reverse_engineer_trigger(
            target_class=0, image_shape=(3, 8, 8)
        )
        self.assertEqual(mask.shape, (1, 1, 8, 8))
        self.assertEqual(pattern.shape, (1, 3, 8, 8))
        self.assertTrue(torch.all(mask >= 0.0) and torch.all(mask <= 1.0),
                         "mask must stay clamped to [0,1]")
        self.assertGreaterEqual(mask_l1, 0.0)
        self.assertFalse(mask_l1 != mask_l1, "mask_l1 must not be NaN")

    def test_full_detect_tiny_end_to_end(self):
        # Full detect() with real (tiny/short) optimisation for all
        # classes — the closest thing to an integration test this
        # module can have without taking minutes.
        detector = _make_detector(
            num_classes=3, num_iterations=5, learning_rate=0.1
        )
        result = detector.detect(image_size=(3, 8, 8))
        self.assertEqual(len(result["per_class"]), 3)
        self.assertIn("backdoor_detected", result)
        self.assertIsInstance(result["backdoor_detected"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)

