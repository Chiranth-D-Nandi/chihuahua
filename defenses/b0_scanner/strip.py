"""
FISSION — defenses/b0_scanner/strip.py

=====================================================
B0 — STRIP Backdoor Detector
(STRong Intentional Perturbation, Gao et al., 2019)

Core idea:
    A backdoor trigger dominates the model's decision regardless of
    what else is in the image. So if you superimpose a suspect image
    onto N different random clean images, a CLEAN image's predictions
    will vary a lot across the N blends (each blend looks meaningfully
    different) — but a TRIGGERED image's predictions will barely move,
    because the trigger pattern still wins every time. That collapse
    in prediction entropy is the backdoor signal.

Unlike Neural Cleanse and Activation Clustering, STRIP needs no access
to model internals or weights — it's inference-only, which makes it the
cheapest method to run against a black-box-ish Ollama-served VLM. Here
it's implemented against the CLIPProbe surrogate for speed/determinism,
but the same superimpose-and-measure-entropy design generalises to
calling the real VLM per perturbation if you want STRIP validated
against LLaVA itself rather than just the surrogate.

Pipeline:
    1. Take a test image (the one you're deciding CLEAN vs TRIGGERED).
    2. Superimpose N random clean images onto it at a fixed blend ratio.
    3. Run each of the N blends through the model, collect predicted
       class for each.
    4. Compute Shannon entropy of the resulting class distribution.
    5. Low entropy (few distinct classes dominate) => likely triggered.
       High entropy (predictions scattered across classes) => likely clean.
    6. Threshold is calibrated against a clean baseline distribution
       (run STRIP on known-clean images first to find where clean
       entropy normally sits).

Metrics reported:
    - entropy per test image
    - triggered: bool (entropy below calibrated threshold)
    - clean_baseline_entropy: mean/std from calibration pass

Usage (from project root):
    python defenses/b0_scanner/strip.py --probe attacks/r1_pixel/probe.pt \\
        --test-image path/to/suspect.png \\
        --clean-glob "adversarial_dataset/synthetic/*_clean.png"

References:
    - STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
      (Gao et al., ACSAC 2019, arXiv:1902.06531)

B0 — STRIP owner: Abhyudhaya M Hegde
"""

import argparse
import glob
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from attacks.r1_pixel.generate_adversarial_dataset import CLIPProbe, SAVE_TRANSFORM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.b0.strip")


# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class STRIPConfig:
    """Hyperparameters for STRIP."""

    n_samples:        int   = getattr(cfg.defense, "strip_n_samples", 20)  # superimpositions per test image (default 20)
    blend_alpha:      float = 0.5    # weight of the clean overlay image (0.5 = equal blend)
    entropy_z_threshold: float = 1.5   # flag if entropy < baseline_mean - z * baseline_std
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed:      Optional[int] = None


# ══════════════════════════════════════════════════════════════
#  STRIP Detector
# ══════════════════════════════════════════════════════════════

class STRIPDetector:
    """
    Runs STRIP superimposition entropy analysis against CLIPProbe.

    Two-step usage:
        1. calibrate(clean_images) once, on a held-out clean set, to
           learn where "normal" entropy sits for this model/tool set.
        2. detect(test_image, clean_pool) per suspect image, comparing
           its entropy against the calibrated baseline.
    """

    def __init__(
        self,
        probe: nn.Module,
        num_classes: int,
        tool_names: List[str],
        config: Optional[STRIPConfig] = None,
    ):
        self.probe = probe
        self.probe.eval()
        self.num_classes = num_classes
        self.tool_names = tool_names
        self.cfg = config or STRIPConfig()
        self.device = torch.device(self.cfg.device)
        self.probe.to(self.device)

        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed)

        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

        log.info(
            f"STRIPDetector initialised | classes={num_classes} | "
            f"n_samples={self.cfg.n_samples} | device={self.device}"
        )

    # ────────────────────────────────────────────────────────────
    #  Core superimposition + entropy step
    # ────────────────────────────────────────────────────────────

    def _superimpose(self, base: Image.Image, overlay: Image.Image) -> Image.Image:
        """Blend `overlay` onto `base` at cfg.blend_alpha, resizing overlay to match."""
        overlay_resized = overlay.resize(base.size)
        return Image.blend(base.convert("RGB"), overlay_resized.convert("RGB"), self.cfg.blend_alpha)

    def _predict_class(self, image: Image.Image) -> int:
        tensor = SAVE_TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.probe(tensor)
            return int(torch.argmax(logits, dim=1).item())

    def _entropy_for_image(self, test_image: Image.Image, clean_pool: List[Image.Image]) -> float:
        """
        Superimpose cfg.n_samples random images from clean_pool onto
        test_image, collect predicted classes, return Shannon entropy
        of the resulting class distribution (in bits).
        """
        n = min(self.cfg.n_samples, len(clean_pool))
        if n < 2:
            raise ValueError(
                f"Need at least 2 clean images in the overlay pool, got {len(clean_pool)}"
            )

        overlays = random.sample(clean_pool, n)
        predictions = []
        for overlay in overlays:
            blended = self._superimpose(test_image, overlay)
            predictions.append(self._predict_class(blended))

        counts = np.bincount(predictions, minlength=self.num_classes)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # drop zero-prob classes before log
        entropy = float(-np.sum(probs * np.log2(probs)))

        return entropy

    # ────────────────────────────────────────────────────────────
    #  Calibration (run once on known-clean images)
    # ────────────────────────────────────────────────────────────

    def calibrate(self, clean_images: List[Image.Image]) -> Dict:
        """
        Compute the baseline entropy distribution for genuinely clean
        images, so detect() has something to compare against. Uses each
        clean image as the "test" image in turn, superimposing the
        REST of the pool onto it.
        """
        if len(clean_images) < self.cfg.n_samples + 1:
            log.warning(
                f"Only {len(clean_images)} clean images available; "
                f"STRIP wants >= {self.cfg.n_samples + 1} for a clean calibration pass "
                f"(1 held out as test + {self.cfg.n_samples} as overlay pool). "
                f"Overlay pool will reuse images — entropy estimate will be noisier."
            )

        entropies = []
        for i, test_img in enumerate(clean_images):
            pool = [img for j, img in enumerate(clean_images) if j != i]
            if not pool:
                continue
            entropies.append(self._entropy_for_image(test_img, pool))

        self.baseline_mean = float(np.mean(entropies))
        self.baseline_std = float(np.std(entropies)) if len(entropies) > 1 else 0.0

        log.info(
            f"STRIP calibration complete | n={len(entropies)} | "
            f"mean_entropy={self.baseline_mean:.4f} | std={self.baseline_std:.4f}"
        )

        return {
            "n_calibration_images": len(entropies),
            "entropies": [round(e, 4) for e in entropies],
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
        }

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    def detect(self, test_image: Image.Image, clean_pool: List[Image.Image]) -> Dict:
        """
        Test a single (possibly triggered) image against the calibrated
        baseline. Call calibrate() first.
        """
        if self.baseline_mean is None:
            raise RuntimeError("Call calibrate() with known-clean images before detect().")

        entropy = self._entropy_for_image(test_image, clean_pool)
        threshold = self.baseline_mean - self.cfg.entropy_z_threshold * max(self.baseline_std, 1e-8)
        triggered = entropy < threshold

        result = {
            "entropy": round(entropy, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
            "threshold": round(threshold, 4),
            "triggered": triggered,
        }

        log.info(
            f"STRIP result | entropy={entropy:.4f} | threshold={threshold:.4f} | "
            f"triggered={triggered}"
        )
        if triggered:
            log.warning("  ⚠️  Entropy collapse detected — likely backdoor trigger dominance")

        return result

    def detect_batch(
        self, test_images: Dict[str, Image.Image], clean_pool: List[Image.Image]
    ) -> Dict:
        """Run detect() over a batch of test images, e.g. the whole adversarial set."""
        results = {}
        for stem, img in test_images.items():
            results[stem] = self.detect(img, clean_pool)

        triggered_count = sum(1 for r in results.values() if r["triggered"])
        summary = {
            "per_image": results,
            "n_tested": len(results),
            "n_triggered": triggered_count,
            "trigger_detection_rate": round(triggered_count / len(results), 4) if results else 0.0,
        }
        return summary


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════

def _load_images(pattern: str) -> Dict[str, Image.Image]:
    images = {}
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        images[stem] = Image.open(path).convert("RGB")
    return images


def main():
    parser = argparse.ArgumentParser(description="FISSION B0 — STRIP Detector")
    parser.add_argument("--probe", default="attacks/r1_pixel/probe.pt",
                        help="Path to trained CLIPProbe")
    parser.add_argument("--clean-glob", default="adversarial_dataset/synthetic/*_clean.png",
                        help="Glob pattern for the clean overlay/calibration pool")
    parser.add_argument("--test-image", default=None,
                        help="Single suspect image to test (optional — omit to just calibrate)")
    parser.add_argument("--output", default="b0_strip_results.json",
                        help="Where to save detection summary")
    args = parser.parse_args()

    if not os.path.exists(args.probe):
        log.error(f"Probe not found: {args.probe}")
        log.error("Run attacks/r1_pixel/train_probe.py first.")
        sys.exit(1)

    probe_data = torch.load(args.probe, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = CLIPProbe(
        probe_path=args.probe,
        num_classes=len(probe_data["tool_classes"]),
        device=device,
    )

    clean_images = _load_images(args.clean_glob)
    if not clean_images:
        log.error(f"No clean images matched {args.clean_glob}")
        sys.exit(1)
    clean_pool = list(clean_images.values())

    detector = STRIPDetector(
        probe=probe,
        num_classes=len(probe_data["tool_classes"]),
        tool_names=probe_data["tool_classes"],
    )

    calibration = detector.calibrate(clean_pool)

    result = {"calibration": calibration}

    if args.test_image:
        test_img = Image.open(args.test_image).convert("RGB")
        result["detection"] = detector.detect(test_img, clean_pool)
    else:
        log.info("No --test-image given — calibration only.")

    print("\n" + "=" * 60)
    print("  FISSION B0 — STRIP Results")
    print("=" * 60)
    print(f"  Baseline mean entropy : {calibration['baseline_mean']:.4f}")
    print(f"  Baseline std          : {calibration['baseline_std']:.4f}")
    if "detection" in result:
        d = result["detection"]
        print(f"  Test image entropy    : {d['entropy']:.4f}")
        print(f"  Threshold             : {d['threshold']:.4f}")
        print(f"  Triggered             : {d['triggered']}")
    print("=" * 60)

    import json
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
