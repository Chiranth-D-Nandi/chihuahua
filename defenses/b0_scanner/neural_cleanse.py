"""
FISSION — defenses/b0_scanner/neural_cleanse.py

=====================================================
B0 — Neural Cleanse Backdoor Detector (Chiranth's module)

Implements the Neural Cleanse algorithm (Wang et al., 2019)
adapted for Vision-Language Model tool-calling pipelines.

Core idea:
    Reverse-engineer a minimal trigger pattern (mask + pattern)
    that forces the model to output the injected target class
    (query_database) for clean inputs. The L1-norm of this mask
    is anomalously small for backdoored models.

Pipeline:
    1. Load surrogate CLIPProbe (or full VLM vision encoder)
    2. For each possible target class, optimize a small trigger
    3. Compute L1 norm of the optimized mask
    4. Use MAD (Median Absolute Deviation) outlier detection
    5. Flag model if any class shows anomalously small mask norm

Metrics reported:
    - Anomaly Index (AI) per class
    - Backdoor detected flag
    - Reverse-engineered trigger (mask + pattern) saved for inspection

Usage (from project root):
    python defenses/b0_scanner/neural_cleanse.py --probe attacks/r1_pixel/probe.pt

References:
    - Neural Cleanse: Identifying and Mitigating Backdoor Attacks
      in Neural Networks (IEEE S&P 2019)
    - Adapted for FISSION multi-agent VLM context

CHIRANTH D NANDI — B0 Neural Cleanse owner
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from attacks.r1_pixel.generate_adversarial_dataset import CLIPProbe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.b0.neural_cleanse")


# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class NeuralCleanseConfig:
    """Hyperparameters for Neural Cleanse reverse engineering."""

    # Trigger size (percentage of image)
    mask_size_ratio: float = 0.03          # 3% of image area
    # Optimization
    num_iterations: int = 300
    learning_rate: float = 0.1
    # Anomaly detection
    mad_threshold: float = 2.0             # MAD multiplier for outlier
    # Regularization
    mask_reg: float = 0.0001               # L1 regularization on mask
    pattern_reg: float = 0.0001            # L2 regularization on pattern
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
#  Neural Cleanse Detector
# ══════════════════════════════════════════════════════════════

class NeuralCleanseDetector:
    """
    Implements Neural Cleanse backdoor detection for FISSION.

    Detects whether the surrogate probe (or VLM vision encoder)
    has been poisoned with a backdoor that forces a specific
    tool call (typically query_database).
    """

    def __init__(
        self,
        probe: nn.Module,
        num_classes: int,
        tool_names: List[str],
        config: Optional[NeuralCleanseConfig] = None,
    ):
        self.probe = probe
        self.probe.eval()
        self.num_classes = num_classes
        self.tool_names = tool_names
        self.cfg = config or NeuralCleanseConfig()
        self.device = torch.device(self.cfg.device)

        # Move probe to device
        self.probe.to(self.device)

        log.info(f"NeuralCleanseDetector initialised | classes={num_classes} | device={self.device}")

    # ────────────────────────────────────────────────────────────
    #  Core reverse-engineering step
    # ────────────────────────────────────────────────────────────

    def _reverse_engineer_trigger(
        self,
        target_class: int,
        image_shape: Tuple[int, int, int] = (3, 224, 224),
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Reverse-engineer a minimal trigger (mask + pattern) that
        forces the model to output the target class for clean inputs.

        Returns:
            mask:      (1, 1, H, W) in [0,1]
            pattern:   (1, C, H, W) in [0,1]
            final_loss: scalar loss after optimization
        """
        H, W = image_shape[1], image_shape[2]

        # Initialise mask (small region) and pattern
        mask = torch.zeros(1, 1, H, W, device=self.device, requires_grad=True)
        pattern = torch.rand(1, 3, H, W, device=self.device, requires_grad=True)

        # Optimiser
        optimizer = Adam([mask, pattern], lr=self.cfg.learning_rate)

        # Target label
        target = torch.tensor([target_class], device=self.device)

        for step in range(self.cfg.num_iterations):
            optimizer.zero_grad()

            # Apply trigger: x' = (1 - mask) * x + mask * pattern
            # We use a dummy clean image (all zeros) for reverse engineering
            dummy_input = torch.zeros(1, 3, H, W, device=self.device)

            triggered = (1 - mask) * dummy_input + mask * pattern
            triggered = torch.clamp(triggered, 0.0, 1.0)

            # Forward through probe
            logits = self.probe(triggered)
            ce_loss = F.cross_entropy(logits, target)

            # Regularization: encourage small mask + smooth pattern
            mask_l1 = torch.sum(torch.abs(mask))
            pattern_l2 = torch.sum(pattern ** 2)

            total_loss = (
                ce_loss
                + self.cfg.mask_reg * mask_l1
                + self.cfg.pattern_reg * pattern_l2
            )

            total_loss.backward()
            optimizer.step()

            # Project mask to [0,1] and keep small
            with torch.no_grad():
                mask.data = torch.clamp(mask.data, 0.0, 1.0)

            if step % 50 == 0 or step == self.cfg.num_iterations - 1:
                log.debug(
                    f"Target {target_class} | step {step:3d} | "
                    f"loss={total_loss.item():.4f} | "
                    f"mask_L1={mask_l1.item():.4f}"
                )

        # Final mask L1 norm (key anomaly signal)
        final_mask_l1 = torch.sum(torch.abs(mask)).item()

        return mask.detach(), pattern.detach(), final_mask_l1

    # ────────────────────────────────────────────────────────────
    #  Anomaly detection (MAD)
    # ────────────────────────────────────────────────────────────

    def _compute_anomaly_index(self, mask_norms: List[float]) -> List[float]:
        """
        Compute MAD-based anomaly index for each class.

        Anomaly Index = |x - median| / MAD
        (mad_threshold, e.g. 2.0, is applied separately in detect() as a
        cutoff on this value — it is not part of this formula itself;
        the previous docstring here incorrectly implied it was a divisor
        term, found while writing this module's first test coverage.)

        NOTE: this uses raw MAD, not the 1.4826-scaled "consistent" MAD
        some Neural Cleanse implementations use to make MAD comparable to
        standard deviation under a normality assumption. That's a
        legitimate implementation choice, not a bug — flagging it here
        since it changes what mad_threshold=2.0 actually means in
        practice, in case it's intended to match literature conventions.
        """
        norms = np.array(mask_norms)
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))

        if mad == 0:
            mad = 1e-8  # avoid division by zero

        anomaly_indices = np.abs(norms - median) / (mad + 1e-8)
        return anomaly_indices.tolist()

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    def detect(self, image_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict:
        """
        Run full Neural Cleanse detection across all classes.

        Returns a dictionary with:
            - per_class: list of results for each class
            - anomaly_indices: list of anomaly scores
            - backdoor_detected: bool
            - poisoned_class: str or None
            - max_anomaly_index: float
        """
        log.info("Starting Neural Cleanse reverse engineering...")

        mask_norms = []
        results = []

        for class_idx in range(self.num_classes):
            tool_name = self.tool_names[class_idx]
            log.info(f"  Reverse-engineering trigger for class {class_idx} ({tool_name})...")

            mask, pattern, mask_l1 = self._reverse_engineer_trigger(class_idx, image_size)
            mask_norms.append(mask_l1)

            results.append({
                "class_idx": class_idx,
                "tool_name": tool_name,
                "mask_l1_norm": round(mask_l1, 6),
                "mask": mask.cpu(),
                "pattern": pattern.cpu(),
            })

            log.info(f"    → mask L1 norm = {mask_l1:.6f}")

        # MAD anomaly detection
        anomaly_indices = self._compute_anomaly_index(mask_norms)

        for i, ai in enumerate(anomaly_indices):
            results[i]["anomaly_index"] = round(ai, 4)

        max_ai = max(anomaly_indices)
        backdoor_detected = max_ai > self.cfg.mad_threshold

        poisoned_class_idx = int(np.argmax(anomaly_indices)) if backdoor_detected else None
        poisoned_tool = self.tool_names[poisoned_class_idx] if poisoned_class_idx is not None else None

        summary = {
            "per_class": results,
            "anomaly_indices": [round(x, 4) for x in anomaly_indices],
            "backdoor_detected": backdoor_detected,
            "poisoned_class": poisoned_tool,
            "poisoned_class_idx": poisoned_class_idx,
            "max_anomaly_index": round(max_ai, 4),
            "mad_threshold": self.cfg.mad_threshold,
            "median_mask_norm": round(float(np.median(mask_norms)), 6),
        }

        log.info("Neural Cleanse complete.")
        log.info(f"  Max anomaly index: {summary['max_anomaly_index']:.4f}")
        log.info(f"  Backdoor detected: {backdoor_detected}")
        if backdoor_detected:
            log.warning(f"  ⚠️  Likely poisoned class: {poisoned_tool} (idx={poisoned_class_idx})")

        return summary

    def save_triggers(self, detection_result: Dict, output_dir: str = "b0_triggers"):
        """Save the reverse-engineered triggers for the flagged class."""
        if not detection_result["backdoor_detected"]:
            log.info("No backdoor detected — no triggers to save.")
            return

        # Directory creation moved here, after the early-return above.
        # Previously this ran unconditionally at the top of the function,
        # so calling save_triggers() on a clean (non-backdoored) result
        # still created an empty output_dir on disk — a surprising side
        # effect contradicting the "no triggers to save" log message
        # right above. Found via tests/test_neural_cleanse.py.
        os.makedirs(output_dir, exist_ok=True)

        poisoned_idx = detection_result["poisoned_class_idx"]
        entry = detection_result["per_class"][poisoned_idx]

        mask_path = os.path.join(output_dir, f"mask_{entry['tool_name']}.pt")
        pattern_path = os.path.join(output_dir, f"pattern_{entry['tool_name']}.pt")

        torch.save(entry["mask"], mask_path)
        torch.save(entry["pattern"], pattern_path)

        log.info(f"Saved reverse-engineered triggers to {output_dir}/")


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FISSION B0 — Neural Cleanse Detector")
    parser.add_argument("--probe", default="attacks/r1_pixel/probe.pt",
                        help="Path to trained CLIPProbe")
    parser.add_argument("--output", default="b0_neural_cleanse_results.json",
                        help="Where to save detection summary")
    parser.add_argument("--save-triggers", action="store_true",
                        help="Save reverse-engineered triggers if backdoor found")
    args = parser.parse_args()

    # Load probe
    if not os.path.exists(args.probe):
        log.error(f"Probe not found: {args.probe}")
        log.error("Run attacks/r1_pixel/train_probe.py first.")
        sys.exit(1)

    log.info(f"Loading probe from {args.probe}...")
    probe_data = torch.load(args.probe, map_location="cpu")

    # Reconstruct CLIPProbe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = CLIPProbe(
        probe_path=args.probe,
        num_classes=len(probe_data["tool_classes"]),
        device=device,
    )

    detector = NeuralCleanseDetector(
        probe=probe,
        num_classes=len(probe_data["tool_classes"]),
        tool_names=probe_data["tool_classes"],
    )

    # Run detection
    result = detector.detect()

    # Print summary
    print("\n" + "=" * 60)
    print("  FISSION B0 — Neural Cleanse Results")
    print("=" * 60)
    print(f"  Backdoor detected : {result['backdoor_detected']}")
    if result["backdoor_detected"]:
        print(f"  Poisoned class    : {result['poisoned_class']} (idx={result['poisoned_class_idx']})")
    print(f"  Max anomaly index : {result['max_anomaly_index']:.4f}")
    print(f"  MAD threshold     : {result['mad_threshold']}")
    print("=" * 60)

    # Per-class table
    print("\n  Per-class anomaly scores:")
    print("  " + "-" * 50)
    print(f"  {'Tool':<25} {'Mask L1':>10} {'Anomaly':>8}")
    print("  " + "-" * 50)
    for entry in result["per_class"]:
        print(
            f"  {entry['tool_name']:<25} "
            f"{entry['mask_l1_norm']:>10.6f} "
            f"{entry['anomaly_index']:>8.4f}"
        )
    print("  " + "-" * 50)

    # Save JSON summary
    import json
    serializable = {k: v for k, v in result.items() if k not in ["per_class"]}
    # Remove tensors from per_class for JSON
    clean_per_class = []
    for entry in result["per_class"]:
        clean_entry = {k: v for k, v in entry.items() if k not in ["mask", "pattern"]}
        clean_per_class.append(clean_entry)
    serializable["per_class"] = clean_per_class

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)

    log.info(f"Detection summary saved to {args.output}")

    if args.save_triggers and result["backdoor_detected"]:
        detector.save_triggers(result)


if __name__ == "__main__":
    main()
