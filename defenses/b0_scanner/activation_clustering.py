"""
FISSION — defenses/b0_scanner/activation_clustering.py

=====================================================
B0 — Activation Clustering Backdoor Detector

Implements Activation Clustering (Chen et al., 2018, "Detecting Backdoor
Attacks on Deep Neural Networks by Activation Clustering") adapted for
FISSION's CLIPProbe surrogate.

Core idea:
    A backdoor trigger causes poisoned samples to reach the target class
    via a DIFFERENT internal pathway than genuinely-clean samples of that
    same class. So even though both groups get labelled identically by
    the model, their penultimate-layer activations separate into two
    distinguishable clusters instead of one homogeneous blob.

Pipeline:
    1. Run the clean probe dataset through CLIPProbe, grouped by the
       class the model predicts for them (not a labelled trigger set —
       we don't have poisoned examples, only clean ones + suspicion).
    2. For each class with enough samples, reduce the 768-d CLIP
       embeddings with PCA, then fit KMeans(k=2).
    3. Compute silhouette score of the 2-cluster split, and the size
       ratio of the smaller cluster.
    4. A class is flagged if the split is BOTH well-separated (high
       silhouette) AND imbalanced (small minority cluster) — a backdoor
       signature. A genuinely bimodal-but-balanced clean class (e.g.
       two visually distinct sub-styles of the same document type)
       should not trip this, which is why both conditions are required
       rather than silhouette alone.

Metrics reported:
    - Per-class silhouette score + minority cluster ratio
    - flagged_classes: classes whose split looks like a backdoor split
    - backdoor_detected: bool (True if any class flagged)

Usage (from project root):
    python defenses/b0_scanner/activation_clustering.py --probe attacks/r1_pixel/probe.pt

References:
    - Activation Clustering (Chen et al. 2018, arXiv:1811.03728)
    - Adapted for FISSION multi-agent VLM context

B0 — Activation Clustering owner: Abhyudhaya M Hegde
"""

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from attacks.r1_pixel.generate_adversarial_dataset import CLIPProbe, SAVE_TRANSFORM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.b0.activation_clustering")


# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class ActivationClusteringConfig:
    """Hyperparameters for Activation Clustering."""

    pca_components:        int   = 10     # dims to reduce 768-d CLIP embeddings to
    min_samples_per_class:  int   = 6      # need at least this many to cluster meaningfully
    silhouette_threshold:  float = 0.25   # above this, the 2-cluster split is "real"
    minority_ratio_max:    float = 0.35   # flag if minority cluster < this fraction
    minority_ratio_min:    float = 0.03   # ignore near-empty clusters (noise, not a trigger group)
    device:                str   = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
#  Activation Clustering Detector
# ══════════════════════════════════════════════════════════════

class ActivationClusteringDetector:
    """
    Runs Activation Clustering against a clean probe dataset.

    Unlike Neural Cleanse (which reverse-engineers a trigger) this method
    only needs forward passes over clean images — no gradient-based
    optimisation — so it's the cheapest of the three SPECTRE methods.
    """

    def __init__(
        self,
        probe: nn.Module,
        tool_names: List[str],
        config: Optional[ActivationClusteringConfig] = None,
    ):
        self.probe = probe
        self.probe.eval()
        self.tool_names = tool_names
        self.cfg = config or ActivationClusteringConfig()
        self.device = torch.device(self.cfg.device)
        self.probe.to(self.device)

        log.info(
            f"ActivationClusteringDetector initialised | "
            f"classes={len(tool_names)} | device={self.device}"
        )

    # ────────────────────────────────────────────────────────────
    #  Feature extraction
    # ────────────────────────────────────────────────────────────

    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Run one image through CLIPProbe's vision encoder and return the
        pre-logit (penultimate) embedding — the same 768-d CLIP feature
        that feeds the linear probe head. This is the "activation" AC
        clusters on, not the final tool-class logits.
        """
        tensor = SAVE_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
            x = (x - self.probe.clip_mean) / self.probe.clip_std
            feat = self.probe.clip_model.encode_image(x).float()
            feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

        return feat.squeeze(0).cpu().numpy()

    def _predicted_class(self, image: Image.Image) -> int:
        tensor = SAVE_TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.probe(tensor)
            return int(torch.argmax(logits, dim=1).item())

    # ────────────────────────────────────────────────────────────
    #  Per-class clustering
    # ────────────────────────────────────────────────────────────

    def _cluster_class(self, embeddings: np.ndarray) -> Tuple[float, float]:
        """
        Reduce embeddings with PCA, fit KMeans(k=2), and return
        (silhouette_score, minority_cluster_ratio).
        """
        n_components = min(self.cfg.pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
        if n_components < 2:
            return 0.0, 0.5  # not enough samples/dims to say anything meaningful

        reduced = PCA(n_components=n_components).fit_transform(embeddings)

        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(reduced)

        try:
            sil = silhouette_score(reduced, labels)
        except ValueError:
            # happens if KMeans collapses to a single effective cluster
            sil = 0.0

        counts = np.bincount(labels)
        minority_ratio = float(counts.min() / counts.sum())

        return float(sil), minority_ratio

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    def detect(self, images_by_stem: Dict[str, Image.Image]) -> Dict:
        """
        Run Activation Clustering across all classes present in the
        clean probe dataset.

        Args:
            images_by_stem: dict of {identifier: PIL.Image} for the
                clean probe set (e.g. loaded from adversarial_dataset's
                *_clean.png files).

        Returns a dictionary with:
            - per_class: list of {tool_name, n_samples, silhouette, minority_ratio, flagged}
            - flagged_classes: list of tool names that look backdoored
            - backdoor_detected: bool
        """
        log.info(f"Extracting embeddings + predictions for {len(images_by_stem)} clean images...")

        embeddings_by_class: Dict[int, List[np.ndarray]] = {}
        for stem, img in images_by_stem.items():
            pred_class = self._predicted_class(img)
            emb = self._extract_embedding(img)
            embeddings_by_class.setdefault(pred_class, []).append(emb)

        results = []
        flagged_classes = []

        for class_idx, embs in embeddings_by_class.items():
            tool_name = (
                self.tool_names[class_idx]
                if class_idx < len(self.tool_names)
                else f"class_{class_idx}"
            )
            n_samples = len(embs)

            if n_samples < self.cfg.min_samples_per_class:
                log.info(
                    f"  {tool_name}: only {n_samples} samples "
                    f"(< {self.cfg.min_samples_per_class}) — skipping, too few to cluster"
                )
                results.append({
                    "tool_name": tool_name,
                    "n_samples": n_samples,
                    "silhouette": None,
                    "minority_ratio": None,
                    "flagged": False,
                    "skipped_reason": "insufficient_samples",
                })
                continue

            stacked = np.stack(embs, axis=0)
            sil, minority_ratio = self._cluster_class(stacked)

            flagged = (
                sil >= self.cfg.silhouette_threshold
                and self.cfg.minority_ratio_min <= minority_ratio <= self.cfg.minority_ratio_max
            )

            log.info(
                f"  {tool_name}: n={n_samples} | silhouette={sil:.4f} | "
                f"minority_ratio={minority_ratio:.4f} | flagged={flagged}"
            )

            results.append({
                "tool_name": tool_name,
                "n_samples": n_samples,
                "silhouette": round(sil, 4),
                "minority_ratio": round(minority_ratio, 4),
                "flagged": flagged,
            })

            if flagged:
                flagged_classes.append(tool_name)

        summary = {
            "per_class": results,
            "flagged_classes": flagged_classes,
            "backdoor_detected": len(flagged_classes) > 0,
            "silhouette_threshold": self.cfg.silhouette_threshold,
            "minority_ratio_range": [self.cfg.minority_ratio_min, self.cfg.minority_ratio_max],
        }

        log.info("Activation Clustering complete.")
        log.info(f"  Backdoor detected: {summary['backdoor_detected']}")
        if flagged_classes:
            log.warning(f"  ⚠️  Flagged classes: {flagged_classes}")

        return summary


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════

def _load_clean_images(pattern: str) -> Dict[str, Image.Image]:
    images = {}
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        images[stem] = Image.open(path).convert("RGB")
    return images


def main():
    parser = argparse.ArgumentParser(description="FISSION B0 — Activation Clustering Detector")
    parser.add_argument("--probe", default="attacks/r1_pixel/probe.pt",
                        help="Path to trained CLIPProbe")
    parser.add_argument("--clean-glob", default="adversarial_dataset/synthetic/*_clean.png",
                        help="Glob pattern for clean probe images")
    parser.add_argument("--output", default="b0_activation_clustering_results.json",
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

    images = _load_clean_images(args.clean_glob)
    if not images:
        log.error(f"No images matched {args.clean_glob}")
        sys.exit(1)

    detector = ActivationClusteringDetector(
        probe=probe,
        tool_names=probe_data["tool_classes"],
    )
    result = detector.detect(images)

    print("\n" + "=" * 60)
    print("  FISSION B0 — Activation Clustering Results")
    print("=" * 60)
    print(f"  Backdoor detected : {result['backdoor_detected']}")
    if result["flagged_classes"]:
        print(f"  Flagged classes   : {result['flagged_classes']}")
    print("=" * 60)

    print("\n  Per-class cluster scores:")
    print("  " + "-" * 60)
    print(f"  {'Tool':<25} {'n':>4} {'Silhouette':>11} {'MinorityRatio':>14}")
    print("  " + "-" * 60)
    for entry in result["per_class"]:
        sil = entry["silhouette"]
        mr = entry["minority_ratio"]
        print(
            f"  {entry['tool_name']:<25} {entry['n_samples']:>4} "
            f"{(f'{sil:.4f}' if sil is not None else 'n/a'):>11} "
            f"{(f'{mr:.4f}' if mr is not None else 'n/a'):>14}"
        )
    print("  " + "-" * 60)

    import json
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Detection summary saved to {args.output}")


if __name__ == "__main__":
    main()
