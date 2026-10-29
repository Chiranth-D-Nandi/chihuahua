"""
FISSION — defenses/b0_scanner/spectre.py

=====================================================
D0 — SPECTRE (Statistical Pattern Examination for Covert Trigger
Recognition): the combined pre-deployment scanner.

This is the module the blueprint describes but that didn't exist yet —
b0_scanner previously only exposed NeuralCleanseDetector directly, with
no ensemble logic tying Activation Clustering + STRIP + Neural Cleanse
together into a single verdict. This file is that wiring.

Why an ensemble at all:
    Each of the three methods catches a different backdoor signature
    and none is individually reliable enough to gate deployment alone:
      - Neural Cleanse can miss triggers that don't localise to a small
        mask (e.g. global/blended triggers).
      - Activation Clustering can miss triggers where the poisoned and
        clean pathways don't cleanly separate in CLIP embedding space.
      - STRIP can miss triggers that are weak enough to still respond
        somewhat to the superimposed clean content (partial dominance).
    Requiring agreement across methods trades recall for precision on
    any single method, in exchange for a much lower false-positive rate
    on the combined verdict — which matters because CONFIRMED_BACKDOOR
    means quarantining a model.

Verdict logic:
    CONFIRMED_BACKDOOR   — >=2 of 3 methods independently flag a backdoor
    SUSPECTED_BACKDOOR   — exactly 1 of 3 methods flags a backdoor
    CLEAN                — 0 of 3 methods flag anything

    CONFIRMED models are quarantined (never enter the pipeline).
    SUSPECTED models proceed but with elevated runtime monitoring
    (i.e. tighter D2 attention-entropy thresholds) — this policy lever
    lives in run_spectre()'s return value, not enforced here; the
    orchestrator/model-loading code decides what "elevated monitoring"
    means operationally.

TDR (Trigger Detection Rate):
    % of known-backdoored models in a held-out test set that SPECTRE
    correctly flags as SUSPECTED or CONFIRMED (i.e. NOT called CLEAN).
    Computed by evaluate_tdr() below, across a set of (model, ground
    truth label) pairs you provide — SPECTRE itself has no way to know
    ground truth, that's supplied externally by whoever curated the
    clean/poisoned checkpoint pairs for the eval.

Usage (from project root):
    python defenses/b0_scanner/spectre.py --probe attacks/r1_pixel/probe.pt \\
        --clean-glob "adversarial_dataset/synthetic/*_clean.png" \\
        --test-image path/to/suspect.png

Owner: joint — Chiranth (Neural Cleanse) + Abhyudhaya (AC + STRIP).
Combined verdict wiring: shared, since it's the D0-as-a-whole contract
the rest of the pipeline (model loading, dashboard) depends on.
"""

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg
from attacks.r1_pixel.generate_adversarial_dataset import CLIPProbe

from defenses.b0_scanner.neural_cleanse import NeuralCleanseDetector, NeuralCleanseConfig
from defenses.b0_scanner.activation_clustering import (
    ActivationClusteringDetector,
    ActivationClusteringConfig,
)
from defenses.b0_scanner.strip import STRIPDetector, STRIPConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.b0.spectre")


# ══════════════════════════════════════════════════════════════
#  Verdict
# ══════════════════════════════════════════════════════════════

class SpectreVerdict(str, Enum):
    CLEAN               = "CLEAN"
    SUSPECTED_BACKDOOR   = "SUSPECTED_BACKDOOR"
    CONFIRMED_BACKDOOR   = "CONFIRMED_BACKDOOR"


@dataclass
class SpectreConfig:
    """
    Ensemble-level knobs. Per-method knobs (mask thresholds, silhouette
    thresholds, entropy z-scores) live in each method's own Config
    dataclass — this only controls how their three bools get combined,
    plus which methods are required to run vs optional.
    """
    confirm_min_flags:  int  = 2      # >=N methods flagging => CONFIRMED
    require_all_methods: bool = True   # if False, missing methods just don't vote
    strip_test_images:  Optional[List[str]] = None  # glob for suspect images to STRIP-test


@dataclass
class SpectreResult:
    verdict:            SpectreVerdict
    methods_flagged:    List[str]
    n_methods_run:      int
    neural_cleanse:     Optional[Dict] = None
    activation_clustering: Optional[Dict] = None
    strip:              Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict.value,
            "methods_flagged": self.methods_flagged,
            "n_methods_run": self.n_methods_run,
            "neural_cleanse": self.neural_cleanse,
            "activation_clustering": self.activation_clustering,
            "strip": self.strip,
        }


# ══════════════════════════════════════════════════════════════
#  Ensemble runner
# ══════════════════════════════════════════════════════════════

def run_spectre(
    probe: CLIPProbe,
    tool_names: List[str],
    clean_images_by_stem: Dict[str, Image.Image],
    test_images_by_stem: Optional[Dict[str, Image.Image]] = None,
    config: Optional[SpectreConfig] = None,
    nc_config: Optional[NeuralCleanseConfig] = None,
    ac_config: Optional[ActivationClusteringConfig] = None,
    strip_config: Optional[STRIPConfig] = None,
) -> SpectreResult:
    """
    Run the full SPECTRE ensemble against a candidate model (via its
    CLIPProbe surrogate) and return a combined verdict.

    Args:
        probe: loaded CLIPProbe for the candidate model.
        tool_names: class names in probe's output order.
        clean_images_by_stem: clean probe-set images, used by both
            Neural Cleanse (image_size only) and Activation Clustering
            (for grouping-by-predicted-class) and STRIP (as the
            calibration/overlay pool).
        test_images_by_stem: optional suspect images to run through
            STRIP's detect() step, beyond just calibration. If omitted,
            STRIP only calibrates (contributes no flag either way,
            since there's nothing to test).
        config: SpectreConfig ensemble knobs.

    Returns:
        SpectreResult with the combined verdict + each method's raw output.
    """
    cfg_ensemble = config or SpectreConfig()
    clean_pool = list(clean_images_by_stem.values())
    num_classes = len(tool_names)

    flagged_methods: List[str] = []
    nc_summary = None
    ac_summary = None
    strip_summary = None
    n_run = 0

    # ── Neural Cleanse ──────────────────────────────────────────
    try:
        log.info("Running Neural Cleanse...")
        nc = NeuralCleanseDetector(
            probe=probe, num_classes=num_classes, tool_names=tool_names,
            config=nc_config,
        )
        nc_summary = nc.detect()
        n_run += 1
        if nc_summary["backdoor_detected"]:
            flagged_methods.append("neural_cleanse")
    except Exception as e:
        log.error(f"Neural Cleanse failed to run: {e}")
        if cfg_ensemble.require_all_methods:
            raise

    # ── Activation Clustering ───────────────────────────────────
    try:
        log.info("Running Activation Clustering...")
        ac = ActivationClusteringDetector(
            probe=probe, tool_names=tool_names, config=ac_config,
        )
        ac_summary = ac.detect(clean_images_by_stem)
        n_run += 1
        if ac_summary["backdoor_detected"]:
            flagged_methods.append("activation_clustering")
    except Exception as e:
        log.error(f"Activation Clustering failed to run: {e}")
        if cfg_ensemble.require_all_methods:
            raise

    # ── STRIP ────────────────────────────────────────────────────
    try:
        log.info("Running STRIP...")
        strip = STRIPDetector(
            probe=probe, num_classes=num_classes, tool_names=tool_names,
            config=strip_config,
        )
        strip.calibrate(clean_pool)

        if test_images_by_stem:
            strip_summary = strip.detect_batch(test_images_by_stem, clean_pool)
            n_run += 1
            if strip_summary["n_triggered"] > 0:
                flagged_methods.append("strip")
        else:
            log.info(
                "STRIP calibrated but no test_images_by_stem given — "
                "STRIP contributes no vote (nothing was actually tested)."
            )
    except Exception as e:
        log.error(f"STRIP failed to run: {e}")
        if cfg_ensemble.require_all_methods:
            raise

    # ── Combine ──────────────────────────────────────────────────
    n_flagged = len(flagged_methods)
    if n_flagged >= cfg_ensemble.confirm_min_flags:
        verdict = SpectreVerdict.CONFIRMED_BACKDOOR
    elif n_flagged >= 1:
        verdict = SpectreVerdict.SUSPECTED_BACKDOOR
    else:
        verdict = SpectreVerdict.CLEAN

    result = SpectreResult(
        verdict=verdict,
        methods_flagged=flagged_methods,
        n_methods_run=n_run,
        neural_cleanse=nc_summary,
        activation_clustering=ac_summary,
        strip=strip_summary,
    )

    log.info("=" * 60)
    log.info(f"SPECTRE verdict: {verdict.value}")
    log.info(f"  Methods run: {n_run}/3 | Methods flagged: {flagged_methods}")
    log.info("=" * 60)

    if verdict == SpectreVerdict.CONFIRMED_BACKDOOR:
        log.warning("  ⚠️  QUARANTINE: model confirmed backdoored, do not deploy.")
    elif verdict == SpectreVerdict.SUSPECTED_BACKDOOR:
        log.warning("  ⚠️  ELEVATED MONITORING: model suspected, deploy with tighter D2 thresholds.")

    return result


# ══════════════════════════════════════════════════════════════
#  TDR — Trigger Detection Rate evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_tdr(labelled_verdicts: List[Dict]) -> Dict:
    """
    Compute TDR (Trigger Detection Rate) across a held-out set of
    (model, ground_truth_is_backdoored, spectre_verdict) triples.

    Args:
        labelled_verdicts: list of dicts, each with:
            {
                "model_name": str,
                "is_backdoored": bool,   # ground truth, supplied externally
                "verdict": SpectreVerdict,
            }

    Returns:
        {
            "tdr": float,              # fraction of TRUE backdoors caught
                                        # (verdict != CLEAN)
            "fpr": float,               # fraction of TRUE clean models
                                        # incorrectly flagged (verdict != CLEAN)
            "n_backdoored": int,
            "n_clean": int,
            "confirmed_among_backdoored": int,
            "suspected_among_backdoored": int,
            "missed_backdoored": list of model_names SPECTRE called CLEAN
                                  despite being genuinely backdoored,
        }
    """
    backdoored = [r for r in labelled_verdicts if r["is_backdoored"]]
    clean = [r for r in labelled_verdicts if not r["is_backdoored"]]

    def _flagged(v) -> bool:
        v = v.value if isinstance(v, SpectreVerdict) else v
        return v != SpectreVerdict.CLEAN.value

    caught = [r for r in backdoored if _flagged(r["verdict"])]
    missed = [r for r in backdoored if not _flagged(r["verdict"])]
    false_positives = [r for r in clean if _flagged(r["verdict"])]

    confirmed = [r for r in caught if
                 (r["verdict"].value if isinstance(r["verdict"], SpectreVerdict) else r["verdict"])
                 == SpectreVerdict.CONFIRMED_BACKDOOR.value]
    suspected = [r for r in caught if r not in confirmed]

    tdr = len(caught) / len(backdoored) if backdoored else 0.0
    fpr = len(false_positives) / len(clean) if clean else 0.0

    result = {
        "tdr": round(tdr, 4),
        "fpr": round(fpr, 4),
        "n_backdoored": len(backdoored),
        "n_clean": len(clean),
        "confirmed_among_backdoored": len(confirmed),
        "suspected_among_backdoored": len(suspected),
        "missed_backdoored": [r["model_name"] for r in missed],
        "meets_target": tdr >= cfg.eval.tdr_min,   # cfg.eval.tdr_min = 0.70
    }

    log.info(f"TDR = {tdr:.4f} (target >= {cfg.eval.tdr_min}) | FPR = {fpr:.4f}")
    if missed:
        log.warning(f"  ⚠️  Missed backdoored models: {result['missed_backdoored']}")

    return result


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
    parser = argparse.ArgumentParser(description="FISSION D0 — SPECTRE combined scanner")
    parser.add_argument("--probe", default="attacks/r1_pixel/probe.pt")
    parser.add_argument("--clean-glob", default="adversarial_dataset/synthetic/*_clean.png")
    parser.add_argument("--test-image", action="append", default=None,
                        help="Suspect image to STRIP-test; repeatable")
    parser.add_argument("--output", default="spectre_verdict.json")
    args = parser.parse_args()

    if not os.path.exists(args.probe):
        log.error(f"Probe not found: {args.probe}")
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

    test_images = None
    if args.test_image:
        test_images = {
            os.path.splitext(os.path.basename(p))[0]: Image.open(p).convert("RGB")
            for p in args.test_image
        }

    result = run_spectre(
        probe=probe,
        tool_names=probe_data["tool_classes"],
        clean_images_by_stem=clean_images,
        test_images_by_stem=test_images,
    )

    print("\n" + "=" * 60)
    print("  FISSION D0 — SPECTRE Verdict")
    print("=" * 60)
    print(f"  VERDICT: {result.verdict.value}")
    print(f"  Methods flagged: {result.methods_flagged}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    log.info(f"Verdict saved to {args.output}")


if __name__ == "__main__":
    main()
