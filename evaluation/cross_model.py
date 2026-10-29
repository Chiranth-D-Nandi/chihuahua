"""
FISSION — evaluation/cross_model.py
=====================================
Helpers for the cross-architecture benchmark deliverable
(LLaVA, Qwen2.5-VL, InternVL2).

Live multi-model inference requires Ollama with each model pulled.
This module:

  1. Knows the canonical model set from config/model_profiles.json
  2. Can merge per-model manifests into one cross-architecture table
  3. Provides a runner script entry that shells out to
     generate_adversarial_dataset.py once per model (transfer attack;
     see Session 17 note) and then builds a unified report + MLflow run

Usage:
    python -m evaluation.cross_model --dry-run
    python -m evaluation.cross_model --models llava:latest qwen2.5vl:7b --stage infer
    python -m evaluation.cross_model --merge-only  # just merge existing manifests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional

log = logging.getLogger("fission.evaluation.cross_model")

# Project root = parent of evaluation/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def canonical_models() -> List[str]:
    """Return model tags from config/model_profiles.json (ordered)."""
    path = os.path.join(ROOT, "config", "model_profiles.json")
    with open(path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return list(profiles.keys())


def manifest_path_for(model: str, base_dir: str = "adversarial_dataset") -> str:
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(base_dir, f"manifest_{safe}.json")


def load_cross_model_results(
    manifest_paths: List[str],
) -> List[dict]:
    """Load and tag entries from many manifests."""
    entries: List[dict] = []
    for p in manifest_paths:
        if not os.path.exists(p):
            log.warning("Missing manifest: %s", p)
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for e in data:
            entries.append(e)
    return entries


def summarise_cross_model(entries: List[dict]) -> Dict[str, dict]:
    """
    Return {model: {attack: {eps: {isr, ssim, sds, n}}}} for tables/plots.
    """
    out: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    buckets: Dict[tuple, List[dict]] = defaultdict(list)
    for e in entries:
        if e.get("isr") is None:
            continue
        key = (e.get("model", "?"), e.get("attack", "?"), e.get("epsilon_int", e.get("epsilon")))
        buckets[key].append(e)
    for (model, attack, eps), group in buckets.items():
        mean_isr = sum(g["isr"] for g in group) / len(group)
        ssim_vals = [g["ssim"] for g in group if g.get("ssim") is not None]
        sds_vals = [g["sds"] for g in group if g.get("sds") is not None]
        out[model][attack][eps] = {
            "isr": round(mean_isr, 4),
            "ssim": round(sum(ssim_vals) / len(ssim_vals), 4) if ssim_vals else None,
            "sds": round(sum(sds_vals) / len(sds_vals), 4) if sds_vals else None,
            "n": len(group),
            "sbi": mean_isr >= 0.5 and (ssim_vals and (sum(ssim_vals) / len(ssim_vals)) >= 0.95),
        }
    return {m: dict(attacks) for m, attacks in out.items()}


def run_per_model(
    models: List[str],
    stage: str = "infer",
    attack: str = "fgsm",
    epsilons: Optional[List[int]] = None,
    n_images: int = 5,
    runs: int = 5,
    output_base: str = "adversarial_dataset",
    python_exe: str = sys.executable,
    dry_run: bool = False,
) -> List[str]:
    """
    Invoke generate_adversarial_dataset.py once per model.

    For stage=generate the adversarial images are model-agnostic (CLIP
    surrogate), so we generate once into output_base and then for infer
    we copy/point each model at that image set with its own manifest.

    Returns list of manifest paths produced (or that would be produced).
    """
    epsilons = epsilons or [1, 2, 4, 8]
    script = os.path.join(ROOT, "attacks", "r1_pixel", "generate_adversarial_dataset.py")
    produced = []

    # Shared image generation (once) when stage needs it
    if stage in ("both", "generate"):
        shared_out = output_base
        cmd = [
            python_exe, script,
            "--output", shared_out,
            "--attack", attack,
            "--epsilons", *[str(e) for e in epsilons],
            "--n-images", str(n_images),
            "--runs", str(runs),
            "--stage", "generate",
            "--model", models[0],
        ]
        log.info("Generating shared adversarial images: %s", " ".join(cmd))
        if not dry_run:
            subprocess.check_call(cmd, cwd=ROOT)
        else:
            log.info("[dry-run] skipped generate")

    for model in models:
        man = manifest_path_for(model, base_dir=output_base)
        # For infer-only: start from the shared manifest if present, else
        # the default manifest.json, rewriting model field.
        if stage in ("both", "infer"):
            src_manifest = os.path.join(output_base, "manifest.json")
            if os.path.exists(src_manifest) and not dry_run:
                with open(src_manifest, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for e in entries:
                    e["model"] = model
                    e["isr"] = None
                    e["hits"] = None
                    e["runs"] = None
                    e["sds"] = None
                    e["sds_drifted_frac"] = None
                os.makedirs(os.path.dirname(man) or ".", exist_ok=True)
                with open(man, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2)
            cmd = [
                python_exe, script,
                "--output", output_base,
                "--attack", attack,
                "--epsilons", *[str(e) for e in epsilons],
                "--n-images", str(n_images),
                "--runs", str(runs),
                "--stage", "infer",
                "--model", model,
            ]
            # The stock script always writes to output/manifest.json —
            # after it finishes we copy to the per-model path.
            log.info("Infer on model=%s: %s", model, " ".join(cmd))
            if not dry_run:
                subprocess.check_call(cmd, cwd=ROOT)
                shared = os.path.join(output_base, "manifest.json")
                if os.path.exists(shared):
                    import shutil
                    shutil.copyfile(shared, man)
            produced.append(man)
        else:
            produced.append(man)

    return produced


def merge_and_report(
    manifest_paths: List[str],
    output_dir: str = "reports",
    log_mlflow: bool = True,
) -> str:
    from evaluation.report_generator import generate_report_from_many
    from evaluation.mlflow_logger import log_manifest_to_mlflow

    # Write a merged manifest for MLflow + report
    entries = load_cross_model_results(manifest_paths)
    merged_path = os.path.join(output_dir, "cross_model_manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    summary = summarise_cross_model(entries)
    summary_path = os.path.join(output_dir, "cross_model_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Cross-model summary: %s", summary_path)

    report_path = generate_report_from_many(
        manifest_paths=manifest_paths,
        output_dir=output_dir,
        title="FISSION Cross-Architecture Benchmark (LLaVA / Qwen2.5-VL / InternVL2)",
    )

    if log_mlflow:
        log_manifest_to_mlflow(
            merged_path,
            experiment_name="fission-cross-architecture",
            run_name="cross-model-merge",
            tags={"deliverable": "cross-architecture-benchmark"},
        )
    return report_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="FISSION cross-architecture benchmark runner")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model tags (default: all in model_profiles.json)",
    )
    parser.add_argument("--stage", choices=["both", "generate", "infer"], default="infer")
    parser.add_argument("--attack", default="fgsm")
    parser.add_argument("--epsilons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-base", default="adversarial_dataset")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip runs; merge existing per-model manifests and write report",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    models = args.models or canonical_models()
    log.info("Cross-architecture models: %s", models)

    if args.merge_only:
        paths = [manifest_path_for(m, args.output_base) for m in models]
        # Also accept the default single manifest if per-model ones missing
        default_man = os.path.join(args.output_base, "manifest.json")
        existing = [p for p in paths if os.path.exists(p)]
        if not existing and os.path.exists(default_man):
            existing = [default_man]
        if not existing:
            log.error("No manifests found to merge under %s", args.output_base)
            sys.exit(1)
        report = merge_and_report(
            existing, output_dir=args.report_dir, log_mlflow=not args.no_mlflow
        )
        print(f"Report: {report}")
        return

    paths = run_per_model(
        models=models,
        stage=args.stage,
        attack=args.attack,
        epsilons=args.epsilons,
        n_images=args.n_images,
        runs=args.runs,
        output_base=args.output_base,
        dry_run=args.dry_run,
    )
    if args.stage in ("both", "infer") and not args.dry_run:
        report = merge_and_report(
            paths, output_dir=args.report_dir, log_mlflow=not args.no_mlflow
        )
        print(f"Report: {report}")
    else:
        print("Manifests (planned):", paths)


if __name__ == "__main__":
    main()
