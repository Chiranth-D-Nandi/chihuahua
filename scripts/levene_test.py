"""
FISSION — scripts/levene_test.py
==================================
Task A1: Statistical assumption validation for ANOVA reported in Section 6.2.

Loads adversarial_dataset/manifest.json and:
  1. Groups ISR values by document (stem) and by epsilon budget
  2. Runs Levene's test for equal variance (ANOVA assumption)
  3. Computes omega-squared (omega^2) effect sizes for both factors
  4. Runs one-way and two-way (randomized block) ANOVA
  5. Runs Kruskal-Wallis non-parametric tests
  6. Prints exact numbers for Section 6.2 and CRC reviewer response

Run from FISSION project root:
    python scripts/levene_test.py
    python scripts/levene_test.py --manifest path/to/manifest.json

Addresses: ICICC-2027 Reviewer 1, Comment 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

# Scipy import
try:
    from scipy.stats import levene, f_oneway, kruskal, f as f_dist, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not installed — Levene's test unavailable.")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def omega_squared(groups: List[List[float]]) -> float:
    """
    Compute omega-squared (omega^2) for a one-way between-groups design.
    Formula: (SS_between - df_between * MS_within) / (SS_total + MS_within)
    """
    all_vals = [v for g in groups for v in g]
    grand_mean = np.mean(all_vals)
    k = len(groups)
    N = len(all_vals)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum((v - np.mean(g)) ** 2 for g in groups for v in g)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = N - k
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    numerator = ss_between - df_between * ms_within
    denominator = ss_total + ms_within

    return max(0.0, numerator / denominator) if denominator != 0 else 0.0


def eta_squared(groups: List[List[float]]) -> float:
    """eta^2 = SS_between / SS_total (upward-biased for small samples)."""
    all_vals = [v for g in groups for v in g]
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    return ss_between / ss_total if ss_total != 0 else 0.0


def analyse(manifest_path: str) -> None:
    print(f"\nLoading manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    entries = [e for e in entries if e.get("isr") is not None]
    print(f"Entries with ISR: {len(entries)}")

    if not entries:
        print("No ISR entries found — run stage_infer first.")
        return

    # Group by document stem
    by_doc: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        by_doc[e.get("stem", "unknown")].append(float(e["isr"]))

    # Group by epsilon budget
    by_eps: Dict[int, List[float]] = defaultdict(list)
    for e in entries:
        eps_key = int(e.get("epsilon_int", e.get("epsilon", 0)))
        by_eps[eps_key].append(float(e["isr"]))

    doc_names = sorted(by_doc.keys())
    eps_keys = sorted(by_eps.keys())

    doc_groups = [by_doc[d] for d in doc_names]
    eps_groups = [by_eps[k] for k in eps_keys]

    # Print summary tables
    print("\n" + "=" * 72)
    print("  PER-DOCUMENT ISR STATISTICS")
    print("=" * 72)
    print(f"  {'Document':<30} {'N':>4} {'Mean ISR':>9} {'SD':>7} {'Min':>6} {'Max':>6}")
    print("  " + "-" * 68)
    for doc in doc_names:
        g = by_doc[doc]
        print(
            f"  {doc:<30} {len(g):>4d} {np.mean(g):>9.4f} "
            f"{np.std(g, ddof=1):>7.4f} {min(g):>6.3f} {max(g):>6.3f}"
        )

    print("\n" + "=" * 72)
    print("  PER-BUDGET ISR STATISTICS")
    print("=" * 72)
    print(f"  {'Epsilon':>8} {'N':>4} {'Mean ISR':>9} {'SD':>7} {'Min':>6} {'Max':>6}")
    print("  " + "-" * 68)
    for k in eps_keys:
        g = by_eps[k]
        print(
            f"  {k:>5}/255 {len(g):>4d} {np.mean(g):>9.4f} "
            f"{np.std(g, ddof=1):>7.4f} {min(g):>6.3f} {max(g):>6.3f}"
        )

    # ANOVA
    print("\n" + "=" * 72)
    print("  ONE-WAY ANOVA & TWO-WAY RANDOMIZED BLOCK ANOVA")
    print("=" * 72)
    print("  Note on Statistical Unit & Independence:")
    print("  ANOVA and Levene tests are evaluated across the 20 cell-level sample")
    print("  proportions (5 documents x 4 budgets). Repeated trials on the same fixed")
    print("  images introduce intra-condition dependence; these tests evaluate between-cell")
    print("  variance and factor stability across conditions, not trial-level Bernoulli independence.")

    if SCIPY_AVAILABLE:
        f_doc, p_doc = f_oneway(*doc_groups)
        f_eps, p_eps = f_oneway(*eps_groups)

        omega2_doc = omega_squared(doc_groups)
        omega2_eps = omega_squared(eps_groups)
        eta2_doc = eta_squared(doc_groups)
        eta2_eps = eta_squared(eps_groups)

        print(f"\n  [One-Way Model]")
        print(f"  Document factor: F = {f_doc:.3f}, p = {p_doc:.4f}")
        print(f"    omega^2 = {omega2_doc:.4f} ({omega2_doc*100:.1f}% variance) [eta^2 = {eta2_doc:.4f}]")
        print(f"  Budget factor:   F = {f_eps:.3f}, p = {p_eps:.4f}")
        print(f"    omega^2 = {omega2_eps:.4f} ({omega2_eps*100:.1f}% variance) [eta^2 = {eta2_eps:.4f}]")

        # Two-Way Randomized Block ANOVA (Document x Budget, df_residual = 12)
        grid = np.array([[by_doc[d][j] for j in range(len(eps_keys))] for d in doc_names])
        grand_mean = np.mean(grid)
        ss_doc = len(eps_keys) * np.sum((np.mean(grid, axis=1) - grand_mean) ** 2)
        ss_budget = len(doc_names) * np.sum((np.mean(grid, axis=0) - grand_mean) ** 2)
        ss_total = np.sum((grid - grand_mean) ** 2)
        ss_error = max(0.0, ss_total - ss_doc - ss_budget)

        df_doc = len(doc_names) - 1
        df_budget = len(eps_keys) - 1
        df_error = df_doc * df_budget

        ms_doc = ss_doc / df_doc if df_doc > 0 else 0
        ms_budget = ss_budget / df_budget if df_budget > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1e-9

        f_doc_2way = ms_doc / ms_error
        f_budget_2way = ms_budget / ms_error
        p_doc_2way = 1 - f_dist.cdf(f_doc_2way, df_doc, df_error)
        p_budget_2way = 1 - f_dist.cdf(f_budget_2way, df_budget, df_error)

        omega2_partial_doc = (df_doc * (f_doc_2way - 1)) / (df_doc * (f_doc_2way - 1) + 20)
        omega2_partial_doc = max(0.0, omega2_partial_doc)

        print(f"\n  [Two-Way Randomized Block Model — Reported in Paper Section 6.2]")
        print(f"  Document factor: F({df_doc}, {df_error}) = {f_doc_2way:.3f}, p = {p_doc_2way:.4f}")
        print(f"    partial omega^2 = {omega2_partial_doc:.4f} ({omega2_partial_doc*100:.1f}% variance)")
        print(f"  Budget factor:   F({df_budget}, {df_error}) = {f_budget_2way:.3f}, p = {p_budget_2way:.4f}")

    # Levene's Test
    print("\n" + "=" * 72)
    print("  LEVENE'S TEST FOR HOMOGENEITY OF VARIANCE (ANOVA Assumption)")
    print("=" * 72)
    print("  H0: Variances across groups are equal (homoscedasticity).")
    print("  Note: Evaluated across cell-level aggregate proportions (N=20 cells).")
    print("  p > 0.05 indicates failure to reject equal variances (assumption viable).")

    if SCIPY_AVAILABLE:
        if len(doc_groups) >= 2 and all(len(g) >= 2 for g in doc_groups):
            lev_stat_doc, lev_p_doc = levene(*doc_groups)
            res_doc = "Fail to reject equal variances (p > 0.05; no significant heteroscedasticity)" if lev_p_doc > 0.05 else "Reject equal variances (p <= 0.05)"
            print(f"\n  Document groups: W = {lev_stat_doc:.3f}, p = {lev_p_doc:.4f} -> {res_doc}")
        else:
            print("\n  Document groups: insufficient data for Levene's test")

        if len(eps_groups) >= 2 and all(len(g) >= 2 for g in eps_groups):
            lev_stat_eps, lev_p_eps = levene(*eps_groups)
            res_eps = "Fail to reject equal variances (p > 0.05; no significant heteroscedasticity)" if lev_p_eps > 0.05 else "Reject equal variances (p <= 0.05)"
            print(f"  Budget groups:   W = {lev_stat_eps:.3f}, p = {lev_p_eps:.4f} -> {res_eps}")
        else:
            print("  Budget groups: insufficient data for Levene's test")

    # Kruskal-Wallis Test
    print("\n" + "=" * 72)
    print("  KRUSKAL-WALLIS TEST (Non-Parametric Alternative)")
    print("=" * 72)

    if SCIPY_AVAILABLE:
        if len(doc_groups) >= 2:
            h_doc, kw_p_doc = kruskal(*doc_groups)
            print(f"\n  Document factor: H = {h_doc:.3f}, p = {kw_p_doc:.4f}")
        if len(eps_groups) >= 2:
            h_eps, kw_p_eps = kruskal(*eps_groups)
            print(f"  Budget factor:   H = {h_eps:.3f}, p = {kw_p_eps:.4f}")

    # Correlation
    print("\n" + "=" * 72)
    print("  ISR-TO-BUDGET CORRELATION")
    print("=" * 72)
    eps_vals = [int(e.get("epsilon_int", e.get("epsilon", 0))) for e in entries]
    isr_vals = [float(e["isr"]) for e in entries]
    if SCIPY_AVAILABLE and len(eps_vals) > 2:
        r, r_p = pearsonr(eps_vals, isr_vals)
        print(f"\n  Pearson r = {r:+.4f}, p = {r_p:.4f} (calculated dynamically across 20 cell proportions)")
    elif len(eps_vals) > 2:
        r = float(np.corrcoef(eps_vals, isr_vals)[0, 1])
        print(f"\n  Pearson r = {r:+.4f}")

    print("\n" + "=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FISSION — Levene test & ANOVA effect sizes")
    parser.add_argument(
        "--manifest",
        default=os.path.join(ROOT, "adversarial_dataset", "manifest.json"),
        help="Path to manifest.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    analyse(args.manifest)


if __name__ == "__main__":
    main()
