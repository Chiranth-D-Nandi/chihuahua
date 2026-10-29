"""
FISSION — attacks/r1_pixel/train_probe.py
==========================================
Step 2 of 3 — Train the linear probe on real LLaVA projector features.

What this does:
    1. Loads features.pt produced by extract_features.py
    2. Trains nn.Linear(4096, 3) on real LLaVA projector features
       with real Ollama tool labels
    3. Saves trained weights to probe.pt

Why this matters:
    With a trained probe, FGSM/PGD gradients point in a direction that
    LLaVA's own feature space associates with tool selection.
    This replaces the random Xavier-initialised probe that was causing
    visible image destruction at large epsilon.

Run from FISSION project root:
    python attacks/r1_pixel/train_probe.py
    python attacks/r1_pixel/train_probe.py --epochs 200
    python attacks/r1_pixel/train_probe.py --features attacks/r1_pixel/features.pt

Expected output:
    Epoch 200/200 | loss=0.0123 | train_acc=100.0% | val_acc=88.9%
    Saved: attacks/r1_pixel/probe.pt

ABHYUDAYA — Session 6.

SESSION 18 FIX — class-weighted loss + held-out validation split.
Root-caused after a live benchmark run showed a 0.500 spontaneous
"clean baseline" ISR (a clean, unperturbed image getting classified
as the injected tool half the time with zero attack applied) and an
FGSM run where ISR *dropped* as epsilon increased (1/255 -> 4/255) —
backwards for a working attack. Investigation traced this to the
real, unforced label distribution coming back from extract_features.py:

    summarise_document: 15 samples
    send_slack_message: 40 samples
    query_database:      45 samples   <- this is INJECTED_TOOL

query_database (the attack's own target class) was the largest class
in the training set with no correction, so the probe's decision
boundary was already biased toward it before any perturbation was
applied — a very plausible direct explanation for the inflated
baseline ISR and the noisy/backwards epsilon trend. Two fixes:

  1. Class-weighted CrossEntropyLoss (inverse-frequency weighting) so
     the majority class (query_database) doesn't dominate the
     gradient direction used to craft "gradient direction toward
     query_database" adversarial perturbations, and so a low-epsilon,
     minimally-perturbed image isn't already halfway there.
  2. A held-out validation split (stratified per class where the
     class has enough samples). The previous version's "99% final
     accuracy" was measured on the exact same 100 samples the probe
     was trained on — a 768 x 3 linear probe (~2.3K params) fit to
     100 points is close to guaranteed to memorize regardless of
     whether the underlying classes are cleanly separable in CLIP
     feature space. A train/val split doesn't fix the class-imbalance
     problem by itself, but it stops the accuracy number from being
     meaningless, and a big train/val accuracy gap is now a visible,
     early warning sign instead of a silent one.

Neither fix guarantees the baseline-ISR issue is fully resolved — that
requires re-running extract_features.py / generate_adversarial_dataset.py
and comparing. See FISSION_Progress_Report for current status/next steps.
"""

import argparse
import logging
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
log = logging.getLogger("fission.r1.train_probe")

TOOL_NAMES  = list(cfg.mcp.registered_tools.keys())
NUM_CLASSES = len(TOOL_NAMES)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features.pt")
PROBE_PATH    = os.path.join(os.path.dirname(__file__), "probe.pt")


def stratified_split(labels: torch.Tensor, val_frac: float, seed: int):
    """
    Per-class shuffle-and-split so every class that has enough samples
    to spare at least one gets one in validation, rather than a single
    random split that could (especially with a 15-sample minority
    class) leave a class with zero validation examples by chance.

    Classes with fewer than 2 samples are kept entirely in train (a
    single sample can't be usefully held out) and a warning is logged.

    Returns (train_idx: LongTensor, val_idx: LongTensor).
    """
    rng = random.Random(seed)
    train_idx, val_idx = [], []

    for cls in range(NUM_CLASSES):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0].tolist()
        if len(cls_idx) < 2:
            if cls_idx:
                log.warning(
                    f"Class '{TOOL_NAMES[cls]}' has only {len(cls_idx)} sample(s) — "
                    "keeping it entirely in train, none held out for validation."
                )
            train_idx.extend(cls_idx)
            continue

        rng.shuffle(cls_idx)
        n_val = max(1, round(len(cls_idx) * val_frac))
        n_val = min(n_val, len(cls_idx) - 1)  # always leave >=1 for train
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])

    return torch.tensor(sorted(train_idx)), torch.tensor(sorted(val_idx))


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Inverse-frequency class weights: weight_i = N / (num_classes * count_i).
    A class with zero samples gets weight 0 (CrossEntropyLoss never sees
    it anyway) rather than dividing by zero.
    """
    counts = torch.tensor(
        [(labels == i).sum().item() for i in range(NUM_CLASSES)], dtype=torch.float
    )
    n = counts.sum().item()
    weights = torch.zeros(NUM_CLASSES)
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = n / (NUM_CLASSES * c.item())
    return weights


def evaluate(probe: nn.Module, feats: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Returns (accuracy_pct, preds). Assumes probe is already in eval mode."""
    if feats.shape[0] == 0:
        return float("nan"), torch.empty(0, dtype=torch.long)
    with torch.no_grad():
        logits = probe(feats)
        preds  = logits.argmax(dim=1)
        acc    = (preds == labels).float().mean().item() * 100
    return acc, preds


def main():
    parser = argparse.ArgumentParser(
        description="FISSION R1 Step 2 — Train linear probe on LLaVA features"
    )
    parser.add_argument("--features", default=FEATURES_PATH,
                        help="Path to features.pt from extract_features.py")
    parser.add_argument("--output",   default=PROBE_PATH,
                        help="Output path for probe.pt")
    parser.add_argument("--epochs",   type=int, default=300,
                        help="Training epochs (default: 300)")
    parser.add_argument("--lr",       type=float, default=1e-2,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of each class held out for validation "
                             "(default: 0.2). Set to 0 to disable and restore "
                             "the old train-on-everything behaviour.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train/val split (default: 42)")
    parser.add_argument("--no-class-weight", action="store_true",
                        help="Disable inverse-frequency class weighting and use "
                             "plain CrossEntropyLoss (Session <18 behaviour).")
    args = parser.parse_args()

    # ── Load features ──────────────────────────────────────────────────
    if not os.path.exists(args.features):
        log.error(f"features.pt not found at {args.features}")
        log.error("Run extract_features.py first.")
        sys.exit(1)

    data = torch.load(args.features, map_location="cpu")
    features = data["features"].float()   # (N, 4096)
    labels   = data["labels"]             # (N,)
    tool_names = data["tool_names"]

    log.info(f"Loaded features: {features.shape}")
    label_counts = {TOOL_NAMES[i]: (labels == i).sum().item() for i in range(NUM_CLASSES)}
    log.info(f"Labels: {label_counts}")

    N = features.shape[0]
    if N < NUM_CLASSES:
        log.error(f"Only {N} samples — need at least {NUM_CLASSES} (one per class).")
        log.error("Re-run extract_features.py with --n-images 30.")
        sys.exit(1)

    # Check all classes are represented
    for i, name in enumerate(TOOL_NAMES):
        count = (labels == i).sum().item()
        if count == 0:
            log.warning(f"Class '{name}' has 0 samples — probe cannot learn this class.")
            log.warning("Consider running extract_features.py with more images.")

    # Flag severe imbalance up front — this is the root cause behind the
    # Session 17 baseline-ISR investigation; surface it every run so it
    # isn't rediscovered from a confusing benchmark result again.
    nonzero_counts = [c for c in label_counts.values() if c > 0]
    if nonzero_counts and (max(nonzero_counts) / min(nonzero_counts)) >= 2.0:
        log.warning(
            f"Class imbalance detected (largest/smallest = "
            f"{max(nonzero_counts) / min(nonzero_counts):.1f}x): {label_counts}. "
            + ("Class weighting is ON (default) to correct for this."
               if not args.no_class_weight else
               "Class weighting is OFF (--no-class-weight) — the probe's "
               "gradient direction will be biased toward the majority class.")
        )

    # ── Normalise features ─────────────────────────────────────────────
    # L2 normalise so probe is not dominated by feature magnitude
    feat_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    # ── Train/val split ────────────────────────────────────────────────
    if args.val_split > 0:
        train_idx, val_idx = stratified_split(labels, args.val_split, args.seed)
        if val_idx.numel() == 0:
            log.warning(
                "val-split > 0 but every class was too small to hold out any "
                "validation samples — proceeding train-only, accuracy below "
                "will be a train-set (not held-out) number."
            )
    else:
        train_idx = torch.arange(N)
        val_idx   = torch.tensor([], dtype=torch.long)

    train_feat, train_labels = feat_norm[train_idx], labels[train_idx]
    val_feat,   val_labels   = feat_norm[val_idx],   labels[val_idx]

    log.info(
        f"Split: {train_idx.numel()} train / {val_idx.numel()} val "
        f"(val_split={args.val_split})"
    )

    # ── Class weights ──────────────────────────────────────────────────
    if args.no_class_weight:
        class_weights = None
    else:
        class_weights = compute_class_weights(train_labels)
        log.info(
            "Class weights (inverse-frequency, computed on train split): "
            + ", ".join(f"{TOOL_NAMES[i]}={w:.3f}" for i, w in enumerate(class_weights))
        )

    # ── Build probe ────────────────────────────────────────────────────
    probe = nn.Linear(features.shape[1], NUM_CLASSES, bias=True)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    optimiser = optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ──────────────────────────────────────────────────
    log.info(f"Training probe: Linear({features.shape[1]}, {NUM_CLASSES})")
    log.info(f"Epochs: {args.epochs} | LR: {args.lr}")

    probe.train()
    for epoch in range(1, args.epochs + 1):
        optimiser.zero_grad()
        logits = probe(train_feat)              # (N_train, NUM_CLASSES)
        loss   = criterion(logits, train_labels)
        loss.backward()
        optimiser.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            probe.eval()
            train_acc, _ = evaluate(probe, train_feat, train_labels)
            val_acc, _   = evaluate(probe, val_feat, val_labels)
            val_str = f"{val_acc:.1f}%" if val_feat.shape[0] > 0 else "n/a"
            log.info(
                f"Epoch {epoch:>4}/{args.epochs} | loss={loss.item():.4f} | "
                f"train_acc={train_acc:.1f}% | val_acc={val_str}"
            )
            probe.train()

    # ── Final evaluation ───────────────────────────────────────────────
    probe.eval()
    train_acc, train_preds = evaluate(probe, train_feat, train_labels)
    val_acc, val_preds     = evaluate(probe, val_feat, val_labels)

    log.info(f"\nFinal train accuracy: {train_acc:.1f}%")
    if val_feat.shape[0] > 0:
        log.info(f"Final val accuracy:   {val_acc:.1f}% ({val_feat.shape[0]} held-out samples)")
        gap = train_acc - val_acc
        if gap >= 15.0:
            log.warning(
                f"Train/val accuracy gap is {gap:.1f} points — likely overfitting. "
                "A high train accuracy alone (e.g. the old 99% figure, measured "
                "on the training set itself) does not mean the gradient direction "
                "generalizes. Consider more samples per class before trusting "
                "this probe's gradients for the benchmark."
            )
    else:
        log.warning(
            "No held-out validation samples — accuracy below is a train-set "
            "number and does not tell you whether the probe generalizes."
        )

    if train_acc < 70.0:
        log.warning("Train accuracy below 70% — probe may not produce useful gradients.")
        log.warning("Consider running extract_features.py with --n-images 50.")
    else:
        log.info("Train accuracy acceptable — gradient direction may be meaningful "
                 "(check val accuracy above before trusting this, though).")

    # Per-class accuracy (train and, where available, val)
    for i, name in enumerate(TOOL_NAMES):
        train_mask = train_labels == i
        val_mask   = val_labels == i
        if train_mask.sum() == 0:
            log.info(f"  {name}: no train samples")
            continue
        t_acc = (train_preds[train_mask] == train_labels[train_mask]).float().mean().item() * 100
        if val_mask.sum() > 0:
            v_acc = (val_preds[val_mask] == val_labels[val_mask]).float().mean().item() * 100
            log.info(
                f"  {name}: train={t_acc:.1f}% ({train_mask.sum().item()} samples) | "
                f"val={v_acc:.1f}% ({val_mask.sum().item()} samples)"
            )
        else:
            log.info(f"  {name}: train={t_acc:.1f}% ({train_mask.sum().item()} samples) | val=n/a")

    # ── Save ───────────────────────────────────────────────────────────
    # Save probe state dict — loaded by generate_adversarial_dataset.py
    torch.save({
        "state_dict":     probe.state_dict(),
        "in_features":    features.shape[1],
        "num_classes":    NUM_CLASSES,
        "tool_classes":   TOOL_NAMES,
        "final_acc":      train_acc,   # kept for backward-compat with old field name
        "final_train_acc": train_acc,
        "final_val_acc":   val_acc if val_feat.shape[0] > 0 else None,
        "class_weighted": not args.no_class_weight,
        "val_split":      args.val_split,
    }, args.output)

    log.info(f"\nSaved: {args.output}")
    log.info("Step 2 complete. Run generate_adversarial_dataset.py next.")


if __name__ == "__main__":
    main()

