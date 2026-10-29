"""
FISSION — attacks/r1_pixel/pgd.py
====================================
Projected Gradient Descent (PGD) pixel-space attack.
(Madry et al., 2018 — "Towards Deep Learning Models Resistant to Adversarial Attacks")

Generates adversarial images by iteratively applying FGSM steps
and projecting back into the L-inf epsilon ball around the clean image.

PGD is strictly stronger than FGSM — it is the multi-step version.
Used in R1 to show ISR improvement over FGSM at the same epsilon budget,
and to find the true adversarial boundary for SBI measurement.

Parameters from config:
  cfg.attack.pgd_steps = 40       — number of iterations
  cfg.attack.pgd_alpha = 2/255    — per-step size
  cfg.attack.epsilon_levels       — outer epsilon budgets

ABHYUDAYA — attack side.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger("fission.r1.pgd")


class PGDAttack:
    """
    Iterative PGD attack within an L-inf epsilon ball.

    At each step:
        x_t+1 = Proj_{x + S}( x_t + alpha * sign(∇_x L(x_t)) )

    where S = {δ : ||δ||_inf ≤ epsilon} and Proj clips back into the ball.

    Args:
        epsilon:   L-inf perturbation budget in [0, 1] (e.g. 8/255).
        alpha:     per-step size (e.g. 2/255). Typically epsilon / 10.
        num_steps: number of PGD iterations. Default 40 (from cfg).
        random_start: if True, initialise δ uniformly in [-epsilon, epsilon]
                      before iterating (standard PGD). Default True.
        device:    torch device string.
    """

    def __init__(
        self,
        epsilon: float,
        alpha: Optional[float] = None,
        num_steps: Optional[int] = None,
        random_start: bool = True,
        device: str = "cuda",
    ):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from config.config import cfg

        self.epsilon     = epsilon
        self.alpha       = alpha     if alpha     is not None else cfg.attack.pgd_alpha
        self.num_steps   = num_steps if num_steps is not None else cfg.attack.pgd_steps
        self.random_start = random_start
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")

        log.info(
            f"PGDAttack initialised | epsilon={self.epsilon:.4f} | "
            f"alpha={self.alpha:.4f} | steps={self.num_steps} | "
            f"random_start={self.random_start} | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perturb(
        self,
        image_tensor: torch.Tensor,
        surrogate_model: torch.nn.Module,
        target_class: int,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Run PGD and return the adversarial image.

        Args:
            image_tensor:    clean float32 (1, C, H, W) in [0, 1].
            surrogate_model: nn.Module mapping (1, C, H, W) → (1, num_classes).
            target_class:    adversarial target class index.
            num_classes:     number of surrogate classes.

        Returns:
            Adversarial image tensor, same shape, clipped to [0, 1].
        """
        surrogate_model.eval()
        x_clean = image_tensor.clone().to(self.device)

        # Initialise adversarial example
        if self.random_start:
            delta = torch.empty_like(x_clean).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()
        else:
            x_adv = x_clean.clone().detach()

        target = torch.tensor([target_class], device=self.device)

        for step in range(self.num_steps):
            x_adv.requires_grad_(True)

            logits = surrogate_model(x_adv)
            loss   = F.cross_entropy(logits, target)

            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            with torch.no_grad():
                # SESSION 20 FIX — same sign error found and fixed in
                # fgsm.py. Ascending cross_entropy(logits, target_class)
                # moves x AWAY from target_class (the untargeted-attack
                # formula); for a targeted attack (push toward
                # target_class) we need to DESCEND that loss instead —
                # i.e. step in the *negative* gradient-sign direction.
                # This bug is more consequential in PGD than FGSM: it is
                # applied and compounded over every one of num_steps
                # iterations rather than a single step, so the drift away
                # from target_class was larger here at equivalent epsilon.
                x_adv = x_adv - self.alpha * x_adv.grad.sign()

                # Project back into L-inf ball around clean image
                delta = torch.clamp(x_adv - x_clean, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

            log.debug(
                f"PGD step {step+1}/{self.num_steps} | "
                f"loss={loss.item():.4f} | "
                f"L-inf delta={delta.abs().max().item():.4f}"
            )

        log.info(
            f"PGD done | epsilon={self.epsilon:.4f} | "
            f"final L-inf={( x_adv - x_clean ).abs().max().item():.4f}"
        )
        return x_adv

    # ------------------------------------------------------------------
    # Convenience: run across all epsilon levels from config
    # ------------------------------------------------------------------

    @classmethod
    def run_epsilon_sweep(
        cls,
        image_tensor: torch.Tensor,
        surrogate_model: torch.nn.Module,
        target_class: int,
        num_classes: int,
        epsilon_levels: Optional[list] = None,
        device: str = "cuda",
    ) -> dict:
        """
        Run PGD at every epsilon level and return a dict of results.

        Args:
            image_tensor:    clean input image (1, C, H, W) in [0, 1].
            surrogate_model: surrogate classifier.
            target_class:    adversarial target class index.
            num_classes:     number of surrogate classes.
            epsilon_levels:  list of epsilon values; defaults to cfg values.
            device:          torch device string.

        Returns:
            {epsilon_value: adv_image_tensor, ...}
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from config.config import cfg

        levels = epsilon_levels or cfg.attack.epsilon_levels

        results = {}
        for eps in levels:
            attacker = cls(epsilon=eps, device=device)
            adv = attacker.perturb(
                image_tensor=image_tensor.clone(),
                surrogate_model=surrogate_model,
                target_class=target_class,
                num_classes=num_classes,
            )
            results[eps] = adv
            log.info(f"PGD sweep | epsilon={eps:.4f} done")

        return results

