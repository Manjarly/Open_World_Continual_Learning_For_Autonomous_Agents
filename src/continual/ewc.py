"""
src/continual/ewc.py
────────────────────
Elastic Weight Consolidation (EWC) for continual learning.

EWC prevents catastrophic forgetting by adding a quadratic penalty to the
loss function. The penalty is proportional to the Fisher Information of each
parameter — parameters critical to the old task (Waymo) are penalized more
when they change during new task training (nuScenes).

Reference:
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks"
    PNAS 2017.  https://arxiv.org/abs/1612.00796

Usage:
    from src.continual.ewc import EWC

    # After training on Task A (Waymo):
    ewc = EWC(model, dataloader_task_a, device="cuda")

    # During Task B (nuScenes) training loop:
    loss = task_b_loss + ewc.penalty(model)
"""

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    Computes the empirical Fisher Information Matrix diagonal over a sample
    of Task A data, then exposes a penalty() method to add to Task B loss.

    Args:
        model:    PyTorch model already trained on Task A.
        dataloader: DataLoader over Task A (used to estimate Fisher).
        device:   "cuda" or "cpu".
        n_samples: Max number of batches to use when estimating Fisher.
                   More samples → better estimate but slower.
        ewc_lambda: Regularisation strength (λ). Typical range: 0.1 – 1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        n_samples: int = 200,
        ewc_lambda: float = 0.4,
    ):
        self.model      = model
        self.device     = device
        self.ewc_lambda = ewc_lambda

        # θ* — parameters at Task A optimum
        self.params_task_a: dict[str, torch.Tensor] = {}
        # F   — diagonal Fisher estimate
        self.fisher:        dict[str, torch.Tensor] = {}

        self._save_task_a_params()
        self._estimate_fisher(dataloader, n_samples)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_task_a_params(self):
        """Snapshot θ* (Task A optimal parameters)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_task_a[name] = param.data.clone().detach()
        logger.info(
            f"Saved {len(self.params_task_a)} parameter tensors from Task A."
        )

    def _estimate_fisher(self, dataloader: DataLoader, n_samples: int):
        """
        Approximate the diagonal of the Fisher Information Matrix.

        F_ii ≈ E[ (∂ log p(y|x, θ) / ∂θ_i)² ]

        We estimate this via Monte-Carlo: for each sample, do a forward pass,
        sample a class from the model's output distribution, compute the
        log-probability of that class, backprop, and accumulate squared grads.
        """
        logger.info("Estimating Fisher Information Matrix (diagonal)...")
        self.model.eval()

        # Initialise Fisher accumulators
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)

        n_batches = 0
        for batch in tqdm(dataloader, desc="Fisher estimation", total=n_samples):
            if n_batches >= n_samples:
                break

            # ── Handle different batch formats ────────────────────────────
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("img"))
            else:
                images = batch

            if images is None:
                continue

            images = images.to(self.device).float()
            if images.ndim == 3:
                images = images.unsqueeze(0)

            self.model.zero_grad()

            try:
                output = self.model(images)

                # Flatten output to (batch, num_classes) if needed
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if output.ndim > 2:
                    # YOLO backbone output — take global average pool
                    output = output.flatten(2).mean(dim=2)

                # Sample pseudo-labels from model distribution
                probs  = F.softmax(output, dim=1)
                log_probs = F.log_softmax(output, dim=1)
                sampled_labels = torch.multinomial(probs.detach(), 1).squeeze(1)
                loss = F.nll_loss(log_probs, sampled_labels)
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher[name] += param.grad.data.clone().pow(2)

            except Exception as e:
                logger.debug(f"Fisher estimation batch skipped: {e}")

            n_batches += 1

        # Normalise
        if n_batches > 0:
            for name in self.fisher:
                self.fisher[name] /= n_batches

        total_params = sum(f.numel() for f in self.fisher.values())
        mean_fisher  = sum(f.mean().item() for f in self.fisher.values()) / max(len(self.fisher), 1)
        logger.info(
            f"Fisher estimation complete. "
            f"Params covered: {total_params:,} | Mean F: {mean_fisher:.4e}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularisation penalty.

        penalty = (λ/2) * Σ_i F_i * (θ_i − θ*_i)²

        Args:
            model: Current model being trained on Task B.

        Returns:
            Scalar penalty tensor (to be added to Task B cross-entropy loss).
        """
        loss = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if name not in self.fisher:
                continue
            fisher  = self.fisher[name].to(self.device)
            theta_a = self.params_task_a[name].to(self.device)
            loss   += (fisher * (param - theta_a).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * loss

    def update_lambda(self, new_lambda: float):
        """Dynamically adjust regularisation strength."""
        self.ewc_lambda = new_lambda
        logger.info(f"EWC lambda updated → {new_lambda}")

    def summary(self) -> dict:
        """Return a summary dict for logging."""
        total_params = sum(f.numel() for f in self.fisher.values())
        non_zero     = sum((f > 0).sum().item() for f in self.fisher.values())
        top_constrained = self._top_constrained_params(n=5)
        return {
            "ewc_lambda":           self.ewc_lambda,
            "total_params":         total_params,
            "non_zero_fisher":      non_zero,
            "top_constrained_params": top_constrained,
        }

    def _top_constrained_params(self, n: int = 5) -> list:
        """Return names of the n most Fisher-constrained parameter tensors."""
        ranked = sorted(
            self.fisher.items(),
            key=lambda x: x[1].mean().item(),
            reverse=True,
        )
        return [name for name, _ in ranked[:n]]


# ── Convenience wrapper for training loop ────────────────────────────────────

class ContinualTrainer:
    """
    High-level wrapper that runs Task B training with EWC penalty.

    Designed for use in train_continual.py.

    Args:
        model:       PyTorch model (already trained on Task A).
        ewc:         Initialized EWC instance.
        optimizer:   PyTorch optimizer for Task B.
        device:      "cuda" or "cpu".
    """

    def __init__(
        self,
        model: nn.Module,
        ewc: EWC,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ):
        self.model     = model
        self.ewc       = ewc
        self.optimizer = optimizer
        self.device    = device

    def train_step(
        self,
        images: torch.Tensor,
        targets,
        task_loss_fn,
    ) -> dict:
        """
        Single training step with EWC penalty.

        Args:
            images:       Batch of images.
            targets:      Ground truth targets (format depends on model).
            task_loss_fn: Callable(model, images, targets) → task loss tensor.

        Returns:
            {"task_loss": float, "ewc_penalty": float, "total_loss": float}
        """
        self.model.train()
        self.optimizer.zero_grad()

        images = images.to(self.device)

        task_loss   = task_loss_fn(self.model, images, targets)
        ewc_penalty = self.ewc.penalty(self.model)
        total_loss  = task_loss + ewc_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "task_loss":   task_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
            "total_loss":  total_loss.item(),
        }
