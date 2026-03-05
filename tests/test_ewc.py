"""
tests/test_ewc.py
─────────────────
Unit tests for the EWC continual learning module.
No GPU or Waymo data required.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.continual.ewc import EWC, ContinualTrainer


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_simple_model():
    """Tiny 2-layer MLP for fast testing."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),   # 4 output classes
    )


def make_dataloader(n=50, input_dim=16):
    X = torch.randn(n, input_dim)
    y = torch.randint(0, 4, (n,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=4, shuffle=True)


# ── EWC tests ─────────────────────────────────────────────────────────────────

class TestEWC:

    def test_fisher_computed(self):
        """Fisher information is non-zero after estimation."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=10, ewc_lambda=0.4)

        assert len(ewc.fisher) > 0
        # At least some Fisher values should be non-zero
        total_nonzero = sum((f > 0).sum().item() for f in ewc.fisher.values())
        assert total_nonzero > 0, "All Fisher values are zero"

    def test_params_task_a_saved(self):
        """Task A parameter snapshot has same keys as model parameters."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5)

        param_names = {n for n, _ in model.named_parameters() if _.requires_grad}
        assert set(ewc.params_task_a.keys()) == param_names

    def test_penalty_zero_at_task_a(self):
        """Penalty should be ~0 when model hasn't moved from Task A params."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5, ewc_lambda=0.4)

        penalty = ewc.penalty(model)
        assert isinstance(penalty, torch.Tensor)
        assert penalty.item() < 1e-6, f"Expected ~0 penalty, got {penalty.item()}"

    def test_penalty_increases_after_param_change(self):
        """Penalty increases after parameters are modified."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5, ewc_lambda=0.4)

        penalty_before = ewc.penalty(model).item()

        # Perturb model parameters significantly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 5.0)

        penalty_after = ewc.penalty(model).item()
        assert penalty_after > penalty_before, \
            f"Penalty did not increase: before={penalty_before}, after={penalty_after}"

    def test_penalty_scales_with_lambda(self):
        """Larger lambda → larger penalty."""
        model   = make_simple_model()
        loader  = make_dataloader()

        ewc_low  = EWC(model, loader, device="cpu", n_samples=5, ewc_lambda=0.1)
        ewc_high = EWC(model, loader, device="cpu", n_samples=5, ewc_lambda=1.0)

        # Perturb model
        model2 = make_simple_model()
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(torch.randn_like(p))

        p_low  = ewc_low.penalty(model2).item()
        p_high = ewc_high.penalty(model2).item()
        assert p_high > p_low, \
            f"Expected p_high > p_low: {p_high:.4f} vs {p_low:.4f}"

    def test_penalty_returns_tensor(self):
        """Penalty returns a scalar tensor (for .backward() compatibility)."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5)
        penalty = ewc.penalty(model)
        assert isinstance(penalty, torch.Tensor)
        assert penalty.shape == torch.Size([])   # scalar

    def test_penalty_backprop(self):
        """Penalty can be used in a backward pass without errors."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        # Simulate task B loss
        x      = torch.randn(4, 16)
        logits = model(x)
        task_loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 4, (4,)))
        total_loss = task_loss + ewc.penalty(model)
        total_loss.backward()    # Should not raise
        optimizer.step()

    def test_update_lambda(self):
        """Lambda can be updated dynamically."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5, ewc_lambda=0.4)
        ewc.update_lambda(0.9)
        assert ewc.ewc_lambda == 0.9

    def test_summary(self):
        """Summary returns expected keys."""
        model  = make_simple_model()
        loader = make_dataloader()
        ewc    = EWC(model, loader, device="cpu", n_samples=5)
        s = ewc.summary()
        assert "ewc_lambda"       in s
        assert "total_params"     in s
        assert "non_zero_fisher"  in s


# ── ContinualTrainer tests ────────────────────────────────────────────────────

class TestContinualTrainer:

    def test_train_step_returns_losses(self):
        """train_step returns task_loss, ewc_penalty, total_loss."""
        model     = make_simple_model()
        loader    = make_dataloader()
        ewc       = EWC(model, loader, device="cpu", n_samples=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer   = ContinualTrainer(model, ewc, optimizer, device="cpu")

        def task_loss_fn(m, imgs, targets):
            out = m(imgs)
            return nn.CrossEntropyLoss()(out, targets)

        images  = torch.randn(4, 16)
        targets = torch.randint(0, 4, (4,))
        losses  = trainer.train_step(images, targets, task_loss_fn)

        assert "task_loss"   in losses
        assert "ewc_penalty" in losses
        assert "total_loss"  in losses
        assert losses["total_loss"] >= losses["task_loss"]
