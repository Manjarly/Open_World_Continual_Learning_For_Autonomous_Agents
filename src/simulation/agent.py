"""
src/simulation/agent.py
───────────────────────
PyTorch Reinforcement Learning Agent for Autonomous Driving.

Features:
  - Multi-head Actor-Critic policy:
      * Motion control: [Hard Left, Gentle Left, Straight, Gentle Right, Hard Right, Brake, Accelerate]
      * Indicator control: [Off, Left, Right]
      * Headlight control: [Off, On]
      * Value estimation: V(s)
  - Open-World Uncertainty Detector:
      * Quantifies novelty/uncertainty of out-of-distribution city states
      * Open-world cautious reflex: slows down, boosts vigilance, activates headlights in dark
  - GPU Execution: Apple MPS / CUDA compliant with strict 80% memory watermark limit
"""

import os
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def resolve_rl_device(requested_device: str = "auto") -> torch.device:
    """
    Resolve compute device adhering to project GPU rules:
      - Apple MPS on macOS
      - CUDA on NVIDIA
      - Strict <80% memory watermark limit
    """
    dev = requested_device.lower() if requested_device else "auto"

    if dev in ("mps", "auto") and torch.backends.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.80"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.60"
        return torch.device("mps")
    elif dev in ("cuda", "auto") and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.80)
        except Exception:
            pass
        return torch.device("cuda")
    return torch.device("cpu")


class OpenWorldUncertaintyEstimator(nn.Module):
    """
    Autoencoder-based feature novelty estimator for Open-World Continual Learning.
    Measures reconstruction error on city state representations.
    High error indicates out-of-distribution (OOD) city conditions (e.g. City B unknown path).
    """
    def __init__(self, feature_dim: int = 128, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(features)
        recon = self.decoder(code)
        # Mean squared error per sample as uncertainty score
        recon_err = torch.mean((features - recon) ** 2, dim=-1)
        # Normalized uncertainty score in [0, 1] using sigmoid scaling
        uncertainty = torch.sigmoid((recon_err - 0.05) * 20.0)
        return recon, uncertainty


class DrivingPolicy(nn.Module):
    """
    Multi-Head Actor-Critic Policy with Open-World Uncertainty Head.
    """
    def __init__(
        self,
        obs_dim: int = 25,
        motion_dim: int = 7,
        indicator_dim: int = 3,
        headlight_dim: int = 2,
        hidden_dim: int = 128,
        uncertainty_threshold: float = 0.55,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.motion_dim = motion_dim
        self.indicator_dim = indicator_dim
        self.headlight_dim = headlight_dim
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or resolve_rl_device()

        # Shared representation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor Heads
        self.motion_head = nn.Linear(hidden_dim, motion_dim)
        self.indicator_head = nn.Linear(hidden_dim, indicator_dim)
        self.headlight_head = nn.Linear(hidden_dim, headlight_dim)

        # Initialize sensible exploration priors:
        # Bias towards cruise/accelerate straight over aggressive hard turns initially
        with torch.no_grad():
            self.motion_head.bias.copy_(torch.tensor([-0.8, 0.2, 1.2, 0.2, -0.8, -0.2, 0.8]))
            self.indicator_head.bias.copy_(torch.tensor([1.0, -0.5, -0.5]))
            self.headlight_head.bias.copy_(torch.tensor([0.5, -0.5]))

        # Critic Head (Value function V(s))
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Open-World Uncertainty Estimator
        self.uncertainty_net = OpenWorldUncertaintyEstimator(hidden_dim)

        self.to(self.device)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from state observations."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits, value, and uncertainty.
        """
        features = self.extract_features(obs)
        motion_logits = self.motion_head(features)
        indicator_logits = self.indicator_head(features)
        headlight_logits = self.headlight_head(features)
        value = self.critic_head(features).squeeze(-1)

        _, uncertainty = self.uncertainty_net(features)
        return motion_logits, indicator_logits, headlight_logits, value, uncertainty

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
        enable_cautious_reflex: bool = True,
    ) -> Tuple[Tuple[int, int, int], float, float, float]:
        """
        Select actions with optional Open-World Cautious Reflex.

        Returns:
          actions: (motion_act, indicator_act, headlight_act)
          total_log_prob: float
          value: float
          uncertainty: float
        """
        self.eval()
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)

        with torch.no_grad():
            m_logits, i_logits, h_logits, value, uncertainty = self.forward(obs_tensor)

            m_dist = Categorical(logits=m_logits)
            i_dist = Categorical(logits=i_logits)
            h_dist = Categorical(logits=h_logits)

            if deterministic:
                m_act = int(torch.argmax(m_logits, dim=-1).item())
                i_act = int(torch.argmax(i_logits, dim=-1).item())
                h_act = int(torch.argmax(h_logits, dim=-1).item())
            else:
                m_act = int(m_dist.sample().item())
                i_act = int(i_dist.sample().item())
                h_act = int(h_dist.sample().item())

            unc_val = float(uncertainty.squeeze().item())

            # ── Open-World Cautious Reflex ─────────────────────────────────────
            # If novel, unfamiliar conditions are detected in City B:
            if enable_cautious_reflex and unc_val > self.uncertainty_threshold:
                # 1. Darkness safety reflex: if ambient light (obs[11]) is low, enable headlights!
                ambient_light = float(obs_tensor.flatten()[11].item())
                if ambient_light < 0.35:
                    h_act = 1  # turn on headlights

                # 2. Avoid aggressive acceleration (action 6) into unknown road; default to cruise/slow
                if m_act == 6:
                    m_act = 2  # cruise straight instead of accelerating

            log_prob = (
                m_dist.log_prob(torch.tensor(m_act, device=self.device)) +
                i_dist.log_prob(torch.tensor(i_act, device=self.device)) +
                h_dist.log_prob(torch.tensor(h_act, device=self.device))
            ).item()

            v_val = float(value.squeeze().item())

        return (m_act, i_act, h_act), float(log_prob), float(v_val), float(unc_val)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        motion_acts: torch.Tensor,
        indicator_acts: torch.Tensor,
        headlight_acts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities, entropy, and values for training batches.
        """
        m_logits, i_logits, h_logits, values, uncertainty = self.forward(obs)

        m_dist = Categorical(logits=m_logits)
        i_dist = Categorical(logits=i_logits)
        h_dist = Categorical(logits=h_logits)

        log_probs = (
            m_dist.log_prob(motion_acts) +
            i_dist.log_prob(indicator_acts) +
            h_dist.log_prob(headlight_acts)
        )

        entropy = m_dist.entropy() + i_dist.entropy() + h_dist.entropy()

        return log_probs, values, entropy, uncertainty
