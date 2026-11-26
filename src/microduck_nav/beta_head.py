"""A Beta action head for rsl_rl 5.0.1, which ships no BetaDistribution, with NVIDIA's guarded mean.

The guarded deterministic output is adapted from agents/navigation_distribution.py (NVIDIA
Isaac Lab exp/microduck 802b8d5, BSD-3-Clause): a forward head whose two concentrations round
to 1.0 returns -1 (brake), and any non-finite logit returns the brake (-1, 0, 0). The Beta
base class is written here, since the one NVIDIA subclasses is not in rsl_rl 5.0.1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Beta, kl_divergence


class BetaDistribution(Distribution):
    """Beta over (0, 1) mapped to the action range; the MLP emits [..., 2, dim] raw concentration logits."""

    def __init__(self, output_dim: int, action_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        super().__init__(output_dim)
        self.low, self.high = action_range
        self._scale = self.high - self.low
        self._log_range_scale = math.log(self._scale) * output_dim
        self._distribution: Beta | None = None

    @staticmethod
    def concentrations(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = torch.unbind(logits, dim=-2)
        return F.softplus(a) + 1.0, F.softplus(b) + 1.0

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        alpha, beta = self.concentrations(mlp_output)
        self._distribution = Beta(alpha, beta, validate_args=False)

    def sample(self) -> torch.Tensor:
        return self.low + self._scale * self._distribution.sample()

    @property
    def mean(self) -> torch.Tensor:
        return self.low + self._scale * self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self._scale * self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1) + self._log_range_scale

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._distribution.concentration1, self._distribution.concentration0)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        unit = ((outputs - self.low) / self._scale).clamp(1e-6, 1.0 - 1e-6)
        return self._distribution.log_prob(unit).sum(dim=-1) - self._log_range_scale

    def kl_divergence(self, old_params, new_params) -> torch.Tensor:
        old = Beta(*old_params, validate_args=False)
        new = Beta(*new_params, validate_args=False)
        return kl_divergence(old, new).sum(dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.concentrations(mlp_output)
        return self.low + self._scale * alpha / (alpha + beta)

    def as_deterministic_output_module(self) -> torch.nn.Module:
        return _PlainBetaMean(self.low, self._scale)


class _PlainBetaMean(torch.nn.Module):
    def __init__(self, low: float, scale: float) -> None:
        super().__init__()
        self.low, self.scale = low, scale

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        alpha, beta = BetaDistribution.concentrations(logits)
        return self.low + self.scale * alpha / (alpha + beta)


class _BrakingBetaMean(torch.nn.Module):
    """Ordinary Beta means; brake when the forward head rounds to uniform or anything is non-finite."""

    def __init__(self, forward_braking_only: bool = False) -> None:
        super().__init__()
        self.forward_braking_only = forward_braking_only

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        alpha, beta = BetaDistribution.concentrations(logits)
        denominator = alpha + beta
        mean = 2.0 * (alpha / denominator) - 1.0
        collapsed = (alpha[..., 0] == 1.0) & (beta[..., 0] == 1.0)
        forward = (mean[..., 0] - 1.0) / 2.0 if self.forward_braking_only else mean[..., 0]
        forward = torch.where(collapsed, -torch.ones_like(forward), forward)
        adjusted = torch.cat((forward.unsqueeze(-1), mean[..., 1:]), dim=-1)
        invalid = ~torch.isfinite(logits).all(dim=-1).all(dim=-1) | ~torch.isfinite(denominator).all(dim=-1)
        braking = torch.stack((-torch.ones_like(forward), torch.zeros_like(forward), torch.zeros_like(forward)), dim=-1)
        return torch.where(invalid.unsqueeze(-1), braking, adjusted)


class MicroDuckBrakingBeta(BetaDistribution):
    """With forward_braking_only the forward residual lives in [-1, 0]; lateral and yaw stay in [-1, 1]."""

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
        forward_braking_only: bool = False,
    ) -> None:
        if output_dim != 3 or tuple(action_range) != (-1.0, 1.0):
            raise ValueError("MicroDuck braking requires three navigation residuals in [-1, 1].")
        super().__init__(output_dim, action_range)
        self.forward_braking_only = forward_braking_only
        self._mean_output = _BrakingBetaMean(forward_braking_only)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self._mean_output(mlp_output)

    def as_deterministic_output_module(self) -> torch.nn.Module:
        return self._mean_output

    def sample(self) -> torch.Tensor:
        samples = super().sample()
        if self.forward_braking_only:
            samples[..., 0] = (samples[..., 0] - 1.0) / 2.0
        return samples

    @property
    def mean(self) -> torch.Tensor:
        mean = super().mean
        if self.forward_braking_only:
            mean[..., 0] = (mean[..., 0] - 1.0) / 2.0
        return mean

    @property
    def std(self) -> torch.Tensor:
        std = super().std
        if self.forward_braking_only:
            std[..., 0] /= 2.0
        return std

    @property
    def entropy(self) -> torch.Tensor:
        entropy = super().entropy
        if self.forward_braking_only:
            entropy = entropy - math.log(self._scale)
        return entropy

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.forward_braking_only:
            outputs = outputs.clone()
            outputs[..., 0] = 2.0 * outputs[..., 0] + 1.0
            return super().log_prob(outputs) + math.log(self._scale)
        return super().log_prob(outputs)
