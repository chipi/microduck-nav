"""The navigation residual on top of the requested velocity, as NVIDIA's MicroDuckWalkingAction.process_actions
(mdp/actions.py:52-76) and the deployment command_from_action (nav_pad.py:104-116) do it."""

from __future__ import annotations

import torch


class ResidualCommand:
    """Velocity command = request + residual, bounded, never faster than the request, slew-limited.

    A zero request, a missing frame or a frame older than max_age zeroes the command at once,
    bypassing the slew limit.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        limits=(0.35, 0.12, 0.75),
        slew=(1.0, 1.0, 3.0),
        max_age: float = 0.25,
        max_lateral_ratio: float = 0.35,
        device: str = "cpu",
    ):
        self.dt = dt
        self.limits = torch.tensor(limits, device=device)
        self.slew = torch.tensor(slew, device=device)
        self.max_age = max_age
        self.max_lateral_ratio = max_lateral_ratio
        self.previous = torch.zeros(num_envs, 3, device=device)

    def reset(self, ids=None) -> None:
        self.previous[slice(None) if ids is None else ids] = 0.0

    def process(
        self,
        requested: torch.Tensor,
        action: torch.Tensor,
        have_frame: torch.Tensor,
        frame_age: torch.Tensor,
    ) -> torch.Tensor:
        cmd = requested + action.clamp(-1.0, 1.0) * self.limits
        cmd[:, 0].clamp_(0.0, self.limits[0].item())
        cmd[:, 1].clamp_(-self.limits[1].item(), self.limits[1].item())
        cmd[:, 2].clamp_(-self.limits[2].item(), self.limits[2].item())
        requested_speed = requested[:, :2].norm(dim=-1, keepdim=True)
        adjusted_speed = cmd[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cmd[:, :2] *= (requested_speed / adjusted_speed).clamp(max=1.0)

        step = self.slew * self.dt
        cmd = self.previous + (cmd - self.previous).clamp(-step, step)
        lateral = cmd[:, 0] * self.max_lateral_ratio
        cmd[:, 1] = torch.maximum(torch.minimum(cmd[:, 1], lateral), -lateral)

        stop = (requested.abs().sum(dim=-1) == 0) | ~have_frame | (frame_age > self.max_age)
        cmd = torch.where(stop.unsqueeze(-1), torch.zeros_like(cmd), cmd)
        self.previous.copy_(cmd)
        return cmd
