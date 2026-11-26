"""The ToF history of a policy that acts every N walker steps: one entry per policy tick, not per walker step.

At N = 1 the task uses mjlab's own per-step history buffer. Above 1 that buffer would fill at the walker's 50 Hz and span
0.2 s, while the navigator keeps one entry per tick (0.1 s at 10 Hz, ten of them: one second). The ring here is pushed only
on the walker step that completes a tick, which is when the policy reads its observation.
"""

from __future__ import annotations

import torch

TOF_HISTORY = 10


class TickHistory:
    """(envs, length, width) ring of ranges and (envs, length) of ages, oldest first; a reset row backfills on its first push as
    mjlab's CircularBuffer does."""

    def __init__(self, num_envs: int, length: int, width: int, device: str):
        self.ranges = torch.zeros(num_envs, length, width, device=device)
        self.ages = torch.zeros(num_envs, length, device=device)
        self.empty = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_step = -1

    def reset(self, env_ids) -> None:
        self.empty[env_ids] = True

    def push(self, ranges: torch.Tensor, ages: torch.Tensor, rows: torch.Tensor, step: int) -> None:
        keep = rows & ~self.empty
        fill = rows & self.empty
        self.ranges[keep] = torch.cat((self.ranges[keep][:, 1:], ranges[keep].unsqueeze(1)), dim=1)
        self.ages[keep] = torch.cat((self.ages[keep][:, 1:], ages[keep].unsqueeze(1)), dim=1)
        self.ranges[fill] = ranges[fill].unsqueeze(1).expand(-1, self.ranges.shape[1], -1)
        self.ages[fill] = ages[fill].unsqueeze(1).expand(-1, self.ages.shape[1])
        self.empty[rows] = False
        self.last_step = step
