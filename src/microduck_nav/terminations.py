"""Obstacle collision and the field boundary (mdp/terminations.py, Isaac Lab exp/microduck 802b8d5, BSD-3-Clause)."""

from __future__ import annotations

import torch

from microduck_nav.obstacles import SLOT_NAMES

CONTACT_PREFIX = "contact_"


def contact_sensor_names() -> tuple[str, ...]:
    return tuple(CONTACT_PREFIX + n for n in SLOT_NAMES)


def obstacle_forces(env, threshold: float = 0.1) -> torch.Tensor:
    """Peak contact force of the robot against each obstacle over the last control step's physics steps, (n, 8)."""
    peaks = []
    for name in contact_sensor_names():
        history = env.scene[name].data.force_history
        if history is None:
            raise ValueError("Obstacle collision detection needs a contact sensor with force history.")
        peaks.append(history.norm(dim=-1).flatten(start_dim=1).amax(dim=-1))
    return torch.stack(peaks, dim=-1)


def obstacle_collision(env, threshold: float = 0.1) -> torch.Tensor:
    """Any robot part touching any obstacle with more than `threshold` N in any physics step of the control step."""
    return (obstacle_forces(env) > threshold).any(dim=-1)


def field_boundary(env, extent: float = 5.0) -> torch.Tensor:
    """Truncate episodes outside the finite field, without calling it a collision."""
    relative = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return relative.norm(dim=-1) > extent
