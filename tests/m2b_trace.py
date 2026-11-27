"""A fixed-seed trace of Mjlab-Nav-Deploy-MicroDuck's observations, hashed: M2d must leave M2b's task byte for byte as it was."""

from __future__ import annotations

import hashlib

import torch

from nav_env import make_env

TASK = "Mjlab-Nav-Deploy-MicroDuck"
TICKS = 6


def trace(task: str = TASK, seed: int = 3, envs: int = 4) -> str:
    env = make_env(task, num_envs=envs, seed=seed)
    try:
        h = hashlib.sha256()
        obs, _ = env.reset()
        torch.manual_seed(seed)
        for tick in range(TICKS + 1):
            for group in ("policy", "tof", "critic"):
                v = obs[group]
                h.update(v.contiguous().numpy().tobytes())
            if tick < TICKS:
                obs, *_ = env.step(torch.full((envs, 3), 0.2))
        return h.hexdigest()
    finally:
        env.close()
