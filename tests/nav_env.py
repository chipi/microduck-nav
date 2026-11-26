"""Builds the navigation task on CPU for the tests."""

from __future__ import annotations

import os

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import mjlab.tasks  # noqa: F401,E402  registers the tasks through the entry points
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

FULL = "Mjlab-Nav-MicroDuck"
ONE_WALL = "Mjlab-Nav-OneWall-MicroDuck"


def make_env(task: str = FULL, num_envs: int = 4, seed: int = 0, edit=None) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if edit is not None:
        edit(cfg)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    return env
