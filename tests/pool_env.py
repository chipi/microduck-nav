"""Builds Mjlab-Nav-Ours-MicroDuck on CPU from pool files the tests write: a few items, each obstacle given in the trunk's start frame."""

from __future__ import annotations

import json
import math
from pathlib import Path

from microduck_nav.pool import FORMAT, TRUNK_START

from nav_env import make_env

OURS = "Mjlab-Nav-Ours-MicroDuck"
SAMPLE = Path(__file__).parent / "fixtures" / "pool_sample.jsonl"


def compose(t, p):
    """The pose p (x, y, yaw) seen from the frame t, as labrunner.frames.compose does."""
    c, s = math.cos(t[2]), math.sin(t[2])
    return (t[0] + c * p[0] - s * p[1], t[1] + s * p[0] + c * p[1], t[2] + p[2])


def box(x, y, hx, hy, z0=0.0, z1=0.5, yaw=0.0, kind="box"):
    return {"kind": kind, "x": x, "y": y, "yaw": yaw, "hx": hx, "hy": hy, "z0": z0, "z1": z1}


def item(boxes, goal=(2.0, 0.0), start=(0.0, 0.0, 0.0), layout="test", seed=0):
    """An item whose boxes and goal are given in the trunk's frame at the start (x ahead, y left); they are written in the item's
    world, where the trunk stands TRUNK_START ahead of the start's frame."""
    trunk = compose(start, (*TRUNK_START, 0.0))
    world = lambda b: {**b, **dict(zip(("x", "y", "yaw"), compose(trunk, (b["x"], b["y"], b["yaw"]))))}
    gx, gy, _ = compose(trunk, (*goal, 0.0))
    return {
        "format": FORMAT, "challenge": 0, "scene": f"{layout}-s{seed}", "layout": layout, "seed": seed,
        "start": dict(zip(("x", "y", "yaw"), start)), "goal": {"x": gx, "y": gy},
        "obstacles": [world(b) for b in boxes], "limits": {"time_s": 60.0}, "sensor": {"impairment": "off", "seed": 0},
    }


def write_pool(path: Path, items: list[dict]) -> Path:
    path.write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in items))
    return path


def ours_env(pool: Path, guards: bool = True, impairments: bool = False, num_envs: int = 4, seed: int = 0, holdout: str = "none", edit=None):
    """The Ours task on CPU over `pool`. impairments=False makes the ToF clean (no bias, dropout, latency or mount draw) so a test
    knows what the sensor sees; guards=False is the task with the switch off (and no guard cost)."""

    def configure(cfg):
        params = cfg.events["obstacle_field"].params
        params["pool"], params["holdout_family"] = str(pool), holdout
        for sensor in cfg.scene.sensors:
            if sensor.name == "head_tof":
                sensor.impairments = impairments
        cfg.actions["nav"].guards = guards
        cfg.commands["twist"].speed_range = (0.3, 0.3)  # Pollen's walker stands still when asked for little over 0.2 m/s (docs/CHECKS.md C3)
        if not guards:
            cfg.rewards.pop("guard_intervention", None)
        if edit is not None:
            edit(cfg)

    return make_env(OURS, num_envs=num_envs, seed=seed, edit=configure)
