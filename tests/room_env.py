"""Builds Mjlab-Nav-Deploy-Room-MicroDuck on CPU for the tests."""

from __future__ import annotations

from nav_env import make_env

ROOM = "Mjlab-Nav-Deploy-Room-MicroDuck"
M2B = "Mjlab-Nav-Deploy-MicroDuck"


def room_env(num_envs: int = 4, seed: int = 0, impairments: bool = False, layout: str | None = None, edit=None):
    """The room task on CPU. impairments=False makes the ToF clean so a test knows what the sensor sees; `layout` pins NVIDIA's family."""

    def configure(cfg):
        for sensor in cfg.scene.sensors:
            if sensor.name == "head_tof":
                sensor.impairments = impairments
        if layout is not None:
            cfg.events["obstacle_field"].params["layout"] = layout
        cfg.commands["twist"].speed_range = (0.35, 0.35)
        if edit is not None:
            edit(cfg)

    return make_env(ROOM, num_envs=num_envs, seed=seed, edit=configure)
