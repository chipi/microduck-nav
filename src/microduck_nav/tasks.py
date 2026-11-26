"""Registers the navigation tasks with mjlab, the way Pollen's tasks.__init__ does (entry point mjlab.tasks)."""

from mjlab.tasks.registry import register_mjlab_task

from microduck_nav.env_cfg import NavOptions, make_nav_env_cfg
from microduck_nav.rl_cfg import NavOnPolicyRunner, make_rl_cfg

FULL = "Mjlab-Nav-MicroDuck"
ONE_WALL = "Mjlab-Nav-OneWall-MicroDuck"

# Milestone M1: one wall across the approach, no impairments.
_ONE_WALL = NavOptions(layout="one_wall", impairments=False)

for _task, _opts, _name in ((FULL, NavOptions(), "microduck_nav"), (ONE_WALL, _ONE_WALL, "microduck_nav_one_wall")):
    register_mjlab_task(
        task_id=_task,
        env_cfg=make_nav_env_cfg(_opts),
        play_env_cfg=make_nav_env_cfg(_opts, play=True),
        rl_cfg=make_rl_cfg(_name),
        runner_cls=NavOnPolicyRunner,
    )
