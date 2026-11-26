"""Registers the navigation tasks with mjlab, the way Pollen's tasks.__init__ does (entry point mjlab.tasks)."""

from mjlab.tasks.registry import register_mjlab_task

from microduck_nav.env_cfg import NavOptions, make_nav_env_cfg
from microduck_nav.rl_cfg import NavOnPolicyRunner, make_rl_cfg

FULL = "Mjlab-Nav-MicroDuck"
ONE_WALL = "Mjlab-Nav-OneWall-MicroDuck"
DEPLOY = "Mjlab-Nav-Deploy-MicroDuck"

# Milestone M1: one wall across the approach, no impairments.
_ONE_WALL = NavOptions(layout="one_wall", impairments=False)
# Trained as robotics-lab's duck navigator runs it: the model is asked at 10 Hz (a 0.1 s tick, five walker steps) and no lateral
# velocity is sent. Layouts and impairments are the full task's.
DEPLOY_DECIMATION = 5
_DEPLOY = NavOptions(policy_decimation=DEPLOY_DECIMATION, lateral_limit=0.0)

for _task, _opts, _name in (
    (FULL, NavOptions(), "microduck_nav"),
    (ONE_WALL, _ONE_WALL, "microduck_nav_one_wall"),
    (DEPLOY, _DEPLOY, "microduck_nav_deploy"),
):
    register_mjlab_task(
        task_id=_task,
        env_cfg=make_nav_env_cfg(_opts),
        play_env_cfg=make_nav_env_cfg(_opts, play=True),
        rl_cfg=make_rl_cfg(_name, _opts.policy_decimation),
        runner_cls=NavOnPolicyRunner,
    )
