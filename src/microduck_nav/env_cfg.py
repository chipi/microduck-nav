"""The navigation task's environment config, built on Pollen's VelStand flat backlash task.

The robot, its actuators, its domain randomisation events and the walker's proprioception terms are
Pollen's, taken from the registered task. What follows NVIDIA's obstacle_avoidance_env_cfg.py (Isaac Lab
exp/microduck 802b8d5, BSD-3-Clause) is the scene, the observation groups, the action, the command, the
rewards and weights, the terminations, the 20 s episode and the 14 m environment spacing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import last_action
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.registry import load_env_cfg

from microduck_nav import observations as obs
from microduck_nav import rewards
from microduck_nav import terminations as terms
from microduck_nav.action import SENSOR, WALKER_GROUP, NavigationActionCfg
from microduck_nav.commands import NavigationCommandCfg
from microduck_nav.field import reset_obstacle_field
from microduck_nav.layouts import NVIDIA_WEIGHTS
from microduck_nav.obstacles import BODY, SLOT_NAMES, exclude_static_contacts, slot_entities
from microduck_nav.tick_history import TOF_HISTORY
from microduck_nav.tof_sensor import ToFSensorCfg

BASE_TASK = "Mjlab-VelStand-Flat-Backlash-MicroDuck"
TOF_SITE = "tof"
TOF_MAX_RANGE = 4.0
ENV_SPACING = 14.0
FIELD_EXTENT = 5.0
PROPRIO_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
KEPT_EVENTS = (
    "reset_base", "reset_robot_joints", "push_robot", "foot_friction", "encoder_bias", "base_com",
    "expand_bam_friction_fields", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
    "randomize_joint_friction", "randomize_armature",
)
KEPT_SENSORS = ("feet_ground_contact", "foot_height_scan")
REWARD_WEIGHTS = {
    "progress": 5.0,
    "goal_arrival": 30.0,
    "barrier_bypass": 5.0,
    "yaw_tracking": 4.0,
    "command_tracking": 1.0,
    "approach": -4.0,
    "command_rate": -5.0,
    "failure": -30.0,
}


@dataclass
class NavOptions:
    """What a task id fixes; every field is one place to change."""

    layout: str = "mixed"
    layout_weights: dict[str, float] | None = None
    one_wall_distance_range: tuple[float, float] = (0.6, 1.0)
    grid: str = "mixed"
    impairments: bool = True
    speed_range: tuple[float, float] = (0.2, 0.35)
    walker_onnx: str | None = None
    collision_threshold: float = 0.1
    policy_decimation: int = 1  # walker steps per policy action: 1 is 50 Hz, 5 the navigator's 10 Hz
    lateral_limit: float = 0.12  # the residual's lateral bound [m/s]; 0 sends no lateral, as the duck navigator does


def _proprio_group(base: ManagerBasedRlEnvCfg) -> dict[str, ObservationTermCfg]:
    actor = base.observations["actor"].terms
    return {name: deepcopy(actor[name]) for name in PROPRIO_TERMS}


def _observations(base: ManagerBasedRlEnvCfg, opts: NavOptions) -> dict[str, ObservationGroupCfg]:
    zeros = lambda dim: ObservationTermCfg(func=obs.zeros, params={"dim": dim})
    policy = _proprio_group(base)
    held = opts.policy_decimation > 1
    action_func = obs.held_action if held else last_action
    policy["actions"] = ObservationTermCfg(func=action_func)
    policy["command"] = deepcopy(base.observations["actor"].terms["command"])
    policy["head_command"] = zeros(4)
    policy["body_command"] = zeros(6)
    policy["applied_command"] = ObservationTermCfg(func=obs.applied_velocity_commands)

    critic = deepcopy(base.observations["critic"].terms)
    critic["actions"] = ObservationTermCfg(func=action_func)
    critic["head_command"] = zeros(4)
    critic["body_command"] = zeros(6)
    critic["obstacle_clearance"] = ObservationTermCfg(func=obs.obstacle_clearance)

    if held:
        # the navigator reads the joint velocities of its previous tick: the lag is a tick of walker steps, not one step
        policy["joint_vel"].delay_min_lag = policy["joint_vel"].delay_max_lag = opts.policy_decimation
        tof = ObservationGroupCfg(
            terms={
                "ranges_and_validity": ObservationTermCfg(func=obs.tof_ranges_history),
                "age": ObservationTermCfg(func=obs.tof_age_history),
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    else:
        tof = ObservationGroupCfg(
            terms={
                "ranges_and_validity": ObservationTermCfg(func=obs.tof_ranges_and_validity),
                "age": ObservationTermCfg(func=obs.tof_age),
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=TOF_HISTORY,
            flatten_history_dim=True,
        )

    return {
        WALKER_GROUP: ObservationGroupCfg(terms=_proprio_group(base), concatenate_terms=True, enable_corruption=True),
        "policy": ObservationGroupCfg(terms=policy, concatenate_terms=True, enable_corruption=True),
        "tof": tof,
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
    }


def _sensors(base: ManagerBasedRlEnvCfg, opts: NavOptions, decimation: int):
    kept = tuple(s for s in base.scene.sensors if s.name in KEPT_SENSORS)
    contacts = tuple(
        ContactSensorCfg(
            name=terms.CONTACT_PREFIX + name,
            primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
            secondary=ContactMatch(mode="body", pattern=BODY, entity=name),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
            history_length=decimation,
        )
        for name in SLOT_NAMES
    )
    tof = ToFSensorCfg(
        name=SENSOR,
        frame=ObjRef(type="site", name=TOF_SITE, entity="robot"),
        max_distance=TOF_MAX_RANGE,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        grid=opts.grid,
        impairments=opts.impairments,
    )
    return kept + contacts + (tof,)


def make_nav_env_cfg(opts: NavOptions | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    opts = opts or NavOptions()
    import mjlab_microduck.tasks  # noqa: F401  registers Pollen's tasks

    cfg = load_env_cfg(BASE_TASK)
    cfg.episode_length_s = 20.0
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.scene.num_envs = 4096
    cfg.scene.env_spacing = ENV_SPACING
    cfg.scene.entities = {**cfg.scene.entities, **slot_entities()}
    cfg.scene.sensors = _sensors(cfg, opts, cfg.decimation)
    cfg.scene.spec_fn = exclude_static_contacts
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.sim.contact_sensor_maxmatch = 128  # a robot overlapping a wall matches more than the default 64 contacts

    cfg.observations = _observations(cfg, opts)
    cfg.actions = {
        "nav": NavigationActionCfg(
            entity_name="robot",
            walker_onnx=opts.walker_onnx,
            limits=(0.35, opts.lateral_limit, 0.75),
            policy_decimation=opts.policy_decimation,
        )
    }
    cfg.commands = {"twist": NavigationCommandCfg(speed_range=opts.speed_range)}

    events = {name: cfg.events[name] for name in KEPT_EVENTS if name in cfg.events}
    events["reset_base"].params["pose_range"].update(x=(-0.15, 0.15), y=(-0.15, 0.15), yaw=(-3.14, 3.14))
    if play:
        events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    events["push_robot"].interval_range_s = (5.0, 10.0)
    events["push_robot"].params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
    for name, final in (("randomize_com", 0.015), ("randomize_head_com", 0.01)):
        if name in events:
            events[name].params["ranges"] = (-final, final)
    events["obstacle_field"] = EventTermCfg(
        func=reset_obstacle_field,
        mode="reset",
        params={
            "layout": opts.layout,
            "layout_weights": opts.layout_weights if opts.layout_weights is not None else dict(NVIDIA_WEIGHTS),
            "one_wall_distance_range": opts.one_wall_distance_range,
        },
    )
    cfg.events = events

    cfg.rewards = {
        "progress": RewardTermCfg(func=rewards.gated_progress, weight=REWARD_WEIGHTS["progress"]),
        "goal_arrival": RewardTermCfg(func=rewards.NavigationGoalArrival, weight=REWARD_WEIGHTS["goal_arrival"]),
        "barrier_bypass": RewardTermCfg(func=rewards.NavigationObstacleBypass, weight=REWARD_WEIGHTS["barrier_bypass"]),
        "yaw_tracking": RewardTermCfg(func=rewards.yaw_tracking, weight=REWARD_WEIGHTS["yaw_tracking"]),
        "command_tracking": RewardTermCfg(func=rewards.command_tracking, weight=REWARD_WEIGHTS["command_tracking"]),
        "approach": RewardTermCfg(
            func=rewards.approach_risk,
            weight=REWARD_WEIGHTS["approach"],
            params={"safety_distance": 0.2, "time_horizon": 2.0, "commanded_risk_weight": 1.0},
        ),
        "command_rate": RewardTermCfg(func=rewards.command_rate, weight=REWARD_WEIGHTS["command_rate"]),
        "failure": RewardTermCfg(func=rewards.terminated_penalty, weight=REWARD_WEIGHTS["failure"]),
    }

    base_terms = cfg.terminations
    cfg.terminations = {
        "time_out": base_terms["time_out"],
        "fell_over": base_terms["fell_over"],
        "nan_state": base_terms["nan_state"],
        "obstacle_collision": TerminationTermCfg(
            func=terms.obstacle_collision, time_out=False, params={"threshold": opts.collision_threshold}
        ),
        "field_boundary": TerminationTermCfg(func=terms.field_boundary, time_out=True, params={"extent": FIELD_EXTENT}),
    }
    if play:
        cfg.scene.num_envs = 16
        events.pop("push_robot", None)
    return cfg
