"""Golden observation vectors for deployment: constructed states and the 1343 numbers the training code builds from each.

A deployed copy of the actor (robotics-lab's `learned.py`) must build exactly the vector this task trains on, or a trained model
there reads garbage. This builds a few constructed states on a one-environment CPU env, sets them into the simulation and the ToF
sensor's channel, and reads the vector back through the task's own observation functions (the proprioception terms of the
`policy` group, then the `tof` group through the manager, so the history is mjlab's own circular buffer). Noise, delays and the
IMU misalignment are Pollen's training randomisation and are switched off here: a deployed robot has neither on top of its sensors.
"""

from __future__ import annotations

import os

os.environ.setdefault("MJLAB_WARP_QUIET", "1")

import math

import torch

from microduck_nav.action import SENSOR
from microduck_nav.env_cfg import PROPRIO_TERMS
from microduck_nav.observations import record_tick
from microduck_nav.tick_history import TOF_HISTORY
from microduck_nav.tof_geometry import zone_directions

TASK = "Mjlab-Nav-MicroDuck"
DEPLOY_TASK = "Mjlab-Nav-Deploy-MicroDuck"
ZONES = 64
CLOCK = 2.0  # the sensor's env time while a state is read; ages need delivered_time >= 0, and float32 is exact enough here
POLICY_TERMS = PROPRIO_TERMS + ("actions", "command", "head_command", "body_command", "applied_command")


def wall(base: float, slope: float = 0.0) -> list[float]:
    """An 8x8 frame, row-major (row 0 the top), of a wall `base` metres away tilted by `slope` m per column."""
    return [round(base + slope * (z % 8) + 0.01 * (z // 8), 4) for z in range(ZONES)]


def with_zones(frame: list[float], zones: dict[int, float]) -> list[float]:
    out = list(frame)
    for z, v in zones.items():
        out[z] = v
    return out


def states() -> list[dict]:
    """Five constructed states. A frame is {axial_m: 64 depths along the optical axis, 0 where the zone gave nothing, age_s: how
    old it is now}; frames are listed oldest first. Every number is exact in float32 so both sides start from the same bits."""
    zero_row = {z: 0.0 for z in range(8)}  # the top row gave nothing
    return [
        {  # episode start: no frame yet, the history is ten empty ticks aged 1 s
            "name": "start", "requested": [0.25, 0.0, 0.3], "applied": [0.0, 0.0, 0.0], "previous_action": [0.0, 0.0, 0.0],
            "joint_pos": [0.0] * 14, "joint_vel": [0.0] * 14, "backlash": [0.0] * 14,
            "quat": [1.0, 0.0, 0.0, 0.0], "gyro": [0.0, 0.0, 0.0], "frames": [],
        },
        {  # walking: three frames 1/15 s apart, some zones empty, some out of range, a drifting wall
            "name": "walking", "requested": [0.3, 0.0, -0.2], "applied": [0.21, 0.0, -0.1], "previous_action": [0.4, -0.2, 0.7],
            "joint_pos": [0.05, -0.12, 0.3, -0.2, 0.15, 0.4, -0.3, 0.1, -0.05, 0.02, 0.09, -0.4, 0.25, 0.08],
            "joint_vel": [0.5, -1.0, 2.0, -0.5, 0.25, 0.0, 1.5, -2.5, 0.75, -0.125, 3.0, 0.5, -1.25, 0.625],
            "backlash": [0.0] * 14,
            "quat": [0.9961946, 0.0, 0.0871557, 0.0], "gyro": [0.1, -0.2, 0.5],
            "frames": [
                {"axial_m": with_zones(wall(1.4, 0.02), {5: 0.0, 18: 0.0, 63: 0.05}), "age_s": 0.17},
                {"axial_m": with_zones(wall(1.3, 0.02), {3: 0.0, 40: 0.0, 41: 4.5}), "age_s": 0.1},
                {"axial_m": with_zones(wall(1.2, 0.02), {9: 0.0, 10: 0.0, 55: 0.0}), "age_s": 0.03},
            ],
        },
        {  # tilted and turning: roll and pitch, far-from-default joints, the encoders reading through the backlash
            "name": "tilted", "requested": [0.2, 0.0, 0.75], "applied": [0.2, 0.0, 0.75], "previous_action": [-1.0, 0.5, -0.6],
            "joint_pos": [-0.2, 0.3, -0.5, 0.4, -0.3, 0.2, 0.1, -0.4, 0.15, 0.3, -0.2, 0.5, -0.1, 0.05],
            "joint_vel": [-3.0, 2.0, -1.0, 4.0, -2.0, 1.0, 0.0, -0.5, 0.25, 1.75, -2.25, 0.125, 5.0, -4.0],
            "backlash": [0.01, -0.005, 0.0, 0.01, 0.0, -0.01, 0.0, 0.005, 0.0, 0.0, -0.005, 0.0, 0.01, 0.0],
            "quat": [0.9, 0.2, -0.3, 0.1], "gyro": [-0.8, 0.6, -1.2],
            "frames": [
                {"axial_m": with_zones(wall(0.6, -0.03), {z: 0.0 for z in (0, 7, 56, 63)}), "age_s": 0.06},
                {"axial_m": wall(0.55, -0.03), "age_s": 0.0},
            ],
        },
        {  # stale: the newest frame is half a second old, the sensor dropped frames; most zones empty
            "name": "stale", "requested": [0.35, 0.0, 0.0], "applied": [0.3, 0.0, 0.0], "previous_action": [0.0, 0.0, 0.0],
            "joint_pos": [0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0],
            "joint_vel": [0.0] * 14, "backlash": [0.0] * 14,
            "quat": [0.9961946, -0.0871557, 0.0, 0.0], "gyro": [0.0, 0.0, 0.0],
            "frames": [
                {"axial_m": with_zones(wall(2.5), {z: 0.0 for z in range(0, 64, 3)}), "age_s": 0.75},
                {"axial_m": with_zones(wall(2.0), {**zero_row, 20: 0.0}), "age_s": 0.5},
            ],
        },
        {  # a long history: frames older than the ten ticks, a far wall near the range limit, a close one
            "name": "long_history", "requested": [0.25, 0.0, 0.1], "applied": [0.25, 0.0, 0.1], "previous_action": [0.25, 0.0, 0.0],
            "joint_pos": [0.01 * i for i in range(14)], "joint_vel": [0.25 * (i - 7) for i in range(14)], "backlash": [0.0] * 14,
            "quat": [1.0, 0.0, 0.0, 0.0], "gyro": [0.0, 0.0, 0.1],
            "frames": [
                {"axial_m": wall(3.9, 0.01), "age_s": 0.5},
                {"axial_m": with_zones(wall(0.13, 0.05), {7: 0.119}), "age_s": 0.4333333},
                {"axial_m": wall(0.9, 0.05), "age_s": 0.2},
                {"axial_m": wall(0.8, 0.05), "age_s": 0.1333333},
                {"axial_m": wall(0.7, 0.05), "age_s": 0.0666667},
                {"axial_m": wall(0.6, 0.05), "age_s": 0.0133333},
            ],
        },
    ]


def history_for(frames: list[dict], step_s: float, steps: int) -> list[tuple[list[float] | None, float]]:
    """What the sensor held at each of the last `steps` control steps, oldest first: the newest frame already captured then (None
    before the first one) and its age at that step. The training buffer records one such pair per control step."""
    out = []
    for k in reversed(range(steps)):
        offset = k * step_s
        fresh = [f for f in frames if f["age_s"] >= offset - 1e-9]
        if fresh:
            f = min(fresh, key=lambda c: c["age_s"])
            out.append((f["axial_m"], f["age_s"] - offset))
        else:
            out.append((None, 1.0))
    return out


def f32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def make_env(task: str = TASK):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab.tasks  # noqa: F401  registers the tasks through the entry points

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    return env


def set_proprio(env, state: dict) -> None:
    from mjlab.utils.lab_api.math import quat_apply

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    pos, vel = torch.zeros(1, len(names)), torch.zeros(1, len(names))
    servo = [i for i, n in enumerate(names) if not n.startswith("passive_")]
    for j, i in enumerate(servo):
        pos[0, i], vel[0, i] = state["joint_pos"][j], state["joint_vel"][j]
        # the encoder reads the servo plus its backlash hinge, so the hinge carries the difference
        pos[0, names.index(f"passive_{names[i]}_backlash")] = state["backlash"][j]
    # the servo position is what the encoder reads less the hinge, so the term's value is joint_pos + backlash
    robot.write_joint_state_to_sim(pos, vel)
    quat = f32([state["quat"]])
    quat = quat / quat.norm()
    omega_w = quat_apply(quat, f32([state["gyro"]]))
    robot.write_root_state_to_sim(torch.cat((f32([[0.0, 0.0, 0.2]]), quat, torch.zeros(1, 3), omega_w), dim=-1))
    robot.data.encoder_bias[:] = 0.0
    env._imu_misalign_quat = f32([[1.0, 0.0, 0.0, 0.0]])
    env.sim.forward()
    env.action_manager.action[:] = f32([state["previous_action"]])
    env.action_manager.get_term("nav")._raw[:] = f32([state["previous_action"]])
    env.command_manager.get_term("twist").vel_command_b[:] = f32([state["requested"]])
    env.action_manager.get_term("nav").residual.previous[:] = f32([state["applied"]])


def policy_group(env) -> torch.Tensor:
    manager = env.observation_manager
    cfgs = dict(zip(manager._group_obs_term_names["policy"], manager._group_obs_term_cfgs["policy"], strict=True))
    assert tuple(cfgs) == POLICY_TERMS, tuple(cfgs)
    return torch.cat([cfgs[n].func(env, **cfgs[n].params) for n in POLICY_TERMS], dim=-1)


def tof_group(env, state: dict, step_s: float) -> torch.Tensor:
    manager = env.observation_manager
    history = env.action_manager.get_term("nav").tof_history
    if history is not None:  # a policy that acts every N steps keeps one entry per tick in its own ring
        history.reset(slice(None))
    for buffer in manager._group_obs_term_history_buffer["tof"].values() if history is None else ():
        buffer.reset()
    sensor = env.scene[SENSOR]
    for frame, age in history_for(state["frames"], step_s, TOF_HISTORY):
        if frame is None:  # as at a reset: no frame delivered, env time 0, so the age is 1 s
            sensor.channel.delivered_frame[0] = 0.0
            sensor.channel.delivered_time[0] = -1.0
            sensor.env_time[0] = 0.0
        else:
            sensor.channel.delivered_frame[0] = f32(frame)
            sensor.env_time[0] = CLOCK
            sensor.channel.delivered_time[0] = CLOCK - age
        if history is None:
            out = manager.compute_group("tof", update_history=True)
        else:
            record_tick(env, force=True)
    if history is not None:
        out = manager.compute_group("tof", update_history=True)
    return out


def build_fixture(task: str = TASK) -> dict:
    env = make_env(task)
    decimation = env.action_manager.get_term("nav").decimation
    step_s = float(env.step_dt) * decimation  # the spacing of the ten history entries: a policy tick
    robot = env.scene["robot"]
    servo = [n for n in robot.joint_names if not n.startswith("passive_")]
    default = robot.data.default_joint_pos[0, [robot.joint_names.index(n) for n in servo]]
    rows = []
    for state in states():
        quat = f32(state["quat"])
        state = {**state, "quat": (quat / quat.norm()).tolist()}  # an IMU's quaternion has unit length; the fixture stores the one set
        set_proprio(env, state)
        policy, tof = policy_group(env), tof_group(env, state, step_s)
        vector = torch.cat((policy, tof), dim=-1)[0]
        assert vector.shape == (1343,) and torch.isfinite(vector).all()
        rows.append({**state, "observation": vector.tolist()})
    env.close()
    return {
        "source": "microduck-nav src/microduck_nav/obs_parity.py: the task's own observation functions on a 1-env CPU env",
        "task": task, "step_s": step_s, "policy_decimation": decimation, "walker_step_s": float(env.step_dt), "history": TOF_HISTORY, "joints": servo, "default_pose": default.tolist(),
        "zone_axial_factor": zone_directions("equal_angle")[:, 0].tolist(),
        "layout": {"policy_terms": list(POLICY_TERMS), "tof_terms": ["ranges_and_validity", "age"], "actor_groups": ["policy", "tof"]},
        "tolerance": 1e-6, "states": rows,
    }
