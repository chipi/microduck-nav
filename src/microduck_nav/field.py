"""Per-environment obstacle placement at reset, through mocap poses with explicit env_ids.

mjlab 1.3.0's write_mocap_pose_to_sim without env_ids fails (entity/data.py:213), so every write
here names its rows. The state the rewards and the command read (centres, yaws, layout id, the front
obstacle) lives on `ObstacleField`, one per env, created on first use.
"""

from __future__ import annotations

import math

import torch

from microduck_nav import layouts
from microduck_nav.obstacles import SLOT_INDEX, SLOT_NAMES, footprints


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=-1)


def root_pose(env, ids=None, asset: str = "robot") -> tuple[torch.Tensor, torch.Tensor]:
    """Root position and quaternion (wxyz) read from qpos: current right after a reset write, before forward()."""
    data = env.scene[asset].data
    adr = data.indexing.free_joint_q_adr
    rows = slice(None) if ids is None else ids[:, None]
    q = data.data.qpos[rows, adr]
    return q[..., :3], q[..., 3:7]


class ObstacleField:
    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.num_envs = n
        self.pos = torch.zeros(n, len(SLOT_NAMES), 3, device=dev)
        self.yaw = torch.zeros(n, len(SLOT_NAMES), device=dev)
        self.layout = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.front_index = torch.full((n,), SLOT_INDEX["obstacle"], dtype=torch.long, device=dev)
        self.half_sizes, self.radii = footprints(dev)

    def geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centres (n, 8, 3), yaw quaternions (n, 8, 4), local half sizes (8, 2), radii (8,), as NVIDIA's obstacle_geometry."""
        return self.pos, quat_from_yaw(self.yaw), self.half_sizes, self.radii


def get_field(env) -> ObstacleField:
    if not hasattr(env, "nav_field"):
        env.nav_field = ObstacleField(env)
    return env.nav_field


def poses_to_world(
    local: torch.Tensor, robot_xy: torch.Tensor, heading: torch.Tensor, origins_xy: torch.Tensor, ring: torch.Tensor,
    spacing_scale: float, jitter: float, slot_order: torch.Tensor, g=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World (x, y) and yaw of every slot: robot-relative for the layouts, origin-relative cells for 'ring'."""
    c, s = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
    x = robot_xy[:, 0:1] + c * local[..., 0] - s * local[..., 1]
    y = robot_xy[:, 1:2] + s * local[..., 0] + c * local[..., 1]
    yaw = heading[:, None] + local[..., 2]
    if ring.any():
        cells = torch.tensor(list(layouts.OBSTACLE_CELLS.values()), device=local.device)[slot_order] * spacing_scale
        r = int(ring.sum())
        noise = torch.empty(r, len(SLOT_NAMES), 2, device=local.device).uniform_(-jitter, jitter, generator=g)
        x[ring] = origins_xy[ring, 0:1] + cells[ring, :, 0] + noise[..., 0]
        y[ring] = origins_xy[ring, 1:2] + cells[ring, :, 1] + noise[..., 1]
        yaw[ring] = torch.empty(r, len(SLOT_NAMES), device=local.device).uniform_(-math.pi, math.pi, generator=g)
    return torch.stack((x, y), dim=-1), yaw


def reset_obstacle_field(
    env,
    env_ids,
    layout: str = "mixed",
    layout_weights: dict[str, float] | None = None,
    jitter: float = 0.15,
    spacing_scale: float = 1.6,
    blocked_front_distance_range: tuple[float, float] = (0.55, 1.05),
    one_wall_distance_range: tuple[float, float] = (0.6, 1.0),
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> None:
    """Place the eight slots for the environments being reset; must run after the robot's reset pose is written."""
    if not 0.0 <= jitter <= 0.15:
        raise ValueError("jitter must be in [0, 0.15] m to keep the layout clearances")
    if spacing_scale < 1.0 or not math.isfinite(spacing_scale):
        raise ValueError("spacing_scale must be finite and at least one")
    field = get_field(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    if len(ids) == 0:
        return
    weights = layout_weights if layout_weights is not None else (layouts.NVIDIA_WEIGHTS if layout == "mixed" else None)
    layout_ids = layouts.sample_layout_ids(len(ids), layout, weights, generator, env.device)
    local = layouts.sample_local_poses(
        layout_ids, jitter, spacing_scale, blocked_front_distance_range, one_wall_distance_range, generator
    )
    front_slot = SLOT_INDEX["obstacle"]
    front = torch.full((len(ids),), front_slot, dtype=torch.long, device=env.device)
    order = torch.arange(len(SLOT_NAMES), device=env.device).expand(len(ids), -1).clone()
    if shuffle:
        order = layouts.shuffle_slots(len(ids), g=generator, device=env.device)
        front = (order == front_slot).long().argmax(dim=-1)
        local = local.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    pos, quat = root_pose(env, ids)
    ring = layout_ids == layouts.LAYOUT_NAMES.index("ring")
    xy, yaw = poses_to_world(
        local, pos[:, :2], yaw_of(quat), env.scene.env_origins[ids, :2], ring, spacing_scale, jitter, order, generator
    )
    field.layout[ids] = layout_ids
    field.front_index[ids] = front
    field.pos[ids, :, :2] = xy
    field.pos[ids, :, 2] = env.scene.env_origins[ids, 2:3]
    field.yaw[ids] = yaw
    for i, name in enumerate(SLOT_NAMES):
        pose = torch.zeros(len(ids), 7, device=env.device)
        pose[:, :3] = field.pos[ids, i]
        pose[:, 3:7] = quat_from_yaw(yaw[:, i])
        env.scene[name].write_mocap_pose_to_sim(pose, env_ids=ids)
