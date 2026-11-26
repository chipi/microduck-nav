"""Per-environment obstacle placement at reset, through mocap poses with explicit env_ids.

mjlab 1.3.0's write_mocap_pose_to_sim without env_ids fails (entity/data.py:213), so every write
here names its rows. The state the rewards and the command read (centres, yaws, layout id, the front
obstacle) lives on `ObstacleField`, one per env, created on first use.
"""

from __future__ import annotations

import math

import torch

from mjlab.managers.event_manager import requires_model_fields

from microduck_nav import layouts, pool as pools
from microduck_nav.obstacles import POOL_BOX, POOL_PREFIX, POOL_SLOTS, SLOT_INDEX, SLOT_NAMES, footprints


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


FAR = 50.0  # where, from its environment's origin, the rewards' geometry puts a slot not in the duck's way: parked, or over its head [m]


def slot_names_of(env) -> tuple[str, ...]:
    """The pool's slots when the scene has them, else NVIDIA's eight."""
    entities = getattr(getattr(env, "scene", None), "entities", ())  # a bare env of the tests has no scene
    names = tuple(n for n in entities if n.startswith(POOL_PREFIX))
    return names or SLOT_NAMES


class ObstacleField:
    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.num_envs = n
        self.names = slot_names_of(env)
        k = len(self.names)
        self.pool = self.names != SLOT_NAMES
        self.pos = torch.zeros(n, k, 3, device=dev)
        self.yaw = torch.zeros(n, k, device=dev)
        self.layout = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.front_index = torch.full((n,), SLOT_INDEX["obstacle"], dtype=torch.long, device=dev)
        if self.pool:
            # the footprints differ per environment: (n, k, 2) half sides and (n, k) radii (none: boxes only)
            self.pos[:, :, :2] = env.scene.env_origins[:, None, :2] + FAR
            self.half_sizes, self.radii = torch.zeros(n, k, 2, device=dev), torch.zeros(n, k, device=dev)
            self.goal_w = torch.zeros(n, 2, device=dev)  # the item's goal in the world
            self.item = torch.full((n,), -1, dtype=torch.long, device=dev)  # which item of the pool each environment holds
            self.geom_ids = torch.tensor([int(env.scene[name].indexing.geom_ids[0]) for name in self.names], device=dev)
        else:
            self.half_sizes, self.radii = footprints(dev)

    def geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centres (n, k, 3), yaw quaternions (n, k, 4), local half sizes, radii, as NVIDIA's obstacle_geometry: (k, 2) and (k,) for
        NVIDIA's fixed slots, (n, k, 2) and (n, k) for the pool's."""
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


PARK_AT = (30.0, 30.0)  # an unused pool slot waits this far from its environment's origin, in a world with nothing else there


def place_boxes(
    local: torch.Tensor, count: torch.Tensor, robot_xy: torch.Tensor, heading: torch.Tensor, origins: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Where an item's boxes stand in the world, given the robot's reset pose (local: (m, k, 7) of pool.FIELDS in the trunk's frame).

    Returns xy (m, k, 2), yaw (m, k), the centre height zc and half height hz (m, k), the half sides (m, k, 2) and `real`
    (m, k): the slot holds one of the item's boxes. An unused slot is a small box parked far off, whose size is POOL_BOX."""
    m, k, _ = local.shape
    real = torch.arange(k, device=local.device)[None, :] < count[:, None]
    c, s = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
    x = robot_xy[:, 0:1] + c * local[..., 0] - s * local[..., 1]
    y = robot_xy[:, 1:2] + s * local[..., 0] + c * local[..., 1]
    park = torch.arange(k, device=local.device, dtype=torch.float32)[None, :] * 0.5
    x = torch.where(real, x, origins[:, 0:1] + PARK_AT[0] + park)
    y = torch.where(real, y, origins[:, 1:2] + PARK_AT[1])
    yaw = torch.where(real, heading[:, None] + local[..., 2], torch.zeros_like(x))
    z0, z1 = local[..., 5], local[..., 6]
    hz = torch.where(real, 0.5 * (z1 - z0), torch.full_like(x, POOL_BOX))
    zc = origins[:, 2:3] + torch.where(real, 0.5 * (z0 + z1), torch.full_like(x, POOL_BOX))
    half = torch.where(real[..., None], local[..., 3:5], torch.full_like(local[..., 3:5], POOL_BOX))
    return {"xy": torch.stack((x, y), dim=-1), "yaw": yaw, "zc": zc, "hz": hz, "half": half, "real": real}


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def reset_pool_scene(
    env,
    env_ids,
    pool: str | None = None,
    holdout_family: str = "doorway",
    slots: int = POOL_SLOTS,
    generator: torch.Generator | None = None,
) -> None:
    """Give each environment being reset one item of the pool: its boxes in the pool slots, resized and placed by the robot's
    reset pose (which must already be written), its goal in the field for the command. Slots the item does not use are parked.

    A box whose underside is over the duck's head (pools.OVERHEAD_Z: a chair's seat) is a real box for the rays and the contacts but
    is not in the rewards' footprints, which are planar: under it the duck is not inside an obstacle."""
    field = get_field(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    if len(ids) == 0:
        return
    data = pools.get_pool(pool, slots, holdout_family, env.device)
    pick = torch.randint(len(data), (len(ids),), generator=generator, device=env.device)
    local, count = data.obstacles[pick], data.count[pick]
    origins = env.scene.env_origins[ids]
    pos, quat = root_pose(env, ids)
    heading = yaw_of(quat)
    placed = place_boxes(local, count, pos[:, :2], heading, origins)
    real, xy = placed["real"], placed["xy"]

    in_the_way = real & (local[..., 5] < pools.OVERHEAD_Z)
    field.pos[ids, :, :2] = torch.where(in_the_way[..., None], xy, origins[:, None, :2] + FAR)
    field.pos[ids, :, 2] = origins[:, 2:3]
    field.yaw[ids] = placed["yaw"]
    field.half_sizes[ids] = torch.where(in_the_way[..., None], placed["half"], torch.zeros_like(placed["half"]))
    field.item[ids] = pick
    field.layout[ids] = data.layout[pick]
    c, s = torch.cos(heading), torch.sin(heading)
    field.goal_w[ids] = torch.stack(
        (pos[:, 0] + c * data.goal[pick, 0] - s * data.goal[pick, 1], pos[:, 1] + s * data.goal[pick, 0] + c * data.goal[pick, 1]), dim=-1
    )

    size = torch.cat((placed["half"], placed["hz"][..., None]), dim=-1)  # (m, k, 3): a box geom's half sides
    rows, cols = torch.meshgrid(ids.to(torch.int), field.geom_ids.to(torch.int), indexing="ij")
    model = env.sim.model
    model.geom_size[rows, cols] = size
    model.geom_rbound[rows, cols] = size.norm(dim=-1)  # the broadphase's bounds, as mjlab's dr.geom_size keeps them for a box
    model.geom_aabb[rows, cols, 1] = size
    for i, name in enumerate(field.names):
        pose = torch.zeros(len(ids), 7, device=env.device)
        pose[:, :2] = xy[:, i]
        pose[:, 2] = placed["zc"][:, i]
        pose[:, 3:7] = quat_from_yaw(placed["yaw"][:, i])
        env.scene[name].write_mocap_pose_to_sim(pose, env_ids=ids)
