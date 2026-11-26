"""The eight obstacle slots as MJCF primitives on mocap bodies.

Slot names, order and planar footprints follow NVIDIA's obstacle_avoidance_env_cfg.py:72-83 (Isaac Lab
exp/microduck 802b8d5, BSD-3-Clause). Their props are SimReady and Nucleus assets; here a prop is a box
or, for the cone and the post, a cylinder. Each body origin sits on the floor, so a pose is (x, y, 0, yaw).
Heights marked "assumed" are not in NVIDIA's code (their assets carry them).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from mjlab.entity import EntityCfg

BODY = "slot"
GEOM = "shape"
PARK_Z = -5.0  # before the first reset a slot sits under the floor, clear of the robot


@dataclass(frozen=True)
class Slot:
    name: str
    kind: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (0.55, 0.55, 0.6, 1.0)

    @property
    def height(self) -> float:
        return self.size[2] if self.kind == "box" else self.size[1]

    @property
    def footprint_half(self) -> tuple[float, float]:
        if self.kind == "box":
            return (self.size[0] / 2.0, self.size[1] / 2.0)
        return (self.size[0], self.size[0])

    @property
    def radius(self) -> float:
        return self.size[0] if self.kind == "cylinder" else 0.0


# box size = full (x, y, z); cylinder size = (radius, height).
SLOTS: tuple[Slot, ...] = (
    Slot("obstacle", "cylinder", (0.1143, 0.3048), (0.95, 0.45, 0.1, 1.0)),
    Slot("crate", "box", (0.6994, 0.5212, 0.40), (0.6, 0.45, 0.3, 1.0)),  # height assumed
    Slot("tall_box", "box", (0.4, 0.3, 0.12), (0.4, 0.4, 0.45, 1.0)),
    Slot("low_block", "box", (0.4, 0.3, 0.35), (0.25, 0.25, 0.28, 1.0)),  # height assumed
    Slot("cylinder", "box", (0.4633, 0.5750, 0.55), (0.5, 0.35, 0.25, 1.0)),  # height assumed
    Slot("thin_post", "cylinder", (0.1143, 0.3048), (0.95, 0.45, 0.1, 1.0)),
    Slot("wall_left", "box", (0.12, 0.9, 0.65), (0.65, 0.66, 0.68, 1.0)),
    Slot("wall_right", "box", (0.12, 0.9, 0.65), (0.65, 0.66, 0.68, 1.0)),
)
SLOT_NAMES = tuple(s.name for s in SLOTS)
SLOT_INDEX = {s.name: i for i, s in enumerate(SLOTS)}


def slot_spec(slot: Slot) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=BODY, mocap=True)
    if slot.kind == "box":
        body.add_geom(
            name=GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[slot.size[0] / 2, slot.size[1] / 2, slot.size[2] / 2],
            pos=[0, 0, slot.size[2] / 2],
            rgba=list(slot.rgba),
        )
    else:
        body.add_geom(
            name=GEOM,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[slot.size[0], slot.size[1] / 2, 0],
            pos=[0, 0, slot.size[1] / 2],
            rgba=list(slot.rgba),
        )
    return spec


def slot_entities() -> dict[str, EntityCfg]:
    park = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z))
    return {s.name: EntityCfg(spec_fn=lambda s=s: slot_spec(s), init_state=park) for s in SLOTS}


def exclude_static_contacts(spec: mujoco.MjSpec) -> None:
    """Slots collide with the robot only: not with the floor, not with each other (36 pairs of wasted contacts)."""
    for i, a in enumerate(SLOTS):
        spec.add_exclude(name=f"{a.name}_floor", bodyname1="terrain", bodyname2=f"{a.name}/{BODY}")
        for b in SLOTS[i + 1 :]:
            spec.add_exclude(name=f"{a.name}_{b.name}", bodyname1=f"{a.name}/{BODY}", bodyname2=f"{b.name}/{BODY}")


def footprints(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """(half sizes (8, 2), radii (8,)): a circle has radius > 0, as NVIDIA's ObstacleFootprintCfg."""
    half = torch.tensor([s.footprint_half for s in SLOTS], device=device)
    radii = torch.tensor([s.radius for s in SLOTS], device=device)
    return half, radii


# The pool task's slots (R7a). NVIDIA's eight are fixed shapes; our scenes hold up to 14 boxes of any size (a corridor's walls are
# 3.8 m long, a chair's leg 3 cm across), so a pool slot is a unit box whose size and height are written per environment at reset.
# The evaluation's simulator builds every obstacle as a box too (robotics-lab labrunner.scene.Box), so a box is the exact shape.
POOL_PREFIX = "pool_"
POOL_SLOTS = 14  # the most obstacles any item of ours-v1 holds (its manifest's max_obstacles)
POOL_BOX = 0.05  # half side of a slot before its first reset [m]


def pool_slot_names(count: int = POOL_SLOTS) -> tuple[str, ...]:
    return tuple(f"{POOL_PREFIX}{i:02d}" for i in range(count))


def pool_slot_spec() -> mujoco.MjSpec:
    """A mocap box centred on its body origin: the pose written at reset is the box's centre, height included."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=BODY, mocap=True)
    body.add_geom(
        name=GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[POOL_BOX, POOL_BOX, POOL_BOX],
        rgba=[0.65, 0.66, 0.68, 1.0],
    )
    return spec


def pool_slot_entities(count: int = POOL_SLOTS) -> dict[str, EntityCfg]:
    park = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z))
    return {name: EntityCfg(spec_fn=pool_slot_spec, init_state=park) for name in pool_slot_names(count)}


def exclude_pool_contacts(spec: mujoco.MjSpec) -> None:
    """The pool slots collide with the robot only: not with the floor, not with each other."""
    bodies = [b.name for b in spec.bodies if b.name.startswith(POOL_PREFIX) and b.name.endswith(f"/{BODY}")]
    for i, a in enumerate(bodies):
        spec.add_exclude(name=f"{a.split('/')[0]}_floor", bodyname1="terrain", bodyname2=a)
        for b in bodies[i + 1 :]:
            spec.add_exclude(name=f"{a.split('/')[0]}_{b.split('/')[0]}", bodyname1=a, bodyname2=b)
