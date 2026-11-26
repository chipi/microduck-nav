"""A small mjlab scene for checking the batched ray path: floor, one mocap box, one mocap head with a ToF site."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

from microduck_nav.tof_geometry import ToFPatternCfg

SITE = "tof"


def _floor_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="plane", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1])
    return spec


def _box_spec(half) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="obstacle")
    body.add_geom(name="obstacle_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half))
    return spec


def _head_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="head")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01], contype=0, conaffinity=0)
    body.add_site(name=SITE, pos=[0, 0, 0])
    return spec


@dataclass
class RayScene:
    scene: Scene
    sim: Simulation
    num_envs: int
    head_height: float

    def place(self, box_pos: torch.Tensor, head_pitch: torch.Tensor, head_xy: torch.Tensor | None = None) -> None:
        """Write the per-world poses of the obstacle and the head, then refresh kinematics."""
        ids = torch.arange(self.num_envs)
        box_pose = torch.zeros(self.num_envs, 7)
        box_pose[:, :3] = box_pos
        box_pose[:, 3] = 1.0
        head_pose = torch.zeros(self.num_envs, 7)
        if head_xy is not None:
            head_pose[:, :2] = head_xy
        head_pose[:, 2] = self.head_height
        head_pose[:, 3] = torch.cos(head_pitch / 2)
        head_pose[:, 5] = torch.sin(head_pitch / 2)
        self.scene["box"].write_mocap_pose_to_sim(box_pose, env_ids=ids)
        self.scene["head"].write_mocap_pose_to_sim(head_pose, env_ids=ids)
        self.sim.forward()

    def sense(self) -> torch.Tensor:
        self.sim.sense()
        return self.scene["tof"].data.distances


def build_ray_scene(
    num_envs: int,
    box_half=(0.05, 0.3, 0.3),
    pattern_mode: str = "pinhole",
    max_distance: float = 4.0,
    head_height: float = 0.22,
    device: str = "cpu",
) -> RayScene:
    cfg = SceneCfg(
        num_envs=num_envs,
        entities={
            "floor": EntityCfg(spec_fn=_floor_spec),
            "box": EntityCfg(spec_fn=lambda: _box_spec(box_half)),
            "head": EntityCfg(spec_fn=lambda: _head_spec()),
        },
        sensors=(
            RayCastSensorCfg(
                name="tof",
                frame=ObjRef("site", SITE, "head"),
                pattern=ToFPatternCfg(mode=pattern_mode),
                max_distance=max_distance,
            ),
        ),
    )
    scene = Scene(cfg, device)
    sim = Simulation(num_envs, SimulationCfg(), scene.compile(), device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(scene.sensor_context)
    return RayScene(scene=scene, sim=sim, num_envs=num_envs, head_height=head_height)


def cpu_reference_distances(rs: RayScene, max_distance: float) -> np.ndarray:
    """The tof.py loop: one mujoco.mj_ray per zone per world, on the same model and poses."""
    model = rs.sim.mj_model
    sensor = rs.scene["tof"]
    site = model.site(f"head/{SITE}").id
    body = int(model.site_bodyid[site])
    local = sensor._local_directions.cpu().numpy().astype(np.float64)
    out = np.full((rs.num_envs, local.shape[0]), -1.0)
    mocap_pos = rs.sim.data.mocap_pos.cpu().numpy()
    mocap_quat = rs.sim.data.mocap_quat.cpu().numpy()
    geom = np.zeros(1, dtype=np.int32)
    for w in range(rs.num_envs):
        data = mujoco.MjData(model)
        data.mocap_pos[:] = mocap_pos[w]
        data.mocap_quat[:] = mocap_quat[w]
        mujoco.mj_forward(model, data)
        origin = data.site_xpos[site].copy()
        rot = data.site_xmat[site].reshape(3, 3)
        for z in range(local.shape[0]):
            vec = np.ascontiguousarray(rot @ local[z])
            hit = mujoco.mj_ray(model, data, origin, vec, None, 1, body, geom)
            if 0 <= hit <= max_distance:
                out[w, z] = hit
    return out
