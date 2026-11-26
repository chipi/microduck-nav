"""The head ToF as an mjlab ray sensor: per-environment zone grid and mount, 15 Hz capture, impairments, delivery.

One sensor, one batched ray call. mjlab's RayCastSensor shares one direction set across environments, so
`prepare_rays` is overridden to write per-environment directions (the grid chosen at reset, rotated by the
episode's mount pitch) and origins (the mount height). Every ray in a world uses that world's grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import warp as wp

from mjlab.sensor import RayCastSensorCfg
from mjlab.sensor.raycast_sensor import RayCastSensor

from microduck_nav.impairments import (
    AcquisitionClock,
    Calibration,
    PacketChannel,
    ToFRandomization,
    corrupt_frame,
    floor_hit_mask,
    sample_calibration,
    task_defaults,
)
from microduck_nav.tof_geometry import ToFPatternCfg, zone_directions

ZONES = 64
GRIDS = ("equal_angle", "pinhole", "mixed")
CAPTURE_PERIOD = 1.0 / 15.0


@dataclass
class ToFSensorCfg(RayCastSensorCfg):
    grid: str = "mixed"
    """equal_angle: Pollen's tof.py (what the simulated duck emits); pinhole: NVIDIA's pixel-centre grid;
    mixed: each environment draws one of the two at every reset."""
    impairments: bool = True
    randomization: ToFRandomization = field(default_factory=task_defaults)
    capture_period: float = CAPTURE_PERIOD
    nominal_pitch_degrees: float = 0.0
    """Pollen's tof site looks level at the home pose; NVIDIA's mount looks 20 degrees down. Positive points down."""

    def __post_init__(self):
        if self.grid not in GRIDS:
            raise ValueError(f"grid must be one of {GRIDS}")

    def build(self) -> "ToFSensor":
        return ToFSensor(self)


def rotate_pitch(directions: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Rotate (n, 64, 3) site-frame directions about +y; a positive angle points the axis down."""
    c, s = torch.cos(pitch)[:, None], torch.sin(pitch)[:, None]
    x, y, z = directions.unbind(-1)
    return torch.stack((x * c + z * s, y, -x * s + z * c), dim=-1)


def draw_grid(n: int, grid: str, device, g: torch.Generator | None = None) -> torch.Tensor:
    """True where an environment uses the pinhole grid."""
    if grid == "pinhole":
        return torch.ones(n, dtype=torch.bool, device=device)
    if grid == "equal_angle":
        return torch.zeros(n, dtype=torch.bool, device=device)
    return torch.rand(n, generator=g, device=device) < 0.5


class ToFSensor(RayCastSensor):
    def __init__(self, cfg: ToFSensorCfg) -> None:
        cfg.pattern = ToFPatternCfg(mode="equal_angle")
        super().__init__(cfg)
        self.cfg: ToFSensorCfg = cfg
        self._pending_dt = 0.0

    # lifecycle

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        n = data.nworld
        self._n = n
        self._dir_sets = torch.stack((zone_directions("equal_angle"), zone_directions("pinhole"))).to(device)
        rand = self.cfg.randomization
        self._pinhole = torch.zeros(n, dtype=torch.bool, device=device)
        self._pitch = torch.zeros(n, device=device)
        self._height = torch.zeros(n, device=device)
        self._env_dirs = self._dir_sets[0].expand(n, -1, -1).clone()
        self.calibration = sample_calibration(n, rand)
        self.calibration = Calibration(**{k: v.to(device) for k, v in self.calibration.__dict__.items()})
        max_latency = rand.latency_frames[1] if self.cfg.impairments else 0
        self.channel = PacketChannel(n, max_latency, device)
        self.clock = AcquisitionClock(self.cfg.capture_period)
        self.env_time = torch.zeros(n, device=device)
        self.captures = torch.zeros(n, dtype=torch.long, device=device)
        self._draw(torch.arange(n, device=device))

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if getattr(self, "_n", None) is None:
            return
        ids = torch.arange(self._n, device=self._device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        self.channel.reset(ids)
        self.env_time[ids] = 0.0
        self.captures[ids] = 0
        self._draw(ids)

    def update(self, dt: float) -> None:
        super().update(dt)
        self._pending_dt += dt

    def _draw(self, ids: torch.Tensor) -> None:
        """Episode draws: grid, mount pitch and height, calibration."""
        rand, n = self.cfg.randomization, len(ids)
        self._pinhole[ids] = draw_grid(n, self.cfg.grid, self._device)
        if self.cfg.impairments:
            self._pitch[ids] = (self.cfg.nominal_pitch_degrees + torch.empty(n, device=self._device).uniform_(*rand.mount_pitch_degrees)) * torch.pi / 180.0
            self._height[ids] = torch.empty(n, device=self._device).uniform_(*rand.mount_height_m)
            fresh = sample_calibration(n, rand)
            for k, v in fresh.__dict__.items():
                getattr(self.calibration, k)[ids] = v.to(self._device)
        else:
            self._pitch[ids] = self.cfg.nominal_pitch_degrees * torch.pi / 180.0
            self._height[ids] = 0.0
            for k in ("distance_bias", "zone_bias", "noise_std", "floor_validity_midpoint_offset_m"):
                getattr(self.calibration, k)[ids] = 0.0
            self.calibration.distance_scale[ids] = 1.0
            self.calibration.latency_frames[ids] = 0
            self.calibration.edge_column[ids] = -1
        chosen = self._dir_sets[self._pinhole[ids].long()]
        self._env_dirs[ids] = rotate_pitch(chosen, self._pitch[ids])

    # rays

    def prepare_rays(self) -> None:
        super().prepare_rays()
        mat = self._cached_frame_mat[:, 0]
        rot = self._compute_alignment_rotation(mat)
        rays = torch.einsum("bij,bnj->bni", rot, self._env_dirs)
        lift = torch.zeros(self._n, 3, device=self._device)
        lift[:, 2] = self._height
        origins = self._cached_frame_pos[:, 0, None, :] + torch.einsum("bij,bj->bi", rot, lift)[:, None, :]
        origins = origins.expand(-1, ZONES, -1)
        wp.to_torch(self._ray_pnt).view(self._n, ZONES, 3).copy_(origins)
        wp.to_torch(self._ray_vec).view(self._n, ZONES, 3).copy_(rays)
        self._cached_world_origins = origins.contiguous()
        self._cached_world_rays = rays

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        dt, self._pending_dt = self._pending_dt, 0.0
        if dt <= 0.0:
            return
        self.env_time += dt
        if self.clock.advance(dt):
            self._capture()

    # acquisition

    def _capture(self) -> None:
        rand, n = self.cfg.randomization, self._n
        radial = self._distances
        axial = torch.where(radial < 0, torch.zeros_like(radial), radial * self._env_dirs[..., 0])
        ids = torch.arange(n, device=self._device)
        if self.cfg.impairments:
            floor = floor_hit_mask(self._hit_pos_w[..., 2], axial)
            frame = corrupt_frame(axial, floor, self.calibration, rand, self.cfg.max_distance)
            received = torch.rand(n, device=self._device) >= rand.frame_dropout_probability
        else:
            keep = (axial >= rand.minimum_distance) & (axial <= self.cfg.max_distance)
            frame = torch.where(keep, axial, torch.zeros_like(axial))
            received = torch.ones(n, dtype=torch.bool, device=self._device)
        self.channel.push(ids, frame, self.env_time.clone(), received, self.calibration.latency_frames)
        self.captures += 1

    # reads

    @property
    def delivered_frame(self) -> torch.Tensor:
        return self.channel.delivered_frame

    @property
    def available(self) -> torch.Tensor:
        return self.channel.available

    @property
    def sample_age(self) -> torch.Tensor:
        return self.channel.age(self.env_time)

    @property
    def uses_pinhole(self) -> torch.Tensor:
        return self._pinhole

    @property
    def env_directions(self) -> torch.Tensor:
        return self._env_dirs

    def world_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The sensor's origin (n, 3), its 64 beams as world unit vectors (n, 64, 3) and each beam's x component in the site frame
        (n, 64), as of the last ray update: what turns an axial depth into a distance along the beam and a point in the world."""
        return self._cached_world_origins[:, 0], self._cached_world_rays, self._env_dirs[..., 0]
