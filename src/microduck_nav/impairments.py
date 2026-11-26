"""Time-of-flight impairments, ported from NVIDIA's MicroDuckToFCamera (sensors.py:27-301).

Same formulas and same draw order as the original, on flat (envs, 64) tensors instead of an
(envs, 8, 8, 1) image, with an explicit torch.Generator. Input depths are optical-axis metres,
0 for a miss (the task's depth_clipping_behavior="zero").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

ZONES = 64
COLUMNS = 8
FLOOR_NEAR_LIMIT = 0.55


@dataclass
class ToFRandomization:
    """Defaults are NVIDIA's ToFRandomizationCfg; task_defaults() adds the task's overrides."""

    enabled: bool = True
    mount_pitch_degrees: tuple[float, float] = (0.0, 0.0)
    mount_height_m: tuple[float, float] = (0.0, 0.0)
    latency_frames: tuple[int, int] = (0, 1)
    frame_dropout_probability: float = 0.02
    minimum_distance: float = 0.12
    distance_bias: tuple[float, float] = (-0.02, 0.02)
    zone_bias: tuple[float, float] = (-0.01, 0.01)
    distance_scale: tuple[float, float] = (0.98, 1.02)
    noise_std: tuple[float, float] = (0.003, 0.01)
    distance_noise_fraction: float = 0.008
    invalid_zone_probability: float = 0.03
    near_floor_edge_impairment_probability: float = 0.0
    near_floor_edge_survival_probability: tuple[float, float] = (0.02, 0.55)
    floor_validity_midpoint_offset_m: tuple[float, float] = (0.0, 0.0)
    floor_validity_transition_m: tuple[float, float] | None = (0.556, 0.015)

    def __post_init__(self):
        lo, hi = self.latency_frames
        if not 0 <= lo <= hi:
            raise ValueError("latency_frames must be an ordered nonnegative interval")
        for name in ("invalid_zone_probability", "near_floor_edge_impairment_probability", "frame_dropout_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")


def task_defaults() -> ToFRandomization:
    """The values obstacle_avoidance_env_cfg.py:230-235 sets for the task."""
    return ToFRandomization(
        mount_pitch_degrees=(-10.0, 10.0),
        mount_height_m=(-0.02, 0.02),
        floor_validity_midpoint_offset_m=(0.0, 0.13),
        near_floor_edge_impairment_probability=0.5,
    )


@dataclass
class Calibration:
    """Episode-level draws, one row per environment."""

    latency_frames: torch.Tensor
    distance_bias: torch.Tensor
    distance_scale: torch.Tensor
    noise_std: torch.Tensor
    floor_validity_midpoint_offset_m: torch.Tensor
    near_floor_edge_survival_probability: torch.Tensor
    zone_bias: torch.Tensor
    edge_column: torch.Tensor

    def __getitem__(self, ids) -> "Calibration":
        return Calibration(**{k: v[ids] for k, v in self.__dict__.items()})


_EPISODE_ORDER = (
    "distance_bias",
    "distance_scale",
    "noise_std",
    "floor_validity_midpoint_offset_m",
    "near_floor_edge_survival_probability",
)


def sample_calibration(n: int, cfg: ToFRandomization, generator: torch.Generator | None = None) -> Calibration:
    low, high = cfg.latency_frames
    latency = torch.randint(low, high + 1, (n,), generator=generator)
    drawn = {}
    for name in _EPISODE_ORDER:
        lo, hi = getattr(cfg, name)
        drawn[name] = torch.empty(n, 1).uniform_(lo, hi, generator=generator)
    drawn["zone_bias"] = torch.empty(n, ZONES).uniform_(*cfg.zone_bias, generator=generator)
    impaired = torch.rand((n, 1), generator=generator) < cfg.near_floor_edge_impairment_probability
    edge = torch.randint(0, 2, (n, 1), generator=generator) * (COLUMNS - 1)
    return Calibration(latency_frames=latency, edge_column=torch.where(impaired, edge, -1), **drawn)


def floor_hit_mask(hit_z: torch.Tensor, ideal: torch.Tensor) -> torch.Tensor:
    """A hit on the flat floor at world z=0 (sensors.py:291); a miss has no hit."""
    return (hit_z.abs() < 1e-3) & (ideal > 0)


def zone_dropout_probability(
    ideal: torch.Tensor, floor_hit: torch.Tensor, cal: Calibration, cfg: ToFRandomization
) -> torch.Tensor:
    p = torch.full_like(ideal, cfg.invalid_zone_probability)
    if cfg.floor_validity_transition_m is not None:
        midpoint, width = cfg.floor_validity_transition_m
        midpoint = midpoint + cal.floor_validity_midpoint_offset_m
        floor_dropout = torch.sigmoid((ideal - midpoint) / width)
        floor_dropout = torch.where(ideal <= FLOOR_NEAR_LIMIT, 0.0, floor_dropout)
        p = torch.where(floor_hit, floor_dropout, p)
    columns = torch.arange(ZONES, device=ideal.device)[None, :] % COLUMNS
    impaired = columns == cal.edge_column
    near = floor_hit & (ideal >= cfg.minimum_distance) & (ideal <= FLOOR_NEAR_LIMIT) & impaired
    edge = 1.0 - (1.0 - p) * cal.near_floor_edge_survival_probability
    return torch.where(near, edge, p)


def corrupt_frame(
    ideal: torch.Tensor,
    floor_hit: torch.Tensor,
    cal: Calibration,
    cfg: ToFRandomization,
    max_distance: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One acquisition: calibration, noise and zone validity. Invalid zones are 0."""
    valid = torch.isfinite(ideal) & (ideal >= cfg.minimum_distance) & (ideal <= max_distance)
    noisy = ideal * cal.distance_scale + cal.distance_bias
    noisy = noisy + cal.zone_bias
    std = cal.noise_std + cfg.distance_noise_fraction * ideal
    noisy = noisy + torch.randn(ideal.shape, generator=generator, device=ideal.device) * std
    p = zone_dropout_probability(ideal, floor_hit, cal, cfg)
    valid &= torch.rand(ideal.shape, generator=generator, device=ideal.device) >= p
    valid &= (noisy >= cfg.minimum_distance) & (noisy <= max_distance)
    return torch.where(valid, noisy, 0.0)


class AcquisitionClock:
    """Deadline scheduler that keeps the sub-step remainder: 15 Hz on average, captured on step edges."""

    def __init__(self, period: float):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be positive and finite")
        self.period = period
        self.time = 0.0
        self.next_capture = period

    def advance(self, dt: float) -> bool:
        self.time += dt
        if self.time + 1e-9 >= self.next_capture:
            periods = math.floor((self.time + 1e-9 - self.next_capture) / self.period) + 1
            self.next_capture += periods * self.period
            return True
        return False


class PacketChannel:
    """Per-environment ring buffer: transport latency in frames, packet loss, and delivered-frame age."""

    def __init__(self, num_envs: int, max_latency: int, device: str = "cpu"):
        self.slots = max_latency + 1
        self.frames = torch.zeros(self.slots, num_envs, ZONES, device=device)
        self.times = torch.full((self.slots, num_envs), -1.0, device=device)
        self.write = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.delivered_frame = torch.zeros(num_envs, ZONES, device=device)
        self.delivered_time = torch.full((num_envs,), -1.0, device=device)

    def reset(self, ids: torch.Tensor) -> None:
        self.frames[:, ids] = 0
        self.times[:, ids] = -1.0
        self.write[ids] = 0
        self.delivered_frame[ids] = 0
        self.delivered_time[ids] = -1.0

    @property
    def available(self) -> torch.Tensor:
        return self.delivered_time >= 0.0

    def age(self, now) -> torch.Tensor:
        return now - self.delivered_time

    def push(
        self,
        ids: torch.Tensor,
        frame: torch.Tensor,
        capture_time: torch.Tensor,
        received: torch.Tensor,
        latency: torch.Tensor,
    ) -> None:
        slot = self.write[ids]
        self.frames[slot, ids] = frame
        self.times[slot, ids] = torch.where(received, capture_time, -1.0)
        read = (slot - latency) % self.slots
        packet_time = self.times[read, ids]
        ok = packet_time >= 0.0
        deliver = ids[ok]
        self.delivered_frame[deliver] = self.frames[read[ok], deliver]
        self.delivered_time[deliver] = packet_time[ok]
        self.write[ids] = (slot + 1) % self.slots
