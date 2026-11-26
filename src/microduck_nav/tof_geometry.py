"""Zone directions of the 8x8 time-of-flight sensor, in the sensor frame (+x forward, +y left, +z up).

Two conventions exist and both are kept: "equal_angle" is Pollen's simulated sensor
(sim/tof.py:50-62, equal angle steps), "pinhole" is the pixel-centre grid of NVIDIA's
pinhole ray camera (the edge zone sits at 19.9 degrees instead of 19.69). Row 0 is the top
of the frame and column 0 the sensor's left, both flattened row-major.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

ROWS = COLS = 8
ZONES = ROWS * COLS
FOV_DEG = 45.0

Mode = Literal["equal_angle", "pinhole"]


def zone_directions(mode: Mode = "pinhole", fov_deg: float = FOV_DEG, dtype=torch.float32) -> torch.Tensor:
    half = math.radians(fov_deg) / 2.0
    index = torch.arange(COLS, dtype=torch.float64)
    if mode == "equal_angle":
        centres = (-half + (index + 0.5) * (2.0 * half / COLS))
        elevation = -centres[:, None].expand(ROWS, COLS)
        azimuth = -centres[None, :].expand(ROWS, COLS)
        d = torch.stack(
            (
                torch.cos(elevation) * torch.cos(azimuth),
                torch.cos(elevation) * torch.sin(azimuth),
                torch.sin(elevation),
            ),
            dim=-1,
        )
    elif mode == "pinhole":
        plane = ((2.0 * index + 1.0) / COLS - 1.0) * math.tan(half)
        y = -plane[None, :].expand(ROWS, COLS)
        z = -plane[:, None].expand(ROWS, COLS)
        d = torch.stack((torch.ones_like(y), y, z), dim=-1)
        d = d / d.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return d.reshape(ZONES, 3).to(dtype)


def radial_to_axial(radial: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Range along the beam to depth along the optical axis; misses (negative) are kept."""
    axial = radial * directions[:, 0]
    return torch.where(radial < 0, radial, axial)


def axial_to_radial(axial: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    radial = axial / directions[:, 0]
    return torch.where(axial < 0, axial, radial)


@dataclass
class ToFPatternCfg:
    """A ray pattern for mjlab's RayCastSensor: 64 zero-offset rays in the site frame."""

    mode: Mode = "pinhole"
    fov_deg: float = FOV_DEG

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        directions = zone_directions(self.mode, self.fov_deg).to(device)
        return torch.zeros((ZONES, 3), device=device), directions
