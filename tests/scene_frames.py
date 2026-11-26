"""Frames of a clean 8x8 ToF in a synthetic world: axis-aligned boxes and a floor, rendered by ray and box intersection.

The sensor sits at `origin` (the trunk's frame of the tests is the world's, the sensor 8.5 cm ahead of the trunk as the duck's is) and
looks along +x. A frame is what the task's channel delivers: axial depths in metres, 0 where the zone read nothing.
"""

from __future__ import annotations

import math

import torch

from microduck_nav.tof_geometry import zone_directions

SENSOR_AHEAD = 0.085  # tof8x8.py sensor_offset: about 8.5 cm ahead of the trunk
HEIGHT = 0.12
MIN_RANGE, MAX_RANGE = 0.12, 4.0


def rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> torch.Tensor:
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    ry = torch.tensor([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])  # positive pitch points the axis down
    rx = torch.tensor([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def render(boxes=(), floor=0.0, floor_ends_at: float | None = None, floor_below: float | None = None, origin=(0.0, 0.0, HEIGHT),
           rot: torch.Tensor | None = None, mode: str = "equal_angle") -> dict:
    """The world's frame: `boxes` are (x0, x1, y0, y1, z0, z1); the floor is at `floor`, and past x = `floor_ends_at` it is gone
    (a hole), or `floor_below` metres lower (a ledge). Returns the frame, the beams in the world and the site-frame x components."""
    site = zone_directions(mode).double()
    rot = (torch.eye(3) if rot is None else rot).double()
    world = site @ rot.T
    o = torch.tensor(origin, dtype=torch.float64)
    ranges = torch.zeros(64, dtype=torch.float64)
    for i in range(64):
        d = world[i]
        best = math.inf
        if d[2] < -1e-9:  # the floor, at its height before the edge and at its lower height past it
            t = (floor - o[2]) / d[2]
            hit = o + t * d
            if floor_ends_at is None or hit[0] < floor_ends_at:
                best = t.item()
            elif floor_below is not None:
                t2 = (floor - floor_below - o[2]) / d[2]
                if (o + t2 * d)[0] >= floor_ends_at:
                    best = t2.item()
        for b in boxes:
            lo, hi = torch.tensor([b[0], b[2], b[4]], dtype=torch.float64), torch.tensor([b[1], b[3], b[5]], dtype=torch.float64)
            t0, t1 = 0.0, math.inf
            for k in range(3):
                if abs(d[k]) < 1e-12:
                    if not lo[k] <= o[k] <= hi[k]:
                        t0, t1 = 1.0, 0.0
                    continue
                a, c = ((lo[k] - o[k]) / d[k]).item(), ((hi[k] - o[k]) / d[k]).item()
                t0, t1 = max(t0, min(a, c)), min(t1, max(a, c))
            if t0 <= t1:
                best = min(best, t0)
        axial = best * site[i, 0].item()
        ranges[i] = axial if MIN_RANGE <= axial <= MAX_RANGE else 0.0
    return {"ranges": ranges.float(), "rays": world.float(), "dir_x": site[:, 0].float(), "origin": o.float()}


class Sensor:
    """What GuardLayer reads of the ToF sensor, for one environment, with a frame clock the test advances."""

    def __init__(self, frame: dict, time: float = 1.0):
        self.set(frame, time)

    def set(self, frame: dict, time: float | None = None) -> None:
        self.frame = frame
        self.delivered_frame = frame["ranges"][None]
        self.available = torch.tensor([True])
        if time is not None:
            self.channel = type("Channel", (), {"delivered_time": torch.tensor([time])})()

    def world_geometry(self):
        return self.frame["origin"][None], self.frame["rays"][None], self.frame["dir_x"][None]
