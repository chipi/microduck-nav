"""The training pool of R7a: robotics-lab's `lab-scenarios pool` file, read into padded tensors in the robot's reset frame.

A line is one item of robotics-lab's scenario format (`robotics-scenario/1`, robotics-evals docs/DATASET-SPEC.md): the start
(x, y, yaw) of the duck's frame, the goal, and the obstacles as boxes (centre, yaw, half sides, the heights z0 and z1) in the seed's
world. robotics-lab's simulator builds every obstacle as a box geom (labrunner.scene.Box), so a box is the shape here too; `kind` is
kept only to count what was read.

Everything is moved once, at load, into the trunk's frame at the start (the duck's frame moved 0.05 m ahead: labrunner.scene.
TRUNK_START, the trunk of a settled duck), the frame the env's robot is reset in. At reset the env places the item by the robot's
own pose, so the item's start, goal and obstacles come out where the item put them relative to the robot, whatever the robot's
random heading.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT = "robotics-scenario/1"
TRUNK_START = (0.05, 0.0)  # labrunner.scene.TRUNK_START: the trunk stands this far ahead of the duck's start frame
OVERHEAD_Z = 0.30  # robots/duck/profile.json footprint.overhead_clear_m: a box whose underside is above this is over the duck
POOL_ENV = "MICRODUCK_POOL"
DEFAULT_POOL = "data/pools/ours-v1.jsonl"
FIELDS = ("x", "y", "yaw", "hx", "hy", "z0", "z1")
ITEM_KEYS = ("format", "layout", "seed", "start", "goal", "obstacles")


def pool_path(path: str | None = None) -> Path:
    """The pool file: the argument, else $MICRODUCK_POOL, else data/pools/ours-v1.jsonl in this checkout."""
    if path:
        return Path(path).expanduser()
    if os.environ.get(POOL_ENV):
        return Path(os.environ[POOL_ENV]).expanduser()
    return Path(__file__).resolve().parents[2] / DEFAULT_POOL


@dataclass
class PoolStats:
    """What the loader did with the file, so nothing is dropped silently."""

    read: int = 0
    kept: int = 0
    held_out: dict[str, int] = field(default_factory=dict)  # by layout name
    skipped_too_many: int = 0  # items with more obstacles than there are slots
    most_obstacles: int = 0  # of the items read
    layouts: dict[str, int] = field(default_factory=dict)  # of the items kept
    kinds: dict[str, int] = field(default_factory=dict)  # of the obstacles read

    def line(self) -> str:
        held = ", ".join(f"{n} {k}" for k, n in sorted(self.held_out.items())) or "none"
        return (f"pool: {self.read} items read, {self.kept} kept; held out: {held}; skipped for more obstacles than slots: "
                f"{self.skipped_too_many}; most obstacles in an item: {self.most_obstacles}")


@dataclass
class Pool:
    obstacles: torch.Tensor  # (N, K, 7): x, y, yaw, hx, hy, z0, z1, in the trunk's reset frame; rows past `count` are zero
    count: torch.Tensor  # (N,) obstacles in each item
    goal: torch.Tensor  # (N, 2) in the trunk's reset frame
    layout: torch.Tensor  # (N,) index into `layouts`
    layouts: tuple[str, ...]
    stats: PoolStats

    def __len__(self) -> int:
        return len(self.count)

    def to(self, device) -> "Pool":
        return Pool(self.obstacles.to(device), self.count.to(device), self.goal.to(device), self.layout.to(device), self.layouts, self.stats)


def parse_holdout(text: str | None) -> tuple[str, ...]:
    """`doorway`, `doorway,corner_in_doorway`, or `none`/empty for nothing held out."""
    names = tuple(n.strip() for n in (text or "").split(",") if n.strip())
    return () if names == ("none",) else names


def to_trunk_frame(start: dict, x: float, y: float, yaw: float = 0.0) -> tuple[float, float, float]:
    """A pose of the item's world in the frame of the trunk at the start (x ahead, y left, yaw)."""
    c, s = math.cos(start["yaw"]), math.sin(start["yaw"])
    tx, ty = start["x"] + c * TRUNK_START[0] - s * TRUNK_START[1], start["y"] + s * TRUNK_START[0] + c * TRUNK_START[1]
    dx, dy = x - tx, y - ty
    return c * dx + s * dy, -s * dx + c * dy, math.remainder(yaw - start["yaw"], math.tau)


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: {value!r} is not a finite number")
    return float(value)


def _read_item(line: str, where: str) -> dict:
    try:
        item = json.loads(line)
    except ValueError as e:
        raise ValueError(f"{where}: not JSON ({e})") from None
    if not isinstance(item, dict) or item.get("format") != FORMAT:
        raise ValueError(f"{where}: not a {FORMAT} item")
    missing = [k for k in ITEM_KEYS if k not in item]
    if missing or not isinstance(item["obstacles"], list):
        raise ValueError(f"{where}: missing {', '.join(missing) or 'obstacles list'}")
    return item


def load_pool(path, slots: int, holdout: tuple[str, ...] = ()) -> Pool:
    """Read the pool. An item of a held-out layout is not kept (and is counted); an item with more than `slots` obstacles is not kept
    either, and is counted in `stats.skipped_too_many`: a pool whose items do not fit is reported, never cut. A line that is not an
    item, or a number that is not finite, is an error with its line number."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pool not found at {path}; set {POOL_ENV} or make it with lab-scenarios pool (docs/R7.md)")
    opener = gzip.open if path.suffix == ".gz" else open
    stats, rows, seen = PoolStats(), [], set()
    with opener(path, "rt", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            item = _read_item(line, where)
            layout = str(item["layout"])
            seen.add(layout)
            stats.read += 1
            stats.most_obstacles = max(stats.most_obstacles, len(item["obstacles"]))
            for o in item["obstacles"]:
                stats.kinds[str(o.get("kind"))] = stats.kinds.get(str(o.get("kind")), 0) + 1
            if layout in holdout:
                stats.held_out[layout] = stats.held_out.get(layout, 0) + 1
                continue
            if len(item["obstacles"]) > slots:
                stats.skipped_too_many += 1
                continue
            start = {k: _number(item["start"][k], f"{where} start.{k}") for k in ("x", "y", "yaw")}
            gx, gy, _ = to_trunk_frame(start, _number(item["goal"]["x"], f"{where} goal.x"), _number(item["goal"]["y"], f"{where} goal.y"))
            boxes = []
            for k, o in enumerate(item["obstacles"]):
                v = {name: _number(o[name], f"{where} obstacles[{k}].{name}") for name in FIELDS}
                if v["hx"] <= 0 or v["hy"] <= 0 or v["z1"] <= v["z0"]:
                    raise ValueError(f"{where} obstacles[{k}]: a box needs positive half sides and z1 > z0")
                x, y, yaw = to_trunk_frame(start, v["x"], v["y"], v["yaw"])
                boxes.append((x, y, yaw, v["hx"], v["hy"], v["z0"], v["z1"]))
            rows.append((layout, (gx, gy), boxes))
    unknown = [h for h in holdout if h not in seen]
    if unknown:
        raise ValueError(f"--holdout-family {', '.join(unknown)}: the pool's layouts are {', '.join(sorted(seen))}")
    if not rows:
        raise ValueError(f"{path}: no item is left to train on ({stats.line()})")
    layouts = tuple(sorted({r[0] for r in rows}))
    obstacles = torch.zeros(len(rows), slots, len(FIELDS))
    for i, (_, _, boxes) in enumerate(rows):
        if boxes:
            obstacles[i, : len(boxes)] = torch.tensor(boxes, dtype=torch.float64).float()
    stats.kept = len(rows)
    for r in rows:
        stats.layouts[r[0]] = stats.layouts.get(r[0], 0) + 1
    return Pool(
        obstacles=obstacles,
        count=torch.tensor([len(r[2]) for r in rows], dtype=torch.long),
        goal=torch.tensor([r[1] for r in rows], dtype=torch.float64).float(),
        layout=torch.tensor([layouts.index(r[0]) for r in rows], dtype=torch.long),
        layouts=layouts,
        stats=stats,
    )


_CACHE: dict = {}


def get_pool(path, slots: int, holdout: str | None, device) -> Pool:
    """load_pool once per (file, slots, holdout), moved to the device; the file's size and mtime are in the key."""
    p = pool_path(path)
    stat = p.stat() if p.exists() else None
    key = (str(p), stat and (stat.st_size, stat.st_mtime_ns), slots, parse_holdout(holdout), str(device))
    if key not in _CACHE:
        pool = load_pool(p, slots, parse_holdout(holdout))
        print(pool.stats.line(), flush=True)
        _CACHE[key] = pool.to(device)
    return _CACHE[key]
