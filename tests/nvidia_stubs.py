"""Loads NVIDIA's rewards, geometry and commands modules (Isaac Lab exp/microduck 802b8d5) without Isaac Lab.

Only the names those files import from isaaclab are stubbed: the manager base classes, `configclass`, and
`isaaclab.utils.math` with xyzw quaternions, written here from the rotation matrix so that it shares nothing
with mjlab's wxyz helpers. The NVIDIA files themselves are imported unmodified from the clone.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

NV_ROOT = Path(
    os.environ.get(
        "ISAACLAB_MICRODUCK",
        "~/.cache/robotics-lab-tests/research/isaaclab",
    )
).expanduser()
NV_MDP = NV_ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_obstacle/mdp"
NV_SHA_PREFIX = "802b8d5"


class Wp:
    """A warp array as the task code reads it: `.torch`."""

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


def _matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_apply(q, v):
    return torch.einsum("...ij,...j->...i", _matrix(q), v)


def quat_apply_inverse(q, v):
    return torch.einsum("...ji,...j->...i", _matrix(q), v)


def yaw_quat(q):
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1)


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


def stub_modules() -> dict[str, types.ModuleType]:
    math = _module(
        "isaaclab.utils.math",
        quat_apply=quat_apply,
        quat_apply_inverse=quat_apply_inverse,
        yaw_quat=yaw_quat,
        euler_xyz_from_quat=None,
        quat_from_euler_xyz=None,
    )
    utils = _module("isaaclab.utils", math=math, configclass=dataclasses.dataclass)
    managers = _module(
        "isaaclab.managers", ManagerTermBase=_ManagerTermBase, RewardTermCfg=object, CommandTerm=object
    )
    velocity = _module(
        "isaaclab_tasks.contrib.microduck.mdp.commands", MicroDuckVelocityCommand=object, MicroDuckVelocityCommandCfg=object
    )
    return {
        "isaaclab": _module("isaaclab", utils=utils, managers=managers),
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math,
        "isaaclab.managers": managers,
        "isaaclab_tasks": _module("isaaclab_tasks"),
        "isaaclab_tasks.contrib": _module("isaaclab_tasks.contrib"),
        "isaaclab_tasks.contrib.microduck": _module("isaaclab_tasks.contrib.microduck"),
        "isaaclab_tasks.contrib.microduck.mdp": _module("isaaclab_tasks.contrib.microduck.mdp"),
        "isaaclab_tasks.contrib.microduck.mdp.commands": velocity,
    }


def available() -> bool:
    return (NV_MDP / "rewards.py").exists()


def load(monkeypatch) -> types.SimpleNamespace:
    """The NVIDIA modules rewards, geometry, commands, events, as attributes of a namespace."""
    for name, mod in stub_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    package = types.ModuleType("nv_mdp")
    package.__path__ = [str(NV_MDP)]
    monkeypatch.setitem(sys.modules, "nv_mdp", package)
    loaded = {}
    for name in ("events", "geometry", "observations", "commands", "rewards", "terminations"):
        spec = importlib.util.spec_from_file_location(f"nv_mdp.{name}", NV_MDP / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"nv_mdp.{name}", mod)
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return types.SimpleNamespace(**loaded)
