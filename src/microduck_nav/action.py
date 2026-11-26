"""The navigation action term: three residuals -> command -> Pollen's walker -> joint targets.

Mirrors NVIDIA's MicroDuckWalkingAction (mdp/actions.py, Isaac Lab exp/microduck 802b8d5, BSD-3-Clause):
the residual maths is `ResidualCommand` (C5), the frozen walker is `Walker` (C3, batched torch rebuilt
from Pollen's ONNX), and the 14 outputs go through mjlab's own joint position term, so the targets are
default pose + output, as in Pollen's task. The walker sees its proprioception group, its own last output,
the command after the residual, and zero head and body pose commands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from microduck_nav.residual_action import ResidualCommand
from microduck_nav.walker import Walker

WALKER_ENV = "MICRODUCK_WALKER_ONNX"
DEFAULT_WALKER = "~/.cache/duck-sim/policies/current/velstand.onnx"
WALKER_PROPRIO_DIM = 34
WALKER_OBS_DIM = 61
WALKER_GROUP = "walker_proprio"
SENSOR = "head_tof"
COMMAND = "twist"


def walker_path() -> Path:
    return Path(os.environ.get(WALKER_ENV, DEFAULT_WALKER)).expanduser()


@dataclass(kw_only=True)
class NavigationActionCfg(ActionTermCfg):
    walker_onnx: str | None = None
    joint_action: JointPositionActionCfg = field(
        default_factory=lambda: JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=True
        )
    )
    limits: tuple[float, float, float] = (0.35, 0.12, 0.75)
    slew: tuple[float, float, float] = (1.0, 1.0, 3.0)
    max_age: float = 0.25
    max_lateral_ratio: float = 0.35

    def build(self, env) -> "NavigationAction":
        return NavigationAction(self, env)


class NavigationAction(ActionTerm):
    cfg: NavigationActionCfg

    def __init__(self, cfg: NavigationActionCfg, env):
        super().__init__(cfg, env)
        path = Path(cfg.walker_onnx).expanduser() if cfg.walker_onnx else walker_path()
        if not path.exists():
            raise FileNotFoundError(f"walker ONNX not found at {path}; set {WALKER_ENV}")
        self.walker = Walker.from_onnx(path).to(self.device).eval()
        if self.walker.obs_dim != WALKER_OBS_DIM:
            raise ValueError(f"the walker takes {self.walker.obs_dim} inputs, expected {WALKER_OBS_DIM}")
        self._joints = cfg.joint_action.build(env)
        self.residual = ResidualCommand(
            self.num_envs, env.step_dt, cfg.limits, cfg.slew, cfg.max_age, cfg.max_lateral_ratio, self.device
        )
        self._limits = torch.tensor(cfg.limits, device=self.device)
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_walker = torch.zeros(self.num_envs, self._joints.action_dim, device=self.device)
        self._zeros10 = torch.zeros(self.num_envs, 10, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def velocity_commands(self) -> torch.Tensor:
        """The command actually delivered to the walker [m/s, m/s, rad/s]."""
        return self.residual.previous

    @property
    def velocity_limits(self) -> torch.Tensor:
        return self._limits

    def walker_observation(self, proprio: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return torch.cat((proprio, self._last_walker, command, self._zeros10), dim=-1)

    @torch.no_grad()
    def process_actions(self, actions: torch.Tensor) -> None:
        env = self._env
        self._raw[:] = actions
        self.previous_velocity_commands.copy_(self.residual.previous)
        requested = env.command_manager.get_command(COMMAND)
        sensor = env.scene[SENSOR]
        command = self.residual.process(requested, actions, sensor.available, sensor.sample_age)
        proprio = env.obs_buf[WALKER_GROUP]
        walker_out = self.walker(self.walker_observation(proprio.float(), command))
        self._last_walker.copy_(walker_out)
        self._joints.process_actions(walker_out)

    @torch.no_grad()
    def apply_actions(self) -> None:
        self._joints.apply_actions()

    def reset(self, env_ids=None) -> dict:
        rows = slice(None) if env_ids is None else env_ids
        self.residual.reset(rows)
        self._raw[rows] = 0.0
        self.previous_velocity_commands[rows] = 0.0
        self._last_walker[rows] = 0.0
        self._joints.reset(env_ids)
        return {}
