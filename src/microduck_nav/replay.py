"""Replays an exported navigation policy in the task and reports whether it stops before the wall (milestone M1).

The policy is the ONNX the runner writes next to each checkpoint, the same file that deploys: (1, 1343) -> (1, 3),
run once per environment per step on CPU. A checkpoint without its ONNX is exported here from the .pt first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from microduck_nav.field import root_pose
from microduck_nav.nav_export import build_actor, export_actor
from microduck_nav import nav_math

STOP_SPEED = 0.03  # m/s, mean over the last STOP_WINDOW_S
STOP_WINDOW_S = 2.0
MIN_PROGRESS = 0.25  # m of the initial gap to the wall closed
SUCCESS_RATE = 0.8


def onnx_for(checkpoint: Path) -> Path:
    onnx = checkpoint.with_suffix(".onnx")
    if onnx.exists():
        return onnx
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor()
    actor.load_state_dict(sd["actor_state_dict"])
    return export_actor(actor, onnx)


@dataclass
class EnvResult:
    initial_clearance: float
    final_clearance: float
    min_clearance: float
    closed: float
    final_speed: float
    collided: bool
    fell: bool
    steps: int

    @property
    def stopped_before_wall(self) -> bool:
        return (
            not self.collided
            and not self.fell
            and self.min_clearance > 0.0
            and self.closed >= MIN_PROGRESS
            and self.final_speed < STOP_SPEED
        )


def replay(task: str, policy: Path, num_envs: int = 8, steps: int = 400, seed: int = 0, device: str = "cpu") -> list[EnvResult]:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg, device=device)
    sess = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    assert tuple(sess.get_inputs()[0].shape) == (1, 1343) and tuple(sess.get_outputs()[0].shape) == (1, 3)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    window = max(1, int(STOP_WINDOW_S / env.step_dt))
    geom = lambda: env.nav_field.geometry()
    start = nav_math.obstacle_clearances(root_pose(env)[0], geom()).amin(dim=-1).clone()
    clearance = [start]
    speed = []
    collided = torch.zeros(num_envs, dtype=torch.bool)
    fell = torch.zeros(num_envs, dtype=torch.bool)
    alive = torch.ones(num_envs, dtype=torch.bool)
    ended = torch.zeros(num_envs, dtype=torch.long)
    for _ in range(steps):
        actor = torch.cat((obs["policy"], obs["tof"]), dim=-1).float().numpy()
        action = np.concatenate([sess.run(None, {name: actor[i : i + 1]})[0] for i in range(num_envs)], axis=0)
        obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action))
        collided |= alive & env.termination_manager.get_term("obstacle_collision")
        fell |= alive & env.termination_manager.get_term("fell_over")
        pos = robot.data.root_link_pos_w
        c = nav_math.obstacle_clearances(pos, geom()).amin(dim=-1).clone()
        v = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1).clone()
        clearance.append(torch.where(alive, c, clearance[-1]))
        speed.append(torch.where(alive, v, speed[-1] if speed else v))
        ended += alive.long()
        alive &= ~(terminated | truncated)
        if not alive.any():
            break
    clr = torch.stack(clearance)
    spd = torch.stack(speed)
    out = []
    for i in range(num_envs):
        k = int(ended[i])
        out.append(
            EnvResult(
                initial_clearance=float(clr[0, i]),
                final_clearance=float(clr[k, i]),
                min_clearance=float(clr[: k + 1, i].min()),
                closed=float(clr[0, i] - clr[k, i]),
                final_speed=float(spd[max(0, k - window) : k, i].mean()) if k else 0.0,
                collided=bool(collided[i]),
                fell=bool(fell[i]),
                steps=k,
            )
        )
    env.close()
    return out


def verdict(results: list[EnvResult]) -> dict:
    ok = [r.stopped_before_wall for r in results]
    return {
        "envs": len(results),
        "stopped_before_wall": int(sum(ok)),
        "rate": sum(ok) / len(ok),
        "collisions": int(sum(r.collided for r in results)),
        "falls": int(sum(r.fell for r in results)),
        "mean_closed_m": float(np.mean([r.closed for r in results])),
        "mean_final_clearance_m": float(np.mean([r.final_clearance for r in results])),
        "mean_final_speed": float(np.mean([r.final_speed for r in results])),
        "done": bool(sum(ok) / len(ok) >= SUCCESS_RATE),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="model_N.pt (its model_N.onnx is used) or an .onnx")
    p.add_argument("--task", default="Mjlab-Nav-OneWall-MicroDuck")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=400, help="control steps at 50 Hz; an episode is 1000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    policy = args.checkpoint if args.checkpoint.suffix == ".onnx" else onnx_for(args.checkpoint)
    t = time.time()
    results = replay(args.task, policy, args.num_envs, args.steps, args.seed)
    summary = verdict(results)
    print(f"{'env':>3} {'start':>7} {'end':>7} {'min':>7} {'closed':>7} {'speed':>7} {'coll':>5} {'fell':>5} {'steps':>6}  stopped")
    for i, r in enumerate(results):
        print(
            f"{i:>3} {r.initial_clearance:7.3f} {r.final_clearance:7.3f} {r.min_clearance:7.3f} {r.closed:7.3f} "
            f"{r.final_speed:7.3f} {int(r.collided):>5} {int(r.fell):>5} {r.steps:>6}  {r.stopped_before_wall}"
        )
    print(json.dumps(summary))
    print(f"policy {policy}  replay {time.time() - t:.0f} s")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "envs": [asdict(r) for r in results]}, indent=1))
    return 0 if summary["done"] else 1


if __name__ == "__main__":
    sys.exit(main())
