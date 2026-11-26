"""Runs the task's real `train` command on CPU and checks what comes out: finite scalars, checkpoints, the ONNX.

This is the C7 check and the M1 dry run: the command line is the one the DGX gets, with smaller numbers.
"""

from __future__ import annotations

import glob
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ITERATION_TIME = re.compile(r"Iteration time:\s*([0-9.]+)s")
COLLECTION_TIME = re.compile(r"Collection time:\s*([0-9.]+)\s*s")
LEARNING_TIME = re.compile(r"Learning time:\s*([0-9.]+)\s*s")
FPS = re.compile(r"Steps per second:\s*([0-9.]+)")


def train_command(task: str, envs: int, iterations: int, save_interval: int, cpu: bool = True) -> list[str]:
    cmd = [str(Path(sys.executable).parent / "train"), task]
    if cpu:
        cmd += ["--gpu-ids", "None"]
    return cmd + [
        "--env.scene.num-envs", str(envs),
        "--agent.max-iterations", str(iterations),
        "--agent.save-interval", str(save_interval),
        "--agent.logger", "tensorboard",
    ]


def scalars(log_dir: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(glob.glob(str(log_dir / "events.out.tfevents.*")))
    acc = EventAccumulator(events[-1])
    acc.Reload()
    return {tag: [s.value for s in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def run(task: str, envs: int, iterations: int, save_interval: int, workdir: Path, timeout: int = 3600) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MJLAB_WARP_QUIET": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = train_command(task, envs, iterations, save_interval)
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    runs = sorted(glob.glob(str(workdir / "logs" / "rsl_rl" / "*" / "*")))
    log_dir = Path(runs[-1]) if runs else None
    result = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "log_dir": log_dir,
        "output": out,
        "iteration_s": [float(x) for x in ITERATION_TIME.findall(out)],
        "collection_s": [float(x) for x in COLLECTION_TIME.findall(out)],
        "learning_s": [float(x) for x in LEARNING_TIME.findall(out)],
        "steps_per_second": [float(x) for x in FPS.findall(out)],
    }
    if log_dir is not None:
        result["checkpoints"] = sorted(p.name for p in log_dir.glob("model_*.pt"))
        result["onnx"] = sorted(p.name for p in log_dir.glob("model_*.onnx"))
        result["scalars"] = scalars(log_dir)
    return result


def all_finite(values: dict[str, list[float]]) -> bool:
    return all(math.isfinite(v) for series in values.values() for v in series)


def checkpoint_finite(path: Path) -> bool:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    tensors = [t for k in ("actor_state_dict", "critic_state_dict") for t in sd[k].values() if torch.is_tensor(t) and t.is_floating_point()]
    return all(torch.isfinite(t).all().item() for t in tensors)


def check_onnx(path: Path) -> dict:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (inp,), (out,) = sess.get_inputs(), sess.get_outputs()
    x = np.random.default_rng(0).normal(size=(1, 1343)).astype(np.float32)
    y = sess.run(None, {inp.name: x})[0]
    return {"input": tuple(inp.shape), "output": tuple(out.shape), "finite": bool(np.isfinite(y).all()), "range": (float(y.min()), float(y.max()))}
