"""C3 evidence: speed got against speed asked, Pollen's walker in mjlab on CPU. Prints a table and writes JSON."""

import argparse
import json
import os

from microduck_nav.walker_speed import measure

parser = argparse.ArgumentParser()
parser.add_argument("--speeds", type=float, nargs="+", default=[0.15, 0.2, 0.3])
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--settle", type=int, default=250)
parser.add_argument("--noise", choices=["on", "off", "both"], default="on")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="docs/walker_speed.json")
args = parser.parse_args()

walker = os.path.expanduser("~/.cache/duck-sim/policies/current/velstand.onnx")
rows = []
print("asked  noise  envs steps  vx_settled(mean)  got/asked  world_speed  per-env vx            resets  secs", flush=True)
for noise in ([True, False] if args.noise == "both" else [args.noise == "on"]):
    for v in args.speeds:
        r = measure(v, args.envs, args.steps, args.settle, noise, args.seed, "cpu", walker)
        rows.append(r.__dict__)
        per = " ".join(f"{x:+.3f}" for x in r.per_env_settled)
        print(f"{v:5.2f}  {'on ' if noise else 'off'}   {r.num_envs:4d} {r.steps:5d}  {r.mean_vx_settled:+.4f}          {r.mean_vx_settled / v:6.1%}    "
              f"{r.mean_world_speed_settled:.4f}      {per}   {r.resets:4d}   {r.seconds:6.1f}", flush=True)
with open(args.out, "w") as f:
    json.dump(rows, f, indent=1)
