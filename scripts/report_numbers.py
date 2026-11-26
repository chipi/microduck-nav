"""The numbers quoted in docs/CHECKS.md, recomputed."""

import math
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from microduck_nav import tof_geometry as geo
from microduck_nav.nav_export import build_actor, export_actor
from microduck_nav.ray_scene import build_ray_scene, cpu_reference_distances
from microduck_nav.walker import Walker
from mjlab_microduck.sim.tof import Tof

print("== C1")
for mode in ("pinhole", "equal_angle"):
    rs = build_ray_scene(4, box_half=(0.05, 0.3, 0.3), pattern_mode=mode)
    box = torch.zeros(4, 3)
    box[:, 0] = torch.tensor([1.0, 1.5, 2.0, 9.0])
    box[:, 2] = 0.3
    rs.place(box, torch.tensor([0.0, 0.0, 0.3491, 0.3491]))
    t = time.time()
    warp = rs.sense().numpy()
    dt = time.time() - t
    ref = cpu_reference_distances(rs, 4.0)
    hit = ref >= 0
    print(mode, "hits", int(hit.sum()), "of", ref.size, "per world", hit.sum(axis=1).tolist(), "miss-mask equal", bool(((warp < 0) == (ref < 0)).all()),
          "max abs diff", float(np.abs(warp[hit] - ref[hit]).max()), "sense s", round(dt, 3), "device", rs.sim.device)

print("== C2 geometry")
ours = geo.zone_directions("equal_angle").numpy()
pol = Tof(None, 0).directions
print("equal_angle vs tof.py max abs", float(np.abs(ours - pol).max()))
for mode in ("equal_angle", "pinhole"):
    d = geo.zone_directions(mode)
    print(mode, "edge azimuth deg", math.degrees(math.atan2(d[0, 1].item(), d[0, 0].item())), "axial factor min", float(d[:, 0].min()),
          "max angle between modes deg", float(torch.rad2deg(torch.acos((geo.zone_directions('equal_angle') * geo.zone_directions('pinhole')).sum(-1).clamp(-1, 1))).max()))

print("== C3 parity")
path = os.path.expanduser("~/.cache/duck-sim/policies/current/velstand.onnx")
w = Walker.from_onnx(path)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
mean, std = w.mean.numpy(), w.denominator.numpy()
for label, make in (("raw N(0,s)", lambda s: rng.normal(0, s, (1, 61))), ("mean+s*std", lambda s: mean + rng.normal(0, s, (1, 61)) * std)):
    for s in (0.5, 1.0, 3.0):
        worst = 0.0
        for _ in range(500):
            x = make(s).astype(np.float32)
            worst = max(worst, float(np.abs(sess.run(None, {"obs": x})[0] - w(torch.from_numpy(x)).numpy()).max()))
        print(label, "scale", s, "max abs diff over 500", f"{worst:.2e}")

print("== C4")
torch.manual_seed(0)
a = build_actor()
a.obs_normalizer._mean.copy_(torch.randn(1, 1343) * 0.3)
a.obs_normalizer._std.copy_(torch.rand(1, 1343) * 2 + 0.5)
export_actor(a, "/tmp/claude-1000/-work-robotics-lab-main/4fb24ee1-5b25-44ab-9cf9-02449c10e291/scratchpad/nav.onnx")
s = ort.InferenceSession("/tmp/claude-1000/-work-robotics-lab-main/4fb24ee1-5b25-44ab-9cf9-02449c10e291/scratchpad/nav.onnx", providers=["CPUExecutionProvider"])
print("input", s.get_inputs()[0].name, s.get_inputs()[0].shape, "output", s.get_outputs()[0].name, s.get_outputs()[0].shape)
worst = 0.0
for _ in range(200):
    x = rng.normal(0, 2, (1, 1343)).astype(np.float32)
    lg = a.mlp(a.obs_normalizer(torch.from_numpy(x)))
    al, be = torch.nn.functional.softplus(lg.unbind(-2)[0]) + 1, torch.nn.functional.softplus(lg.unbind(-2)[1]) + 1
    ref = (2 * torch.distributions.Beta(al.double(), be.double()).mean - 1).float().detach().numpy()
    worst = max(worst, float(np.abs(s.run(None, {"obs": x})[0] - ref).max()))
print("max abs diff onnx vs torch Beta mean over 200", f"{worst:.2e}")
