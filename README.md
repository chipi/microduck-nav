# microduck-nav

NVIDIA's Microduck obstacle-avoidance task (Isaac Lab, branch `exp/microduck`) ported to mjlab, so that
the navigation controller can be trained against Pollen's walking policy on one physics. This is
research R6 of the robotics-lab programme. The state today: the CPU checks C1 to C5 are written
and pass; the task itself (scene, observations, rewards, training) is not written.

## Where it comes from

| Source | Commit | Licence | Used as |
| --- | --- | --- | --- |
| NVIDIA Isaac Lab `exp/microduck` | `802b8d5` | BSD-3-Clause | adapted: ToF impairments, residual command, guarded Beta mean |
| Pollen `microduck_rl` (`mjlab_microduck`) | `0daab160` | Apache-2.0 (meshes CC-BY-NC-SA-4.0) | git dependency pinned to the commit, read only |

This repository is Apache-2.0 (`LICENSE`); `NOTICE` and `licenses/` carry the notices. Part of [robotics-lab](https://github.com/chipi/robotics-lab)
(research plan: `docs/wip/RESEARCH-LEARNED-AVOIDANCE.md` there); its evaluations are kept with eval-harness.

## Layout

| Module | Check | What |
| --- | --- | --- |
| `src/microduck_nav/tof_geometry.py` | C1, C2 | zone directions (Pollen's equal-angle and the pinhole pixel-centre grid), radial/axial depth, an mjlab ray pattern |
| `src/microduck_nav/ray_scene.py` | C1 | 4-world mjlab scene (floor, mocap box, mocap head) and the CPU `mj_ray` reference |
| `src/microduck_nav/impairments.py` | C2 | NVIDIA's ToF corruption, episode calibration, packet channel (latency, loss, age), 15 Hz clock |
| `src/microduck_nav/walker.py`, `walker_speed.py` | C3 | the walker as a batched torch module; the mjlab CPU speed run |
| `src/microduck_nav/beta_head.py`, `nav_export.py` | C4 | Beta head for rsl_rl 5.0.1 with the brake guard; ONNX export |
| `src/microduck_nav/residual_action.py` | C5 | residual velocity command: limits, speed cap, slew, stale stop |

## Run

```bash
uv sync                                  # 2.1 GB .venv, CPU torch
uv run pytest -q                         # 72 tests, about 10 s
uv run python scripts/report_numbers.py  # the numbers quoted in docs/CHECKS.md
uv run python scripts/walker_speed_table.py --speeds 0.1 0.15 0.2 0.25 0.3 0.4 --envs 8 --noise both
```

The walker tests need `~/.cache/duck-sim/policies/current/velstand.onnx` (sha256 `ce894d30...`) or
`MICRODUCK_WALKER_ONNX`; without it they skip. The file is not committed.

## Results

`docs/CHECKS.md` holds each check with its numbers and its PASS, FAIL or BLOCKED.
