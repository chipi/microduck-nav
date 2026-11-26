# microduck-nav

NVIDIA's Microduck obstacle-avoidance task (Isaac Lab, branch `exp/microduck`) ported to mjlab, so that
the navigation controller can be trained against Pollen's walking policy on one physics. This is
research R6 of the robotics-lab programme. The state today: the CPU checks C1 to C7 pass and the task
is written and registered with mjlab (`Mjlab-Nav-MicroDuck`, and `Mjlab-Nav-OneWall-MicroDuck` for milestone M1);
nothing has run on a GPU yet. The ToF zone grid is a config option, `grid`: `mixed` (the default; each environment
draws Pollen's equal-angle or NVIDIA's pinhole grid at reset), `equal_angle` or `pinhole`.
The DGX command for M1 is in `docs/M1.md`. M2 (`docs/M2.md`) has two: `Mjlab-Nav-MicroDuck` (50 Hz, NVIDIA-faithful) and
`Mjlab-Nav-Deploy-MicroDuck` (a 10 Hz policy and no lateral residual, matched to robotics-lab's navigator).
R7 (`docs/R7.md`) is `Mjlab-Nav-Ours-MicroDuck`: the deploy task over a pool of robotics-lab's own challenge scenes (one family held
out) with the navigator's reflex and cliff stop in the loop, each intervention costing reward.

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
| `src/microduck_nav/obstacles.py`, `layouts.py`, `field.py` | C6 | eight mocap slots (MJCF primitives), NVIDIA's layout families with seeded sampling, the per-env placement at reset |
| `src/microduck_nav/tof_sensor.py` | C6 | ToF ray sensor with a per-env zone grid (`mixed`, `equal_angle`, `pinhole`), impairments, ages |
| `src/microduck_nav/action.py`, `commands.py` | C6 | residual command, then the walker, then joint targets; the destination request |
| `src/microduck_nav/nav_math.py`, `rewards.py`, `terminations.py`, `observations.py` | C6 | NVIDIA's reward maths ported, contact termination, the 1343-wide actor observation |
| `src/microduck_nav/env_cfg.py`, `rl_cfg.py`, `tasks.py` | C6 | the task, the PPO config (Beta head, gamma 0.998), the registration |
| `src/microduck_nav/train_check.py` | C7 | runs the real `train` command on CPU and checks its outputs |
| `src/microduck_nav/replay.py` | M1 | replays an exported policy on the one-wall task and reports whether it stops |

## Run

```bash
uv sync                                  # 2.1 GB .venv, CPU torch
uv run pytest -q                         # 159 tests, about 5 min (149 in 50 s with -m "not slow")
uv run python scripts/report_numbers.py  # the numbers quoted in docs/CHECKS.md
uv run python scripts/walker_speed_table.py --speeds 0.1 0.15 0.2 0.25 0.3 0.4 --envs 8 --noise both
```

The walker tests need `~/.cache/duck-sim/policies/current/velstand.onnx` (sha256 `ce894d30...`) or
`MICRODUCK_WALKER_ONNX`; without it they skip. The file is not committed.

## Results

`docs/CHECKS.md` holds each check with its numbers and its PASS, FAIL or BLOCKED. `docs/M1.md` is the GPU run: the exact commands,
what done means, how to replay.
