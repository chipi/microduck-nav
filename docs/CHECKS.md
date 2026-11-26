# CPU checks C1 to C7 and the M1 dry run (plan `RESEARCH-R6-PLAN.md`, section 5)

Run on the workstation, no NVIDIA GPU (Warp reports `CUDA driver not available`, `devices ['cpu']`),
2026-11-26. Stack: mjlab 1.3.0, mujoco 3.10.0, mujoco-warp 3.8.1, warp-lang 1.12.0, rsl-rl-lib 5.0.1,
torch 2.9.1+cpu, onnx 1.23.2, onnxruntime 1.32.0, numpy 2.4.1. `uv run pytest -q`: 159 passed in 309 s (149 in 51 s with `-m "not slow"`).
Numbers below come from `scripts/report_numbers.py` and `scripts/walker_speed_table.py`.

| Check | Result |
| --- | --- |
| C1 rays | PASS (ray path runs on CPU) |
| C2 sensor maths | PASS |
| C3 walker, parity | PASS (max abs diff 5.3e-5, typical 1e-6 to 3e-5) |
| C3 walker, speed in mjlab on CPU | DONE, evidence below |
| C4 export | PASS |
| C5 residual action | PASS |
| C6 task: scene, sensor, observation, rewards, terminations | PASS |
| C7 8 envs x 50 PPO iterations on CPU | PASS (2.6 s per iteration) |
| C7 deploy task, 8 envs x 20 PPO iterations on CPU | PASS (10.9 s per iteration, five walker steps per policy step; `docs/M2.md`) |
| M1 dry run on CPU, one wall | PASS (command works end to end; the 5-iteration policy does not stop, as expected) |

## Answers to the plan's open questions

- mjlab and its BVH ray path run on CPU: yes. `Scene`, `Simulation`, `SensorContext`, `RayCastSensor`
  and `mujoco_warp.rays(..., rc=...)` ran on Warp's CPU device. No CUDA error occurred, so no torch
  reference of the ray geometry was needed.
- Per-world mocap moves are seen by the rays: yes (C1), for the box and for the head.
- `onnx` is in Pollen's lock (1.22.0, through an override) and is installed in this repo's venv.
- rsl_rl_lib 5.0.1 does not ship a `BetaDistribution` (`rsl_rl/modules/distribution.py` has
  `Distribution`, `GaussianDistribution`, `HeteroscedasticGaussianDistribution` only). NVIDIA's
  `MicroDuckBrakingBetaDistribution` subclasses one that is missing, so its Beta base is written
  here (`beta_head.py`). The `Distribution` base does accept a custom class by `class_name` (used
  by `MLPModel`, C4). Not run: the PPO loop with this head.

## C1 rays: PASS

Scene: floor plane, one box (0.05 x 0.3 x 0.3 half sizes, mocap), one head body (mocap) carrying a
`tof` site, one `RayCastSensor` with 64 rays, `max_distance` 4.0, 4 worlds on `cpu`.
Per-world poses: box x = 1.0, 1.5, 2.0, 9.0 m; head pitch 0, 0, 0.3491, 0.3491 rad.

- Distances against a CPU `mujoco.mj_ray` loop (same model, same poses, same body exclusion):
  hits 210 of 256 rays (54, 40, 60, 56 per world); the set of misses is identical; max abs
  difference 2.6e-07 m (pinhole pattern) and 4.4e-07 m (equal-angle pattern).
- Analytic: with the head level, the centre zones read `(box_x - 0.05) / dir_x` within 1e-4 m for
  worlds 0 to 2; world 3 (box at 9 m) does not see it.
- Moving the box in world 1 only (1.5 m to 0.8 m) changes world 1 only (others bit-identical) and
  shortens its centre zone by more than 0.6 m.
- Worlds with different head pitch (0 and 0.3491 rad) each agree with `mj_ray`; the bottom row hits the floor when pitched down.
- Floor hits are recoverable from `hit_pos_w` z (|z| < 1e-3) and are not flagged for the box or misses.
- First `sense()` call in a process: 25 ms for 4 x 64 rays (Warp kernels cached; a cold compile takes several seconds).

Findings:
- `Entity.write_mocap_pose_to_sim(pose)` without `env_ids` raises `RuntimeError: expand(...[4, 1, 3]...)`
  in mjlab 1.3.0 (`entity/data.py:213`); passing `env_ids=torch.arange(n)` works. The placement
  event must pass explicit ids.
- mjlab's `PinholeCameraPatternCfg` is not the ToF layout: it uses the MuJoCo camera convention
  (-z forward) and `linspace(-1, 1)` end points (edge ray at 22.5 degrees, `raycast_sensor.py:198-211`).
  `ToFPatternCfg` replaces it, in the site frame (+x forward, +y left, +z up).

Not covered: cost beyond 4 worlds (nothing measured at 4096), geoms other than box and plane,
a site offset or pitch randomised per world inside the head (only the head body's pose per world),
obstacle bodies that move during physics stepping (poses were written, then `forward()`).

## C2 sensor maths: PASS

- Zone directions: `zone_directions("equal_angle")` equals `Tof(...).directions` of Pollen's
  `tof.py` (max abs difference 2.6e-08). Row 0 is the top, column 0 the left, row-major.
- Pinhole pixel-centre grid (NVIDIA's convention): edge zone at 19.92 degrees, against 19.69 for
  equal angle; the largest angle between corresponding zones of the two layouts is 0.90 degrees.
  mjlab's own pinhole pattern would give 22.5. A decision for the operator: train with the pinhole
  grid (NVIDIA's recipe, the default here) or with `tof.py`'s grid (what the simulated duck emits).
- Radial to axial: `radial * dir_x` (smallest factor 0.887 equal-angle, 0.890 pinhole), misses stay
  negative, round trip exact, hand case 2 m at 30 degrees gives 1.732 m.
- Impairments (`impairments.py`), on (envs, 64) tensors with a `torch.Generator`:
  - hand cases: range gate before and after noise (0.10 m and 0.119 m invalid, 0.12 m valid, 4.0 m
    valid, 4.01 m invalid, 0, inf, NaN invalid); scale 1.02, bias 0.01, zone bias 0.005 gives 1.035;
    noise equals a seeded `randn * (noise_std + 0.008 * depth)`, and over 4000 x 64 draws its
    std is within 2 % of 0.026 at 2 m; invalid-zone rate 0.03 within 0.004; floor sigmoid at
    0.55, 0.556, 0.58, 0.65 m is 0, 0.5, 0.832, over 0.997; midpoint shift 0.13 moves it;
    edge column: dropout `1 - (1 - p) * survival` only for floor hits between 0.12 and 0.55 m in the
    impaired column;
  - against NVIDIA's formulas: `sensors.py:206-301` restated on their (n, 8, 8, 1) layout and run
    with the same seeded draws agrees with `corrupt_frame` to 1e-6 on 4 seeds x 64 envs; the
    episode draw order matches `_reset_randomization` exactly; ranges and the 0.5 edge-episode rate
    (within 0.03 over 4000) hold;
  - delivery: latency 0 and 1, packet loss keeps the previous frame with its true age, reset clears
    only the chosen rows; the 15 Hz clock fires every 3 or 4 steps (0.06 or 0.08 s), 149 or 150
    captures in 500 steps, first at step 4.
- Discrepancy with R1: the far-floor sigmoid at 0.58 m leaves 16.8 % of floor returns valid
  (`1 - sigmoid(1.6)`); R1 quotes 16 %.

Not covered: the mount randomisation (pitch +-10 degrees, height +-2 cm: configured, not
implemented), the 1343-wide observation assembly (history, flags, ages), the wiring from the ray
sensor through axial conversion to the impairments. Impairment values are NVIDIA's priors, as R1 says.

## C3 walker

Parity (PASS). `Walker.from_onnx` rebuilds `velstand.onnx` (sha256 `ce894d30...`, policies v10; 61 ->
512 -> 256 -> 128 -> 14, ELU, normaliser as Sub and Div) as a batched torch module. Against
onnxruntime on 500 random 61-vectors per case, max abs difference: 8.8e-07 / 3.1e-06 / 5.3e-06 for
inputs mean + s * std with s = 0.5 / 1 / 3, and 8.6e-06 / 2.7e-05 / 4.9e-05 for raw N(0, s)
vectors (huge after normalisation). The "around 1e-5" expectation holds for plausible inputs and
is exceeded by 5x on implausible ones. A batch of 4096 runs; row-by-row equals batched; on the
real actor observation of the mjlab task (2 envs) torch and onnxruntime agree within 1e-4.

Speed in mjlab on CPU. `Mjlab-VelStand-Flat-Backlash-MicroDuck` play config (the walker's task, flat
terrain, backlash, BAM actuators, domain randomisation of the reset kept), 8 envs, 500 steps at 50 Hz
(10 s), constant forward command, lateral and yaw zero, pushes, prone starts and the curriculum
removed, 0 falls or resets in every run. Speed = body-frame forward velocity averaged over steps
250 to 500. Seed 0, one run per row. Raw: `docs/walker_speed_table.txt`, `docs/walker_speed.json`.

| asked m/s | got, observation noise on | got / asked | got, noise off | got / asked |
| --- | --- | --- | --- | --- |
| 0.10 | 0.005 | 5 % | 0.004 | 4 % |
| 0.15 | 0.034 | 23 % | 0.007 | 5 % |
| 0.20 | 0.092 | 46 % | 0.092 | 46 % |
| 0.25 | 0.118 | 47 % | 0.116 | 46 % |
| 0.30 | 0.142 | 47 % | 0.140 | 47 % |
| 0.40 | 0.193 | 48 % | 0.193 | 48 % |

Per env at 0.15 with noise on: 0.067, 0.065, 0, 0, 0, 0.062, 0, 0.081 (4 of 8 walk); with noise off
7 of 8 stand still (one at 0.059). At 0.20 and above all 8 walk, spread 0.084 to 0.103 at 0.20.
Evidence for the question: in mjlab's training physics the walker tracks about 46 to 48 % of the
asked speed from 0.2 to 0.4 m/s (Pollen's duck-sim figure: about 44 %), and below 0.2 it mostly does
not walk (0.15 asked: 5 to 23 % got; Pollen's duck-sim: 0.001 got). The two physics agree. This is
a measurement, not a verdict on cause. The walker's task samples forward commands of at least 0.3
for the `rel_forward_envs` share of envs (0.2 in the play config; `velocity_command.py:91-99`), which
is code, not shown to be the cause.

Not covered: one seed, 10 s runs, flat terrain only (the published walker was trained on
`Rough-Backlash`), no lateral or yaw commands, no commands between 0.15 and 0.2 finer than 0.05,
no GPU run, no comparison with duck-sim in this repo (the duck-sim figures are quoted from the plan).

## C4 export: PASS

`build_actor` makes an rsl_rl `MLPModel`: 1343 -> 128 -> 128 -> (2 x 3 Beta logits), tanh, empirical
normaliser (random mean and std set for the test), `MicroDuckBrakingBeta` head.
`export_actor` uses the settings of mjlab's runner export (opset 18, legacy exporter, static batch).

- onnxruntime input `obs` shape `[1, 1343]`, output `actions` shape `[1, 3]`; one input, one output;
  the normaliser mean is an initialiser of the graph.
- Output against the torch Beta mean (`2 * Beta(alpha, beta).mean - 1`, float64 reference) on 200
  random inputs: max abs difference 1.75e-07 (limit 1e-5); equals the actor's own deterministic forward.
- Brake: forward concentrations forced to 1.0 (logits -60) give `[-1, 0, 0]` when the other heads are
  symmetric; a NaN in the observation or in a weight gives exactly `[-1, 0, 0]`; a symmetric healthy
  forward head gives 0, not a brake.
- Sampling, log probability (including the log 2 Jacobian), entropy and KL against `torch.distributions.Beta`
  agree within 1e-4; `forward_braking_only` keeps forward in [-1, 0].

Findings:
- A collapsed forward head alone gives `[-1, lateral_mean, yaw_mean]`, not `[-1, 0, 0]`: the guard
  replaces only the forward value. Only a non-finite logit gives the full brake. This is NVIDIA's
  behaviour (`navigation_distribution.py:21-30`).
- An infinite observation does not brake: tanh saturates and the logits stay finite (tested for
  +-inf). Only NaN propagates to the brake. A deployment wrapper that must brake on out-of-range
  inputs needs its own check (`nav_pad.py` raises on non-finite state before the network runs).

Not covered: the PPO update with this head (log probability and KL are checked against torch, the
loop is not run), the ONNX of an actor trained on real observations, a dynamic batch (static batch of 1, as the contract says).

## C5 residual action term: PASS

`ResidualCommand` (`residual_action.py`), batched torch, state = previous command per env. 17 tests:
- limits (0.35, 0.12, 0.75) clamp each axis; no reverse (forward -0.25 clamps to 0); the action is
  clipped to [-1, 1];
- never faster than requested: the translation norm is scaled to the request, hand case request 0.3
  with a (0.35, 0.12) target gives norm 0.300; holds over a grid of residuals at request 0.15;
- slew (1, 1, 3) per second at 50 Hz: first step 0.02 forward, 0.06 yaw; 10 steps reach a 0.2 request;
  braking from 0.3 gives 0.28;
- lateral stays inside 0.35 of forward;
- zero request, no frame, or age over 0.25 s zeroes the command at once and the next command restarts
  from zero; age exactly 0.25 is fresh;
- agrees with a restatement of the deployment `command_from_action` (`nav_pad.py:104-116`) to 1e-6
  on 300 random cases;
- rows are independent; `reset(ids)` clears only those rows.

Finding: the speed cap acts on the target before the slew, so after a request drop the slewed command
can exceed the new request for a few ticks (request 0.1 after 0.3: 0.28 on the first tick). Same
order in NVIDIA's code (`actions.py:60-67`) and in `nav_pad.py`.

Not covered: the mjlab `ActionTerm` wrapper (this is the command maths only), the walker inside
the term, the joint-target path, per-env sensor age from the real channel.

## C6 the navigation task: PASS

Registered as `Mjlab-Nav-MicroDuck` (all layouts, impairments) and `Mjlab-Nav-OneWall-MicroDuck`
(layout `one_wall`, no impairments, the M1 task) through the `mjlab.tasks` entry point; both are built on Pollen's
`Mjlab-VelStand-Flat-Backlash-MicroDuck` config (its robot, DR, pushes and walker observation noise).
Tests: `test_c6_layouts.py` (25), `test_c6_scene.py` (8), `test_c6_terminations.py` (4),
`test_c6_rewards.py` (35), `test_c6_wiring.py` (5).

- **Scene.** Eight mocap entities (`wall_left`, `wall_right`, ... in NVIDIA's slot order), one MJCF per slot with
  one primitive (cylinder or box), compiled with the walker and the floor. Slot-vs-floor and slot-vs-slot contacts
  are excluded in the MJCF (36 pairs); the robot-vs-slot contact stays. `reset_obstacle_field` runs after the
  robot reset and writes each slot's mocap pose with explicit `env_ids` (the mjlab call refuses without them).
  Parked slots sit 5 m below the floor. The ten layout families (ring, open, barrier, slalom, passage, thin_low,
  blocked, corner, scattered, one_wall) are NVIDIA's sampling (`events.py`) with an explicit `torch.Generator`: the
  same seed gives the same layouts, different seeds differ, the shuffle never moves the walls, and over 3000 draws per
  layout no obstacle spawns closer than the spawn clearance.
- **ToF.** `ToFSensor` subclasses mjlab's `RayCastSensor` and overrides `prepare_rays`, so each world has its own
  ray directions and origin. 64 zones, radial to axial, floor mask, NVIDIA's corruption, dropout, a 15 Hz clock and
  a packet channel with latency. Rays match a `mj_ray` reference that uses each env's own grid and mount.
- **Zone grid, `mixed` is the default.** Options: `equal_angle` (Pollen's), `pinhole` (NVIDIA's), `mixed`
  (each env draws one of the two at every reset and keeps it for the episode). In the test `mixed` gives both
  grids in about equal numbers over many resets; a given env and seed always gets the same grid; each env's
  rays follow its own grid. **Per-env grid works with the batched ray sensor** (the sensor takes a per-world
  direction array), so the fallback of two sensors was not needed.
- **Observation.** Actor group = `policy` (53) + `tof` (1290: 10 ticks of 64 ranges and 64 validity flags, newest last, then
  10 ages) = 1343, in NVIDIA's term order; `critic` adds the true obstacle clearances; the walker has its own 34-wide
  `walker_proprio` group with Pollen's noise.
- **Action.** `NavigationAction`: `ResidualCommand` (C5), then the walker (C3) fed 61 inputs (proprio 34, its last output
  14, command 3, zeros 10, so head and body inputs are zero), then the joint-position term. Checked against an
  independent assembly. The requested speed is drawn from `speed_range = (0.2, 0.35)`, a config field.
- **Rewards (weights).** progress +5, goal_arrival +30, barrier_bypass +5, yaw_tracking +4, command_tracking +1,
  approach -4, command_rate -5, failure -30. **Each term is tested against NVIDIA's own functions**: their
  `geometry.py`, `commands.py`, `rewards.py` and `terminations.py` are imported unmodified (isaaclab stubbed, with its own
  xyzw quaternion maths in `tests/nvidia_stubs.py`) and run on the same random scenes and the same constructed
  trajectories as ours (3 seeds x 256 envs: clearance, requested-path clearance, progress, gated progress, velocity
  and yaw tracking, approach risk with three parameter sets, clearance risk, command rate; the stateful
  goal arrival and obstacle bypass step by step on walks to the goal, around either end, through and short of a wall;
  the contact threshold). Agreement 2e-5 absolute. `failure` is `terminated / step_dt` times -30 so that mjlab's
  `dt` scaling leaves -30 per terminated step.
- **Terminations.** A trunk contact over 0.1 N terminates: one contact sensor per slot (the secondary must be a single
  body) reading the peak over the substeps; a wall pushed into the robot terminates that world only. The field
  boundary (5 m from the course origin) is a time-out (truncation).

Findings:
- **Mount.** Pollen's `tof` site is level at HOME and sits at z about 0.249 m; NVIDIA's sensor looks 20 degrees down from 0.22 m.
  `ToFSensorCfg.nominal_pitch_degrees` (default 0) is the knob; not tuned.
- **Prop heights.** NVIDIA's obstacle footprints are exact (box sizes, radius 0.1143 m); the heights of `crate`, `low_block`
  and `cylinder` are assumed (0.40, 0.35, 0.55 m) because the footprint is all NVIDIA's rewards use. `cylinder` is a box here
  and `thin_post` a cylinder (NVIDIA's code treats it as a cone).
- **Trunk contact.** The contact regex covers the whole robot subtree, as the plan's "any robot geom" reads, so a hand or foot touching a prop also terminates.
- **Walker speed.** About 46 % of the request (C3), so reaching the 2.6 m goal needs a request of about 0.28 m/s or more;
  the 0.2 m/s floor is config.
- **Order.** `terminated` is read before the reset events, `env.obs_buf` holds the post-step observation and the poses
  are read from qpos during reset (derived poses are stale there).

Not covered: GPU, the Beta head's behaviour on a trained policy, per-env grid on CUDA, the impairment model against
real ToF data, prop heights (assumed), the `Mjlab-Nav-MicroDuck` task beyond the 50 iterations of C7, any layout in M1
except `one_wall`, the walker under heavy obstacle contact.

## C7 8 envs x 50 PPO iterations on CPU: PASS

`tests/test_c7_training.py` (7 tests, `slow`) runs the real command
`train Mjlab-Nav-MicroDuck --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 50 --agent.save-interval 25 --agent.logger tensorboard`.
- 50 iterations finish; every logged scalar (rewards, losses, std) and every actor and critic tensor in the last
  checkpoint is finite; the 8 reward terms and the 5 termination terms are logged each iteration; progress is nonzero;
  mean action std stays under 5 and the learning rate stays 3e-4.
- Checkpoints `model_0`, `model_25`, `model_49` are saved with a `model_N.onnx` each. The ONNX loads in onnxruntime,
  input `[1, 1343]`, output `[1, 3]`, finite, inside [-1, 1], and equals the checkpoint's own Beta mean to 1e-5
  (the runner exports on save through `nav_export.export_actor`).
- **CPU iteration time: 2.63 s mean (median 2.58 s, max 3.87 s)** over iterations 2 to 50, of which
  2.53 s is collection (8 envs x 24 steps = 192 env steps) and 0.10 s the PPO update: 75 env steps/s. This is CPU
  with 8 envs on this 32-thread machine; it says nothing about the GB10 (the plan's estimate is 3.2 to 4.5 s per iteration at 4096 envs).

Not covered: that the policy learns anything (50 iterations at 8 envs is a plumbing test), GPU, multi-GPU, the
W&B logger.

## M1 dry run on CPU

`tests/test_m1_dry_run.py` (3 tests, `slow`): the M1 command (`docs/M1.md`) at 16 envs x 5 iterations on
`Mjlab-Nav-OneWall-MicroDuck` runs, saves `model_0` and `model_4` with their ONNX, and the replay
(`python -m microduck_nav.replay`) loads the checkpoint, runs 4 envs for 150 steps and prints a verdict. The
5-iteration policy does not stop before the wall (0 of 4 stopped, no collision, mean closed 0.25 m, final speed 0.15 m/s);
that is the expected result and not a statement about M1 on the GPU. `docs/M1.md` is checked against the commands the
repository builds.

Not covered: the 64 x 20 smoke and 4096 x 200 run themselves, a replay of a trained policy, the "done" thresholds
(chosen here, not the plan's).

## Environment

- `.venv`: 2.1 GB, 141 packages (`uv sync`), CPU torch 2.9.1+cpu from the PyTorch
  CPU index (the PyPI x86 wheel pulls CUDA libraries; the CPU wheel avoids them).
- Pollen's pins reproduced for: mujoco 3.10.0, numpy 2.4.1, tensordict 0.10.0, scipy 1.18.0, BAM
  `62bd8ce` (the branch tip `57d13ea` made `mjlab_microduck.tasks` fail to import with
  `FrictionDRBamActuatorCfg ... 'vin_drop_gain_range'`). Other packages were not diffed against
  Pollen's `uv.lock`; known differences: onnx 1.23.2 (lock 1.22.0), onnxruntime 1.32.0 (lock 1.24.4).
- Added for this repo: onnx, onnxruntime, pytest. Pollen's checkout was only read (the editable
  install writes ignored `__pycache__`).
