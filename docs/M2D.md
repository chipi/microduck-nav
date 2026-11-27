# Milestone M2d: the deploy task with a room round each scene

`Mjlab-Nav-Deploy-Room-MicroDuck` is `Mjlab-Nav-Deploy-MicroDuck` (M2b, `docs/M2.md`) with four wall slots forming a rectangle
round the field at every reset. Nothing else changes: `policy_decimation` 5, `lateral_limit` 0, the rewards and their weights,
the impairments, NVIDIA's layouts and their eight obstacle slots, the terminations, the 20 s episodes, the 1343-wide actor input
and its ONNX. `NavOptions(room=True)` is the one switch; it is exclusive with `pool=True` (R7's scenes bring their own walls).

## Why

`/work/microduck-nav-runs/r5/standstill-diagnosis.md`: the trained policy brakes in robotics-lab's rooms because its training scenes
are open fields. In a 4-env sample of the training task's scattered layout no depth return was beyond 1.99 m (lines 79-81), while the
navigator's frame shows room walls at 2.5-3.1 m (line 71); on recorded frames, removing the far walls made M2b's late checkpoints
drive (the table at line 89). M2d gives training those far returns. Its Q4 fix (a) asks for "room walls at 2.5 to 3.5 m in the scenes"
(line 221).

M2d does **not** touch the other half of the diagnosis: training's impairment drops floor returns beyond 0.556 m
(`impairments.py:39`, `:108-112`) and the navigator keeps them (lines 83-87). That stays as M2b has it.

## What it adds

- Four slots `room_0` to `room_3` (`obstacles.py`): the unit mocap box R7a built for the pool (`pool_slot_spec`), resized per
  environment at reset through `geom_size` with `geom_rbound` and `geom_aabb` kept consistent. The write is `field.write_box_slots`,
  factored out of `reset_pool_scene` (same code, same order of writes; R7's tests are unchanged and green). Each wall has a contact
  sensor, `contact_room_N`, and is excluded from contact with the floor, the other slots and NVIDIA's eight (`exclude_box_slot_contacts`,
  `exclude_pool_contacts` is now a one-line call of it).
- `room.py`, an event `room` after `obstacle_field`: reads the goal and the eight obstacles the layout placed, finds what the room must
  hold, draws a room round it, writes the walls.
  - The room is a rectangle in the robot's reset-heading frame (the frame NVIDIA's layouts are drawn in), axis-aligned there.
  - It must hold the start and the goal with 0.3 m (`DUCK_MARGIN`) and every one of NVIDIA's eight obstacles, in their real
    footprints (box rotated by its yaw, circle by its radius), with 0.05 m (`OBSTACLE_MARGIN`).
  - Each side is drawn uniform in `ROOM_SIDE_RANGE = (1.6, 6.0)` m, the inner side. A room that does not hold the scene is drawn again;
    this is done in closed form (a side is uniform in `[max(need, 1.6), 6.0]`, the same distribution as rejection). The room's place
    is uniform over where the scene stays inside.
  - A scene wider than 6 m on an axis gets **no room**: its walls are parked as the pool parks a slot, and it is counted
    (`field.room_counts` = rooms built, of them drawn again, environments without one; a line is printed every 100,000 resets).
  - The goal is computed by `commands.destination_direction`, the NVIDIA rule moved out of `NavigationCommand._sample_destinations`
    unchanged, so the room and the command read the same goal.
- Wall thickness 0.04 m and height 0.5 m, robotics-lab's: `labrunner.scene.wall(..., thick=0.04, height=0.5)`
  (`mission-control/libs/runner/src/labrunner/scene.py:58`, used by every `_walled` room at `challenges.py:19-22`). The walls stand on
  the outside of the inner sides, so the inner sides are the room's.
- The walls are obstacles for **the sensor** (they are geoms, group 0, in every world's ray cast), **the collision termination**
  (`obstacle_collision` takes the four `contact_room_N` sensors with NVIDIA's eight; threshold 0.1 N as before) and **the reward
  footprints** (`field.geometry()` is now `(n, 12, 2)` half sizes: approach, clearance risk, path clearance, the critic's obstacle
  clearances, which are 12 wide instead of 8; the actor is untouched, 1343 -> 3).
- One exception, on purpose: `barrier_bypass` ("pay once for passing the nearest obstacle blocking the initial request") never takes
  a room wall as that obstacle (`field.blocks_request`; `rewards.py`): a wall round the field cannot be passed, and a straight
  request toward the goal would otherwise name the far wall in every scene with a free path. For M2b and R7 the mask is all true.

## Room-size ranges and where they come from

`ROOM_SIDE_RANGE = (1.6, 6.0)` m is the range the task asked for. robotics-lab's challenge **generator** (`mission-control/libs/runner/
src/labrunner/challenges.py` at mission-control 8560dba; the evaluation sets' items were not read) gives, by its own draws:

| What | Range | Lines |
| --- | --- | --- |
| corridor width (challenge 1; the U-turn; the dead end) | 0.7-0.9 m; 0.6 m; 0.45-0.55 m | 26, 90, 64 |
| one walled room, ten goals (21) | 2.9-3.3 x 2.9-3.3 m | 347 |
| one room with a drop (22) | 2.8-3.2 x 3.3-3.7 m | 409 |
| two rooms (23) | each 2.4-3.0 x 2.4-3.0 m | 443-444 |
| `_room(goal_x)` (24-28): "a walled room from behind the start to past the goal, 3 m across" | 3.0 wide, up to 4.6 long (goal_x up to 3.4, line 563) | 174-176, 563 |

Read plainly, the generator's rooms go from 2.4 to 4.6 m a side and its corridors down to 0.45 m. **1.6 m and 6.0 m are not
generator values**: 1.6 is between the corridors and the smallest rooms, 6.0 is above the largest room (4.6 m). They are the task's
stated range and are used as stated; if the generator's own range is wanted, change `ROOM_SIDE_RANGE`, and expect most scenes to get
no room (below).

What the task really draws is narrower, because NVIDIA's eight obstacles stand within 2.6 m of the duck and rotate with its random
heading. Over 4,224 resets of the task (128 envs, seed 11, the first and 32 more):

- 3,571 got a room, all of them drawn again to fit (every scene needs more than 1.6 m); **653 (15.5%) got none**, the scene being
  wider than 6 m: the layouts that park their props round the duck (`open` 54% of its draws, `barrier` 49%, `thin_low` 48%, `passage` 18%,
  `corner` 36%, `blocked` 11%); `scattered`, 62% of all draws, never (0 of 2,622).
- sides of the rooms built: min 3.20, 10th percentile 3.93, median 5.45, 90th 5.96, max 6.00 m; the short side's median 4.93 m, the
  long side's 5.71 m. The farthest wall from the duck, median 3.28 m (max 5.64 m).

So the sampler covers 1.6-6 m (`test_room_sides_stay_in_the_range...`: a small scene draws min 1.6, max 6.0, mean 3.8), but with NVIDIA's
layouts no room smaller than 3.2 m is ever built. Small rooms need a different layout set.

## Checks (CPU, `uv run pytest -q -m "not slow"`: 232 passed)

`tests/test_m2d_room.py`, written first (against the commit before M2d the file fails at import: no `room_slot_names`, no task), then
green:

- the sensor reads a wall 3.0 m from it as 3.0 m (median of the zones that see it, +-0.2 m: the head is pitched);
- a duck asked to walk into a wall 0.45 m ahead ends the episode as `obstacle_collision` (not `fell_over`); one with the wall 5 m ahead does not;
- over 6 seeds x 64 envs x 2 resets the start, the goal (the command's own `destination_w`) and all eight obstacle footprints are inside
  the room, with the margins; every reset is counted as built or not;
- the sampler's range, the cut at the scene's size, the no-room case, the wall geometry (thickness 0.04, height 0.5, inner sides);
- a sample of 4,096 resets (128 envs x 32), rays as cast at the reset, impairments on: **return farther than 2 m that is not the
  floor: M2d 61.1% of frames (6.59% of zones); M2b 2.1% (0.09% of zones)**;
- the task is M2b's with 12 slots: same observation groups (policy 53, tof 1290), rewards, decimation 5, limits (0.35, 0, 0.75); the
  wall's clearance is its distance minus the duck's radius; the bypass target is never a wall;
- `tests/test_m2b_unchanged.py`: M2b's observations (policy, tof, critic) for a fixed seed over 6 ticks hash to what they did before
  M2d (sha256 taken at 4248b0e); the parity fixture (`test_obs_parity.py`), the C-tests and R7's tests are unchanged and green.

C7, `tests/test_c7_room.py` (slow): the real `train` command, 8 envs x 10 iterations, finite, checkpoints at 5 and 9, the ONNX
`(1, 1343) -> (1, 3)`, metadata `task = Mjlab-Nav-Deploy-Room-MicroDuck`, `policy_step_s = 0.1`, `lateral_limit = 0.0`.

### CPU time

| Task | Iterations | Mean iteration | Median | Max | Collection | PPO update |
| --- | --- | --- | --- | --- | --- | --- |
| M2d | 10 | **8.08 s** | 8.07 | 8.16 | 8.04 | 0.04 |
| M2b, same machine, same session | 20 | 7.95 s | 7.89 | 8.67 | 7.91 | 0.04 |

M2d costs 1.6% more per iteration than M2b measured today. `docs/M2.md`'s 10.9 s for M2b was measured on another day, under another
load, and is not the right comparison: against it M2d would look 26% faster, which it is not. The GB10 is not measured.

## The DGX command

```bash
MICRODUCK_WALKER_ONNX=/path/to/velstand.onnx uv run train Mjlab-Nav-Deploy-Room-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.save-interval 100 --agent.logger tensorboard
```

Watch the log for `room: N resets, ... without (x%)`: about 15% is expected.

## Not covered

- Nothing ran on a GPU or at 4,096 envs; the far-return sample is 4,096 *resets* in batches of 128 envs, and the time is CPU, 8 envs.
- No model was trained on this task past 10 iterations: that the far returns stop the braking, or that M2d beats M2b at the navigator, is
  untested. The diagnosis's other cause (floor returns that training drops and the navigator keeps) is not addressed.
- The far-return share counts rays as cast at the reset, not the delivered, impaired frames over an episode, and not the navigator's frames.
- 15.5% of resets have no room (open field, as M2b); no room under 3.2 m is built; rooms are axis-aligned with the duck's reset heading
  (as the layouts are), so the start never faces a corner or a wall at a random angle to the room; the room's yaw is not randomised.
- Walls have one height (0.5 m) and one thickness; doors, corridors and rooms joined to rooms (the generator's challenges 2, 3, 23) are not built.
- `barrier_bypass` ignores walls by choice; a duck that goes round the field is not rewarded for it, nor is it penalised beyond the
  approach and clearance terms.
- The critic's privileged input is 12 clearances, not 8; a checkpoint trained on M2b cannot be resumed on M2d for its critic.
