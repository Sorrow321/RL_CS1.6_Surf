# `--obs-potential abs|rel`: the race potential as a second image channel

Branch `contyaw-abs` (2026-09-07). Default OFF; with the flag off the trainer,
the renderer and every eval tool are byte-identical to before (no config key
is written, the depth kernel is untouched, `tests/python/test_unstuck.py`'s
flag-off identity against the git-history trainer still passes). Built and
smoked on the local 5090; nothing rented, no arm run.

## The ask

The user's words: "render another image, on which you show the potential
field measured at the same point, for which we render depth. And train with
2 input channels." Then, on the encoding: two separate experiments, an
ABSOLUTE channel and a RELATIVE one, "it will see positive values forward
and negative backward".

So: for every lidar ray of the 64x32 depth image, take the ray's hit point
and sample the race potential there - the geodesic distance-to-finish
(`surfgym.goalfield.build_goal_field`, the field the race shaping reward
walks down, cell 32 on cannonball, `maps/surf_src_cannonball.goal_32.npz`).
The value becomes the second channel of the image the conv trunk reads, so
the policy can see which parts of its view lead toward the goal.

## The picture

`docs/potential_view.png` (`python tools/demo/potential_view.py`), six states
of `C:/RL_Surf_base/runs/research/cyABSV/traj_8020557824.jsonl` episode 0
(the 8.0B absolute-view checkpoint, a non-finisher that dies at the 88.8 %
wall): the spawn, 15 / 30 / 45 / 60 s, and the wall entry - the last tick at
which the map pushed back (vertical acceleration departing from the gravity
step, `tools/pick_selfline.py`'s rule; tick 6,287 = 62.9 s here, the start
of the final free fall). Four columns per state: the depth image; the abs
channel on the policy's scale; the same abs frame stretched to its own
range; the rel channel on the policy's scale. One `GpuLidar.render` call per
mode produced all of it, so the picture IS the observation, pixel for pixel.

The value ranges, from the tool's own table (`d_eye` = the field at the eye):

| state | tick | pos | d_eye / d0 | abs min .. max (mean) | in-frame span | rel min .. max (mean) | share > 0 |
|---|---|---|---|---|---|---|---|
| 0 s (spawn) | 0 | (-14336, 3079, 10528) | 1.000 | 0.971 .. 1.000 (0.996) | 5,625 u | +0.024 .. +2.000 (+0.41) | 100 % |
| 15 s | 1500 | (-5727, 376, 4191) | 0.838 | 0.804 .. 0.839 (0.833) | 6,762 u | -0.066 .. +2.000 (+0.48) | 98 % |
| 30 s | 3000 | (-4448, 885, 1885) | 0.608 | 0.579 .. 0.609 (0.601) | 5,891 u | -0.096 .. +2.000 (+0.68) | 98 % |
| 45 s | 4500 | (7526, 600, 2130) | 0.398 | 0.375 .. 0.398 (0.394) | 4,477 u | +0.029 .. +2.000 (+0.43) | 100 % |
| 60 s | 6000 | (-908, -3017, -4837) | 0.146 | 0.128 .. 0.149 (0.143) | 4,125 u | -0.291 .. +1.771 (+0.29) | 72 % |
| wall entry | 6287 | (2683, 4493, -4725) | 0.095 | 0.077 .. 0.101 (0.091) | 4,672 u | -0.596 .. +1.740 (+0.32) | 87 % |

No ray in any of the six frames sampled unreachable space (bad share 0.00
in both channels).

What the two encodings look like, read off the picture:

* **abs is a LEVEL, not a picture.** Within one 64x32 frame the field spans
  4,100-6,800 u, i.e. 2-3.4 % of d0, so on the policy's [0, 1.5] scale each
  frame is nearly flat and its value says where in the run the agent is
  (1.0 at the spawn, 0.84 at 15 s, ..., 0.095 at the wall). The structure
  inside a frame - which is what the user wants the agent to steer by - is
  there (third column) but 30-50x smaller than the channel's range. A conv
  trunk can still read it (the first layer is linear and the numbers are
  exact float32), but it has to learn to subtract the level first.
* **rel is the picture.** The goal-ward part of the view is a bright blob
  (up to the +2 clip = 4,000 u nearer the finish than the eye, one field
  cell short of an 11,500 u ray), the rest sits near 0, and the parts that
  lead back read negative (-0.3 at 60 s, -0.6 at the wall entry, both on
  the edges of the view that look back up the track). The sign convention
  is the user's: positive forward, negative backward.

## Design

### Where the sample is taken: one field cell short of the hit

The march can only stop INSIDE a solid voxel (`hit_eps` is under one cell
and every air voxel's EDT is at least one), and the field holds its
sentinel on solid voxels. Sampled at the raw hit point, **16 % of the rays
on the picture's poses have no honest corner** and would read "unreachable"
- depending on which half of the wall voxel the march landed in, i.e. on
nothing the agent can see. The sample is therefore taken one field cell
short of the hit along the ray, `t_s = max(t - cell, 0)`: the air the
player would occupy at that surface. One cell back is always air (the last
live step was `0.9 d` from a point at least `0.6 cell` clear); measured, 0 %
of the same rays are bad there. A ray that ran to range samples one cell
short of its end point, in the open air it crossed, so a clear ray still
says whether that direction descends the field; a ray stopped inside its
own start voxel samples at the eye. The trilinear weights renormalise over
honest corners exactly as `GoalField.sample` does (a sample beside a wall
is a one-sided extrapolation, never a mixture with the wall). The
difference between "at the surface" and "one cell short" is at most 32 u =
0.016 of the rel scale.

### The two encodings

`surfgym.vision.LidarPotential(field, mode, d0=...)`:

| mode | channel | clip | no honest corner | eye unreachable |
|---|---|---|---|---|
| `abs` | `d_hit / d0`, d0 = the map's start geodesic (the trainer's own `race_d0`, the mean field over the raw map spawns, 198,380 u on cannonball) | [0, 1.5] | 1.5 | ignored |
| `rel` | `(d_eye - d_hit) / 2000 u`, d_eye = the field at the eye (origin + 17 u standing / 12 u ducked, the ray origin) | [-2, 2] | -2 | the whole frame -2 |

Goal-ward is POSITIVE under rel (a hit whose field is lower than the eye's
leads toward the finish). An eye in unreachable space (the agent already
falling out of the world) has no goal-ward direction, so the frame reads
-2 everywhere rather than a meaningless difference against the sentinel.
`d0` is recorded per map in the config (`obs_potential_d0`) so the eval
tools scale the abs channel exactly as it was trained; they recompute the
same number from the same spawns when a config predates the key.

### The renderer

* `GpuLidar(..., potential=LidarPotential)` -> `(N, H, W, 2)` interleaved
  (depth, potential), channel-fastest like `--surf-mask`, so
  `Policy.forward_split`'s restride into the channels_last trunk stays a
  free view. `lidar.channels` becomes 2 and everything downstream follows
  (`FRAME`, `img_ch`, `Policy(in_ch=2)`, the rollout buffer, the truncation
  bootstrap's `render_rows`, the BC render, the eval wrappers - all of them
  call `lidar.render`).
* Triton: `_march_kernel_pot`, a copy of `_march_kernel` (the copy-not-flag
  rule of the other channel kernels: the depth encoding is warm-start ABI
  and the single-channel kernel stays untouched) with the sample as its
  tail - 8 int16 gathers, `GoalField.sample`'s float32 arithmetic in its
  order. The eye's own field is computed once per env in torch and passed
  in, not once per ray. The torch fallback (`_render_torch`, the CPU smoke
  path) does the same in torch and is the reference: the two agree on 100 %
  of the pixels of the picture's poses (max |diff| 2.4e-7 abs, 2.3e-5 rel),
  and the depth channel is bit-exact against the plain kernel on both.
* The grid rides on the device as the cache's own uint16 codes (`code *
  quant`, quant = cell / 8 = 4 u on cannonball), viewed as int16 and read
  back through `& 0xFFFF` because triton has no unsigned 16-bit load: 1.34
  GB for the 908x881x839 grid, next to the 1.34 GB fp16 SDF. The upload
  checks that `code * quant` reproduces the reward's grid bit for bit.
* Exclusive with `--surf-mask`, `--normals`, `--pinhole`, `--frame-stack`
  and the goal ball (no combined kernels); needs `--reward race` with the
  geodesic field (`--race-dist euclid` has no grid; `--goals` shapes on a
  per-env field). Multi-map and held-out slots each upload their own map's
  field at their own `d0`.

### The trainer, the checkpoint, the tools

* `--obs-potential abs|rel` in `train_fast.py`; the mode string is written
  into `run.json` / the checkpoint config ONLY when set, with
  `obs_potential_d0 = {map tag: d0}`. A resume restores it; a resume asking
  for the other mode, or for no channel on a channel checkpoint, is refused
  (conv1 is `(16, in_ch, 5, 5)` and the two channels mean different things).
  `obs_potential` is in `ARCH_KEYS`.
* `tools/record_ckpt.py`, `tools/beam_tas.py`, `tools/diversity_bench.py`,
  `tools/expert_dagger.py` mirror it through `LidarPotential.from_cfg`;
  `tools/demo/wr_scan.py` refuses such a checkpoint (it is discrete-only
  and behind on the vision flags anyway).
* `tools/run_arm.sh`'s SCRATCH branch and `launch_local.ps1 scratch_ablate`
  take it as a trailing flag (`"$@"` / `$Extra`).

## Throughput

Render kernel alone, 2048 envs x 64x32 on the 5090 (`_march_kernel` vs
`_march_kernel_pot`, same poses): 0.37 ms -> 1.19 ms per batch (+0.82 ms
per decision; the 8 random int16 gathers into a 1.34 GB grid are latency
bound).

Trainer, the scratch preset at 2048 envs through the launcher (below),
local 5090, `time/fps` from `progress.csv` at matched steps:

| run | flag | steady-state steps/s (40M-120M) | cumulative at 120M | wall to 120M |
|---|---|---|---|---|
| cyPOT0 | (control) | 791k | 655k | 184 s |
| cyPOT0b | (control, again) | 747k | 683k | 177 s |
| cyPOTA | `--obs-potential abs` | 672k | 605k | 199 s |
| cyPOTR | `--obs-potential rel` | 642k | 573k | 210 s |

`time/fps` is cumulative (steps since start over wall since start), so the
steady-state rate is the slope of steps against wall between two rows. The
two controls differ by 6 % between themselves (a foreign process held ~4 GB
and 10-25 % of the card throughout; nothing was rented), so the read is:
**the channel costs 10-19 % of steady-state throughput on this box, ~0.85x
wall-clock** - the render kernel accounts for about 7 % of it (0.82 ms per
decision against a 10.3 ms decision at 792k steps/s), the rest is the
twice-as-wide image through the rollout buffer, the update's gathers and
the trunk's first layer, plus the noise floor. abs and rel run the same
kernel (one constexpr branch apart); their 30k gap is noise.

## What is pinned

* `tests/python/test_obs_potential.py` (this branch): the sampler against
  `GoalField.sample`; the encodings; the synthetic-scene render (depth
  bit-identical to the depth-only lidar, the sample one cell short of the
  hit recomputed from the march's own `t`, goal-ward positive / backward
  negative / unreachable -2 under rel, abs in [0, 1.5]); the exclusivity;
  the CUDA kernel against the fallback on cannonball (both channels, the
  depth ABI, the eye sample, the 16 % raw-hit sentinel share); the trainer
  smokes on the toy scratch set (finite losses at obs width 15 + 2x16x8,
  in_ch 2, the config keys, record_ckpt's mirror, the resume restore, the
  mismatch refusal, the euclid / surf-mask / bad-mode refusals).
* Flag off: `tests/python/test_unstuck.py::test_flag_off_is_bit_identical_
  to_the_trainer_before_unstuck` (run.json, progress.csv, the eval
  trajectory bytes, the weights, the Adam moments against the git-history
  trainer) and `test_view_absolute.py`'s identity smokes, both re-run on
  this commit.

## The arms

Two arms, one seed each, the from-scratch ablation baseline (CLAUDE.md
section 2), the control is the same line without the flag:

    SCRATCH=1 bash tools/run_arm.sh cyPOTA --obs-potential abs
    SCRATCH=1 bash tools/run_arm.sh cyPOTR --obs-potential rel

Both confirmed through the SCRATCH branch locally (a 2048-env launch with a
small budget: the config carries the key, the trainer reports `in_ch 2`,
the eval wrappers render the channel). Judge them as CLAUDE.md says: the
gate cleared and the step it was cleared at, `tools/eval_honesty.py
--order-only 16` on the trajectories, never `race/eval_progress` alone.

## Not done

* The arms themselves (the user launches them).
* The abs channel's flatness within a frame (2-3 % of its range) is a
  property of the encoding the user asked for, reported above, not changed.
* `tools/render_pov.py` has no potential panel (a channel checkpoint
  renders its depth panel from channel 0 as before).
