# `--obs-potential abs|rel|norm|logabs`: the race potential as a second image channel

Branch `contyaw-abs` (2026-09-07). Default OFF; with the flag off the trainer,
the renderer and every eval tool are byte-identical to before (no config key
is written, the depth kernel is untouched, `tests/python/test_unstuck.py`'s
flag-off identity against the git-history trainer still passes). Built and
smoked on the local 5090; nothing rented, no arm run.

Branch `contyaw-norm` (2026-09-07, on top of it) adds the THIRD mode, `norm`:
the abs sample standardised per frame - contrast without the level. It is a
post-process of the rendered channel (no kernel change), described in its
own section below. Built and smoked on CPU only (the local GPU was busy with
a trainer, so the torch fallback path is what was exercised; the CUDA kernel
did not change); nothing rented, no arm run.

Branch `contyaw-norm` (2026-09-07, later the same day) adds a FOURTH mode,
`logabs`, and a flag that applies to all four, `--obs-potential-curtain`.
Both come out of the same reading of the first arm: `cyPOTA2` (`abs`) stalled
at the wall like every untreated control, and the channel it was given cannot
say anything about the end of the map. `abs` is LINEAR and the conv trunk in
use (`trunk: plain`) has no normalisation layer, so on cannonball the wall
region - 88.8 % of the route, d = 6,568 u - reads 0.033, the finish-room
walls 0.003-0.007 and the goal 0: the whole last eighth of the run is one
value. And the finish is a `trigger_multiple` CURTAIN, not a solid, so rays
pass straight through it and stop on the far wall 550-1,100 u behind - the
goal is never a pixel at all. `logabs` fixes the first (the level, on a log
axis); `--obs-potential-curtain` fixes the second (the goal becomes a pixel).
Sections below. Both are OFF by default and bit-identical off.

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

The third mode is the follow-up to the picture below: the abs frame on its
own range (the third column) as a trainable input.

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
* **norm is the third column as an input** - the abs frame with its level
  removed and its own spread filling the channel. See its section.

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

### The encodings

`surfgym.vision.LidarPotential(field, mode, d0=...)`:

| mode | channel | clip | no honest corner | eye unreachable |
|---|---|---|---|---|
| `abs` | `d_hit / d0`, d0 = the map's start geodesic (the trainer's own `race_d0`, the mean field over the raw map spawns, 198,380 u on cannonball) | [0, 1.5] | 1.5 | ignored |
| `rel` | `(d_eye - d_hit) / 2000 u`, d_eye = the field at the eye (origin + 17 u standing / 12 u ducked, the ray origin) | [-2, 2] | -2 | the whole frame -2 |
| `norm` | `(d_hit - mean_frame) / (std_frame + 50 u)`, mean and population std over the frame's honest pixels (per env, per render call) | [-3, 3] | +3 | ignored; a frame with fewer than 8 honest pixels reads 0 everywhere |
| `logabs` | `log1p(d_hit / 1000 u) / log1p(d0 / 1000 u)`, the same `d0` as abs | [0, 1.5] | 1.5 | ignored |

Goal-ward is POSITIVE under rel (a hit whose field is lower than the eye's
leads toward the finish). An eye in unreachable space (the agent already
falling out of the world) has no goal-ward direction, so the frame reads
-2 everywhere rather than a meaningless difference against the sentinel.
`d0` is recorded per map in the config (`obs_potential_d0`) so the eval
tools scale the abs channel exactly as it was trained; they recompute the
same number from the same spawns when a config predates the key. rel and
norm use no scale; the recorded value rides along unused.

### norm: the abs frame on its own range, as an input (branch `contyaw-norm`)

**The exact semantics.** For every frame (one env, one render call) take
the abs sample of every ray - `d_hit`, the field one cell short of the hit,
in map units, the same honesty as abs (at least one honest corner; the eye
plays no part) - and

1. over the frame's HONEST pixels compute `mean` and `std` (the population
   std, divided by n, accumulated in float64);
2. every honest pixel becomes `(d_hit - mean) / (std + 50 u)`, clipped to
   [-3, 3];
3. every pixel with no honest corner becomes **+3**;
4. a frame with **fewer than 8** honest pixels becomes **0 everywhere**,
   its bad pixels included.

Larger = farther from the goal, abs's sign: the near floor under the eye
reads positive, the goal-ward part of the view negative.

**Why +3 for a bad pixel and not -3.** The sentinel is ABOVE every honest
value of the field (`reach_max + 2 cell`): unreachable space is, in the
field's own arithmetic, the farthest thing there is. abs already encodes it
that way (1.5, its ceiling), and so does rel (-2, its LEAST goal-ward end
under the flipped sign). norm keeps abs's sign, so the honest continuation
of "d = sentinel" through the standardisation is the top clip. -3 would put
unreachable space at the goal-ward end of the range, where a policy would
read it as the most attractive direction in view - the opposite of what it
is.

**Why 0 for a frame under 8 pixels.** With no statistics there is no
picture; +3 everywhere would say "everything is far" and -3 "everything is
near", both false. abs and rel keep their bad values there (1.5 / -2); norm
alone reads 0, which is what its own zero-mean rule says about a frame with
nothing in it.

**Why the +50 u floor.** A frame that is nearly flat (the sky-heavy view at
the spawn: 197 u of std) would otherwise amplify its rounding into the full
range; with the floor, a flat frame stays flat (an exactly flat one reads
0, not 0/0) and a spread of 150 u with no other structure is the most the
channel will ever push to the clip. It is not a scale: no `d0` is needed,
and shifting the whole field by a constant leaves the channel unchanged
(pinned on the synthetic scene).

**How it differs from the picture's third column.** That column is a
min-max stretch of the abs frame (`imshow` on the frame's own range); norm
is a z-score with a floor. Same information - the abs frame minus its level
- but a single far pixel cannot flatten the rest of the frame the way a
min-max stretch lets it, and the level is removed by the mean rather than
by the minimum.

**What the numbers look like on cannonball** (`GpuLidar.render` on the CPU
fallback with the main checkout's caches, the fixture poses of
`test_obs_potential.py`: the spawn, four states of the cyABSV episode, a
ducked state, and a sky-heavy view at the spawn; every pixel honest, so
+3 never fires here and the range's lower end is the goal-ward blob):

| state | frame mean u | frame std u | abs min .. max | norm min .. max (mean, std) | at -3 |
|---|---|---|---|---|---|
| spawn 0 s | 197,505 | 1,240 | 0.952 .. 1.000 | -3.00 .. +0.65 (+0.009, 0.922) | 1.2 % |
| 15 s | 165,245 | 917 | 0.805 .. 0.839 | -3.00 .. +1.14 (+0.014, 0.889) | 1.4 % |
| 30 s | 119,257 | 983 | 0.579 .. 0.609 | -3.00 .. +1.52 (+0.006, 0.930) | 1.5 % |
| 45 s | 78,141 | 852 | 0.375 .. 0.398 | -3.00 .. +0.89 (+0.010, 0.910) | 2.1 % |
| 60 s | 28,395 | 776 | 0.128 .. 0.149 | -3.00 .. +1.40 (+0.002, 0.933) | 0.8 % |
| ducked (-2667, 2972, -1453) | 11,145 | 329 | 0.049 .. 0.059 | -3.00 .. +1.21 (+0.007, 0.840) | 1.8 % |
| sky-heavy (spawn, yaw 90 pitch 25) | 198,590 | 197 | 1.000 .. 1.003 | -1.10 .. +1.94 (+0.000, 0.798) | 0 % |

So: the level (0.05 .. 1.0 of d0 across these states) is gone, every frame
is a zero-mean picture with a std of 0.80-0.93 (= `s / (s + 50)` minus the
clip's bite), and the goal-ward blob rel shows at its +2 clip is here the
-3 end - 0.8-2.1 % of the pixels sit at -3, being 2,300-3,700 u nearer the
finish than the frame's mean. The near side never reaches +3 on these
frames: the far end of an honest frame is the eye's own level plus a few
hundred u. The floor is 4 % of the std on the widest frame and 25 % on the
sky-heavy one.

**Implementation: a post-process of the abs sample, no kernel change.** The
kernel's tail takes its constants at run time (`scale_inv`, the clip, the
bad value), so under norm the same `REL=False` tail runs with scale 1 (raw
map units), the clip [0, `valid_max`] (a no-op: a trilinear mean of honest
corners is under `valid_max` by construction) and the bad marker **-1**
(out of band: a geodesic is never negative). `GpuLidar.render` then calls
`LidarPotential.normalise` on the rendered channel, on the triton path and
the torch fallback alike (one function, float64 statistics, float32 out).
`_march_kernel_pot` is byte-identical to `contyaw-abs`; abs and rel are
untouched (the constants are the same numbers as before, set through one
`_set_mode`). The trainer, the checkpoint and the tools carry the mode
string exactly as for abs and rel: `obs_potential: "norm"` in `run.json`
and the checkpoint config (with `obs_potential_d0`, unused), restored on a
resume, refused against abs / rel / off (`--obs-potential changes the conv
trunk's input channels`), `ARCH_KEYS`, `record_ckpt` / `beam_tas` /
`diversity_bench` / `expert_dagger` through `LidarPotential.from_cfg`,
`wr_scan` refuses. Exclusive with `--surf-mask` / `--normals` / `--pinhole`
/ `--frame-stack` / the goal ball, needs `--reward race` with the geodesic
field - the same refusals as the other two modes.

### logabs: abs's LEVEL, log-compressed (2026-09-07)

`value = log1p(d_hit / 1000 u) / log1p(d0 / 1000 u)`, clipped to [0, 1.5],
a sample with no honest corner 1.5, no per-frame statistics. The two
anchors are exact by construction: `d = d0` reads **1.0** and `d = 0` reads
**0.0**, so the number still means "where in the run this ray points" the
way abs's does - unlike `norm`, which throws the level away.

**Why.** `abs` is linear in a geodesic that spans 198,380 u, and the part
of the map the agent is stuck on lives in the bottom 3 % of it. Measured on
cannonball:

| place | d (u) | abs | logabs |
|---|---|---|---|
| spawn (d0) | 198,380 | 1.000 | 1.000 |
| the wall, route vertex 1601 | 6,568 | 0.033 | 0.38 |
| finish-room wall, far | 1,100 | 0.0055 | 0.13 |
| finish-room wall, near | 550 | 0.0028 | 0.075 |
| the goal | 0 | 0.000 | 0.000 |

The trunk carries no normalisation layer, so those abs numbers reach the
first conv as they are: the entire end of the map is one value to the
network, and 0.033 against 0.0028 is a difference of 0.03 in an input whose
other channel swings over [0, 1.25]. Under logabs the same span opens to
0.38 against 0.075. `log1p` is concave and both curves pass through (0, 0)
and (d0, 1), so logabs is >= abs everywhere inside the run, <= abs past the
start, and strictly monotone in d throughout - it re-weights the scale
without reordering anything (pinned by a test).

**How.** Exactly like `norm`, as a POST-PROCESS of the rendered abs sample,
so no kernel changed: the kernel tail runs its abs branch at scale 1 (raw
map units, the clip a no-op, the bad marker -1 out of band), and
`GpuLidar.render` calls `LidarPotential.postprocess` ->
`log_compress` on the channel, on the triton path and the torch fallback
alike. `d0` is required (the mode is refused without it) and is the same
per-map `race_d0` abs uses, recorded in `obs_potential_d0`.

### `--obs-potential-curtain`: the finish catches rays (2026-09-07)

**The defect.** The finish on a type-1 map is a `trigger_multiple` - an
invisible curtain, not geometry. It is not in the SDF (a trigger volume
draped over a ramp must not become a wall), so the lidar march flies
through it and stops on the far wall of the finish room, 550-1,100 u
behind. Whatever the encoding, the most goal-ward value the channel can
ever show is that wall's, and the GOAL ITSELF is never a pixel. On
cannonball the box is `end` = mins (-14720, 7487, -1824), maxs (-8064,
7488, -352): **one unit thin in y**, against a 32 u march step - stepping
onto it is hopeless, which is why this is an analytic test and not a second
grid.

**The rule.** For every ray, slab-test the segment from the eye to the
depth hit (or to the lidar range, if the ray ran clear) against the finish
AABB. If the ray enters the box at or before its hit, that pixel's
potential sample is replaced by the goal value **d = 0** and marked honest.
Everything downstream is the mode's own encoding: `abs` and `logabs` read
**0**, `rel` reads `d_eye / 2000` clipped at +2 (its goal-ward end), `norm`
reads the frame's most goal-ward value. The **depth channel is untouched** -
the curtain is a fact about the potential, not about geometry, and the
policy must not learn to expect a surface there.

Details that matter:

* The box used is the PADDED `mins`/`maxs`, i.e. the trainer's own
  `slot.goal_box`, the same AABB `surf_set_goal_box` scores a finish
  against - so the channel and the +50 success bonus agree on where the
  goal is. (`true_aabb`, the unpadded honest finish line, is not used.)
* The slab test handles the degenerate axis explicitly rather than by
  `1/0`: a ray parallel to a slab either starts inside it (no constraint)
  or misses the box entirely. With a box 1 u thin in y that is the common
  case, not a corner one.
* Off, the render is bit-identical: the kernel's block is behind a
  `CURTAIN: tl.constexpr` and is not compiled at all, and the six box
  scalars are passed as zeros. Pinned per mode by
  `test_curtain_off_is_bit_identical`, which also renders with a box no ray
  can reach and demands the same bits.
* Multi-map and held-out slots each get their OWN finish box.
* The key is `obs_potential_curtain: 1` in `run.json` / the checkpoint
  config, written only when set, restored on a resume (it changes no tensor
  shape, so a mismatch is not refused the way the mode is). The eval tools
  rebuild the box from the map's own `zones.json` in
  `LidarPotential.from_cfg`. `--obs-potential-curtain` without
  `--obs-potential` is refused.

### The renderer

* `GpuLidar(..., potential=LidarPotential)` -> `(N, H, W, 2)` interleaved
  (depth, potential), channel-fastest like `--surf-mask`, so
  `Policy.forward_split`'s restride into the channels_last trunk stays a
  free view. `lidar.channels` becomes 2 and everything downstream follows
  (`FRAME`, `img_ch`, `Policy(in_ch=2)`, the rollout buffer, the truncation
  bootstrap's `render_rows`, the BC render, the eval wrappers - all of them
  call `lidar.render`, which is also where norm's post-process lives).
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

* `--obs-potential abs|rel|norm|logabs` in `train_fast.py`; the mode string is
  written into `run.json` / the checkpoint config ONLY when set, with
  `obs_potential_d0 = {map tag: d0}`. A resume restores it; a resume asking
  for another mode, or for no channel on a channel checkpoint, is refused
  (conv1 is `(16, in_ch, 5, 5)` and the channels mean different things).
  `obs_potential` is in `ARCH_KEYS`.
* `tools/record_ckpt.py`, `tools/beam_tas.py`, `tools/diversity_bench.py`,
  `tools/expert_dagger.py` mirror it through `LidarPotential.from_cfg`;
  `tools/demo/wr_scan.py` refuses such a checkpoint (it is discrete-only
  and behind on the vision flags anyway).
* `tools/run_arm.sh`'s SCRATCH branch and `launch_local.ps1 scratch_ablate`
  take it as a trailing flag (`"$@"` after `shift` / `$Extra`; checked for
  `--obs-potential norm` by reading both and by binding the PowerShell
  parameter block: `scratch_ablate cyPOTN --obs-potential norm` lands as
  `Arg1 = cyPOTN, Arg2 = <empty>, Extra = [--obs-potential, norm]`).

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

norm's throughput was NOT measured (no GPU free). Its extra work is the
post-process: a float64 mean and std over 2048 pixels per env and one
elementwise pass, on a (2048, 32, 64) tensor per decision - small next to
the kernel's 0.82 ms, but unmeasured.

## What is pinned

* `tests/python/test_obs_potential.py` (this branch, 13 tests, 12 run on
  CPU and 1 CUDA test skipped here): the sampler against `GoalField.sample`;
  the encodings; **norm against a hand computation** (the kernel tail's
  constants under norm; the rule on a synthetic frame set: bad pixels +3,
  7 honest pixels -> the whole frame 0, exactly 8 -> standardised, a flat
  frame 0, the floor damping a near-flat frame's outlier, a far outlier
  clipping at +3, zero mean and `s / (s + 50)` std); the synthetic-scene
  render (depth bit-identical to the depth-only lidar, the sample one cell
  short of the hit recomputed from the march's own `t`, goal-ward positive
  / backward negative / unreachable -2 under rel, abs in [0, 1.5]); **norm
  on the same scene** (the hand rule on the recomputed raw sample, the
  unreachable rays +3, the near row positive, a view with no honest ray 0
  where abs reads 1.5 and rel -2, a constant shift of the field leaving the
  channel unchanged); the exclusivity under rel and norm; the CUDA kernel
  against the fallback on cannonball in all three modes (the norm leg is
  written but was not executed - no GPU); the trainer smokes on the toy
  scratch set for rel, abs and norm (finite losses at obs width 15 +
  2x16x8, in_ch 2, the config keys, record_ckpt's mirror, the resume
  restore, EITHER other mode refused on resume, the euclid / surf-mask /
  bad-mode refusals for rel and norm).
* Flag off: `tests/python/test_unstuck.py::test_flag_off_is_bit_identical_
  to_the_trainer_before_unstuck` (run.json, progress.csv, the eval
  trajectory bytes, the weights, the Adam moments against the git-history
  trainer) and `test_view_absolute.py`'s identity smokes, both re-run on
  the `contyaw-abs` commit. abs and rel on `contyaw-norm`: their constants
  and code paths are unchanged (the kernel is byte-identical; the same
  synthetic-scene and smoke tests pass), no bit-identity run was repeated.

## The arms

Three arms, one seed each, the from-scratch ablation baseline (CLAUDE.md
section 2), the control is the same line without the flag:

    SCRATCH=1 bash tools/run_arm.sh cyPOTA --obs-potential abs
    SCRATCH=1 bash tools/run_arm.sh cyPOTR --obs-potential rel
    SCRATCH=1 bash tools/run_arm.sh cyPOTN --obs-potential norm

(locally: `powershell -File tools/launch_local.ps1 scratch_ablate cyPOTN
--obs-potential norm`.) abs and rel were confirmed through the SCRATCH
branch locally on a 2048-env launch with a small budget; norm through the
toy-size CPU smoke of the same argument set (`SMOKE_FLAGS`, the scratch set
at 64 envs, 16x8) plus the launcher reading above. Judge them as CLAUDE.md
says: the gate cleared and the step it was cleared at, `tools/eval_honesty.py
--order-only 16` on the trajectories, never `race/eval_progress` alone.

## Not done

* The arms themselves (the user launches them).
* The abs channel's flatness within a frame (2-3 % of its range) is a
  property of the encoding the user asked for, reported above, not changed;
  norm is the encoding that removes it.
* norm on the GPU: the triton path was not executed on this branch (the
  kernel is unchanged; the post-process is device-agnostic torch), its
  throughput is unmeasured, and the CUDA test's norm leg is unrun.
* `tools/demo/potential_view.py` has no norm column (its third column is
  the min-max stretch, not this encoding); `tools/render_pov.py` has no
  potential panel.
