# Open questions after the first 108-map run (rounds 24-25)

State when written: the 34e9-step 4x3090 pool run and the 3x3 size x lr
grid are harvested (runs/mmPOOL_harvest/, runs/ablation_results.tar.gz)
and both boxes destroyed. Headline facts these questions live against:
mean map_pct 22.7 / median 11.8 (24 maps under 5%), **zero finishes on
all 108 maps in 18 evals**, reservoir min-depth at 0 with train win_rate
4.8% (the harvest artifact), width monotone-positive at matched steps
(+31% for 4x at 1.9x step cost), lr 3e-4 right at every width.

Each item: what was observed -> what the code/data actually says
(verified, with places) -> the experiment that decides it -> cost under
the bundle protocol (section 10).

## 1. Map and zone bugs (P0, $0 to diagnose)

Observed: a few maps saturate map_pct almost immediately; on one, the
agent drops to a floor that should be lethal, walks, and jumps into the
end cuboid.

Verified: **the sim has no fall damage at all** (nothing in `src/pm.c`);
in GoldSrc the drop would kill. The three maps at >85% by eval 2
(1.8e9 steps) are **bucetation (100.0), shortbox (100.0), ut0pia
(99.4)** - the drop-exploit map is one of these (check each in the
viewer). bucetation/shortbox were already the documented "map_pct
liars".

Decides it: watch the three trajs locally ($0). Then per map: remove
from pool (v3 archive) or keep knowingly.

Fix directions, separate arms:

* Fall damage in pm.c (landing-speed threshold -> fail). Changes physics
  for every map; bit-identity break; own arm.
* Finish-box padding: user rule 2026-08-25 - do not inflate so much,
  roughly 10% less than the current 64 u on button maps. CLAUDE.md 4b
  caveat stands: below pad ~32 some finishes have no standable point.
  Requires regenerated zones + v3 pool.

## 2. Ramps invisible in depth but visible in the 3D viewer (P0, $0)

Observed on gi_rino (traj_25268060160): ramps on the round wall render
in the 3D viewer, not in the POV depth.

Verified mechanics: the viewer mesh export renders **all brush
entities, every face** (tools/export_map.py, "brush entities (model
*N), all faces rendered"), while collision solidity is the explicit
class list (src/bsp.c:169 - func_wall/breakable/pushable/button/
train/conveyor/wall_toggle/rotating/door_rotating/door), and
func_illusionary is SOLID_NOT. The lidar SDF is sampled **through the
core's own solidity** (vision.py), so depth should equal collision
truth, not visual truth.

Two hypotheses:

* (a) the ramps are func_illusionary decoration: nonsolid, correctly
  invisible to depth - the VIEWER over-renders. Cosmetic, but it
  poisons human analysis of every traj.
* (b) GoldSrc clip-brush pairing: visual illusionary ramp + invisible
  CLIP collision. If the clip contents are not in the SDF, the agent
  collides with geometry it cannot see - unlearnable by vision.

Decides it ($0, local, pool is on disk): at the ramp coordinates, query
core point-solidity and cast one lidar ray; compare with the traj (did
the agent ever stand on them?). Then sweep all 108 maps with a script
diffing rendered-entity faces vs solid classes to bound how widespread
(a) is, and grep for maps where trajs collide where depth shows free
space to bound (b).

## 3. Is the depth actually used? (P1, $0)

Question: could the policy be running on scalars alone?

Verified context: without `--gps` (our config) the scalars are strictly
egocentric - velocity in the **local yaw frame** (fwd/left/up, /1000),
hspeed, flags, pitch, previous action deltas; absolute position AND
absolute heading (sin/cos yaw) are excluded (SCALAR_NOGPS drops
indices 7,8,12-14 of surfcore.h). So any map-specific behavior must
route through depth recognition - but that is an argument, not a
measurement.

Decides it ($0 on the local 5090, using ckpt_32e9 on a 5-map bundle):
eval the SAME checkpoint with (i) depth forced to 1.0 (no-hit), (ii)
depth frozen at the spawn frame, (iii) rows shuffled. map_pct collapse
under (i)-(iii) = depth is load-bearing; survival = the policy is a
scalar automaton and the vision work is wasted.

## 4. Time pressure, free death, and the broken telescope (P0 - the big one)

Observed (user + a second agent): deeper runs can score WORSE reward
than short safe ones; and "doesn't applying gamma break the telescopic
series?"

Verified, and the answer to the gamma question is YES:

* rewards.py: `r = (d_prev - d_cur)*scale - time_pen*every` - the
  shaping is the **undiscounted** potential difference F = Phi_next -
  Phi.
* Ng's invariance theorem needs F = gamma*Phi_next - Phi. With
  gamma < 1 the discounted sum of the undiscounted delta does not
  telescope; the residue is a per-step leak of about
  (1-gamma) * banked_potential.
* Magnitude: gamma per decision = 0.9995^4 ~ 0.998. At 88% of a map the
  banked potential is ~88 reward units (scale normalizes a full map to
  ~100), so the leak is ~0.176 per decision - **~9x the explicit time
  penalty** (0.005 * 4 = 0.02). The effective time pressure GROWS
  linearly with progress: a uniform mechanism for deep-map stalls and
  turn-backs (round 18's final-descent barrier arithmetic is a special
  case of it).
* Death is free and can even be transport: `r[ended] = 0`, fail_pen
  defaults to 0 (the launcher does not set it), and with respawn-frac
  0.9 a death relocates the env to a reservoir state - possibly deeper
  down the map. So yes: there exist regions where dying beats
  continuing, by construction.

Standard fixes, each its own arm on a bundle (~$0.25/arm):

* (a) Ng-conformant shaping `F = gamma*Phi_next - Phi` - the principled
  fix; bit-identity break with all history (same class as the known
  truncation-bootstrap bug - never a drive-by patch).
* (b) `--fail-pen > 0` (the flag exists) - prices death directly.
* (c) Arc-length coordinate (the proven finish-maker on cannonball;
  xSELF built champion-free lines from the policy's own episodes - we
  hold 1,958 trajectories across all 108 maps as pick_selfline input).

(a) vs control is the single highest-information cheap experiment in
this document.

## 5. Normals, or better slope visibility (P2)

Low-res depth makes shallow ramps hard to see; depth+normals is
classic. Feasible cheaply: the march already stops at an SDF surface -
central differences of the SDF at the hit point are 6 extra taps per
ray in the same triton kernel, giving (nx,ny,nz) channels with no new
bake. Alternative for $0 first: probe whether a trained net already
infers slope (sensitivity of actions to ramp-angle perturbation).
Experiment: +3 channels on a bundle, matched steps.

## 6. Model size, seriously (P1)

The grid says capacity helps at matched steps (9.38 / 11.30 / 12.26;
trigger subset +89%; cannonball 3x) and costs 1.9x wall per step. Open:

* matched-WALL 1x vs 4x - the decision-relevant comparison;
* LR scaling with width (the x10 collapse at 4x confirms
  bigger-needs-lower; sweep 3e-4 vs 1.5e-4 at 4x);
* the memorize-vs-generalize fear: hold out 8-10 maps from training and
  eval on them every arm - cheap to add to any bundle and it converts
  "generalization is bad" from a feeling into a curve.

## 7. Strafing needs state the policy does not have (P2)

Verified: obs = ONE depth frame + egocentric scalars; velocity is in
the local yaw frame; the only temporal signal is the previous action
delta pair (indices 10, 11). A strafe (oscillating yaw + lateral input
against air-accel) is representable in principle from (velocity,
previous action), but high-frequency alternation without memory means
learning a razor-thin vector field - plausibly why it never emerges.
Options, in effort order: action-chunk decoder (the H>0 codepath
exists in train_fast - sample_code/decoder - historically "did not
train", needs its own debugging arm), a small GRU (new arm class; note
frame-STACKING is validated dead on single-map, which is evidence
against naive history, not against recurrence), scripted-strafe action
prior (a surf-specific cheat worth one bundle arm).

## 8. Seeing past walls (P3)

Ramps hidden behind walls are invisible until commitment. Ideas:

* (i) second-surface depth: continue the march past the first hit - a
  second channel, "distance to what is BEHIND the first surface".
  Kernel change only, no new bake. The most direct x-ray.
* (ii) goal-field-at-hit channel: sample the geodesic field at each
  ray's hit point - "does what I see lead anywhere". Semantic, cheap,
  uses existing fields.
* (iii) route-lookahead scalars exist for single-map (the xROUTE
  machinery) but need per-map routes - pick_selfline output would
  serve double duty here.

Rank: (ii) then (i); both bundle-sized.

## 9. One policy or one per map? (settled)

**One united policy.** A single net, DDP-broadcast and checksummed
identical across ranks every iteration; all 108 slots share it; there
is no map-id input anywhere in the obs; with NOGPS there is no absolute
position or heading either. Any per-map specialization lives inside
geometry recognition from the depth image - which is exactly what
question 3 measures and question 6's held-out maps quantify.

## 10. The cost protocol these run under (user, 2026-08-25)

Ablations went from "a few dollars" to "dozens" because every arm
carried 108 maps and 2-4 GPUs. Standing protocol from here:

* **1x RTX 3090** (the standing research rule returns) - $0.13-0.20/h
  conforming boxes are plentiful; the 5-minute trial-readiness window
  applies to these rentals.
* **The bundle: 5 fixed maps**, one per difficulty band from the final
  108-map eval (>75%, 25-75%, 10-25%, 5-10%, <5%), chosen once and
  never changed so arms stay comparable; matched-step eval at 0.85e9
  exactly like round 25.
* An arm is then ~45-60 min, roughly **$0.15-0.25**, and a 4-arm
  question costs about a dollar.
* $0 tier first: questions 1, 2, 3 and the normals probe need only the
  local machine and the harvested artifacts - no rental at all.

Priority order: 4(a) shaping A/B and 1+2 map hygiene first; then 3 and
the question-6 matched-wall + held-out pair; then 5/7/8 by what those
change.
