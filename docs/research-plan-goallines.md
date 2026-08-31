# Goal-lines: teaching "the line marks the destination" as a general pattern

Status: PLAN (2026-08-29). Nothing below has run yet. Each experiment states
its expected effect BEFORE running; the ledger (`docs/research-results.md`)
gets the outcomes, and a mismatch between expectation and evidence is a
finding to dig into, not a failure to bury.

## 0. The claim under test

The shaping reward currently carries all of the "where to go" signal, and the
agent cannot observe it - `--obs-reward` feeds one scalar per timestep, which
in a memoryless policy contains no direction. The proposal is the
goal-conditioned decomposition:

* **WHERE lives in the observation**: a line, extracted per map from the
  baked geodesic field, exposed as the ego-frame lookahead fan
  (`surfgym/route.py`, the observation every superhuman racing agent has).
* **Reward pays for advancing ALONG that same line and never charges the
  detour**: the `--race-arc` corridor reward (arclen branch) - signed arc
  delta inside the corridor, zero with a frozen anchor outside it.
* **The pattern to be learned is general**: "go where the visible line
  points; the path need not follow it." If that pattern is actually learned
  (rather than the map memorized), it must transfer to lines the agent has
  never seen - new maps, truncated goals.

The falsifiable core: **an agent trained with the fan on several maps should
make progress on a HELD-OUT map's line, and an identical agent trained
without the fan should not.** Everything else in this plan is the ladder
that gets us to that test cheaply and tells us where it broke if it fails.

Why the current setup forbids this pattern: potential shaping charges every
step that raises geodesic d. From route vertex 1601 on cannonball the
champion's own path RAISES d by 8,408 u (-4.24 reward), which is why all 234
control episodes turn back exactly there. A detour ("the stairs") can never
look good under a potential defined on the line itself. The corridor-frozen
arc reward removes the charge; the fan supplies the direction the scalar
potential never could.

## 1. Design primitives (what exists, what is new)

| piece | status | where |
|---|---|---|
| ego-frame lookahead fan, speed-scaled, per-point normalized | EXISTS | `python/surfgym/route.py` (`RouteLine`) |
| warm resume onto a pre-route checkpoint, bit-identical start | EXISTS | `train_fast.py widen_for_route` (zero-pad trailing columns) |
| corridor arc reward, potential inside / frozen outside | EXISTS (arclen branch) | `--race-arc`, `tests/python/test_race_arc.py` |
| arc farmability diagnostics (`arc_gain`, `arc_off_frac`) | EXISTS (arclen branch) | `RaceReward.pop_stats` |
| line FROM THE FIELD (no champion, no demo, no self-episodes) | **NEW** | `tools/build_route.py --from-field` (E0) |
| per-map lines in fleet mode | **NEW** | `mapfleet.py` per-slot `RouteLine` (E3) |
| eval-side obs ablation for the fan | **NEW** | `record_ckpt.py --route-mode off/frozen/shuffle`, mirroring `--depth-mode` (E2) |

The field-descent line is the keystone. A greedy strictly-descending walk on
the goal field's voxel graph from any reachable seed MUST terminate at the
goal set - BFS distance guarantees a strictly closer neighbor at every
non-goal voxel, so unlike `eval_progress` (which has its reachable minimum at
the wall) the TRACE cannot stop at vertex 1601; it crosses the wall region by
the known 8,700 u level glide over the void. That glide is a lie about HOW
(nothing can fly it) but under this design the line only claims WHERE, which
is exactly the xAUTO lesson: a line with 24.8% of vertices inside solid and
1,131 u max deviation still supplied the ordering and finished the map.

## 2. The experiment ladder

Ordering principle: each rung is the cheapest experiment that can kill the
design, and each rung's failure has named suspects to dig into. E0 is free.
E1 is the load-bearing one-hour arm. E3 is the actual claim; E5 reaches the
same claim through goal diversity on one map, needs no fleet plumbing, and
can run before E3.

---

### E0 - field-descent line geometry (local, no training, ~zero cost)

**Do:** add `--from-field` to `tools/build_route.py`: seed at the map spawn,
walk steepest-descent on the baked goal field lattice (strictly decreasing d,
26-neighborhood), resample at 128 u via `resample_polyline`, save
`<map>.fieldroute.npz`. Deterministic tie-breaking so the line is
reproducible. Run it on cannonball + every pool map that has a baked field.

**Measure:**
1. Termination: fraction of traces reaching d <= 2 cells. Guaranteed on the
   lattice; verify the implementation honors the guarantee.
2. On cannonball: max/mean deviation from the champion route, % vertices
   inside solid (SDF < 0), and the wall crossing (expect the level glide).
3. Per pool map: line length vs spawn d0 (ratio >> 1 flags a pathological
   detour), % in solid.

**Expected:** 100% termination; cannonball deviation bounded (< ~1,500 u)
everywhere except the void crossing; in-solid fraction in the same class as
xAUTO's 24.8%. **If a trace does not terminate** the tie-break or sentinel
handling is buggy - fix before anything trains. **If deviation or in-solid
far exceeds xAUTO's line**, note it: it becomes the prime suspect if E1
nulls.

---

### E1 - xFARC: the field line as REWARD ordering (warm resume, 1 h, 3090)

**The load-bearing rung.** xARC/xAUTO/xSELF proved the reference line
supplies the ordering, not the path - but every line so far came from a
recording (champion, decimated champion, the policy's own episodes). If a
bake-only line works, every map in the pool gets a line for free, which is
what E3 and the multi-map plan need.

**Do:** branch from the current integration branch, merge/cherry-pick
arclen's `--race-arc`, run the standard warm resume of
`runs/sOBSR2/ckpt_latest.pt` via `tools/run_arm.sh` with `--race-arc` on the
E0 cannonball line. Same card class (3090), same launcher, one seed, one
hour. Direct comparators on this exact checkpoint and card already exist:

| arm | line source | finishes | corridor MAX |
|---|---|---|---|
| xARC | champion recording | 63/102 | 231,680 |
| xAUTO | champion, 58 chords | 62/102 | 231,680 |
| xSELF | own episodes, stops 1,280 u short | 47/102 | 231,680 |
| controls (4 arms) | - | 0/333 | 205,312-205,440 |

**Expected:** finishes > 0 and corridor MAX ~231,680, landing between xSELF
and xAUTO - the line is full-length (unlike xSELF) but geometrically worse
(void glide, wall-hugging). Even 10/102 finishes proves the point: the
ordering came from the bake alone.

**If 0 finishes, dig in this order:**
1. `arc_off_frac` high near the wall -> the corridor around the void-glide
   segment excludes the flyable path (the real line around the void may sit
   further from the trace than the corridor radius). Suspect first; remedy:
   widen the corridor on that segment or measure deviation of xARC's
   finishing trajectories from the field line.
2. `arc_gain` collapsing early -> in-solid vertices break the nearest-point
   projection; compare against the E0 in-solid number and xAUTO's 24.8%.
3. Neither -> the recording-derived lines carried something beyond ordering
   after all; that contradicts the xAUTO conclusion and is worth its own
   ledger entry.

---

### E2 - xFOBS: the fan observation, and whether the policy reads it
(warm resume, 1 h, 3090)

**Do:** E1's config + `--route <fieldroute>` (fan features from the SAME
line the reward pays along). `widen_for_route` zero-pads the checkpoint, so
step 0 is bit-identical to E1's step 0 - verify locally. Also add
`--route-mode off/frozen/shuffle` to `record_ckpt.py` (the `--depth-mode`
idiom) and probe both E1's and E2's final checkpoints.

**Expected:** on a single already-mastered map, E2 vs E1 lands INSIDE the
27% seed-noise floor - the net can solve one map without the fan, so "no
harm" is the pass criterion, not "improvement". The probe is the real
measurement: E2's checkpoint should degrade under `--route-mode shuffle`
somewhat (it has no reason to fully ignore a free feature), E1's checkpoint
is unaffected by construction. **If E2 is a LARGE regression**, the fan's
normalization/scale is fighting the trunk (check feature saturation against
`clamp=4.0`) - fix before E3, because E3 is meaningless with a broken fan.

---

### E3 - xGEN: the generalization claim (from scratch, multi-map, 2 arms x 1 h)

**The actual hypothesis test.** One pattern - "advance along the visible
line" - learned on training maps, applied on a held-out map.

**Do (engineering first):** per-slot `RouteLine` in `mapfleet.py` (each map
slot gets its own field line; the featurizer is already per-batch, the
plumbing is the new part). Then two from-scratch arms per the hyperparameter
baseline (64x32 depth, no `--obs-reward`, `--act-every 4`, SCRATCH launcher),
on 6-8 pool maps with baked fields and E0 lines:

* **xGEN-F**: fan + `--race-arc` per map.
* **xGEN-C**: identical minus the fan (`route_dim 0`). Same reward. The ONLY
  difference is whether the line is visible.

**Eval:** zero-shot on 2-3 held-out maps (fields and lines baked, never
trained on), plus the `--route-mode` probe on xGEN-F.

**Metric:** fraction of the held-out line's arc reached (greedy, corridor
scoring via `eval_honesty.py --order-only 16`), and on training maps
time-to-arc-threshold (step at which each arc gate is first cleared), NOT
end-of-run `eval_progress` means - the gate-ladder retraction applies with
full force to scratch arms.

**Expected:** xGEN-F reaches a nonzero arc fraction on held-out maps and
xGEN-C reaches roughly spawn-neighborhood arc only; the shuffle probe
collapses xGEN-F's held-out performance (proving the line is READ, not
coincidental). **The honest null:** per the standing rule, scratch curves
may not separate inside one hour - "not separated yet" is a legitimate
result to report, and the fallback is comparing time-to-first-arc-gate on
TRAINING maps, which separates earlier than transfer does.

**If xGEN-F transfers no better than xGEN-C:** the suspects, in order:
(a) the probe says the fan is ignored even on training maps -> too few maps
to force reliance, increase map count before concluding anything (goal
diversity is the HER-side requirement); (b) the fan is read on training
maps but transfer fails -> the pattern learned is map-entangled; try
`--route-critic-only 0` vs 1 and stronger map count; (c) reward-side
transfer works (arc gained on held-out under xGEN-C too) -> the depth
image alone already localizes the track and the fan is redundant - an
interesting result in itself.

---

### E4 - xTRUNC: truncated goals and relabeling (DESIGN ONLY until E3 reads)

The HER-flavored extension, cheap with ONE baked field: a goal is "reach
field value <= d_g", and the observation line is the descent trace truncated
where d < d_g. No per-goal bake needed.

* **Curriculum:** sample d_g per episode from trivial (just below spawn d,
  the "curb nearby" that grounds the symbol) toward 0 (the map goal).
  Per the sweep (section 5), prefer the Florensa-shaped rule - d_g chosen
  where the current policy succeeds 10-90% of the time - over a fixed anneal
  schedule, which can outrun or lag the policy.
* **Relabel:** a failed episode that reached min d = m becomes a SUCCESS for
  the goal d_g = m; recompute the truncated-line features offline and train
  on the relabeled transitions. **Relabeled goals pay SPARSE success only**
  (HER's own ablation: dense shaping on relabeled goals hurts goal-reaching)
  - the dense arc reward stays tied to the full line and is never recomputed
  against a truncated one.

**Expected (when run):** faster grounding of the fan on fresh maps (probe
sensitivity appears earlier in training than without relabeling).

**Confounds to keep out, per the ledger:** d_g annealing overlaps with
respawn-margin harvesting (both shrink the effective start-goal gap) - one
treatment at a time; and any success-paying mechanism near the goal is the
trivial-win trap that fired on xPSSR - report win_rate ONLY next to
reservoir min-depth.

---

### E5 - xPOINT: arbitrary point goals on ONE map (goal diversity instead
of map diversity)

E3 forces the net to read the line by varying the MAP; this forces the same
thing by varying the GOAL on a single map, which is cheaper (no fleet
plumbing, no multi-map bakes) and purer: the map is constant, so the line
observation is the ONLY thing that predicts where reward lives. It is the
standard goal-conditioned move (HER's random per-episode goals) applied
here, and it can run BEFORE E3 - if it works, E3's fan arm starts from a
policy that already treats the line as the task. A mid-air goal is scored
the way type-2/3 buttons already are: arrival inside a padded box
(`surf_set_goal_box` + AABB test), pass-through counts.

**The generation problem ("hard to automatically generate them") has three
answers, in order of trust:**

1. **Truncate the descent line** (= E4's goals). Free, guaranteed sensible,
   but the goal family is one 1D curve - weakest diversity.
2. **Harvest visited states - do not generate.** Any state the agent has
   actually reached (reservoir, recorded episodes) is PHYSICALLY reachable
   by construction, difficulty is measurable from visit statistics, and
   flight states give exactly the mid-air goals wanted here. This is the
   Go-Explore/HER logic the reservoir already implements for respawns,
   pointed at goals instead of starts.
3. **Sample free voxels + a per-goal A\* line.** A single-pair A* on the
   baked occupancy lattice costs seconds (vs minutes-hours for a full
   field bake), so arbitrary goals get a line without a per-goal bake.
   CAVEAT: finite field distance is NECESSARY, not SUFFICIENT - the BFS
   believes free lateral flight (the void glide), so a sampled voxel can be
   physically unreachable. Budget a timeout/skip rule, or filter candidates
   to within ~512 u of some harvested state (generator 2 as the validator).

**Reward:** dense arc along that episode's line + sparse success at the
goal box. Relabeled goals (a failed episode's reached state) pay SPARSE
success only, per section 5's HER finding. Line per goal: generator 1 uses
the truncated trace; generators 2-3 use the A* polyline (generator 2 may
instead use the harvested episode's own recorded path - policy information,
xSELF-adjacent, acceptable).

**Curriculum:** per the user's framing and Florensa's rule - begin with
goals the current policy reaches 10-90% of the time (near spawn / near the
visited manifold), widen outward as the band moves. Distance-to-goal is the
difficulty knob; the success band is the controller, not a fixed schedule.

**Engineering gates (before renting):** per-env lines - `RouteLine` today
is one line shared by all envs, so E5 needs a batched multi-line featurizer
(pad to max length, per-env index) and a per-env goal box; both are local
shape-test items.

**Expected:** the `--route-mode` shuffle probe turns positive (policy reads
the line) EARLIER in training than in E3, because goal diversity per sample
is denser than map diversity; held-out test = goals in regions never used
for training goals. **If the probe stays negative here too**, the fan
representation itself is the suspect (section 6's alternatives, starting
with the tri-plane minimap), because E5 removes every excuse E3 leaves -
one map, one physics, nothing to transfer.

**Guards:** win_rate is meaningless without the goal-distance distribution
next to it (trivial-win trap, xPSSR); success-band selection self-corrects
but REPORT the band's position over time. One treatment at a time versus
respawn-margin - goal curriculum and start curriculum both shrink the
start-goal gap and must not run together in one arm.

---

## 3. Metrics discipline (unchanged, restated because every rung uses it)

* Verdicts come from `eval_honesty.py` corridor MAX + finishes (+
  `--order-only 16`), never `race/eval_progress` alone - it has been
  anti-correlated with truth once and blind twice.
* Scratch arms: report which gate and the STEP it was cleared, prefer the
  matched-step early point (525M-class); end-of-run differences under the
  27% floor are noise by definition.
* Same card class per comparison. One seed. One hour.
* `arc_gain` / `arc_off_frac` from `pop_stats` in every arc arm's report -
  they are the farm detector.

## 4. Local gates before renting anything

1. E0 termination + determinism test (new, `tests/python`), runs in CI.
2. `test_race_arc.py` green with a `.fieldroute.npz` file.
3. `widen_for_route` bit-identity: E2 step-0 logits identical to E1 step-0
   on a few hundred local steps.
4. Fleet plumbing (E3): per-slot fan features shape/frame test + a few
   hundred local multi-map steps before any box is rented.

## 5. Prior art (literature sweep, 2026-08-29)

Sweep ran over goal-conditioned RL with path observations, HER variants,
distance-field subgoals, and progress-vs-tracking rewards. Curated to what
changes THIS plan; racing lookahead (Sophy/Linesight/Fuchs/Swift) is already
in `research-litsurvey.md` section 0. Items below marked (unverified) came
back with numbers the sweep could not pin to the paper - treat as leads, not
constants.

**Directly load-bearing:**

* **HER (Andrychowicz et al., NeurIPS 2017): their own ablation found SHAPED
  reward HURTS goal-reaching vs sparse success.** Shaping biases the policy
  toward local progress that does not compose into reaching the goal. Two
  consequences for us: (1) it independently predicts the vertex-1601
  turn-back we already measured; (2) **E4's relabeled goals must pay SPARSE
  success at d <= d_g, not dense progress along the truncated line** - the
  dense arc reward stays tied to the FULL line only. Relabel strategies:
  "future" states within the episode worked best in follow-ups; "final"
  state is the simplest and what E4 specifies (goal = min d reached).
* **Progress > tracking (Song & Scaramuzza, drone racing, Sci. Robotics
  2023): rewarding task progress beats rewarding reference-line tracking
  outright** under model uncertainty (their line-tracking baseline fails
  where gate-progress succeeds; exact figures unverified). Confirms the core
  split: the line is WHERE (observation + progress coordinate), never HOW
  (no tracking term, no off-line penalty). If anyone proposes adding an
  off-corridor penalty to `--race-arc`, this is the citation against it.
* **Ego-frame continuous waypoint encodings beat discrete ones** (waypoint
  models for instruction-guided navigation, ICLR 2022: continuous
  heading+distance outperformed discretized by ~3-8%). `RouteLine`'s
  continuous ego-frame fan is the right representation class; no change.
* **Florensa reverse curriculum (ICML 2018) selects starts by SUCCESS
  PROBABILITY band [0.1, 0.9], not by distance.** No published ablation of
  explicit distance-threshold annealing vs success-band selection exists -
  so E4's d_g schedule has two candidate rules: fixed anneal (simple,
  confounded with learning speed) and success-band (d_g chosen so the
  current policy succeeds 10-90% of the time; self-paced, Florensa-shaped).
  Prefer the success-band rule if E4 runs: it cannot outrun or lag the
  policy, which the fixed anneal can.

**Supporting / contrarian:**

* SoRB (Eysenbach 2019) and functional-distance planning (ICLR 2021) both
  use a LEARNED value as the distance metric and report that hand-crafted
  metrics lose to it. Our field is pre-baked and lies about free flight; the
  design absorbs that by demoting the field to observation + monotone
  coordinate, but if E3 stalls on line quality, learned-distance is the
  literature's next step.
* FeUdal Networks is the contrarian on representation (latent direction
  subgoals, geometry suppressed); every system in the racing/navigation
  cluster exposes explicit geometry instead. We follow the cluster.
* Quasimetric RL formalizes forward/backward asymmetry of goal distances; a
  surf ramp is exactly a quasimetric edge. No action now - the corridor
  reward is already asymmetric (forward pays, backward charges, off-line
  freezes) - but it is the frame if reward asymmetry ever needs its own arm.

**Confirmed gaps (nobody has published this combination):** pre-baked
descent-line observation + monotone corridor reward + intermediate-goal
relabeling on the same field. E1/E3/E4 are not reinventions; the closest
prior art is Linesight's self-bootstrapped centerline, which is xSELF's
territory, already run.

## 6. How to EXPOSE the line: representation candidates (sweep 2, 2026-08-29)

A second sweep asked specifically how embodied/3D agents are told "where to
go" (Habitat PointNav, CARLA route conditioning, BEV/minimap rasters,
first-person overlays, FOV-loss studies). Headline: **no paper compares a
vector lookahead fan against a route rendered into first-person pixels** -
the comparison E3 would run is not in the literature. Numbers below came
from a fast sweep; verify against the paper before quoting any in the
ledger.

Three candidate representations, all fed from the SAME field line:

| | encoding | FOV-safe | binds to geometry | prior support |
|---|---|---|---|---|
| **fan** (baseline) | 27 scalars, ego-frame lookahead points | YES | no | universal in racing (litsurvey s.0) |
| **trail** (first-person channel) | line points splatted into the 64x32 view, intensity ramped by arc distance, off-screen points clamped to the border | no - blank when facing away, and the policy is memoryless | YES | game racing lines; no RL ablation found |
| **minimap** (top-down channel) | agent-centered, yaw-aligned top-down raster (~32x32) around the agent: route line + arc ramp, optionally occupancy from the existing slabocc bake; route z-delta as a second intensity | YES | 2D only (surf is very 3D - the vertical axis is squashed) | the strongest-supported raster in the sweep: EgoMap-style egocentric grids beat implicit/vector memory for long-horizon navigation; ChauffeurNet/TransFuser-class driving stacks all rasterize the route into a BEV channel |
| **tri-plane minimap** (3 channels) | three agent-centered ORTHOGRAPHIC projections, yaw-aligned: top-down (forward/left), forward-elevation (forward/up), side-elevation (left/up), each ~32x32 with route splat + arc ramp + an occupancy slice from the slabocc bake | YES | partial in each plane, full across the three | the standard tri-plane factorization of a 3D volume (EG3D-style); resolves the BEV objection - the two vertical planes carry exactly the climb/descent structure a top-down raster deletes, which on this task is the whole game (ramps, the final descent) |
| **command token** (discrete) | the line tangent over the next ~500 u quantized to 8-16 maneuver classes (yaw-delta x pitch-delta), one-hot or embedded | YES | no | Codevilla CIL: discrete commands beat a continuous goal vector for driving; in-repo: xLATCH (below) |

Sweep findings that bear on the choice:

* **Egocentric rasterized spatial maps generalize better than pure vector
  encodings** in navigation (EgoMap; BEV fusion in driving). This is the
  literature's vote FOR a raster arm in E3, and specifically for the
  minimap variant, which keeps the raster's binding without the trail's
  FOV failure.
* **A visual goal cue that can leave the field of view is a measured
  failure mode** (limited-FOV target navigation work: large drop when the
  target leaves view; attention/memory recovers most of it). We have no
  memory, so the trail channel MUST keep its border-clamp trick, and the
  fan or minimap should accompany it in any serious arm.
* **Codevilla CIL: discrete high-level commands beat a continuous goal
  vector for driving.** Probably about topological forks (turn left/right),
  not fine geometry, so not directly actionable here - but it is a warning
  that the obvious continuous encoding is not always the winner, and a
  cheap discrete variant (quantized descent direction) exists if the fan
  underperforms.
* Rasterizing VECTOR map predictions as an auxiliary loss helped driving
  stacks (MapVR) - if a raster channel wins E3, a cheaper hybrid is a
  fan observation with an auxiliary raster prediction head, not a raster
  input.
* **In-repo evidence for DISCRETIZATION: xLATCH (round 19, ledger).** The
  wall's local minimum is continuous reward arithmetic (-4.02 to climb the
  valley), and the arm that beat it with no reference line of any kind was
  a BINARY structure: once an episode first reaches d <= 6,996 the shaping
  switches OFF for the rest of the episode - 52/102 finishes, best 80.35 s,
  vs 0 finishes for the continuous floored potential (xCLAMP) that flattened
  the same region without the switch. And because a latch is episode
  history, the latch BIT had to be exposed as an observation feature (obs
  column 15, zero-padded via `widen_for_route`) - which makes xLATCH also
  the in-repo precedent for a discrete goal-related observation input. This
  is the same shape as HER's sparse-beats-shaped ablation (section 5): at
  the decision point that matters, a discrete signal was learnable where the
  continuous gradient argued for turning back. The command-token candidate
  above is this observation generalized from one bit to a maneuver
  vocabulary.

**Consequence for E3:** the treatment axis is representation, same line,
same reward: fan / trail / tri-plane minimap / command token vs the no-line
control, each with the `--route-mode` shuffle probe. Budget says run fan
first (proven format, cheapest), then ONE raster arm - **tri-plane minimap**
over trail (FOV-safe, keeps the vertical structure, carries the sweep's
generalization evidence) and over flat BEV (surf lives in the vertical
planes) - unless the trail's first-person geometry-binding is the specific
hypothesis under test. The command token is the fallback if the continuous
fan probes as unread, not a first-line arm.

## 7. Execution status (2026-08-31)

E0 executed and scored (ledger round 28): termination MET, in-solid
EXCEEDED, deviation PARTIALLY MISSED (25.3% beyond xAUTO's max).
E1/E2 ran as a from-scratch 2x2 on the local 5090 per the user's
directive (xsCTL / xsFARC / xsFAN / xsFULL) - full numbers and the
expectation-vs-evidence scoring in the ledger. Standouts: the field
line carries the ordering at matched steps; xsFAN (fan obs + geodesic
reward) posted 122,752u corridor MAX = 1.57x control but INSIDE the
identical-run spread, so it is a direction pending E3/E5; xsFULL
exposed the off-corridor plateau failure mode (86.8% off, intrinsic
exhausted, no gradient back to the line). Engineering before any rented
arc arm: numba-fuse ArcProgress.advance (~30% fps tax measured).
