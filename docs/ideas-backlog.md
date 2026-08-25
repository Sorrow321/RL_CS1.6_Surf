# Ideas backlog

Working list, ordered by what I would spend the next box on. One idea per
run; move an item into `research-results.md` when it has a verdict, and
leave a one-line pointer here.

Two open problems drive everything below:

* **P1 - convergence.** A scratch agent gets stuck at walls. The walls are
  not exploration failures; they are places where the shaping's potential
  has an interior minimum, because free-space distance and *feasible-route*
  progress disagree there.
* **P2 - the timer.** 78.70 s against a 68 s human record. Reward scaling
  is exhausted; the rest is line geometry.

---

## Closed - do not spend another box on these

| idea | verdict | why |
|---|---|---|
| Binary / sparse reward from scratch | dead | At gamma 0.9995 a terminal +50 is worth **0.925** after 8,000 ticks. V(race) = -8.89 vs V(quit) = 0 beyond 35.8 s from the finish. `xNOSHP` collapsed to 0/9 in one eval; `xBIN3` reached 2.4% of the map in 2e9 steps. Dense positive income is what carries value backwards through the discount - this is arithmetic, not luck. |
| Bigger `time_pen` | dead above ~0.0125 | Break-even speed = `time_pen / scale`. At 0.015 it exceeds the policy's mean speed, at 0.020 it exceeds the champion's *peak*. Both collapsed to zero finishes. **0.010 is the optimum** and is worth keeping (78.70 s vs 79.78 s). |
| Shorter horizon (gamma) | null | The shaping is a telescoping potential with a **fixed** total (96.47), so gamma scales the lap's value and the value of a second saved together. Measured pressure 2.26 / 2.18 / 2.16 % per second at 20 / 10 / 5 s. `xHZ999`: best 79.50 s, pooled mean 80.18 s - variance, not time. |
| Linesight 7 s hard mini-race | regression | Frontier went backwards 115,328 u. The constant is a transplant: 7 s is 27% of their median lap and 8% of ours. `xHZ999` proved the fault is the hard window, not the short horizon - corridor MAX held at 100% under a plain gamma change. |
| Gravity-directional goal field | dead | Baked and gated: makes the barrier **14% deeper**, `>=` the old field at all 1,811 route vertices. The deception is lateral free flight through open air, not one-way climbs, and `gravity_dir` gates only `dz > 0` edges. |
| RND, frame stacking, BC warm start, bigger nets, per-episode time bonuses, strided rollouts | dead | See `research-litsurvey.md` section 8. |
| `explore_phase1.py` as a route source | not as it stands | Froze at **6.67% of arc** in both configurations. The ledger's "92.4% of the track" was geodesic progress on a non-injective field. |

---

## 0. MEASURED 2026-08-23: no transfer at all, and the timer ceiling is provable

**Transfer (free, no rental).** The cannonball finisher, recorded on the
unseen `surf_petrus_lite`: **0.7-0.9% of the map covered, peak speed
197-290 u/s** across 6 greedy episodes, all dead in 4.7-7.0 s. On
cannonball the same weights hold ~2,900 u/s and peak ~4,000. Peak 290 is
barely above walk speed (250) - it never catches a ramp. Not a map
artifact: the spawn is at z=592 with terrain below, and the policy simply
wanders a ~100 u patch and falls off. **There is no generalised surfing
skill; the policy memorised one map.** Idea 4 is therefore not optional
for the 1000-map goal, it IS the goal.

**The timer ceiling is a theorem, not a tuning problem.** Potential-based
shaping is time-blind by construction: `sum(Phi(s') - Phi(s)) =
Phi(end) - Phi(start)`, path- AND time-independent for ANY potential. So
no choice of potential rewards speed - which retires idea 1 below as
originally written (Necto's `exp(-d/v_max)` is a reshaping of distance,
not a change of units). Speed pressure can only come from non-potential
terms, and it is capped by **racing must beat quitting**: shaping income
(96.47/8,113 = 0.0119/tick) must exceed `time_pen`. That is exactly the
measured 0.0125 cliff, and scaling shaping down forces `time_pen` down
with it, so the trap is conserved.

**`--fail-pen` is the missing degree of freedom** - it moves the
constraint rather than trading inside it, and unlike the +50 bonus
(discounted to 0.925 over 8,000 ticks) a death penalty is paid
immediately. Running as `xFPEN` (`--fail-pen 50 --time-pen 0.020`).

---

## 1. Time-to-go potential instead of distance-to-go  (P2 - RETIRED, see 0)

**The idea.** `Phi = -(d / v)` rather than `Phi = -d`. The per-step reward
then equals the **time saved**, not the distance covered.

**Why.** Our own survey already contains this and we never used it - Necto /
RLGym: *"Distance potentials are `exp(-d/max_speed)` = time-to-go, not
distance-to-go."* It is the structural fix for P2: today the collectible
total is fixed at 96.47 regardless of route or speed, so a longer-but-faster
line is punished and every speed knob divides into the same fixed pie.
Under time-to-go, accelerating raises `Phi` directly and a second saved is
worth the same everywhere on the map.

**Cost.** ~5 lines in `RaceReward`. Needs a `v` floor so `d/v` cannot blow
up at rest, and the collectible total is no longer normalised to 100, so the
`time_pen` / `success_bonus` scales need re-deriving (the break-even
arithmetic in the closed table is the template).

**Test.** From the 78.70 s checkpoint, one arm, measure pooled mean finish
time of all finishers (not best - `xHZ999` showed best-of moves on variance
alone).

**Risk.** It does NOT fix P1: `d/v` is still monotone in `d`, so the
interior minima survive. This is a timer fix only.

---

## 2. Self-bootstrapping frontier ratchet  (P1 + P2, and it is the one I would run first)

**The idea.** Shape on arc length along a reference line extracted from the
agent's **own best episodes**, and **re-extract it every N steps** so the
reference ratchets forward as the policy improves.

**Why.** Both fixes that have worked were the same fix - make the coordinate
monotone along something the agent can actually traverse. `xSELF` proved one
iteration: a line built only from the stuck policy's own *non-finishing*
runs, stopping 1,280 u short of the wall, still finished 47/102. Nobody has
run the loop twice. Running it *continuously from scratch* is what removes
walls **during** training instead of patching the last one, which is the
objection to the latch being an endgame hack.

**Literature.** Linesight re-extract their reference line from the AI's own
previous runs as records fall. It is also a landmark graph degenerated to a
single path (see 3).

**Cost.** Medium. `--race-arc` exists (branch `arclen`), `tools/build_route.py`
and `tools/pick_selfline.py` exist, and the champion-free trim rule (last
tick the map pushed back, read off the departure of vertical acceleration
from the gravity step) exists and cuts five independent episodes within
25 u of each other. The new part is doing it *in the trainer* on a schedule.

**Test.** Scratch run, ratchet on, against the geodesic-from-scratch
baseline. Metric: does it wall at 88%, and how many steps to first finish.

**Watch for.** Self-reinforcement - the reference is built from the policy,
which is trained on the reference. Keep the corridor wide (1,500 u worked)
and never pay for *distance to* the line, only progress *along* it
(Song & Scaramuzza: penalising distance-to-line gives 0% success under a
realistic model).

---

## 3. Reachability metric from a landmark graph  (P1, the general answer)

**The idea.** Replace the static voxel geodesic with a graph over states the
policy has actually visited, where an edge exists only if the policy got
from A to B. Distance = hop count, i.e. **reachability**, monotone along
feasible routes by construction.

**Why.** Every wall we have hit is one bug: free-space distance disagrees
with feasible progress. A metric built from what actually happened cannot
disagree with itself. It also improves as the policy improves, which a
static field cannot.

**Literature.** SPTM (Savinov 2018), **SoRB** (Eysenbach 2019 - distances
from a goal-conditioned value function, planned over a graph of replay
states), Mapping State Space using Landmarks (Huang 2019), L3P / *World
Model as a Graph* (Zhang 2021), DHRL (2022).

**Cost.** Large. This is the principled version of 2 and should follow it.

---

## 4. Multi-map training  (P1, user idea)

**The idea.** Train on ~10 maps simultaneously so the agent learns *surfing*
rather than *this map's ramps*. A generalised skill shrinks the search
space to pathing.

**Why the premise is right.** Measured, not speculated: the ledger's
"GENERALIZATION TEST (2026-08-21): the champion has not learned to surf" -
the champion transfers badly to unseen geometry. And the observation is
already honest-perception (no absolute position, no heading), so nothing
*forces* memorisation except that there is only one map to memorise.

**Literature.** Procgen (Cobbe 2020): level-generalisation needs hundreds to
thousands of levels; **10 maps will most likely be memorised as 10 maps**.
But the *mechanic* here is identical on every map - the physics is one
control law - so skill generalisation may need far fewer than level
generalisation does. The Sonic benchmark datapoint in our survey is the
encouraging one: joint pretraining + per-map fine-tuning roughly **doubles**
low-budget scores. Linesight's warning is the discouraging one:
multi-map generalisation was *brutally* expensive versus single-map records.

**Cost.** Real engineering: per-map SDF + goal field + zones, and the env
needs per-env map assignment or per-run rotation. We have 3 maps today
(`surf_src_cannonball`, `surf_ski_2`, `surf_petrus_lite`).

**How I would de-risk it.** Do not start at 10. Start at 2-3 with a
**held-out map** and measure transfer, because the whole claim is about
transfer and it is cheap to falsify. If 3 maps do not improve held-out
performance at all, 10 will not either.

### BUILT 2026-08-23: `--maps` - joint training over several maps

`python\train_fast.py --maps a.bsp,b.bsp` trains **one shared policy** on
several maps at once. `--envs` is split evenly, one `SurfCore` per map, and
the PPO update is untouched - it just sees a batch whose rows come from
different maps. Branch `multimap`; the machinery is
`python/surfgym/mapfleet.py` (`MapSlot` + `MapFleet`), tests in
`tests/python/test_multimap.py`.

**A slot is everything map-shaped plus the env range it owns:** bsp, core,
lidar/SDF, goal field, reward function, respawn reservoir, spawn pool,
novelty count table, `d0`, goal box, `map_center`, eval core, env
`[lo, hi)`.

**What that buys, concretely.**

* **`scale = 100/d0` is per map**, so finishing *any* map is worth 100:
  cannonball's `d0` is 198,380 u and petrus_lite's 35,637 u, and one shared
  scale would have made petrus's entire race worth **18 points**.
* **`--race-latch-frac`** replaces the absolute `--race-latch` on multi-map
  runs: a fraction of *each map's own* `d0`. The arm's 6,996 u is 3.53% of
  cannonball and **19.6%** of petrus - the same number is two different
  treatments. Absolute `--race-latch` and `--race-dfloor` are now refused
  with more than one map rather than silently meaning two things.
* **Per-map eval, logging and dashboard.** One eval core per map;
  `race/eval_progress.<map>`, `race/eval_finish_s.<map>`,
  `race/eval_finishes.<map>` are appended to `CSV_COLS` (so an old
  `progress.csv` header is still a prefix and still migrates), recordings
  land as `traj_<step>_<map>.jsonl`, and the dashboard sorts each map's
  series next to its pooled metric.
* **The checkpoint records the map list** (`config["maps"]`,
  `config["map_cells"]`, `config["race_latch_frac"]`) and a resume restores
  it; the novelty counts and the respawn reservoirs are checkpointed **per
  map** (reservoir states are raw map coordinates - one map's are garbage in
  another). `tools/record_ckpt.py` knows the new keys and re-derives the
  latch distance for whichever map you record.

**Evidence it did not break the single-map path** (the thing every arm in
flight depends on): a one-slot fleet is driven beside a hand-rolled
reference core for 300 ticks in two regimes (stall-kill deaths, and
truncations) and compared on obs, base rewards, done, trunc, terminal obs,
the shaped reward and the torch RNG cursor - all identical. End to end, the
cannonball finisher (`runs/research/xLAT3/xLAT3_final.pt`) resumed at
`--lr 0` produced a **byte-identical** greedy recording under `--map` and
under `--maps cannonball,petrus_lite` (md5
`e62167967d1717e91eb8613bc38214e1`, 333,828 bytes both ways).

**First joint numbers (local 5090 smoke, not an experiment).** The same
weights, one process, both cores stepping: cannonball 41,028 u of 198,380
and petrus_lite 315 u of 35,637 (0.88%, peak 379 u/s) - i.e. the plumbing
reproduces the no-transfer measurement in item 0 rather than hiding it.

### What is still missing before this can test GENERALISATION

**The scientific limit, plainly: with only two goal-maps this measures
whether joint training WORKS, not whether it GENERALISES.** Training on
cannonball + petrus and then evaluating on cannonball + petrus can only
tell us that one network can hold two maps at once (and what it costs the
single-map score). Generalisation is a claim about a map the weights have
never seen, and it needs a **third map held out of training** - trained on
A+B, evaluated on C, against a control trained on A alone. Without C, a
rise in the petrus number is memorisation of petrus, which is exactly the
result item 0 already has for cannonball.

**What it takes to get C.** A held-out map needs three things, none of them
free:

1. **A BSP with a labelled finish.** `surfgym/zones.py` auto-extracts start
   and end from timer triggers; that worked for both current maps. A map
   without them needs a hand-written `maps/<map>.zones.json`
   (`"source": "manual"`), which is ~10 minutes with the viewer.
   `surf_ski_2` did **not** qualify on this branch - it is freestyle and
   auto-extraction finds no end zone.
2. **A geodesic goal field**, one GPU bake per map+zone (~10 min at cell 32
   for a cannonball-sized map, seconds for a petrus-sized one), plus the
   occupancy/SDF grids vision needs. All cached next to the `.bsp`, all
   gitignored.
3. **Nothing else** - the trainer takes the third map with one more entry in
   `--maps`, and the held-out evaluation is `tools/record_ckpt.py --map C`,
   which already resolves that map's own cell and latch.

The cheapest source is another ported Source/GoldSrc linear surf map with
timer triggers, the same way `surf_petrus_lite` arrived. Two more would be
better than one: with a single held-out map, "it transfers" and "C happens
to resemble A" are the same measurement.

**Two candidate maps are already in flight on sibling branches** and should
be checked before sourcing anything new: `thirdmap` hand-labels
`surf_ski_2` as a held-out navigation map, and `sidistic` adds
`surf_src_sidistic`. If both hold up at integration the held-out test is
unblocked immediately - train `--maps cannonball,petrus_lite`, evaluate on
the third with `tools/record_ckpt.py --map`, against a control trained on
cannonball alone.

**Also still open, in rough priority order.**

* **`ep_ticks` is one number for every map.** It is 12,000 (120 s), tuned
  for cannonball; petrus_lite's whole route is ~5x shorter, so most of an
  episode there is a dead agent waiting for the stall-kill. Per-map
  `ep_ticks` (`max_episode_ticks` is already a per-core config field, so
  this is a small change) is the obvious follow-up and was deliberately not
  done here.
* **`--int-coef` is not map-normalised.** Shaping is fixed at 100 per map by
  construction, but the intrinsic-novelty budget grows with map size (cells
  on the line ~ `d0/256`), so one `--int-coef` is a bigger share of income
  on the long map than the short one. The same fix as the latch: express it
  per map.
* **The env split is even and static.** No curriculum over maps - a map the
  policy has mastered keeps half the fleet. Weighting the split (or
  resampling it on eval progress) is the natural next knob.
* **Single-map-only flags:** `--route` (one map's reference line),
  `--demo-file` (one map's demo spine), `--race-dfloor` (absolute
  distance). All three refuse loudly under `--maps`.
* **The truncation bootstrap feeds raw scalar slot 12 into V(s_T) under
  `--obs-reward`** (pre-existing: `term_obs` carries position there, while
  training writes `tanh(reward)`). It was already an out-of-distribution
  feature on one map; with several it is also map-relative. Left alone
  because fixing it would break single-map bit-identity - worth its own
  small change.

---

## 5. Goal render + goal-conditioning  (P1, user idea)

**The idea.** A second depth image showing only the goal volume, rendered
through walls (wallhack-style), so the goal is *visible* rather than implied
by a scalar field.

**Why it is a good idea.** It converts the task from "follow this map's
potential" to "go to the thing you can see", which is map-independent and
therefore the natural substrate for 4. Swift feeds the next gate's four
corners; Sophy feeds course points; both are the same move.

**The honest caveat.** Information-theoretically a goal render is ~4 numbers
(ego-frame direction, distance, apparent size). **An ego-frame goal vector
would carry nearly the same signal at nearly zero cost**, and
`surfgym/route.py` already builds exactly that kind of ego-frame feature. The
render earns its keep only if (a) you want the conv trunk to handle goals
uniformly with geometry, or (b) goals become extended/complex shapes. Worth
testing the cheap version first; if the vector works, the render is a
compute tax.

**The curriculum idea inside it is the strong part.** With easy maps where
the goal is a few seconds away, the agent learns *what the goal channel
means*, and that meaning is what transfers. That is a real curriculum, not a
representation trick.

**Cost.** Small-medium. A cuboid render with no occlusion is analytic and
much cheaper than the SDF march.

---

## 6. Hindsight Experience Replay  (P1, user idea - and the caveat matters)

**The idea.** Relabel failed episodes with the goal the agent actually
reached, so every episode becomes a success at *something*.

**Why it fits.** It is the standard answer to sparse goal-reaching, and it
requires exactly what 5 provides: the goal in the observation and a reward
that is a function of `(state, goal)`.

**Three obstacles, in order of severity.**

1. **Our reward is not goal-conditioned.** It is a precomputed geodesic
   field for one fixed goal. You cannot relabel it - there is one field per
   goal and baking one costs GPU-minutes. HER therefore forces the reward to
   be something recomputable per goal.
2. **The obvious recomputable reward is distance-to-goal, which is the bug
   this whole round diagnosed.** Euclidean or free-space distance to an
   arbitrary relabelled goal has interior minima behind walls, and on this
   map "fell into the pit" would be relabelled as a *success at reaching the
   pit*. That teaches falling.
3. **PPO is on-policy.** HER is a replay-buffer method; relabelled
   transitions are off-policy and cannot enter a PPO batch without
   importance-weighting surgery. Round 16 already lost an arm to exactly
   this (burst transitions excluded, 5x fewer optimizer steps).

**Where it does work.** Short horizons and *nearby* goals: "reach a point
2-5 seconds away" is well-posed, has no wall problem at that range, and is
precisely the generalised surf skill. So HER belongs to a **low-level
goal-reaching skill**, not to the full-map objective.

**And that composition is a named architecture.** Goal-conditioned
low-level policy + high-level choice of which point to aim at **is SoRB /
SPTM** (idea 3). Ideas 4, 5, 6 and 3 are one program, not four.

---

## 7. Contact-loss penalty for line quality  (P2)

Running as `xCSPD`. Charges the kinetic energy `PM_ClipVelocity` destroys -
exactly the plane-normal component, so a grazing ride pays zero. 71% of all
energy supplied is destroyed at contacts (7,363,534 against gravity's
9,392,448 and strafing's 1,039,768), and 68 s needs a 35-40% cut. Losses
concentrate in route deciles 8, 10 and 3 (48% of the total). It is the one
signal that rewards the geometric property of a fast line **without knowing
the line**. Ran once before the wall was solved and was a null on the
barrier while demonstrably working on its own terms (speed decay through the
approach went -81 u/s to 0).

---

## 8. Search then distil  (P2, the heavy artillery)

Savestate hill-climbing in **strafe-parameter space** (bxt-rs / TMInterface
frame bulks, not per-tick inputs), then distil the found line into the
policy via the backward curriculum we have already proved. The earlier
attempt failed because per-segment gains do not compose - each window's
improvement invalidates the next window's script - and the identified fix is
to re-optimise each window **from the state actually reached**. Rigorous
floor for the champion's route is **59.00 s** with zero contact loss, so
"<60 s" is essentially at the physical limit of that line; a different line
would have a different floor.

---

## 9. Built but never run

* **Quantile / distributional critic** (`--quantiles 32`, branch `qr-critic`,
  14 tests green, `train/q_spread` diagnostic). Never ran - the local GPU was
  held all night by another agent's zombie process. Both superhuman racers in
  the survey use one. Run it **on the shaping that works**, and `q_spread` at
  the wall is the measurement nobody has.
* **Physics codebook** (`tools/make_physics_codebook.py`, `--codebook`) -
  never used; all chunk runs used the learnable decoder, which collapsed 2/2.

## 10. Untested runners-up from the survey

`gSDE` / pink noise (poor fit - our action space is discrete), SIL / SAIL
`(R-V)+` filter, kickstarting / QDagger from the 78.70 s policy into a fresh
seed, Seer's KL-to-imitation with a decaying coefficient, Swift's perception
term for the aimable camera, n-step returns, gamma parameterised in seconds.

## 11. Hull-1 vision rebake - fold into pool v3 (P0 for multi-map, round 26)

**The finding (ledger round 26, confirmed at user-marked coordinates
2026-08-25):** the player collides via BSP hull-1 clipnodes, but every
grid vision and reward read - occupancy, slab occupancy, the SDF the
depth march samples, the goal field - is built from hull-0 point queries
(`surf_occupancy_grid`, env.c). GoldSrc CLIP brushes exist ONLY in hull
1, so collision can be invisible to depth BY CONSTRUCTION. Measured:
37/106 pool maps have invisible clip volume in open space; 28/106 have
clip-backed func_illusionary ramps (gi_rino: 102/105 models - its round
wall tower is entirely invisible to the policy); skids2's final traj
stands on invisible geometry in 12/16 landings via a SECOND mechanism, a
1u-thick WORLD brush threading the cell/4=8u slab lattice. User surfed
gi_rino and marked (7167.7,-1366,4889) and (2275.9,-6263,4353): both are
illusionary wedges whose collision is **worldspawn CLIP** (trace ent 0),
so no entity-level fix can ever see it - only sampling hull 1 of the
world model can.

**The fix, held back deliberately (bit-identity class):**

* (b1) occupancy from a zero-length standing-hull trace per voxel center
  (the player's C-space). Geometry comes out hull-fattened (~16u lateral
  / ~36u vertical) - the BSP does not store unexpanded clip brushes, so
  C-space IS the available truth.
* (b2) rasterize thin WORLD faces into the slab grid (the thin-entity
  net already exists; world panes have none).
* Both bump `_OCC_SEMANTICS`/`_SDF_SEMANTICS` -> full rebake of every
  map, goal fields included (geodesics stop tunneling through clip walls
  on the 37 maps, so d0/map_pct histories break comparability too).

**Plan:** ONE cut, bundled into pool v3 with the tighter finish pads and
the liar-map removals, so there is a single comparability break. Gate
adoption on one 5-map-bundle A/B (fixed SDF vs current, matched steps,
~$0.25) plus local validation that gi_rino's ramps appear in the depth
march and skids2's floor appears in the SDF. Rebake cost: overnight
local or a few hours on one cheap box.
