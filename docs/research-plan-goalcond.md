# Goal-conditioning: the experiment set (PRE-REGISTERED)

Status: PLAN, 2026-09-02. Nothing below has run. Every experiment states
its expected outcome before it runs; evidence that differs is a finding to
dig, and the dig question is written next to each expectation. Companion:
docs/research-plan-goallines.md (the machinery validated so far) and ledger
round 28 (what it produced and where it drifted).

## 0. The claim, and why the last two days did not test it

The claim: an agent can learn ONE transferable pattern - "the goal signal
marks the destination; go there, finding the path yourself" - and apply it
to goals it has never been given. Round 28 validated the ingredients on a
single (map, goal) pair: the goal representation is read and load-bearing
(route-mode probe: 90-98% collapse, three policies), grounding requires
reward-observation consistency (xsFAN vs xsFULL), and a warm policy is
brittle to its guidance contract (addenda 6/7) - which is precisely why
diversity must be present DURING training. What was never tested is the
claim itself: every policy so far saw exactly one goal per lifetime, so it
may be treating the line as a map-specific position encoding. The
experiments below exist to distinguish "goal-conditioned" from
"map-solved", and only held-out goals can do that.

Three variables (user, 2026-09-02):

1. **Goal representation.** Point goals mid-map have no per-goal path (a
   path per goal is a bake per goal), so the fan is unavailable; the
   minimal general encoding is the EGO-FRAME GOAL VECTOR (unit direction in
   the agent's frame + log distance, normalized like the fan's points) -
   Habitat PointNav's GPS+compass. The minimap/tri-plane raster with the
   goal marked is the one representation that binds to geometry WITHOUT a
   path, so it is the representation arm, second.
2. **Goal diversity.** Two generators, no manual labels: ACHIEVED states
   (the reservoir already holds thousands of visited states per map; a
   goal drawn from the same recorded episode k seconds ahead is reachable
   by construction and difficulty-tunable by k - HER's "future" strategy
   applied to the goal DISTRIBUTION, which is what composes with on-policy
   PPO, since transition relabeling needs a replay buffer we do not have)
   and RANDOM reachable air (SDF > 0 and GoalField.reachable, both baked;
   what survives and is still dynamically unreachable is the tolerated
   remainder). Goal = sphere, radius 192u (the type-2/3 button-box logic,
   pass-through counts); entry ends the episode with the bonus. The map's
   own finish is one goal among many.
3. **Goal-to-reward without a rebake.** Sparse success + time penalty as
   the primary reward: nothing is more consistent with the arrow than
   "reward fires exactly when the arrow shrinks to zero", and grounding
   is gated by consistency (round 28) - HER's own ablation says the same.
   Euclidean potential (EuclidField, zero precompute) is a CAPPED
   tiebreaker held in reserve; per-goal geodesic bakes are off the table
   for good; learned distances are the long-term answer if range ever
   forces it.

## 1. Metrics (fixed before any run)

* **success rate** per goal-distance band, and **time-to-goal** for
  successes. Never race/eval_progress (goals are not the map finish).
* **"stairs needed" flag** per goal: straight-line-through-solid (SDF
  sampled along the start-to-goal chord) marks goals the arrow lies
  about. Success as a function of that flag is the quantity the building
  analogy predicts. On successes, path-flown / straight-line is the
  realized detour factor.
* **the goal-block probe** (`--goal-mode off/frozen/shuffle` in
  record_ckpt, the route-mode idiom): a policy that reads the goal
  collapses when it is zeroed.
* **held-out success**: goals from a map sector masked out of the training
  sampler, plus fresh random air. This is the only number that says
  "goal-conditioned". Reported next to the zeroed-goal control on the
  same held-out set.
* Guards inherited: one seed; 27%+ noise floor on end-of-run levels;
  matched-step early comparison preferred; success-rate numbers only
  next to their sampling distribution (trivial goals excluded: k >= 1 s).

## 2. Engineering prerequisites (local, tested before any GPU hour)

* Per-env goal store (center, radius) and a per-tick vectorized reach
  test in python (the C goal box is ONE box per core; `core.force_fail`
  already terminates per-env from python, so goal-reach = force-terminate
  + bonus paid by the reward fn on the goal mask - the stall-kill idiom).
* Goal block in `_obs`: 4 scalars through the route-block widening path
  (trailing columns, so route_dim 0 stays byte-identical); record_ckpt
  mirror + `--goal-mode` probe. Bit-identity test: goal block disabled ==
  current code path.
* Sampler: achieved-state goals from RespawnBuffer rows (origin per state
  is stored; episode/time provenance is needed for the k-horizon draw -
  add episode id + tick to the stored row if absent) and random air with
  the SDF/reachable filter. Curriculum controller: per-band success kept
  in the Florensa 10-90% window.
* Held-out mask: one map sector (by geodesic band or by x-range) excluded
  from training goals, reserved for G3.

## 3. The ladder

### G0 - machinery correctness (local, no GPU)
Tests: episode ends on sphere entry and pays the bonus; ego-frame block
rotates with yaw (a goal straight ahead reads (1,0,0,log d)); sampler
never emits solid/unreachable goals; disabled block == baseline path
bit-for-bit; per-tick cost < 2% of a rollout tick.
**Expected:** all pass. A failure here is a bug, never a result.

### G1 - grounding on NEAR goals (1 h, one map, scratch)
Arrow block; goals = achieved states at k in [1 s, 5 s] from the start
state (easy), no random air yet; sparse +50 on entry, time penalty
0.005/tick, no shaping; scratch_ablate constants otherwise.
**Expected:** success rate rises from ~0 to > 50% within the hour on this
band; probe: zeroing the goal block collapses success (the arrow is
read). Time-to-goal falls over the hour.
**If success stays ~0:** sparse is too sparse even at 1-5 s -> G1b adds
the capped Euclidean potential (scale = a small fraction of the bonus
per unit, so it can never outrank reaching the goal). **If success is
high but the probe does NOT collapse:** the goals are solvable by
wandering (sphere too large or k too short) -> shrink the sphere, raise
the minimum k, rerun. Either dig is one hour.

### G2 - the curriculum to FAR goals (3 h, one map, scratch, --steps set)
G1's arm with the band controller: k widens while band success stays in
10-90%; 30% random reachable air mixed in; the map finish included as a
goal. The held-out sector is masked out of the sampler from step 0.
**Expected:** success stays 30-60% as k grows; the reached-distance
distribution shifts outward monotonically; success vs the "stairs
needed" flag decays smoothly rather than cliff-edging (the arrow lies
through walls and the agent learns to route around them); the map
finish is reached as a special case once the band covers its distance
- the first finishes of a scratch lineage, arriving as a by-product.
**If success collapses at some k:** find the k, and whether the failing
goals are the high-"stairs" ones (then the arrow's lie is the limit ->
G4's raster is the answer) or the high-speed ones (then it is motor,
not guidance). **If the finish is never reached** while far goals are:
the finish box specifically is hard (approach geometry), not
goal-conditioning - report it as such, do not patch it.

### G3 - HELD-OUT goals (eval only, ~20 GPU-min)
The G2 checkpoint on (a) the masked sector's goals, (b) fresh random
air, each with the goal block live vs zeroed.
**Expected:** live >> zeroed on both sets, and held-out success within
~2x of in-distribution success at matched distance bands. This is the
claim; a positive here is the first goal-conditioned result in the
ledger.
**If held-out ~ zeroed:** the policy memorized (goal, path) pairs ->
the dig is diversity: more goals per lifetime (shorter episodes, more
random air share), smaller spheres, and G3b re-evaluates. **If live >>
zeroed but held-out << in-distribution:** the arrow generalizes at
short range only -> measure the range, it is the representation's
limit and G4's question.

### G4 - representation: tri-plane minimap vs arrow (3 h)
G2 repeated with the goal marked in ego-centric occupancy rasters (the
tri-plane from research-plan-goallines.md section 6), arrow removed or
kept as a second arm if budget allows.
**Expected:** on high-"stairs" held-out goals raster > arrow (geometry
binding); on low-"stairs" goals equal. **If arrow >= raster everywhere:**
the binding hypothesis is dead at 32x32; drop rasters from the program.

### G5 - cross-map (E3 proper, 3 h + evals)
Two maps jointly (cannonball + petrus_lite via mapfleet, per-slot goal
sampling, per-map reservoirs), held-out goals on a THIRD map.
**Expected:** nonzero held-out-map success on near goals (arrow + depth
transfer at short range), decaying with distance; the petrus zero-shot
null (round 28) says motor transfer is the confound, so report success
vs required speed too. **If zero on the third map:** the pattern is
map-bound at this diversity (two maps) - the dig is more maps, which is
what the pool is for.

## 4. Order and budget

G0 (local) -> G1 (1 h) -> G2 (3 h) -> G3 (20 min) decide -> G4 or G5
(3 h). One arm at a time on the local 5090; every arm gets its probe;
expectations above are frozen as of this commit and scored in the ledger
by "met / partially met / missed - and why".
