# Second-opinion review: the ramp-command search design (Fable, 2026-09-27 evening)

Read-only review. Sources: CLAUDE.md 0 / 0b / 0c, docs/ramp-search-design.md (the consolidated
draft), the ledger docs/research-results.md from 2026-09-27 06:07 to 19:18 (the uf2 night, the
utopia diagnosis and its corrections, the ramp graph, Codex's reply), docs/planner-design.md,
docs/review-fable-2026-09-27.md, the surveys (litsurvey-game-agents-hierarchy.md 3.7,
research-litsurvey-hier.md 1-7 and its one-day validation section), and the code: src/pm.c
(pm_fly_move, pm_categorize_position), src/env.c (surf_set_state), python/surfgym/core.py
(SurfState), python/surfgym/vision.py (LidarPotential), python/surfgym/goalball.py (PlanLineLidar),
python/surfgym/goalfield.py (sentinel), tools/edge_archive.py, tools/ramps.py, tools/ramp_route.py.
One CPU check on the saved prototypes runs/research/ramps_{blue025,uf2,utopia}.npz (numpy only).
No GPU, no training, no simulator runs, no file edited.

## 0. Verdict

The ramp vocabulary is the right search space and Codex's two structural corrections (exact states
as nodes, never (ramp, speed); an event contract with a miss on C charged to command B) are
correct and I would not relax either. Where I disagree is on what the design is FOR and on the
order of work. The draft conflates two products: (1) a move operator that lets the archive WITNESS
ramp-to-ramp transitions it never draws today, and (2) a deployed command-following executor that
takes "go to ramp B" reliably from a highlight in its image. Only (1) is needed to close utopia,
and (1) can be tested this week with the existing line-following movers and the existing archive:
make "go to ramp B" a LINE (an arc into B's plane, then B's own down-slope centreline) and run
edge_archive with that operator against the rays and primitives at an equal flight budget. If that
does not move utopia past the R5 -> R8 kicker, a trained highlight executor will not either,
because the ledger already shows the mover can fly the right line when it is drawn (07:53, 09:48)
and cannot fly what no line draws (the U-turn, the launch). The deployed artefact should stay the
flat policy on the route the search found (efTGT200 / rpCTL / xSELF / jt3ANCHU are the only things
in this ledger that finish maps), which drops the executor's requirement from "reliable capture"
to "capture at least once in K seeded tries". Three more things need changing before uf2 is even
a fair test: the executor's reward as written ("landing on B, time penalty") reproduces the utopia
failure (arrive slow and still succeed); the highlight channel alone cannot carry a command whose
target is behind the agent, which is the case at both uf2 links that fail today; and v1's
deterministic greedy executor is the weaker policy at exactly the hard links (landing bench:
greedy dead by 1 s where sampled lands 23/64). The uf2 pit will still fail after edgeflow and
utopia pass, for a reason the design does not address: the shaft launch is contactless air
strafing from an approach that needs ~1,700 u/s, and no policy-owned state carries that energy;
the ramp graph fixes proposal geometry, not that skill. Section 4 says what could.

## 1. Q1 - is "ramps as actions + exact states as nodes + implicit search first" right?

Mostly yes, and the reasons are in this project's own numbers, not only in the literature.

What I agree with, and why:
* Exact states as nodes. research-litsurvey-hier.md lesson 7: a 0.2% velocity change at the
  cannonball room entry killed 95% of replays while +-4 u of position was tolerated 70%. Any
  (ramp, speed) aggregation would splice edges that never happened; Codex is right and the
  abstraction-sufficiency probe it asks for is the right way to ever relax this.
* Ramps as the vocabulary. The contact bench (09:32) is the decisive measurement: from the
  agent's own pre-contact states, random primitives of either kind keep 24/192 alive, while the
  same mover shown a working line survives 64/64. The proposal geometry is the limit, and ramps
  are the only generic object on a surf map that tells you where a proposal should END.
* An implicit (exhaustive, budgeted) search before any learned planner. The ledger's learned
  planners never beat a route (review 2026-09-27 03:46, section 3). More important: a search
  with a budget is the only anti-deception mechanism in the design. The heuristic h ranks uf2's
  R1 (potential-worse) last, but a search that eventually probes every UNKNOWN command still
  tries it; a policy following the same potential never does. The correctness comes from the
  budget rule, not from h. Write that rule down as a constant: "every UNKNOWN command of a node
  is probed once before any child of that node is probed twice" (progressive widening as the
  default, not a fallback for maps with hundreds of targets).
* Energy as an ordering only. Agreed; and the 17:20 correction stands (the max-E state per cell
  is 92.5% a fast fall).

What the literature adds that the design should borrow:
* Options / semi-MDPs (Sutton, Precup, Singh 1999): "go to ramp B" is an option with an
  event-defined termination (first new targetable contact). Event-defined, not learned,
  termination is exactly what research-litsurvey-hier.md 1 recommends (option-critic's
  termination collapse; Harb's deliberation cost). Good.
* Skill chaining (Konidaris and Barto 2009) and Deep Skill Chaining (Bagaria and Konidaris,
  ICLR 2020): options are built BACKWARD from the goal and each option's target is not "touch
  the next region" but "land inside the INITIATION SET of the next option", the set of states
  from which the next option succeeds. This is the exact formalisation of the design's open
  problem ("arrive with enough speed for the NEXT transition") and it is what the executor's
  reward must encode (section 2). The control-theory version is the funnel picture the game
  survey already cites (Burridge, Rizzi, Koditschek 1999; Tedrake's LQR-trees): command B's
  region of success must contain command A's landing cloud.
* Go-Explore (Ecoffet et al. 2019 / Nature 2021), policy-based variant: the archive is already
  it; "return then explore" with a goal-conditioned policy IS a ramp-commanded executor. The
  robustification phase (RL from the archive's states, never cloning) is the generations loop,
  and the attribution control (11:34) showed that practice from archive states is what teaches
  the pit skills. Keep that as the executor's spawn distribution.
* PRM-RL (Faust et al. 2018): an edge is accepted at >= 85% of 20 rollouts, and route success is
  p^n (0.85^6.05 = 37% predicted vs 38% observed). Use this to separate DISCOVERY (one witness
  in K tries) from ROUTE (capture probability per edge), which the design's ledgers 1 and 2
  already do; the number K and the acceptance threshold are recipe constants, set once.
* Hindsight (HER, Andrychowicz 2017; HAC's hindsight action transitions, Levy et al. 2019;
  GCSL, Ghosh et al. 2021): every flight commanded to B that ended on C is a correct training
  example for "go to C". Codex's ban is right for the SEARCH ledger (a C-miss is an outcome of B)
  but not for the EXECUTOR's training set. Caveat: PPO is on-policy and cannot relabel; the
  supervised path is self-imitation on the policy's own recordings (allowed by section 0; the
  project has sil.py), and it is optional.
* Kinodynamic planning: with a learned local planner and exact nodes this is a kinodynamic
  roadmap. RRT / SST need a state metric for nearest-neighbour; best-first over exact nodes
  does not, which is a real advantage. Lattice planners (Pivtoraiko and Kelly) discretise
  controls instead of states; the ramp vocabulary is a control lattice that the map supplies.
* For the amortizer later: Gumbel MuZero (Danihelka et al. 2022) for few-simulation policy
  improvement over a variable candidate set, Stochastic MuZero for chance nodes if the executor
  stays sampled. Not now.
* Racing agents (GT Sophy, Linesight): flat policies with a monotone course coordinate, no
  hierarchy. Together with efTGT200 / xSELF / jt3ANCHU, this is the evidence that the
  hierarchy's job here is DISCOVERY of the coordinate, and the deployed policy is flat.

What to avoid: learned termination; two learning levels chasing each other (HIRO's whole
correction machinery exists because of that); learned world models when the simulator forks at
200k steps/s; position-only gates; any (ramp, speed) value table.

Simpler formulation than the draft's: the ramp graph as a MOVE OPERATOR of the existing archive
(section 5), with the found chain handed to the flat recipe as a self-route. The command-
conditioned executor is a second stage, justified only if the line transport is shown to fail on
flyability while independent witnesses exist (the design's own "executor incompetence" rule).

## 2. Q2 - training the executor and the command interface

### 2.1 The reward as drafted re-creates the utopia failure

Section 4: "Command reward: landing on B (the next contact), time penalty; the arrival state's
quality (speed) is what the search values downstream." The executor never sees that downstream
value, so it learns the cheapest way to touch B, which on a surf map is slow and safe. That is
the 16:46 diagnosis in one sentence ("the +50 success bonus dwarfs the time penalty. Arriving at
half speed still succeeds"), and the corridor pair (11:50, 12:05) showed the same objective shape
decides precision and speed retention. Then the search cannot witness speed-gated transitions
because its executor never arrives fast, and the generations loop has nothing to select.

The generic fix is the option-chaining return, the same lesson as `--exec-cut` in the previous
review (2.1): the executor's episode does NOT end at B. It continues with the next command of the
chain the search found from that node (2-3 commands ahead, a fixed constant), each capture pays
+1, the finish pays its bonus, the time penalty runs, and death forfeits every capture still
ahead. Nothing about speed appears in the reward; arriving at B in a state from which C is
capturable is what the return values, which is Konidaris' initiation-set target. Constants: the
number of lookahead commands, set once. Do not bootstrap the executor's critic with the search's
values instead: before the first finish they are ~0 everywhere (the refund_i lesson).

### 2.2 Spawn and command distribution (no map term)

* Spawns: the search's exact nodes (policy-owned), weighted to intermediate difficulty as the
  loop's `--dump-weights goid` already does, plus true starts. This is the one thing the night's
  attribution control (11:34) showed to teach a skill.
* Commands: witnessed transitions from those nodes for the main share; a fixed share of
  unwitnessed commands ordered by the search's prior, so the executor is also asked for things
  the search wants to know; hindsight labels for the supervised auxiliary if it is used.
* Evaluate greedy AND sampled (review 2.5), and run the override ablation (shuffled targets must
  cut capture) periodically: a plan-conditioned policy that ignores its input is the collapse
  risk planner-design.md 9 already names.

### 2.3 The interface: scalars first, the mask second, both in the end

A highlight channel alone is a poor command carrier for THIS policy, for a reason specific to
surf: under `--view-absolute velocity` the camera is the steering input. Air strafing is yaw plus
a side key; the policy must look where it accelerates, not where the target is. A command that
exists only in pixels forces "see the command" and "steer" to compete for the same output. Two
concrete cases in tonight's data: the U-turn at the north end of uf2's lower A-frame (the target
is behind the agent for most of the turn) and the shaft launch (the target R9 is above and behind
during the contactless loop). Both render an all-zero highlight for most of the command. Design
step 4 would catch that on edgeflow only if edgeflow had such a transition; A-frames in a zigzag
chain always show the next ramp ahead, so the smoke test passes and uf2 fails on the interface.

The project's own precedent says the same: the scalar fan carried the route on every map that
finished; `--goal-obs fanline` (PlanLineLidar) helped "modestly at one seed", and its docstring
states the rule: "No off-screen marker: the fan's scalars still carry a plan that has left the
view." So: an ego-frame (velocity-frame) target block of scalars - vector to the target's nearest
point, vector to its down-slope end, its mean normal, its extents, and a visible-fraction number -
always defined, plus the mask channel for the take-off decision. Add one more generic scalar
the memoryless executor lacks: the ramp id / ticks since the last real contact from the
instrumented core, because `onground` is -1 while surfing and in flight alike, so the executor
cannot tell "still on A" from "airborne" and the event contract's "declared departure" is
invisible to it.

## 3. Q3 - the biggest risks, and why uf2 fails after edgeflow and utopia pass

Ranked:

1. **The executor's skill, not the search, is uf2's limit, and ramps do not supply skill.** The
   night's benches say it from the record's own states with the record's own line: the U-turn at
   1,730 u/s kills every mover within 0.5 s (11:11) except K128 (61/64, at the cost of survival
   elsewhere); the launch reaches z >= 450 on 0-10/64 (11:04); the archives arrive at the shaft
   approach with <= ~64% of the record's speed (11:04 energy profile). The launch loop itself has
   no contact (19:11: "air-strafed, vz exactly ballistic"), so "go to R9" from R5 is one command
   spanning a 250 u radius loop the highlight cannot show. A ramp-command search from
   policy-owned nodes cannot witness R5 -> R9 until some executor flies it, and the design's
   generations loop trains only on selected (witnessed) segments. This is a chicken-and-egg the
   draft acknowledges with the "independent primitive search" witness rule but does not solve:
   tick-level searches here have a record of 0/18,900 leaving the cannonball bowl and
   explore_phase1 froze. The honest expectation is: edgeflow passes (plumbing), utopia is the
   real test of the vocabulary, uf2 tests something else (energy retention over a 4-link chain),
   and the ladder does not transfer.
2. **The reward gap of 2.1.** Without the chain return the executor is trained to be slow, the
   search inherits a slow executor, and the negative result on utopia will be misread as "the
   search cannot find it" when the executor could not arrive.
3. **Greedy executor in v1.** archive_greedy_b200 died 17 of its first 24 flights and exhausted
   at depth 2 while the sampled executor survived the same first plan 78-100% (review 2.5); on
   the uf2 landing bench greedy is dead by 1 s for most movers where sampled lands 23/64. A
   deterministic best-first search with one outcome per (node, command) will label capturable
   transitions as misses. Use K seeded samples with common random numbers from the start (the
   design's own plan for the stochastic case), cache all K outcomes, and treat one capture as the
   witness. K is a recipe constant.
4. **The interface risk of 2.3**, which the edgeflow test cannot reveal.
5. **Search degeneracy from the value scheme.** Untried commands carry optimism 1 and expanded
   non-finishing children carry h <= 0.5, so every node's untried commands outrank every
   expanded child and the search is breadth-first over commands (16-70 per node) before it is
   deep. Best-first by the prior with progressive widening as the default, and a duplicate rule
   over exact nodes (the archive's key is the natural one), is what will actually run.
6. **Card-dependent witnesses.** CLAUDE.md 3: the lidar march is not bit-exact across GPU
   architectures, so a witness is a fact about (executor hash, render device). Namespace ledger
   2 by both, not by the executor hash only.
7. **Time is ignored by the value.** A finish at 40 s and at 60 s are both 1. Fine for
   discovery, but score edges in time for the route that goes to the flat policy (hier survey
   lesson 9: variable-duration edges bias a search).

## 4. Q4 - the ramp extraction: sound in principle, and the saved files show four defects

The definition is generic: n_z < 0.7 is the engine's own ground test (pm.c 252, 264, 362, 440),
the same number on every map, and grouping collision hits into whole ramps needs no map constant.
Codex's corrections (standing hull, CLIP geometry, categories, validity mask) are right. The saved
prototypes add data to them (runs/research/ramps_*.npz, CPU only):

* **Sentinel contamination is real, not hypothetical.** Four utopia ramps (R11, R12, R50, R51)
  share d_max = 167,812 exactly, and uf2 R19 (a 1,136 u ramp) reads d 11,690..34,360: the goal
  field returns a finite sentinel (reach_max + 2 cell, goalfield.py 170), so `np.isfinite` keeps
  it and every mean and range in the 19:11 table is biased upward. Use the validity mask.
* **The 25 deg chaining merges what the search needs separate.** uf2 R9 spans 2,416 x 1,616 u
  with normals 65 deg from its mean; six utopia ramps exceed 45 deg. On blue025 every ramp is one
  plane (spread 0 deg, extent <= 416 u), so edgeflow will never show this. Merge rule fix that is
  still generic: cap a component's normal spread (e.g. 45 deg from its running mean; a constant
  set once) and keep atomic patches beneath the ID as Codex proposes. uf2's pit is a bowl - the
  curved south wall, the A-frame face and the north wall must be distinct targets or the
  commands R5 -> north wall -> R5' -> south wall have no vocabulary.
* **A mean potential is not an ordering on long ramps.** Median within-ramp spread on utopia is
  38,477 u of a 166k field (median ramp extent 3,904 u); on uf2 the p90 is 23,217 u of 30k. Use
  the down-slope END quantile (the state you leave the ramp in) for ordering, and the range only
  as the search's bound.
* **Duplicate pairs** (utopia R50/R51 with 11,192 / 11,195 samples and identical extents, R11/R12
  likewise) are either mirrored geometry or both sides of a thin brush; check before counting
  transitions.

Further pitfalls: consecutive ramps in a long descent are ride-contiguous (utopia's finisher
visits 18 of 70), so "go to the next ramp" while already sliding onto it is a trivial command
and the search depth is 18; merge ride-contiguous ramps into corridors, or add a "continue"
command. Floors and platforms must be TARGETS, not only terminators (the edgeflow start
platform, landing platforms), or many routes have no direct ramp-to-ramp edge; ceilings (the
block underside that killed the movers at the launch) must be an outcome class. Maps with no
ramp in reach (bhop, boosters, teleports, ladders, water) need a defined fallback: the flat
policy alone. A 16 u point grid misses thin and CLIP geometry, as Codex said; the contact
instrumentation itself is cheap - pm_fly_move already has every plane it hits in `planes[]`
(pm.c 258) and `trace.ent`; recording the last contact normal, ent and tick per env is a few
lines and gives both the search's contact truth and the executor's phase scalar of 2.3.

## 5. Q5 - the minimal decisive experiment on edgeflow (and the one that matters, on utopia)

Do not build the executor first. Run the VOCABULARY test with the movers and archive that exist:

1. `--moves ramp` in tools/edge_archive.py: for each candidate ramp B (all ramps within a fixed
   time horizon at the current speed, ordered by the prior), the move's line is a Hermite arc
   leaving along the current velocity and arriving in B's plane (edge_archive.tangent_curve
   already builds this for the first surface a trace meets; generalise the end point to a chosen
   ramp), continued along B's own down-slope centreline from the extractor's points. Contact
   labels from the point-hull ramps and the proximity rule are adequate on edgeflow (single
   planes) and for a first utopia pass; contact truth can land in parallel.
2. Controls at an equal FLIGHT budget, same mover and seed: `--rays 3` and `--moves prim`.
   Baselines already on disk: blue200 rays 97,654 expansions / ~293k flights / 268 s
   (09:20), blue025 rays 5.3k expansions; utopia no finish at 597,834 expansions, best 9.8%
   geodesic gain, live endpoints to t ~10 s of the finisher's timeline (16:46, 17:20).
3. Measure, pre-registered: flights to the first finishing chain (a time-to-event number, the
   only kind that survives one seed here) and the chain's 32-replay rate; on utopia, the
   furthest live node along the policy-owned finisher's timeline (jt3ANCHU, allowed as a ruler
   because it is the policy's own recording) and whether any node passes the R8 landing
   (t 10.75 s) with speed within 20% of the finisher's; the distinct ramp-to-ramp transitions
   witnessed (the vocabulary's own yield).
4. The steerability half of design step 4, with lines instead of a channel: from the same exact
   blue025 nodes, command ramp k vs ramp k+1 (and the two faces of one A-frame); report the
   target-contact confusion matrix. That is the "can this vocabulary be commanded at all"
   number, and it needs no training.

Outcomes:
* CONFIRMS: blue200 found within the rays' budget AND utopia passes R5 -> R8 (or finishes) at
  <= 2x the 597,834 budget. Then hand the chain to the flat recipe as a self-route (SELF_STATES=1,
  the efTGT200 flags) and run the unchanged pipeline on utopia end to end. Only then is a
  command-conditioned executor worth training, and only if the line transport is the limit.
* KILLS: no gain over rays on blue200 and no node past R8 on utopia at 2x budget, with the
  steerability matrix showing the lines ARE followed to the commanded ramp. Then the ramp
  vocabulary is not the lever; the limit is the executor's speed retention and the work goes to
  section 2.1 (the chain return) rather than to the search.
* AMBIGUOUS (the likely case): the confusion matrix is clean but utopia stalls at the same
  kicker with the mover arriving slow. That is the 2.1 result and it decides the executor
  program's first objective.

Note that edgeflow cannot discriminate the search: greedy-by-potential ramp commands pass a
zigzag chain of A-frames whose potentials fall monotonically 2,235 -> 230. Edgeflow is plumbing;
utopia is the first real test, as the draft says; uf2 is a different question (section 3.1).

## 6. Q6 - what the user and Codex are missing

* **Two products in one design** (section 0). Decide now that the deployed policy is the flat
  policy on the found route, and the ramp machinery is the discovery operator. That choice
  removes the reliability requirement from the executor, removes the search from deployment,
  and matches every finish in the ledger.
* **The reward gap** (2.1) is the pivotal unknown for uf2. The test that decides it is small and
  policy-owned: from archive states on R5, train the executor on the 2-command chain (ride R5
  north, U-turn, ride south) with the chain return, and measure the arrival speed at the shaft
  against the ledger's 64% (the record as a ruler only). If the chain return does not raise it,
  no amount of search will.
* **Phase information** (2.3): the executor cannot see contact; give it the instrumented
  contact scalars.
* **Off-screen commands** (2.3) and the view-as-steering conflict.
* **The edgeflow -> utopia -> uf2 ladder tests three different mechanisms** (3.1); pre-register
  what each rung can and cannot show.
* **Sampled executor with common random numbers in v1** (3.3), not greedy.
* **Render-device namespacing of witnesses** (3.6).
* **The extractor defects in the saved files** (section 4): fix the sentinel and the chaining
  before any potential-ordered search reads those numbers.
* **Ride-contiguous ramps and floors as targets** (section 4).
* **The generations loop is the same expert-iteration loop that plateaued at 2,46x-2,69x u
  tonight** with a different proposal vocabulary. What produced skills was practice from archive
  states under an objective in which survival and progress mattered (landing 0 -> 27/64) and the
  objective's shape (K128). The ramp vocabulary changes what the loop can witness; only the
  executor's objective changes what it can learn.

## 7. Recommended order of work (vs the draft's section 7)

1. Contact instrumentation in pm_fly_move (cheap, in parallel with everything below); the
  extractor fixes of section 4 (validity mask, normal-spread cap, end-quantile potential).
2. `--moves ramp` in the archive with the line transport; the vocabulary test and steerability
  matrix of section 5 on blue025 / blue200 / utopia, locally, no rental.
3. If utopia passes: self-route -> flat recipe on utopia, unchanged flags; then the second map.
4. If utopia stalls with a clean confusion matrix: the executor with the chain return, spawned
  from archive nodes, scalar target block + mask; the causal test of design step 4; then the
  search with K seeded samples.
5. uf2 only after 4, with the R5 chain measurement of section 6 as the gate, and the whole
  9-ramp route as the verdict.
6. Generations, then the amortizer, as the draft says.

Where this differs from Codex: extractor v2 and the observation audit come after the vocabulary
test, not before it, because a line-transport search tests the vocabulary without them; v1 uses
the sampled executor; the executor's reward is the chain return, not capture plus time; the
deployed artefact is the flat policy on the found route.
