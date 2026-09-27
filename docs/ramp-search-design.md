# Ramp search: ramps as the command vocabulary (design, 2026-09-27)

Status: DRAFT consolidating the user's proposal (2026-09-27 evening) with Codex's reviews (agent bus
15:19Z-17:35Z; the 17:35Z corrections are folded into sections 2, 5 and 7). Nothing here is built beyond the offline prototypes named in section 7. It extends
[planner-design.md](planner-design.md) (the planner's mission: macro exploration, CLAUDE.md 0c) and
obeys CLAUDE.md 0 / 0b (no demo in training; one recipe, the same constants on every map).

## 1. Why

- Search with a binary objective (reach the finish or nothing) cannot scale to large maps (uf2), and
  it regressed on utopia: the planner-free archive stopped at 9.8% of utopia's geodesic gain, at
  the first kicker, where the flat policy with geodesic shaping had finished.
- The geodesic potential is a good signal but not the objective: taken literally it walks into
  uf2's pit (the greedy next ramp is reachable but you arrive without the speed to continue).
- Surf reduces to ramps. A run alternates riding a ramp (where speed is kept or gained and the
  decisions are made) and flying between ramps (near-ballistic). Known routes are short ramp
  sequences on small graphs (prototype measurement, ledger 19:0x): blue025 16 ramps; uf2 29 ramps
  and the record's route 9 of them, R1 -> R5 -> R9 -> R11 -> R16 -> R17 -> R20 -> R24 -> R28, via the
  potential-WORSE pit ramp R1 first; utopia 70 ramps and our own finisher's route ~18, potential-
  monotone, with the archive's stall exactly at the R5 -> R8 kicker transition.

## 2. The abstraction (Codex's correction)

- **Ramps are the ACTION vocabulary, exact simulator states are the NODES.** A node is an exact
  contact/landing state (full velocity, heading, clock and controller state), labelled with its
  ramp ID. Never collapse the state to (ramp, speed): position along the ramp, side, velocity
  direction, v_z, contact phase and remaining time all change the successors.
- **A command is a ramp ID**, "your next contact is ramp B" (a DIRECT transition, the user's rule).
- **The event contract** (fixed before any search; Codex 17:35Z): a command starts on a controller
  decision boundary; continuing contact with the source ramp A is ignored until a declared
  departure; the FIRST new targetable whole-surface contact ends it. {B only} is a direct_touch,
  and a chainable direct_capture only if the agent is alive and action-ready at the next decision
  boundary. Contact with C, re-contact of A after departure, death, timeout and a simultaneous
  {B, C} are separate outcomes; a route touching C before B is A -> C -> B, never a direct A -> B.
  Facets of one whole ramp must not create false intermediate contacts; walls, floors and kickers
  are targetable classes when they can end a transition.
- **A miss on C stays an outcome of command B.** It is NOT a success of command C (the controls were
  generated under the B observation; counting it would let the search learn to say B to get C). It
  may seed the exact state s_C as a new search node and enqueue a real replay of command C, and only
  that replay populates the command-C edge.
- **The command reaches the policy as an observation channel** that highlights ramp B in the depth
  render (one next-ramp channel first; later fixed-rank channels next / next+1 / next+2). The
  channel is transport, not the action: the action set is the ramp list, not the view-dependent
  image.
- **Where and when to take off is the policy's decision**, inside the command.

## 3. Ramps and contacts from the collision geometry (generic)

- Contact truth first: per tick, the planes the player's movement actually hits (instrumented from
  the core's movement code, or the fallback: a non-ballistic velocity impulse plus a swept
  standing-hull trace). `onground == -1` is not contact evidence (true in flight and while surfing).
- Extraction from the STANDING-player collision (`core.trace(..., hull=0)`), not the point hull or
  visible faces (collision-only CLIP geometry exists on 28/106 surveyed maps, utopia and gi_rino
  among them). Categories: walkable support (n_z >= 0.7, the engine's ground test), surf slope, and
  near-vertical wall / kicker contacts (uf2's shaft). Atomic collision patches kept beneath the
  whole-ramp IDs. One global resolution and rule for every map.
- Per-ramp potential: quantiles of the goal field over valid contact points (the field's validity
  mask, not isfinite), used only as a bounded heuristic (section 5).
- Validation before use: contact recall, false contacts in free flight, unassigned contacts, ID
  flicker and seam switches, stability at 16 vs 32 u, a gi_rino CLIP regression; policy-owned
  finishers for checking, human records only as rulers.

## 4. The executor

- ONE policy (the flat surf policy family) conditioned on the highlighted target ramp. Command
  reward: landing on B (the next contact), time penalty; the arrival state's quality (speed) is
  what the search values downstream.
- Trained only on commands whose transitions EXIST (observed at least once from an exact state).
- Frozen within a search generation; trained between generations (section 6).
- Its steerability must be shown causally (section 7, step 4) before any search result counts.

## 5. The planner: implicit first (a search), learned later

- **Energy margin is an ORDERING prior, not a prune** (Codex 17:35Z): base velocity, push triggers,
  teleports, ladders, water and grounded motion all break a z + |v|^2 / (2 g) bound, and air
  acceleration adds up to 900 to |v|^2 per 10 ms tick (so E_max = 0.5 |v|^2 + g z + 450 N only holds
  for a restricted, finite-horizon, dry, push-free airborne subsystem). v1 never deletes a candidate
  on energy: fail-open, UNKNOWN rather than IMPOSSIBLE.
- **v1 search: exact-state BATCHED BEST-FIRST, not UCT**, while the executor is deterministic
  (greedy): every candidate ramp is commanded once per exact node, the exact outcome cached, children
  expanded best-first (ordered by bounded potential, geometric distance, energy margin and novelty,
  never deleting skip or potential-worse ramps), replanning after every direct capture; progressive
  widening with eventual budget for every UNKNOWN command if a map has hundreds of targets. With a
  stochastic executor: UCT / chance search over a fixed common set of policy RNG streams, keeping
  the full outcome distribution. Values: an
  actual finish 1, death 0; the potential only as a bounded heuristic at unexpanded leaves,
  h = 0.25 (z + 1) in [0, 0.5] with z = clip((d0 - d) / d0, -1, 1), so no potential value can tie a
  real finish; action-level optimism for untried commands (upper value 1), a UCB bound after
  samples. Backups use ALL outcomes of a command (B, C, death, timeout), never the lucky child.
- **Three separate ledgers** (Codex 17:35Z): (1) immutable physical witnesses and sound
  impossibility certificates; (2) per-executor-hash outcome counts N(s, commanded B, actual
  contact / death / timeout); (3) coarse novelty counts, only for allocating discovery probes.
  Statuses: UNKNOWN, WITNESSED_TOUCH, WITNESSED_VIABLE, CERTIFIED_IMPOSSIBLE. One success proves
  feasibility from that exact state, not reliable execution; finite failures never prove
  impossibility (namespaced by executor hash, retryable after an update). "Executor incompetence"
  is claimed only when an independent fixed-budget primitive / action search finds a direct viable B
  witness from the same exact state that the executor cannot reproduce. Discovery may use the
  existential witness; route execution uses current-executor capture probabilities (or a
  conservative bound).
- **The next ramp is the search's argmax** (the user's "implicit, like value iteration"): no trained
  planner at first. With few physically plausible candidates per state the search can try each.
- **Later, an AlphaZero-style amortizer**: a permutation-equivariant set / pointer network scoring
  the variable-size candidate set from generic features (never map-local ramp ID embeddings) (relative position, normal, height / energy margin,
  bounded potential), trained on the working search's own statistics - it can speed up a search
  that works, never replace one that does not.

## 6. Generations

freeze the executor (hash) -> build / revalidate the exact-state search -> freeze the selected
command segments -> train the next executor on them (plus true starts and rehearsal) -> clear or
revalidate transition statistics under the new hash -> repeat. One policy lineage, no learned
planner.

## 7. Order of work (Codex 17:17Z, adopted), edgeflow first

1. Contact truth (instrumented collisions or the swept-hull fallback), with analytic fixtures.
2. Extractor v2 (standing hull, categories, atomic patches, validity-masked potential), validated
   on the four edgeflow maps plus utopia, with the gi_rino CLIP regression.
3. Observation audit: render the next-ramp highlight; the fraction of approach decisions where the
   target is visible (a memoryless depth policy cannot obey an invisible target).
4. Causal target-following test on edgeflow: from the same policy-owned states, commanding two
   different independently witnessed reachable ramps must give distinct observations before the
   controls need to diverge, and different landings (two off-screen targets that both render an
   all-zero highlight are an interface failure, not two actions). Controls: zero channel, shuffled /
   wrong target, the current line interface. Outputs: target-contact confusion matrix, alive at the
   next decision, arrival speed / heading / time.
5. Frozen-executor search: exact-state ramp-command UCT against flat primitive search and against
   greedy-by-potential commands, at an equal simulator budget. Edgeflow smoke, utopia the first
   discriminative test, then uf2 / cannonball.
6. Only then the generations of section 6; the amortizer last.

Prototypes so far (vocabulary sketches, not evidence): tools/ramps.py (point hull - to be redone
per section 3), tools/ramp_route.py (proximity contacts - to be replaced by step 1).

## 8. Second review (Fable, docs/review-fable-ramps-2026-09-27.md) and the merged order

Fable agrees with sections 2 and 5's structure (ramps = actions, exact states = nodes, a C-miss is an
outcome of command B) and disagrees on what the design is FOR and on the order:

- **Two products, keep them apart.** (1) A move OPERATOR that lets the archive witness ramp-to-ramp
  transitions it never draws today; (2) a deployed command-following executor. Only (1) is needed
  to close utopia, and it needs no new training: "go to ramp B" as a LINE (a Hermite arc arriving
  in B's plane, as edge_archive.tangent_curve already does for the first surface hit, then B's own
  down-slope centreline), flown by the existing mover inside edge_archive (`--moves ramp`). The
  deployed artefact stays the flat policy on the found self-route (the only thing in the ledger
  that finishes maps), so the executor needs "one witness in K seeded tries", not reliable capture.
- **Executor reward, when one is trained:** "landing on B + time penalty" re-creates the utopia
  failure (arrive slow, still succeed). Use an option-CHAIN return: the episode continues through
  the next 2-3 commands of the search's chain, +1 per capture, death forfeits the rest (skill
  chaining); no speed term needed.
- **Command interface:** a highlight channel alone cannot carry targets behind or above the agent
  (uf2's U-turn and launch), and under --view-absolute the camera IS the steering. Ego-frame scalars
  (vector to the nearest point and the down-slope end, normal, extents, visible fraction) plus the
  mask, and a contact-phase scalar from the core.
- **Sampled, not greedy:** the ledger's greedy clones die where sampled flights land (landing bench,
  greedy dead by 1 s vs 23/64 sampled), so v1 uses K seeded samples with common random numbers.
- **Extractor defects in the prototypes:** the unreachable sentinel in potentials (uf2 d_max 34,360,
  utopia 167,812), 25-deg chaining merging uf2's arch (65 deg normal spread), a large within-ramp
  potential spread on utopia (a mean is not an ordering: use the down-slope end quantile),
  duplicate pairs; floors must be targets, ceilings an outcome class.
- **Search details:** untried-command optimism 1 vs h <= 0.5 makes the search breadth-first
  (progressive widening by default); namespace witnesses by executor hash AND render device (the
  lidar march is not bit-exact across cards).
- **uf2 will still fail after edgeflow and utopia pass:** the shaft launch is contactless air
  strafing needing ~1,700 u/s arrival that no policy-owned state carries. Ramps fix proposal
  geometry, not that skill; the gate is the R5 chain (ride R5, U-turn, ride south) trained with the
  chain return, measured as arrival speed at the shaft (the record as a ruler only).
- **Edgeflow cannot discriminate the search** (greedy-by-potential passes its monotone A-frame
  chain): it is plumbing. Utopia R5 -> R8 is the pre-registered first real test.

**Merged order (proposed to the user):**
1. In parallel: contact instrumentation from the core's movement collisions (Codex step 1) and the
   extractor fixes (validity mask, normal-spread cap, down-slope end quantile, floors as targets).
2. `--moves ramp` in edge_archive with line transport; the vocabulary test at an equal flight budget
   against `--rays 3` and `--moves prim` on blue025 / blue200 (plumbing) and utopia (the test);
   the steerability confusion matrix from the same blue025 exact nodes (command ramp k vs k+1).
   Pre-registered: flights to the first finishing chain and its 32-replay rate; on utopia, the
   furthest live node on the finisher's timeline and whether any node passes the R8 landing
   (t 10.75 s) within 20% of the finisher's speed; the number of distinct witnessed transitions.
   CONFIRM = blue200 within the rays' budget and utopia past R5 -> R8 at <= 2x 597,834 expansions;
   KILL = no gain and no node past R8 at 2x with a clean confusion matrix.
3. If utopia passes: its chain -> the unchanged flat self-route recipe on utopia, then a second map.
4. If utopia stalls with a clean matrix: the executor (chain return, scalar target block + mask,
   spawned from archive nodes), Codex's causal interface test, then the search with K seeded samples.
5. uf2 after 4, gated by the R5 chain measurement.
6. Generations, then the amortizer.

## 9. Codex on the merged order (bus 18:15Z)

- Agrees with the two-product split and with a line-transport operator before any trained
  executor; no objection to an edgeflow smoke or a provisional utopia run in parallel with the
  contact / extractor work.
- **Asymmetric reading of the provisional utopia run.** A replayable crossing of R8 (or a finish) is
  positive evidence for this geometry-line operator. A null kills nothing, because the treatment
  changes four things at once (vocabulary, target point, Hermite arrival, down-slope tail) on ramp
  objects known to be defective (CLIP / floors / walls missing, merged curves, duplicates,
  sentinel values). A decisive utopia negative needs contact truth and the minimal extractor fixes.
- **Protocol:** budget by total simulated physics ticks, not expansions (a ramp operator has ~70
  actions per node), counting every sampled attempt, with the controls rerun on the same
  checkpoint, code and device. A direct command must persist until the first new contact or a fixed
  global timeout (R5 -> R8 takes ~4.5 s against 2 s archive moves); candidates may be ranked but not
  deleted, so R8 gets budget. Add `prim_tangent` (the same Hermite arrival aimed at generic surface
  points) as the attribution control. Drop the "within 20% of the finisher's speed" gate (it echoes
  the withdrawn 80%-pace bench) and report the continuous arrival state; jt3ANCHU only scores the
  timeline afterwards.
- **Restored nodes are not exact states:** SurfState lacks PmPersist, push-once bits, the last
  yaw / pitch deltas and the side-hold latch, and the Flyer rebuilds the wrapper. Any R8 witness is
  validated by an uninterrupted true-start replay of the stored chain (the 32-replay panel).
- **Common random numbers** need counter-based random tapes keyed by (node hash, sample k, decision
  index, action head), every candidate on all K tapes, and a disjoint held-out panel for the chain
  probability; namespace outcomes by executor hash, render device, restore contract and panel.
- **Option-chain return,** if tested: the full command suffix [B, C, D] observable (the key is the
  command stack), one fixed horizon H, reward only direct captures, terminate on wrong contact /
  death / timeout, no repeated-credit farming; H = 1 vs 2 / 3 on search-owned chains.
- **A rules issue in Fable's review:** its uf2 "R5 north / U-turn / south" training chain comes from
  the human record's anatomy. Under CLAUDE.md 0 / 0b the record may only SCORE behaviour; executor
  training chains must be produced generically by the policy and the search. Section 8's point 5
  gate is therefore a measurement, not a training design.
- **Target block:** no raw ramp IDs, world boxes, mean normals or potential. One ego frame:
  normalized direction and log-distance to the nearest and exit points, the local patch normal /
  tangent, log extent, phase bits (on source, departed, on target, other) and clipped time since
  contact; the scalar command stays distinct when the mask is all zero; include a behind-the-agent
  case (edgeflow cannot expose it).
- **Four verdicts kept separate:** extractor / contact validity, line-transport proposal value,
  learned command following, and the final flat self-route recipe (from the true start, repeated on
  a second map with unchanged constants).
