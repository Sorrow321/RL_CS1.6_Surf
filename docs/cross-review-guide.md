# Cross-review guide: one exploration recipe for four surf maps

For an outside reviewer (code + ideas). Repository `RL_CS1.6_Surf`, branch
**`petrusnight`** (head at the time of writing: 3115437). Everything below is
on that branch; the ledger `docs/research-results.md` is append-only and holds
every run with its numbers; `CLAUDE.md` holds the operating rules the results
were produced under.

## 0. The goal, and the one hard rule

**Goal.** A single, robust, champion-free exploration mechanism for a
from-scratch PPO surfer that beats four maps from the true start:
`surf_src_cannonball` and `surf_src_celestial` (easy-medium),
`surf_petrus_lite` (medium), `surf_unitfarmer2` (hardest - the open one).
"Beats" = the greedy policy finishes the map (or, on unitfarmer2 today,
passes its start gate: contact -> in-pit speed >= 1,400 u/s -> exit).

**The rule.** Human demos (world-record recordings) never enter RL training in
any form: no spawn states, windows, spines, behaviour cloning, route lines,
arc coordinates, shaping targets, or warm starts from checkpoints trained on
any of those. Demos are for analysis and measurement only (`CLAUDE.md`
section 0). On 2026-09-13 a demo-window run "passed" unitfarmer2's start and
was disqualified for exactly this. Anything derived from the POLICY'S OWN
recordings is allowed.

**The second hard rule (user, 2026-09-14): UNITFARMER IS A BENCHMARK, NOT
THE GOAL. NO MAP-SPECIFIC METHODS.** "We need generic exploration mechanism
(or algorithm redesign with optimal search space) that will allow us to
beat any map." A reward term, spawn rule, threshold or schedule built after
looking at this map (the dip-speed bonus and its cap, a climb bonus, a
pit-tuned speed gate, anything that mentions a gate box) is an ad-hoc method
and is disqualified as a recipe component; such runs are map ANALYSIS, never
results. The recipe must run with the SAME flags and constants on every map
and repeat on a second map before it is called a winner. Full text:
`CLAUDE.md` section 0b.

## 1. What the system is

* **Simulator**: a C reimplementation of GoldSrc surf physics (`src/`),
  batched over thousands of envs; 10 ms tick; the agent decides every 4 ticks.
* **Observation**: a 64x32 lidar depth image plus a second image channel with
  the geodesic goal potential, plus scalars (velocity, view, ...). Action:
  held movement keys (forward/side/duck/jump) as categorical heads with a
  "keep" bin, and an absolute continuous view direction in the velocity frame
  (`docs/contyaw.md`).
* **Reward** (`python/surfgym/rewards.py`, `RaceReward`): potential shaping on
  the geodesic distance-to-goal field (`python/surfgym/goalfield.py`, a BFS
  over reachable voxels at a 32-48 u cell), a per-tick time penalty, +50 on
  finishing, no death penalty. `--race-ratchet` pays only NEW progress records
  within an episode (a detour costs only time). Optional count-based novelty
  `--int-coef` (bonus `c/sqrt(visits)` on entering a 256 u cell keyed by 8 yaw
  sectors x 3 speed buckets; counts global, self-annealing).
* **PPO** (`python/train_fast.py`): 2,048 envs, rollout 32-128 decisions,
  4 epochs x 16 minibatches, lr 3e-4, gamma 0.9995 per physics tick (20 s
  horizon; raised to the decision rate internally), GAE 0.95, clip 0.2,
  entropy 0.005, value coef 0.5. Episodes capped at 30-120 s; a stall kill
  after 15 s without progress (training only, never in evals).
* **Start-state machinery**: 90% of episodes respawn from a reservoir of the
  policy's own states harvested `--respawn-margin` seconds before an
  episode's end (1 s on the gate work), 10% from the true start; the reservoir
  can be replaced by a fixed window of states (`--demo-file`, ONLY from the
  policy's own recordings now), a forward frontier curriculum
  (`--respawn-frontier`, `docs/respawn_frontier.md`), uniform random reachable
  states (`--respawn-random`, `docs/respawn_random.md`), Go-Explore /
  Florensa / backward bin weightings (`--respawn-mode`).
* **Exploration machinery**: `--unstuck` (`docs/unstuck.md`): a plateau
  detector that raises a sampling temperature T on the keys heads (logits /
  (1+T)) when the run's progress measure stalls, and decays it on improvement;
  `--unstuck-reach alive` makes that measure the deepest point an episode
  reached and survived 3 s later (dives and deep spawns cannot saturate it).
  `--curiosity-cond` (`docs/curiosity_cond.md`): Agent57-style T-conditioned
  family - each env draws its own T per episode, T is an observation column,
  reward mix `(1 - T/T_max) x shaping + T x novelty`, the T = 0 member is pure
  race reward and is what the greedy eval runs. `--spawn-burst`: random
  actions for N decisions after a respawn, excluded from the update.
* **Evaluation**: greedy episodes from the true map start; the recorder
  `tools/record_ckpt.py`; the gate bench `tools/gate_bench.py` (pass rules,
  ramp-box contact, in-box speed, ends plotted over the potential);
  `tools/gate_report.py` (per-run table of the gate columns); the anatomy
  tools `tools/bev_potential.py` (potential field + record + policy line
  around a death) and `tools/uf2_entry_bench.py` (which sub-skill a policy
  lacks, spawning it from record states - measurement only).

## 2. The method that works on the easier maps (all champion-free)

Every gate found so far is a place where the potential's steepest descent
runs into a void (a wall, a fall) while the real line goes sideways or up. The
recipe that passes them ("THE METHOD" in the ledger, 2026-09-13):

1. **Base recipe to the gate** (potential shaping, held keys, absolute view,
   reservoir respawn). The greedy line reaches the gate and dies there.
2. **Window + temperature**: cut a window of the policy's OWN states from its
   greedy line 1.5-4 s before the death (`record_ckpt --dump-states` +
   `gate_bench.py spine`), spawn 80-95% of episodes from it, and run the keys
   temperature (`--unstuck`, cap 1). The sampled behaviour discovers the
   continuation because entrants are sustained until the continuation is
   learned (discovery density, not credit).
3. **T = 0 consolidation** of the same window (the tempered policy and the
   greedy policy differ; PPO optimises the tempered one).
4. **Continuation from the start** with the window share fading (0.5 -> 0.2)
   and 60-90 s episodes.

Results: celestial finished from the true start (`gsCELunstuck6`: 14/16
sampled at 42.3 s, best 42.04; its window was cut from the greedy line of
`jt3ANCHU`, a joint scratch run with no demo, BC or route - clean); petrus
finished 9/9 at 32.0 s in a joint cannonball+petrus scratch run (`jtANCHU`,
round 40 - clean); utopia finished 9/9 at 53.66 s (record 52.67) in
`jt3ANCHU` - clean. **Cannonball is NOT a champion-free finish** (GPT
cross-review, 2026-09-13): `gbCANfin3/4` trained on a window cut from
`exitABS` round 9, and `exitABS` descends through BC rows and a spine from
`cySPINEW`, whose spine came from a finisher of the champion-line era; the
human record on cannonball is 68.60 s, the demo-tainted finisher 77.3 s.
Cannonball must be re-run with a window from a CLEAN failing policy's own
line before the method is credited with it. Ledger entries: "gsCELunstuck6",
"Round 40, arm jtANCHU", "THE METHOD in one place", and the correction dated
2026-09-13 19:20.

**What is manual in it, and therefore not yet "a mechanism":** finding the gate
(the death point of the greedy line), cutting the window by hand, and choosing
when to switch T on and off. A robust mechanism has to do all three by itself.

## 3. Why unitfarmer2 breaks the method

Its start gate is not a void gate but a **dip gate**: the record dives south
into a pit (potential rises +2,840 u over 2 s, speed 765 -> 1,765 u/s), surfs
the pit's ramps and flies back out; the potential says north, where a slide
banks +2,777 u by 6.3 s and dies. The policy's own line never contains the
alternative, so a window cut from it cannot teach it. Facts established by
measurement (details and every number in `docs/uf2-exploration-review.md`):

* the decision is the heading in the first 2.5 s off the platform;
* the dive must be uncharged (ratchet) or the policy steers out of the pit
  and dies even from inside it;
* entrants die within 2 s unless the pit surf has been practised from states
  inside the pit; the surf and the exit are learnable in < 1B from there;
* the gate has three rungs: pit contact -> in-pit speed >= 1,400 u/s (the
  first ramp surfed down) -> exit (depth >= 5,600 alive);
* the only champion-free mechanism that produced entries from the true start
  is count-based novelty at 10x the base coefficient (greedy 6/8 enter), and
  the same bonus then keeps the agent crawling in the pit (878-928 u/s inside;
  never the first ramp). Eight other mechanisms were null at 1B each
  (temperature on keys / all heads, frontier curriculum, speed reward, surf
  reward, time-penalty bias, random reachable starts, spawn-heading
  diversity). In flight at the time of writing: novelty annealed, 4x novelty,
  position-only novelty, the T-conditioned family at 10x, spawn bursts,
  curiosity-cond at base, Go-Explore bin weights, keys T cap 2.

## 4. What we would like reviewed

### 4a. Code (please read, not just the docs)

* `python/surfgym/rewards.py` - `RaceReward`: the shaping, the ratchet, the
  count-based novelty (cell key, `1/sqrt(visits)`, the count decay under the
  temperature), the curiosity-cond mix, the stall kill. Question: is the
  novelty as implemented a sound primitive for a one-time heading decision,
  and is its interaction with the ratchet and the count decay what keeps the
  agent camping?
* `python/train_fast.py` - `UnstuckSchedule` and `AliveReach` (the plateau
  detector and its progress measure), `--curiosity-cond` plumbing (search
  `CC_COL`, `cc_encode`), the respawn/reservoir harvest (search
  `respawn_margin`, `stash_depth_bins`), the GAE and the truncation bootstrap
  (a KNOWN defect: under `--obs-reward` the bootstrap reads a raw scalar; not
  used on these arms), the per-decision reward accumulation (`rpd`).
* `python/surfgym/goalfield.py` - the potential's construction; the known
  failure class is that BFS distance believes the player can fly across
  voids. `docs/credit_diag.md`: GAE at lambda 0.95 anti-credits a manoeuvre
  whose payoff arrives seconds later - is the credit assignment itself part
  of why a 10-second dip is never learned from the race reward alone?
* `tools/run_arm.sh` - the launcher every run goes through (the demo guard
  at the top; the SCRATCH argument set).
* `tools/gate_bench.py`, `tools/gate_report.py` - the verdict machinery.
  Are the pass rules (depth threshold + 3 s survival, box contact, in-box
  speed) measuring what they claim?

### 4b. Ideas (ranked list with evidence in `docs/uf2-exploration-review.md` section 4)

The candidates: novelty with an anneal; the T-conditioned family at 10x;
time-windowed or state-triggered curiosity (T high for the first W seconds
or while the potential is rising, then 0); curiosity-weighted training
(sample as now, weight episodes' race-reward advantages by their summed
novelty); a position-keyed Go-Explore reservoir (the current `goex` mode
weights depth bins, which cannot separate pit from slide); then the
two-stage recipe on the agent's own pit states once entrants survive.

### 4c. The questions

1. Given sections 2-3, what would a mechanism look like that (a) finds the
   gate itself, (b) generates the alternative when the own line does not
   contain it, and (c) hands the discovery to the greedy policy - without a
   demo, and without a per-map knob? Which of the candidates is closest,
   and what is missing from all of them?
2. Is there evidence in the code or the numbers that the failure is credit
   assignment (GAE lambda, horizon, the ratchet's accounting) rather than
   exploration? The demo-assisted diagnostic says the pit route returns 26
   against the slide's 10 for the SAME policy once the exit is learned.
3. The greedy line is fragile even when learned (the trainer's jittered eval
   read 9,089 u where the recorder's greedy was 4/4; dropping the spawn
   support lost the line in 300M steps). Is that a sign the consolidation
   stage is under-specified (entropy, sampling noise, spawn jitter)?
4. Which of the eight nulls would you re-run longer or differently before
   believing them, given one seed per arm and a near-binary gate metric
   (identical configs have differed 2.7x on it)?
5. Anything in the PPO setup you would change for a two-skill gate (a
   heading decision whose payoff is gated by a skill learned only from
   inside the region the decision leads to)?

## 5. Reading order

1. `CLAUDE.md` sections 0-4 (rules; the metric caveats in section 3 matter).
2. `docs/uf2-exploration-review.md` (the open problem, all numbers).
3. `docs/research-results.md`, entries dated 2026-09-12/13 ("THE METHOD in
   one place", the gate benchmark batches, the unitfarmer2 batches 2-11).
4. `docs/unstuck.md`, `docs/curiosity_cond.md`, `docs/respawn_random.md`,
   `docs/respawn_frontier.md`, `docs/credit_diag.md`.
5. The code in 4a.

Caveats the reviewer should carry: one seed per arm (user's rule); the
frontier metric is a ladder of physical gates, so a run's score is mostly
which gate it cleared; 1B-step arms are 25-35 minutes on a 5090; every
start-line verdict is 16 episodes; and nothing measured on a rented card is
bit-identical to the local one (the lidar march differs across GPU
architectures).
