# Adversarial review: the planner / executor work since the edgeflow benchmark

Written 2026-09-25 15:29 (+0200, machine clock) against branch `petrusnight` at commit c3aae27,
with the UNCOMMITTED `set_flat` edits then on disk noted where they matter (goalarc.py,
goals.py, goalprimplan.py, goalsys.py, goalsearch.py, test_goal_primlearn.py). Nothing was
trained or launched. Reviewed: docs/planner-work-since-edgeflow.md (the document under review),
docs/planner-design.md, docs/hrl-reward-survey.md, the ledger entries they cite, and the code:
python/surfgym/goalprimplan.py, goalprim.py, goalsearch.py, goalsys.py, goallearn.py (plan_gae),
goalarc.py (MultiArcProgress), respawn.py (weight_fn), rewards.py (success bonus, spawn pool),
record.py (trailer), python/train_fast.py (primlearn wiring, --exec-cut GAE), tools/record_ckpt.py
(search / MCTS wiring), tests/python/test_goal_primlearn.py, tools/run_arm.sh. Numbers marked
[computed] were re-derived here with a short script (map bounds through the core, start->finish
distances from the zones files, tanh-squash statistics by sampling, progress.csv medians).
Line numbers are at c3aae27. ASCII only.

## 0. Verdict

1. No defect was found that invalidates the edgeflow numbers (blue025 / blue050 from step 1,
   blue100 from the chain). The goal-system call order, the --exec-cut GAE, the death / cap
   refund, the late-end patch, the search's continuation of the executor wrapper and the finish
   flag all check out (section 1.12 lists what was checked so nobody re-checks it).
2. THE RECIPE cannot run on the benchmark it exists for. On a map of cannonball's size
   `--plan-cover` is refused by its own memory budget (1.2); on any map longer than the 30 s cap
   the refund rule halves the forward pull from the start and hides the finish (1.3); the
   planner's bank input clips on cannonball, unitfarmer2 and utopia (1.4). None of this is
   visible on edgeflow.
3. The planner's action distribution is built to saturate: the entropy bonus is computed on
   the PRE-squash Gaussian, so it rewards the pinned-at-the-bounds behaviour blue200 died of.
   `--plan-mu-bound` treats the symptom (1.1).
4. What was built is a 2-second latent-action hierarchy with a Euclidean progress reward, not
   the planner docs/planner-design.md specifies: no search in training, no memory, no plan
   continuity, a fixed curve family the executor does not fly (completion 0.4%). The forward
   pull is the shaping bias that exact potential-based shaping removed, i.e. the flat agent's
   Euclidean potential one level up (2.1-2.3).
5. The evidence that the planner beats a flat agent on blue050 / blue100 is missing: every
   flat null on those maps predates the spawn fix and ran with different flags (3.1).

## 1. Defects in the code (ranked)

### 1.1 The entropy bonus rewards saturation at the action bounds
Where: goalprimplan.py:269 (`mix_entropy`), :255 (log_std clamp [-3, 0.5]), :244 (init
log_std = log 0.5), `PRIMLEARN_DEFAULTS` plan_ent 0.01.

`mix_entropy` is the differential entropy of the pre-squash mixture; the action is tanh(u).
As sigma grows the pre-squash entropy rises to its clamp while the entropy of the SQUASHED
action falls and its mass piles up at +-1 [computed, one dimension]:

| mu | sigma | H pre-squash | H of tanh(u) | samples with abs(a) > 0.9 |
|---|---|---|---|---|
| 0 | 0.50 (init) | 0.73 | 0.50 | 0% |
| 0 | 1.00 | 1.42 | 0.67 | 14% |
| 0 | 1.65 (the clamp) | 1.92 | 0.31 | 37% |
| 1 | 1.65 | 1.92 | -0.09 | 45% |
| 2 | 1.65 | 1.92 | -1.15 | 64% |
| 3 | 0.50 | 0.73 | -3.85 | 100% |

So the bonus is a constant push toward the clamp in every update, and once samples saturate
there is no restoring force on the mean (u = 4 and u = 6 are the same action, so their
advantage difference is noise). The logged plan/entropy of rec_b025 / rec_b050 (median 9.2)
is sigma about 1 per dimension (14-33% of samples in the outer 10% of the range); flat_b050's
11.9 is at the clamp. The ledger's diagnosis of blue200 (nothing penalises the pre-squash
means) is half of it: the entropy term actively pushes. The `plan/entropy` "collapse" alarm
cannot fire on this proxy either.

Fix: compute the log density and the entropy on the squashed action (SAC's correction,
log pi(a) = log N(u) - sum log(1 - tanh(u)^2), entropy by sampling), or a Beta policy on
[-1, 1]. Either removes the need for `--plan-mu-bound` and restarts the planner: rerun on
blue025 + blue050 first (the restart the 10:06 ledger entry feared is needed regardless).

### 1.2 `--plan-cover` (hence `--plan-return`, hence THE RECIPE) is refused on a large map
Where: goalprimplan.py:91 (`COVER_MAX_BITS` 4e8), :385-390 (the ValueError), train_fast.py
:7452 (`--plan-return` refused without cover).

The episodic visited set is a dense bool bitmap n_envs x (cells of 128 u over the map's whole
box) [computed with core.map_bounds()]:

| map | box cells at 128 u | x 2048 envs | budget 4e8 |
|---|---|---|---|
| surf_edgeflow_blue025 | 5,096 | 1.0e7 | ok |
| surf_edgeflow_blue200 | 14,196 | 2.9e7 | ok |
| surf_petrus_lite | 124,800 | 2.6e8 | ok |
| surf_src_cannonball | 10,249,200 | 2.1e10 | REFUSED (max 39 envs) |

Fix: keep each env's episode cells sparsely (a primitive adds ~10-30 cells: a per-env key
array tested with np.isin when the primitive closes, or a global hash of (env, cell) -> episode
id), or index only the map's reachable cells (the `cover` grid is about 1% dense).

### 1.3 `--ep-secs 30` under `refund` changes meaning with the map's length
Where: goalprimplan.py:557 (a time cap is a FAILED end, pays -max(bank, 0)); run.json ep_ticks
3000, stall_secs 30.

Under refund progress is paid at once and refunded at the cap, discounted by 0.95 per
primitive in between: on a route the cap cuts (cannonball finishes in ~76 s, celestial ~43 s)
a start episode keeps only about half of its progress pay and never sees the +10, while
mid-route reservoir spawns see both. The value head has no time input (hrl-reward-survey.md
section 7 already names Pardo 2018), so the cap is hidden state. On edgeflow (10-25 s routes)
none of this applies, which is why the constant looks generic. Also `stall_secs` = `ep_secs`,
so the stall kill is inert under the recipe (harmless, but say so in the ledger). The survey's
version (B) - refund at death only, bootstrap V at the cap - or a time-left scalar is the fix;
neither is implemented.

### 1.4 Absolute units in the planner's reward and observation
Where: goalprimplan.py:207 (bank clipped to [-5, 5] per 1000 u); plan_progress per 1000 u vs
plan_finish_bonus 10.

Start->finish Euclidean distance [computed from the zones files]: edgeflow 2,731 u, petrus
4,693 u, unitfarmer2 6,006 u, cannonball 12,949 u, utopia 19,256 u. On the last three the
planner cannot see the stake the refund rule will charge, and the whole route pays 6-19 of
progress against a fixed +10 finish (2.7 on edgeflow): the finish / progress balance flips
with map size. Fix: progress and the bank as fractions of d_spawn (what `plan/ep_prog`
already does), log-scaled in the observation. The same applies, less urgently, to every other
absolute constant (192 / 384 u corridors, 128 u cells, 4,000 u rays, 300 u/s floor, 2 s).

### 1.5 The completion judge has the loophole `--plan-strict` closed on the maze
Where: goalprimplan.py:336 and :899 (`MultiArcProgress(..., window=16)`), :493 (`comp`);
goalarc.py:243-282.

The tracker searches +-16 vertices (2,048 u) around its anchor and takes the nearest point
inside the 192 u corridor. A 2 s primitive at 300-1,000 u/s is 600-2,000 u = 5-16 vertices,
so the whole line is inside the window and "arc >= 0.9" means arriving within 192 u of the
end by ANY route - the judge loophole of the 2026-09-23 "(afternoon)" entry, closed on the
maze with window 2 and left open here. `exec/complete` is therefore lenient exactly for the
slow primitives, and the executor still completes 0.4% of the planner's primitives (rec_b050
median [computed]). The completion corridor (192 u, `--goal-radius`) also differs from the
executor's pay corridor (384 u, `race_arc_corridor`).

### 1.6 The strict tracking metric measures speed change, not obedience
Where: goalprimplan.py:357 (`ncurve = secs / 0.01`), goalprim.py `curve` (DT 0.01, traced at
max(start speed, 300 u/s)).

The curve is time-indexed at a hard-coded 10 ms per tick and at the START speed. Any speed
change (every ramp) grows the time-aligned error on a perfectly followed path, and under
`--tick-ms 7.63` the index runs 30% fast. "strict 0.16 / lenient 0.54" in the ledger reads as
disobedience; only the lenient number supports that.

### 1.7 No launcher and a wrong default
Where: tools/run_arm.sh (no primlearn branch); train_fast.py:7444 (`plan_shaping` defaults
to "pbrs", the variant that wandered).

Every recipe run was hand-typed (runs/research/gate_bench/research_rec_b050_launcher.txt).
CLAUDE.md section 4 exists because of exactly this. A primlearn run that omits
`--plan-shaping refund` is a different arm with no warning.

### 1.8 Every planner recording's trailer says "fail"
Where: record.py:208 (`end = "done" if r0 >= 25`): r0 is the core's step reward; the +50
lives in the wrapper. The document knows; the fix is one line (read `core.goal_hits`).

### 1.9 The flat_b050 entry in the ledger (930ddfa) is confounded
The executor's arc reward, fan and completion used a 3D corridor of 384 u around a flat line
drawn at the spawn height, so the executor was paid only while staying level: "the planner
shrinks to small safe moves" and `plan/adv_real` -140 u describe that reward, not flat
primitives. The uncommitted `set_flat` edits on disk say the same ("the first flat run circled
on the spawn platform"). The ledger entry needs a CORRECTION when that lands, and flat_b050 a
rerun; until then it is not a result.

### 1.10 The search continues the wrapper through private attributes
Where: goalsearch.py `_row_of` / `_policy` (`_held`, `_tick`, `view`, `keys`, `_keys_tick`
through getattr with None defaults).

A rename silently reverts the simulation to released keys and a fresh decision - the 82c955a
failure (0/9 with search vs 2/9 without). `test_trainer_and_recorder_run_primlearn` only
asserts that the fidelity line prints; assert the median end error there (it is 0-1 u today).

### 1.11 Hygiene
* goalprimplan.py:608 rebinds `pos` (the positions array) to a bool mask inside `on_tick`;
  nothing reads `pos` after it today.
* `load_state_dict_all` drops a mismatched optimizer state and the counts silently; a
  cross-map resume prints nothing about it.
* `_Edge.q` recomputes the subtree max recursively at every selection (fine at 16-48
  expansions; cache before scaling).
* The scratch core steps all K slots even when fewer candidates exist (unused slots fly a
  straight primitive).
* `describe()` and viewer/runs.js:76 still print the old reward description (doc 7.3).

### 1.12 Checked and NOT a bug
* `goalsys.on_step` (the death charge reads the bank) runs before `goalsys.assign` (the new
  spawn zeroes it): train_fast.py:14147 vs :14442, same tick.
* `--exec-cut` marks decision t, the last under the old plan (:14644-14649), and the GAE
  masks both the bootstrap and the recursion (:14695-14697).
* `goalsys.map_center` is set under primlearn (:12571), so an ended episode's terminal
  position decodes and the finish pays the right progress.
* `goal_hit` is per step (src/env.c:709), so a scratch slot teleported with `set_state`
  cannot inherit a finish; `pending_fail` is consumed per step too.
* The wrapper's decision phase (`_tick % k`, train_fast.py:2483) matches the eval's
  `(t + 1) % K` and MCTS's `ready` gate; the state's `tick` field carries the episode clock
  into the scratch core, so a simulated cap truncates like the real one.
* The late-end patch (goalprimplan.py:618-640) and a mid-primitive death agree: both net
  -bank at the primitive's start under refund and under pbrs.
* `v_boot` = 0 for a waiting env cannot bite: `replan()` runs before `update()`, so no env is
  waiting at an update.

## 2. Design flaws of the recipe and the search

### 2.1 It is not the planner the design doc specifies
docs/planner-design.md sections 6-7 (the user's requirements): trajectories of jump points
(full states), exact jumps by forked simulation, MCTS with the tree re-rooted between
decisions, pi learned from visit counts, memory in a global archive queried at decision time,
no predefined trajectory set in the long run. THE RECIPE: PPO over 6 continuous numbers every
~2 s from 24 rays + the finish vector + speed + the bank, no memory of any kind in the
observation, no search in training, the tree rebuilt at every decision at eval only, a fixed
parametric curve family. The requirement met in spirit is "velocity in the state". The "no
fixed vocabulary" requirement is met only because the executor ignores the curves (2.3). The
write-up should say this plainly; today it is spread over 5.2-5.4.

### 2.2 The forward pull is the flat agent's Euclidean potential, one level up
The module docstring proves it: exact potential-based shaping (v3) telescopes to zero and the
planner wandered (3.3% of the route); `refund` pays Euclidean progress as it comes and refunds
it only at a failed end - a non-potential bias toward a smaller Euclidean distance. On edgeflow
the finish is roughly ahead, so it works. On cannonball (the route folds; the 88.8% wall is
where the potential says turn back) and unitfarmer2 (the pit) Euclidean progress is the very
signal the planner was built to escape, now at 2 s granularity with a value head that has
never seen the +10. The only exploration term is count-based coverage over 128 u cells, the
mechanism family CLAUDE.md section 0 records as eight nulls at the tick level on unitfarmer2.
The untested hypothesis is exactly "does 2 s commitment make count-based exploration work?".
Only unitfarmer2 answers it; no such run exists, and 1.2-1.4 mean it cannot be launched as is.

### 2.3 The executor decodes; the curve is not load-bearing; the code pays for it anyway
Completion 0.4%; planned +32 u vs realised +264 u per primitive (rec_b025); the override
ablation (the executor needs the planner's numbers and tracks meaningless curves BETTER).
So the 6 numbers are a latent action and everything downstream of the curve - polyfit,
resampling, the arc judge, the tracking metrics, the obedience gate, "MCTS over primitives" -
describes a nominal object. Two consistent ways out: (a) make following load-bearing: a
following reward with no dead zone (hrl-reward-survey.md section 7: a pull toward the plan off
the corridor) plus the obedience gate on the planner, so the curves mean something and MCTS's
children are geometry; (b) drop the curve: hand the 6 numbers (or a learned latent) to the
executor directly with the same 2 s commitment, HIRO / Director style, and stop measuring
curve-following. Today it is (b)'s semantics at (a)'s cost, and the viewer draws curves nobody
flies.

### 2.4 Zero exploration at eval, no memory in the planner
The verdict is greedy planner + greedy executor from the start with no visit information: a
planner that fails from the start fails identically nine times. In training the exploration is
the mixture's sigma (with 14-33% of samples per dimension at the bounds, 1.1), count-weighted
coverage the policy cannot condition on (it sees no counts), and count-weighted reservoir
draws. The requirement "no fixed context window" was met with no context at all.

### 2.5 The start decision is starved by construction
`respawn_frac` 0.9 x `plan_uniform` 0.5 (applied to every fresh episode, the start included)
x the count-weighted return (which down-weights the most-visited cell of all, the start): the
decision the verdict tests is about 5% of episodes and about 1% of planner transitions. The
ledger found it on blue200; it is generic and wants a generic fix (exempt map-start spawns
from the uniform opener, or a floor on the start share), not `--respawn-frac 0.7` per map.

### 2.6 No time in the planner's objective
No per-primitive cost; gamma 0.95 per primitive whatever its duration (goallearn.py:679
`plan_gae`), while the verdict on a finished map is wall time. A 3 s time-out and a 1.5 s
completion are discounted alike; `--plan-mcts-time` patches it at eval only. An SMDP discount
by duration in `plan_gae` and in the tree is the consistent fix (the survey says so).

### 2.7 The executor's hard cut discards momentum
`nonterm = 0` at every re-plan (train_fast.py:14695): the executor's value at the end of a
primitive is 0, so arriving fast or well placed for the next primitive is worth nothing to it.
On surf, speed is the state. Bootstrapping under the SAME plan (HIRO, Director) closes the
farm equally - the arc pay per line is bounded - and keeps the speed incentive. One arm.

### 2.8 Coverage and novelty live outside the bank
They are never refunded, so a death after collecting them keeps them. Early in a run they
dominate (first log rows: cover 0.22 + novelty 0.13 per primitive vs progress ~0); later they
are ~0.5% of the reward, so count decay handles it in practice. The survey's own remark (a
coverage bonus even inside the bank rewards delaying the finish) applies more strongly
outside it.

### 2.9 MCTS as built cannot answer the blue200 question
blue200's route is 8,583 u; at the ~235 u/s these evals average (blue050: 3,058 u in 13 s)
that is over 30 s and 15 or more primitives, against a tree 4-6 primitives deep. "No finish
in any tree" is the horizon, not the executor. Below the horizon the tree rests on a value
head trained on blue200 without a single start-to-finish, with coverage and novelty baked
into its targets. Max backup over K noisy value-head leaves is optimistically biased with
depth (max of K estimates), so deeper trees look better regardless of the map. The measured
"MCTS = depth-1 search" is what this predicts. The fidelity numbers (median 0-1 u) confirm a
bit-exact continuation of a deterministic simulator, i.e. the plumbing, not a model; depth 2+
is not checked but would read the same.

### 2.10 Reservoir harvest under the recipe
`respawn_margin` 1 s with whole-chain harvesting of finished episodes (`success_margin` off,
respawn.py:214-220): once the recipe finishes, the reservoir piles up near the goal and
`plan/finish` (37-44%) is the harvest, not the policy - the trap CLAUDE.md section 3 records
for `race/win_rate`. `plan/finish_start` (25-35%) is the honest number; the document uses
it. Do not quote `plan/finish`.

## 3. What the evidence shows and does not

### 3.1 The matched flat control is missing
After the spawn fix (0e6b320, 2026-09-24 19:26) no flat / no-planner arm ran on blue050 or
blue100 (ledger grep: only tpFLAT, a 40M-step timing probe). The pre-fix blue050 nulls (waves
1-5, efCTLnp_blue050) ran with 4 of 16 spawns embedded and different flags (episode length,
`--int-coef 0.25`, respawn settings). The override ablation shows that an executor TRAINED
with a planner needs it at eval; it does not show that a flat agent cannot learn blue050. The
control is prim1_b025's executor with the recipe's flags and no planner (straight or uniform
primitives, or the plain race reward) on blue050 and blue100.

### 3.2 "The recipe from step 1" holds for blue025 and blue050 only
blue100 needed a 2.4B-step chain through the retracted variants (p2_b100 -> ret_b100w ->
ret_b100L) and blue200 failed under three variants. One seed each and near-binary verdicts:
rec_b050's evals went 0, 0, 1, 2, 2, 7, 8, 9, 4, 8, 5, 8, 8, 8, 9 of 9, so CLAUDE.md's
gate-ladder retraction applies to any two-arm comparison on these maps.

### 3.3 The write-up's own inconsistencies
Section 7 of the document lists them (heading times, ledger vs data, code vs docs) and they
are real; add 1.9 (flat_b050 confounded) and the missing control (3.1).

## 4. Recommended order of work (one arm each; same flags on blue025 + blue050 first)

1. Fix the squash (1.1): entropy and log density on the squashed action, drop
   `--plan-mu-bound`. Rerun the recipe on blue025 / blue050, only then blue200.
2. Make the recipe launchable and scale-free: sparse episodic coverage (1.2), progress and
   bank as fractions of d_spawn (1.4), refund at death only with a bootstrap at the cap or a
   time-left scalar (1.3), a `run_arm.sh` branch with `refund` pinned (1.7).
3. Run the matched flat control on blue050 / blue100 (3.1) before any claim that the planner
   is what passed them.
4. Then unitfarmer2 from the true start - the benchmark the planner exists for. Until it
   runs, the edgeflow ladder has measured a 2 s latent-action PPO with a Euclidean reward,
   not macro exploration.
5. Decide 2.3 (a) or (b) before spending more on MCTS over curves; if (a), the following
   reward without a dead zone comes first, and MCTS's children then mean something.
