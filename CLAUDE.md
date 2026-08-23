# RL_Surf - operating rules for agents

These are the user's standing rules. They are not suggestions and they do not
expire at the end of a session. Read them before renting anything or launching
anything. Violating the GPU rules costs the user real money; violating the
experiment rules costs a whole night of evidence.

## 1. Rented GPUs: running or deleted, never stale

* **Ready in 60 seconds or it dies.** Time from `create instance` to a usable
  ssh session is 60 s. Not ready -> blacklist the machine/host in
  `tools/bad_hosts.json` (`python tools/vast_pick.py --block <id> --reason ...`)
  and `vastai destroy instance <id> -y`. No second chances, no "let me wait a
  bit more".
* **Underpowered or underperforming dies the same way.** Run
  `python tools/gpu_health.py --all`. A card that is the right model but
  clock-capped, memory-capped or slower than the reference is a defect.
  Blacklist BEFORE destroying - the identifiers vanish with the instance.
* **A box with no load for 5 minutes must be destroyed.** Deploying, baking,
  training, harvesting all count as load. "I might use it later" does not.
* **`vastai destroy instance <id> -y`.** Without `-y` the command silently
  aborts and the instance keeps billing. Always verify with
  `vastai show instances` afterwards, and re-issue until the box is gone.
* **Every rented box gets a deadline watchdog at launch time**, running
  locally, that destroys the instance when the run's budget expires or the
  trainer dies. The agent's own attention is not a safety mechanism - the
  session can end, the context can run out, the network can drop. Set the
  watchdog first, then start the run.
* **Single 3090 for research arms.** Not 2x, not a 5090, not a 4090. The
  baseline numbers below were measured on one 3090 and comparisons across
  card types are not comparable in wall-clock.

### Ops details that have already cost time

* `vastai` needs `PYTHONIOENCODING=utf-8`. The console here is cp1251 and the
  CLI crashes on its own output without it.
* **Race candidates, don't queue them.** Serially waiting 60 s per offer
  burns the night. Create 3-4 at once, register all of them, keep the first
  to reach `running`, destroy the rest. Round 16 destroyed 24
  instances this way for about $0.08 total. Stay inside the cap of SIX
  while racing (user raised it from 4 on 2026-08-23; the local GPU is
  additional and does not count against it). The cap is shared across
  ALL agents and sessions, not per agent - check
  `python tools/fleet_watchdog.py list` before creating anything. **Blacklist only the ones that actually failed readiness.** A
  loser that reached `running` and simply lost the race is not defective,
  and recording a false defect against a healthy host shrinks the pool for
  every future agent. Blacklist the ones still `loading` past the window;
  destroy the rest unblocked.
* **Never `pkill -f <pattern>` over ssh when the pattern appears in your own
  command line** - it matches the shell running it and kills the session,
  which looks exactly like a dead box. Same family as the `pgrep` self-match
  that deadlocked the remote watchers.
* The `git clone` is ~170 MB because the repo carries `video_demo.mp4` and a
  30 MB `.npz`; on a 1 MB/s box that is three minutes before anything else
  starts. `--filter=blob:none` would fix it if this ever becomes the pole.
* **Working in a git worktree silently triggers a 30-minute goal-field
  re-bake.** `train_fast.py` and `record_ckpt.py` resolve a checkpoint's map
  as `<repo>/maps/<stem>.bsp`, and in a worktree that is a *copy* with
  different mtimes. The cache signature embeds size + `mtime_ns`, so it
  misses and the trainer starts baking. It also rewrites `zones.json`. **Use
  absolute paths into the main checkout** (`C:/RL_Surf/maps/...`) for the
  map and the caches when running from a worktree, and watch the first
  minute of any run for a bake line.
* **A pre-existing bug, do not "fix" it casually:** the truncation bootstrap
  feeds raw scalar slot 12 into `V(s_T)` under `--obs-reward` instead of the
  fed reward value. Correcting it changes the single-map numbers and would
  break bit-identity with every checkpoint trained so far, so it needs its
  own arm, not a drive-by patch.
* **`scp` can silently truncate a large file and still exit 0.** With two
  agents pushing at once the shared uplink collapsed from 1.6 MB/s to
  44 KB/s and a 153 MB checkpoint arrived as 3.9 MB with a success exit
  code; only `deploy_box.sh`'s md5 check caught it. **Never trust an scp
  exit code on the checkpoint - always verify the md5 on the box.** When
  `fleet_watchdog list` shows another box already up, prefer
  `SEED_HOST=/SEED_PORT=` box-to-box seeding (28 s in that incident versus
  62 minutes of retries), and probe with a single 16 MB scp before pushing
  147 MB.
* The watchdog is the safety net, not the plan: register on create, release
  on finish. `python tools/fleet_watchdog.py list` is the shared view of what
  is rented, across every agent and session.
* **The watchdog itself destroyed a healthy training box on 2026-08-23, and
  the failure mode generalises.** Instance 48446220 had trained for 16
  minutes when the vast API reported it `offline` for ONE poll; the
  readiness rule keyed on current status plus age-since-create and killed it
  as "never came up (status offline, age 28.1m)" - **41 minutes before its
  own deadline** - taking the checkpoint, `progress.csv` and every
  trajectory. Fixed: the registry latches `ready` on the first `running`
  sighting, and the readiness kill now requires that flag to be absent; a
  later blip is logged and left to the deadline, which is the bound meant to
  hold it (`tests/python/test_fleet_watchdog_ready.py`). **The general rule
  for anything with a destroy button: a single observation is not evidence,
  and an unrecoverable action needs a bound that a transient cannot trip.**
* **Branching from an OLD branch silently REVERTS fixes and DELETES assets,
  and this has now cost real time three times.** Arm branches carry their own
  `viewer/app.js`, `tools/*.py`, `CLAUDE.md` and tracked map assets. Checking
  out a branch that predates a fix takes the fix away without saying so:
  branching from `multimap` re-broke the viewer's map resolution (the user
  had reported it twice already), reverted the watchdog to the version that
  destroys live boxes, and **deleted `maps/surf_petrus_lite.bsp` and
  `viewer/assets/surf_petrus_lite.*`**, which failed two launches before
  anyone noticed. **After ANY branch switch, run
  `git diff HEAD <integration-branch> --stat -- tools/ python/ viewer/ CLAUDE.md tests/`
  and restore what is behind.** `tests/python/test_viewer_map_resolution.py`
  now fails loudly for the viewer half of this.
* **Restart the daemon after editing `fleet_watchdog.py`.** It holds the old
  code in memory and will keep applying the old rule to every agent's boxes.
  Check with `Get-CimInstance Win32_Process` for `fleet_watchdog` and make
  sure there is exactly ONE.

## 2. Experiments: one paper, one run, one seed

* **One hour of training per ablation.** Not two, not "let it run overnight
  and see".
* **One paper = one run. One seed. More than one seed is forbidden.** The
  point is to test whether a published mechanism moves this task, not to
  produce statistics.
* **Test locally for correctness before renting.** Shape checks, a
  bit-identity check against the baseline code path when the arm is supposed
  to start identical, a few hundred local steps. A rented box is not a
  debugger.
* **Watch the run while it trains.** Obligatory, not optional: that it is
  alive, that it is not consistently degrading, what the agent is actually
  doing and where it is stuck. A run nobody watched produced no evidence.
* **Stationary for more than 10 minutes = the experiment failed.** Stop it,
  destroy the box, write down what happened.

## 3. The single metric

* Runs that do **not** finish the map: `race/eval_progress` (in
  `runs/<run>/progress.csv`, and on the dashboard).
* Runs that **do** finish: wall-clock time from start to finish.

Nothing else decides an arm. Training reward, episode length and win rate are
diagnostics for *why*, never the verdict.

**But `race/eval_progress` on its own has already been wrong once, so pair it
with the honest metric.** Round 18's xROUTE arm posted three evals at
~195,2xx - the best figures ever recorded on this checkpoint on a 3090,
better than any control - and was a complete null: all 99 episodes across 11
evals stopped at the same route vertex and none finished. The rise was
consistency (weak episodes disappearing), not progress. Always run:

```
python tools/eval_honesty.py --route maps/surf_src_cannonball.route.npz \
    runs/research/<ARM>/traj_*.jsonl
```

**Corridor MAX and finishes are the frontier.** Round 18's progression, all
from the same stuck checkpoint:

| arms | corridor MAX | past 205,440 u | finishes |
|---|---|---|---|
| xROUTE / xSP / xNECTO / xCONTACT | 205,312-205,440 | 0 / 333 | 0 |
| xMARGIN (`--respawn-margin 2`) | 208,640 | 6 / 72 | 0 |
| **xARC** (`--race-arc`, arc-length reward) | **231,680 (100%)** | **84 / 102** | **63 / 102** |
| **xAUTO** (same, line decimated to 58 chords) | **231,680 (100%)** | **81 / 102** | **62 / 102** |
| **xSELF** (line from the checkpoint's OWN runs, reaching 88.12%) | **231,680 (100%)** | **77 / 102** | **47 / 102** |

**The reference line supplies the ORDERING, not the line.** xAUTO's line was
58 straight chords of 4,096 u - 1,131 u max deviation from the champion,
24.8% of its vertices inside solid map geometry, not flyable - and it
matched the full champion line on every axis, with a better best time
(80.51 s vs 81.04 s, and the champion recording's 81.35 s). Linesight's
"does not need to be fast... usually the centerline", reproducing literally.
Neither of those two lines licenses an autonomy claim: both came from a
recording of a finisher.

**xSELF closes that.** Its line was built from the stuck checkpoint's own
270 recorded episodes (0 finishes between them) plus the constancy of
gravity - no champion, demo, goal field or human route - and **stops
1,280 u SHORT of the wall**, leaving 11.88% of the route with no reference
at all. It still finishes 47/102 at the fastest times of any arm
(best 80.06 s). **The monotone coordinate does not have to COVER the hard
part, only get the agent to it without charging it.** Truncation costs
rate, not frontier, and the direct observable is the off-corridor share
rising 9.5% -> 35.5% while both full-line arms drove it down.

Two things that matter if you rebuild such a line. Every episode of a
non-finishing policy **ends in a fall**, and an untrimmed line PAYS for
falling (that checkpoint's fallers reach 99.3% of it, its finishers 96.9%)
- trimming is what stops a second treatment contaminating the result. Of
the champion-free trim rules, the geodesic field lands *inside* the fall
(its basin is goal-adjacent airspace) and consensus across the policy's own
episodes fails because a deterministic policy falls the same way every time,
so the falls corroborate each other. What works is **the last tick at which
the map pushed back** - vertical acceleration departing from the gravity
step - which cut five independent episodes within 25 u of each other
(`tools/pick_selfline.py`).

**What is still open:** the seed is a policy that already flies 88% of the
map. Where that comes from on an unflown map is the remaining gap, and
`tools/explore_phase1.py` does not currently close it (see above).

**`tools/explore_phase1.py` does NOT reach 92.4% of the track - do not plan
around that number.** Re-run in both its default and its queued speed-keyed
configuration, it froze at the same physical gate at **6.67% and 6.68% of
arc** across 32.5M exploration runs. The old figure was GEODESIC progress,
and that field is non-injective, so it maps to either 84.5% or 92.4% of arc
and cannot be read as "past the wall". Also measured: both champion-free
ways of picking a leaf out of a Go-Explore archive are broken on this map -
the geodesic one stops a line AT the wall by construction (that field's
minimum is there), and the deepest-node one picks a dead-end pocket at 3-5%
of arc.

Always pass **`--order-only 16`**: a global argmin credits a fall with up to
46,000 u where the route folds back on itself. `tools/wall_profile.py` then
says what went wrong (speed, height, off-line error versus the champion
line).

**The standing metric has now been anti-correlated with the truth, not just
blind to it.** Through the middle of xARC, `race/eval_progress` FELL
184,390 -> 156,305 while the honest frontier ROSE 205,362 -> 223,909 and the
agent started finishing the map. Do not report an arm on `eval_progress`
alone, ever.

**`race/win_rate` is now the THIRD deceptive metric, and it has fired.**
Round 19's xPSSR posted petrus's first non-zero win rate, 0 -> **18.46%**,
while the greedy frontier sat flat at 68.6-68.8% with **0/45 finishes**. The
wins were 6.3-7.2 s long from a reservoir whose minimum depth had fallen to
1,485 u: the agent was being respawned next to the goal and walking in. This
is exactly the trivial-win trap this file predicted for `--respawn-margin 2`,
observed for the first time. **A win rate that rises while reservoir
min-depth falls is measuring the harvest, not the policy.** Report the two
together or not at all.

**Evals do NOT stall-kill. Training does.** `core.force_fail` has exactly one
call site (`train_fast.py:2921`) and it is inside the training rollout, so
`--stall-secs` never applies to an eval episode. A policy whose training
`ep_len_mean` is pinned at 1,502.6 - i.e. every episode killed at 15 s - will
still show **120 s crawling episodes in its evals**, because nothing stops
them there. Two agents (this one included) misread the fail-pen arms on
exactly this, concluding the agent had found a way to evade the stall-kill
when it was being killed every single episode in training. **Check training
`ep_len_mean` before drawing any conclusion from eval episode length.**

**The stall threshold is PER-CALL and scales with `--act-every`.**
`rewards.py:735-737`: `_best` is a running minimum updated every call, so the
timer re-arms only when a *single decision* improves the episode's best by
more than `stall_eps` (32 u). It is not a rate and not a budget over the
window. Measured on a real petrus flight: at `act_every 3`, 13.7% of calls
clear 32 u (longest gap 92 of a 500-call window). At `act_every 1` the peak
is 24.97 u and **0.0%** clear it, so the detector would kill legitimate
flight. `--stall-eps` is now a flag (default 32.0, bit-identical); scale it
with the decision rate.

### What is known about the stuck checkpoint

* It is **not lost and not stalling**. It tracks the champion line to within
  1-2 units for 88% of the map, then leaves the ramp in the 256 u between
  route vertices 1596 and 1598, entering 6% slower than the champion
  (2,820 vs 2,930 u/s) after a ~0.45 s precursor of small growing error, and
  free-falls where the champion accelerates to 3,728 u/s. **A
  control-precision problem at one place, not an exploration problem across
  the map.**
* **Three** independent mechanisms (lookahead route geometry; soft
  shrink-and-perturb; Necto difficulty-weighted respawn) have now stopped at
  that same vertex with **0 finishes in 234 greedy episodes** between them.
  In all three the eval_progress movement was consistency, never frontier.
  **The fourth (xMARGIN, `--respawn-margin 2`) is the first to get past it**:
  6 of 72 episodes crossed 205,440 u, reaching 208,640 u (90.06%), and the
  ramp departure at vertex 1598 went from the controls' 1,735-3,019 u
  off-line to 177-254 u. Still 0 finishes, and the crossings appeared only in
  evals 2-4 of 8, so it is an intermittent capability, not a new frontier.
* **The reservoir could not reach the wall, and that - not the sampling rule
  - was the cap. FIXED by `--respawn-margin 2` (round 18, xMARGIN).** At the
  default 10 s margin the wall's bin held 0 states of the checkpoint's own
  reservoir and 0.07% of starts, and reservoir min-depth plateaued at
  12,180-16,867 u across xROUTE / xSP / xNECTO (24 readings). At 2 s it holds
  ~20% of the reservoir and ~19% of starts, min-depth 4,470-5,273 u, i.e.
  past the wall. **Every start-state arm before that one was measuring the
  harvest margin rather than its own mechanism** - Florensa, Salimans-Chen,
  Go-Explore cell selection and Necto difficulty weighting are all
  implemented and none has actually been tested; rerun each on top of
  `--respawn-margin 2`. Caveat: at a 2 s margin, the moment an arm starts
  FINISHING the reservoir will harvest states 2 s from the goal and can
  self-reinforce on trivial wins; win rate was 0.00% throughout xMARGIN so
  this never fired.
* **`race/eval_progress` cannot see progress past the wall, at all.** It is
  `mean(d at spawn - min d reached)` and the geodesic field's minimum ALONG
  THE ROUTE is at route vertex 1601 (d = 6,568 u) - the wall itself. Any
  route-following episode saturates at **191,812 u**; readings above that
  came from off-route dives. At or past 88%, only `tools/eval_honesty.py`
  corridor MAX and finishes mean anything.
* **The gravity-directional field does NOT fix the barrier - do not re-bake
  it.** `build_goal_field(gravity_dir=True)` was baked and gated in round 18:
  it makes the barrier 14% deeper (rise above the route minimum 8,344 ->
  9,555 u), is `>=` the old field at all 1,811 route vertices and lower at
  none, and the validator scores it as cutting a failure's banked potential
  by 0.0%. **Why it cannot help:** a greedy trace on the grid from vertex
  1600 is 191 level steps, 5 down, 0 up, **zero climb** - a straight
  ~8,700 u level glide through open air with 3,584 u of floor clearance. The
  BFS believes the player can fly laterally across a void, and `gravity_dir`
  gates only `dz > 0` edges, so it disconnected exactly 0 voxels. **The
  deception is free flight through open space, not one-way climbs.** Fixing
  it needs the velocity dimension or a different progress coordinate
  entirely (arc length along a route is monotone by construction and cannot
  have a mid-route minimum). The baked field is at
  `maps/surf_src_cannonball.goalg_32.npz` (39.6 MB, gitignored) if anyone
  wants to re-check; a rule change bumps `_GRAVITY_RULE_VERSION` and costs
  one 32-minute bake.
* **The final descent is a potential BARRIER in the shaping reward.** From
  vertex 1601 to 1680 the champion's own line RAISES geodesic d by 8,408 u,
  charged at `scale = 100/d0` = -4.24 reward, plus ~-4.6 of time penalty to
  run the rest of the route, against a +50 success bonus this policy has
  never once observed. Turning back at vertex 1601 is locally optimal and
  the reward says so - which is where all 234 control episodes stopped.
  `build_goal_field(gravity_dir=True)` already exists and was written for
  this class of defect elsewhere on the map; check whether it is monotone
  down the descent before running more exploration arms.
* Its weights sit at **2.9x the norm of a fresh draw** from its own init
  distribution (conv trunk head 4.0x, towers 2.3-2.8x, action head 283x).
  At the paper's beta = 1e-6 the shrink term cancels only about three
  quarters of the ongoing norm growth and the norm still rises, so if
  effective-LR decay is the wall here the published constant is one to two
  orders of magnitude too small at this scale.

### Ledger hygiene with parallel arms

Arms run on separate branches and each appends to the same append-only
`docs/research-results.md`, so their tails conflict by construction. That is
expected: append your section on your branch, and it gets folded into the
round's integration branch in arm order. Never edit someone else's section.

### The SEED-NOISE FLOOR: 3.0x at 750M, and it is BIMODAL (2026-08-23)

**One seed per arm is the standing rule, so the noise floor decides what a
difference is allowed to mean.** It has been measured twice, and the second
measurement is much worse than the first.

**First estimate (Round 21):** `xEP4` and `xNS128` are the SAME from-scratch
config run by two agents on two boxes - 18,001 vs 18,082 corridor MAX at 525M
(0.4% apart) but 27% apart at 750M.

**Corrected (Round 22):** there are now **four** runs of that configuration,
verified identical field-by-field from `run.json` (`xSH1` vs `xNS128` differ
in NO key at all). Order-only corridor MEAN:

| step | xNS128 | xGC32 | xEP4 | xSH1 | spread |
|---|---|---|---|---|---|
| 525M | 17,747 | 17,534 | 17,643 | **30,038** | **1.71x** |
| 750M | 18,126 | 18,209 | 15,961 | **48,482** | **3.04x** |

So, on from-scratch cannonball arms of about an hour:

* **a difference under ~3x at the end of the run is inside the noise** and
  must not be reported as an effect; at 525M the floor is ~1.7x;
* the "0.4% at 525M" reproducibility above is real for three of the four
  runs and **false for the fourth** - it describes the seeds that stayed in
  the low mode, not the protocol;
* **the distribution is not a band, it is a LADDER.** Greedy episodes stop at
  a few fixed places on the map (end z ~5,500-6,130 = 17-18k u; ~3,320 =
  26-27k u; ~760-800 = 47-49k u). A seed either clears a gate inside the hour
  or does not, so the corridor MEAN is nearly a small integer and the whole
  2.7x is one gate crossing.
* **Therefore "MEAN tracks MAX" is NOT corroboration.** Inside a mode it is
  automatic - a deterministic greedy policy falls the same way every time, so
  9 eval episodes of one seed are ~1 sample. More eval episodes cannot fix
  this. Report **which gate** and **the step at which the seed cleared it**,
  not just a mean.

Round 20's `n_steps` effect (2.2x) and Round 21's density effect are at or
below this floor and are no longer safe as stated. Round 20's `--goal-cell 64`
result is **retracted as a training claim** (its field measurements stand):
its final-eval MEAN 47,787 is matched by `xSH1`, an untreated control, at
48,482, which also leads xGC32 at 11 of 11 evals with MEAN tracking MAX.
See the Round 22 entry in `docs/research-results.md`.

### Hyperparameter ablations: the FROM-SCRATCH baseline (user, 2026-08-23)

**This supersedes the stuck-checkpoint rule below for hyperparameter
ablations** (`--n-steps`, `--epochs`, and anything else about the optimizer
rather than about exploration). The user set it explicitly:

* `surf_src_cannonball`, depth render, **standard 64x32**
* **NO `--obs-reward`** - which also removes the known truncation-bootstrap
  bug that flag carries
* **FROM SCRATCH**, not a warm resume
* **one hour maximum** per ablation

Launch it with `SCRATCH=1 bash tools/run_arm.sh <name> <flag>`. That branch
of the launcher carries the COMPLETE argument set, because a scratch run
restores nothing from a checkpoint and a partial line is exactly how Round 17
lost two runs. `respawn_margin` stays at the pinned 10.0 rather than the 2.0
Round 18 preferred: in an ablation the only thing that matters is that it is
IDENTICAL across arms.

**Two consequences for reading these arms.** The documented baseline table
below (the 140k-195k band, the stuck-checkpoint corridor figures) does NOT
apply - a scratch arm's only reference is its own control, so the control is
not optional. And this file's own warning that scratch runs need ~2.5 h to
say anything still stands against a 1 h budget, so **"the curves have not
separated yet" is a legitimate result** and must be reported as such rather
than squinted past.

### Every run starts from the STUCK checkpoint

`runs/sOBSR2/ckpt_latest.pt` - an agent that gets most of the way down the
map and then fails, for want of exploration, not for want of capability
(Round 16 proved the capability half: placed on-route by a demo curriculum
the same weights finish at champion pace). Its win rate had been 0.00% for
~2e9 steps with `race/eval_progress` oscillating in a band. **Do not start
arms from scratch and do not start them from a finisher** - scratch runs need
~2.5 h to say anything and a finisher has already solved the thing under
test.

### The baseline every arm is compared against

Warm resume of `runs/sOBSR2/ckpt_latest.pt`
(md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920) on
`surf_src_cannonball`, which is the checkpoint Round 16 used.
`race/eval_progress` by steps consumed, from `runs/research/live/*.csv`:

| steps after resume | xCTL (local 5090) | xGE (3090) | xEZ (3090) |
|---|---|---|---|
| 0 | 24,307 | 193,802 | 172,480 |
| +150M | 155,696 | 160,234 | 109,259 |
| +300M | 157,288 | - | 49,797 |
| +450M | 137,972 | - | - |
| +600M | 151,012 | - | - |
| +751M | 174,159 | - | - |

**Read the right column.** The 24,307 first eval is xCTL's alone; both
3090 arms opened at 172-194k. Surf is chaotic and the lidar march is not
bit-identical across GPU architectures (`test_march_is_bit_exact_against_
the_legacy_kernel` fails on a 3090), so one differing depth pixel forks the
whole greedy trajectory. **Compare an arm only against runs on the same
card**, and treat the opening eval as a coin flip between "ran the route
and fell short" and "died early", not as a signal.

Working band on a 3090: **~140k-195k**. Above ~195k sustained is a
positive; a decaying series like xEZ's 172k -> 109k -> 50k is a clear
negative and gets killed on sight.

**`race/eval_progress` is flattered by death-dives.** The shaping field's
reachable minimum is not the goal: an agent that falls past the finish
into goal-adjacent space scores ~178k of 198,380 without finishing. Always
open the eval's `traj_*.jsonl` and check whether the episodes ended inside
the finish box (`maps/surf_src_cannonball.zones.json` `end`) or below it -
5 of 9 episodes ending near z = -4,200 is a dive, not progress. A real
positive shows up as **finishes**, or as the eval band rising while
episodes still end on the route.

A 3090 delivers roughly 0.75-0.9e9 steps/hour on this config, so one hour
reaches about the +751M row.

## 4. One launcher

All rented runs start from **`tools/run_arm.sh`**, which reproduces the
baseline exactly. An arm is a **small edit** to that script (or a flag passed
through it) - never a hand-typed command line. This exists because Round 17
lost two scratch runs to hand-typed launches that silently omitted
`--respawn-frac` and `--int-coef`: a resumed run restores those from the
checkpoint, so an incomplete line looks fine until the first run with no
checkpoint behind it.

`tools/launch_local.ps1` is the same rule for local runs.

## 5. Where things are written down

* `docs/research-litsurvey.md` - the papers, with the constants, and what is
  already validated dead (RND, frame stacking, BC warm start, bigger nets,
  per-episode time bonuses, strided rollouts - do not retest these).
  **Add to that list: a 7-second temporal mini-race window.** Round 18 ran
  Linesight's mechanism faithfully and it was the round's only regression -
  the frontier went backwards 115,328 u, two thirds of the map the agent
  already had. The idea is not what failed, the constant is: 7 s is 27% of
  Linesight's median 26.3 s lap and 8% of an 82.8 s run here, and a randomly
  phased window that short puts the cost of a 2.3-4.2 s climb inside the
  window every time while the payoff often lands past an edge where the
  return is defined to be zero. Scaled to the same fraction of race length
  that Linesight uses, the window would be ~22 s - which is what this
  project's `gamma = 0.9995` (2,000 physics ticks = **20.0 s**) already is.
  **Treat the horizon question as answered and do not shorten it.**
  (`gamma` is PER PHYSICS TICK and the trainer raises it to `act_every`
  itself, so the horizon does not change with the decision rate; the
  survey's "60.6 s" figure is wrong and an earlier round wrecked an arm on
  this.)
* `docs/research-results.md` - the ledger. Every round appends: what was run,
  the numbers, and the verdict. Corrections are appended, never edited in
  place.
* `tools/bad_hosts.json` - blacklisted vast machines/hosts.
* `DEPLOY.md` - how a box is brought up (`tools/deploy_box.sh`).
