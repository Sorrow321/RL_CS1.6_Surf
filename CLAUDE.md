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
  to reach `running`, blacklist+destroy the rest. Round 16 destroyed 24
  instances this way for about $0.08 total. Stay inside the cap of 4 while
  racing.
* **Never `pkill -f <pattern>` over ssh when the pattern appears in your own
  command line** - it matches the shell running it and kills the session,
  which looks exactly like a dead box. Same family as the `pgrep` self-match
  that deadlocked the remote watchers.
* The `git clone` is ~170 MB because the repo carries `video_demo.mp4` and a
  30 MB `.npz`; on a 1 MB/s box that is three minutes before anything else
  starts. `--filter=blob:none` would fix it if this ever becomes the pole.
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

**Corridor MAX and finishes are the frontier.** The reference frontier to
beat is **205,312-205,440 u of 231,680 u**; anything past that is the first
real movement of the barrier from a config change. `tools/wall_profile.py`
then says what went wrong there (speed, height, off-line error versus the
champion line).

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
