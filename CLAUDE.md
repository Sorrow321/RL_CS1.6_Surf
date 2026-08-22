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
`surf_src_cannonball`, which is the checkpoint Round 16 used. Control run
`xCTL` (`runs/research/live/xCTL.csv`), `race/eval_progress` by steps
consumed:

| steps after resume | eval_progress |
|---|---|
| 0 | 24,307 |
| +150M | 155,696 |
| +300M | 157,288 |
| +450M | 137,972 |
| +600M | 151,012 |
| +751M | 174,159 |

So: **first eval ~24k (the resume trough is real and reproducible - seven
independent resumes landed 19.8k-24.3k), then a 138k-174k band.** Above
~180k is a positive (the accidental distance-flattened respawn arm reached
178-184k); a decaying series like xEZ's 172k -> 109k -> 50k is a clear
negative and gets killed on sight.

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
* `docs/research-results.md` - the ledger. Every round appends: what was run,
  the numbers, and the verdict. Corrections are appended, never edited in
  place.
* `tools/bad_hosts.json` - blacklisted vast machines/hosts.
* `DEPLOY.md` - how a box is brought up (`tools/deploy_box.sh`).
