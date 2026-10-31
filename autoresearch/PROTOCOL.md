# Autoresearch protocol - branch agents (fork-and-flush on surf_unitfarmer2)

You are ONE branch agent of a fork-and-flush round (Cai et al. 2026, arXiv 2610.07447,
"Escaping Idea Basins in Autoresearch Agents"). Read this file, then `CLAUDE.md` (binding,
every rule in it applies to you), then `autoresearch/NOTES.md` (what this lineage knows),
then `AR_BRANCH.json` (who you are, your deadlines, the incumbent). Then start working.

## The method, and your place in it

* Each round forks K = 4 branch agents from the SAME workspace: this git branch (code,
  `autoresearch/NOTES.md`, the ledger) plus the protected incumbent (the best checkpoint and
  eval so far). Each fork starts with a FRESH context. That is the "flush", and it is the
  point: independent runs stall in different "idea basins", and a fresh context escapes the
  one the previous lineage was in.
* You do not talk to the other branches, and you do not read what they do (see
  **Independence** below). You are not asked to "be different" either (the paper measured
  that directive: it does not help). Do the best research you can.
* At the round's end the orchestrator scores every branch with `autoresearch/score.py`. That
  score is the best greedy eval from the TRUE map start that your runs produced this round.
  The winner's branch head seeds the next round's four forks. The other branches are archived
  and pruned. So what you leave in this branch (code, NOTES.md, RESULT.json) is what the next
  round inherits if you win.

## Independence: read only your own branch

Ideas cross between lineages only at the round boundary, through the winner's workspace. So
read only:

* your worktree (this branch: its code, NOTES.md, RESULT.json, AR_BRANCH.json) and its git
  history;
* your runs (`$AR_HOME/runs/$AR_AGENT/`) and your box (`box.py status --agent $AR_AGENT`,
  `box.py budget`);
* the incumbent (`$AR_HOME/incumbent/`, its `refs/` included);
* the WR demo's analysis files (`$AR_HOME/wr/`: ANALYSIS ONLY, CLAUDE.md section 0).

Never read, list or grep any of these:

* another branch's worktree (`$AR_HOME/wt/b<k>`), runs (`$AR_HOME/runs/<agent>`) or box
  record;
* another branch's git branch (an `ar/*` ref that is not yours);
* another branch's tmux window or Claude transcript;
* the archive of earlier rounds (`$AR_HOME/archive/`);
* the orchestrator's files: `$AR_HOME/status.md`, `state.json`, `orchestrator.log`. They are
  the user's and the operator's view of every branch.

This holds even when the goal is to avoid repeating another branch's idea. Two branches trying
one idea is a cost the method accepts. Copying the leader mid-round only pulls a branch into
the leader's idea basin. In round 1, r1b4 read the other branches' status and notes and spent
its last six hours on the leader's recipe, and it finished below the leader. A losing branch's
findings are archived, not inherited: that is the flush.

## The goal and the one number

Pass surf_unitfarmer2 from the true start, champion-free and generic (**This lineage** below):
first OUT OF THE START PIT, then a finish, then finishing reliably, then faster than 35.17 s,
the human world record (Fizz, 2024-01-28, speedrun.eu; its demo times 35.21 s on our zones,
start curtain to finish box; the demo is an analysis instrument only, CLAUDE.md section 0).
The record's time is the bar a result is read against, never a constant in the reward, the
spawn rule or a schedule (CLAUDE.md 0b). Your score is
`python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/`. It is the best eval's
max-episode progress on the goal field (a finish counts 100), plus 10 x the finish fraction,
plus a time term once something finishes (10 x (1 - best finish s / 300)). One more finishing
episode of the 9 is worth 1.11 and one second off the best finish 0.033, so the score ranks
reliability before speed. Only greedy evals from the true start count; the grader rejects
anything else.

The true start. The map's own `info_player_start`, (-2176, -2272, 1192) yaw 270, on the start
platform (floor z 1,152). The start curtain (trigger_multiple *16 -> counter_start_button) is
the slab y -2,408..-2,400 just south of it; the finish is trigger_multiple *23 ->
counter_stop_button, the box (3328, -2592, -640)..(3936, -1888, 96). Both come from
`maps/surf_unitfarmer2.zones.json` (source auto: the BSP's own timer triggers), and every
trainer, recorder and the grader start at that spawn. The map's 11 trigger_teleports are fall
catches (4 back to the spawn, `recomeco`; 7 to a restart at (-2176, 3200, 640), `part2`). The
record touches none of them, and every one ends the episode in our core (`--teleport-fail`).
The server runs sv_maxvelocity 3500 (the WR demo's movevars; airaccelerate 100 and gravity
800 are the core's defaults), so the recipe carries `--maxvel 3500`.

* the field is `config.field` = the no-corner-cut goal graph, the field the recipe shapes on
  (`--race-no-corner-cut 1`); `config.start_d` is its distance at the spawn, 30,647.5 u.
* only the first `config.score_episodes` (9) episodes of an eval count, by header episode
  index: a bigger eval is not a better score.
* **Round 1 has no incumbent** (AR_BRANCH.json `incumbent` is null, and
  `$AR_HOME/incumbent/ckpt.pt` does not exist yet). From round 2 on it is the previous
  winner's checkpoint. `config.incumbent_recipe` is the scratch recipe (the sidistic finisher's
  flags, carried through both princessburglar lineages). Three policies that have never seen
  this map are kept as optional warm starts in `$AR_HOME/incumbent/refs/`. All three are
  champion-free and have the recipe's observation and action layout:
  * `pb_r26`: the princessburglar expert loop's final policy, 9/9;
  * `pb_r15`: that loop's best eval, 9/9 at 62.11 s best / 65.62 s mean;
  * `sid_ref`: the sidistic lineage's finisher. It has no `run.json`; its config is inside
    the checkpoint.

  The two pb policies come with their `run.json`. They run at a 7.64 ms tick (a 10 -> 7.63 ms
  ramp that ended long before), and a resume keeps it. The scratch recipe runs at 10 ms.

  Like every checkpoint, a warm start restores its own shape flags; never pass them to a
  resume. A recipe that only works warm is not yet a recipe.

## READ FIRST - the finish is the trigger, not its box (operator, 2026-10-31)

* **The bug.** Until round 2 the core completed an episode when the player crossed the finish
  trigger's BOUNDING BOX (zones.json `end`, hull-inflated). GoldSrc fires a brush trigger only
  with the player's origin inside the trigger model's CLIP HULL at the end of a frame
  (SV_TouchLinks; the box is just the pre-check). surf_unitfarmer2's finish (`*23`,
  counter_stop_button) is three 1 u curtains along the finish platform's edges - the west face
  x 3328 (y <= -2144), a diagonal (3328, -2144)-(3584, -1888), the north face y -1888
  (x >= 3584) - and its box adds an EMPTY triangle north-west of the diagonal, in open air beside
  and below the platform. The policies and the searches learnt to clip that triangle: of the
  1,312 episodes the core called finished up to 13:00 UTC, 191 (15%) entered the real trigger;
  all 193 search lines on the round-2 boxes, and 8 of the round-1 incumbent's 9 "finishes", were
  box-only. The user saw it in a video (the record crosses the line, the AI "finishes" before).
  Found independently by uf2r2b1, uf2r2b2 and uf2r2b4 the same hour; their fixes are merged here.
* **The fix (the harness core, src/env.c).** surf_set_goal_box finds the brush model whose
  bounds ARE the goal box and, unless that model is a plain box, completes on its clip hull at
  the tick's end (trigger_contains, the test teleports and hurts already used). A plain box
  trigger keeps the swept box - byte-identical on such maps (cannonball, petrus). The APIs of all
  three branches work: `SurfCore.set_goal_hull(False)` / `SURF_GOAL_AABB=1` = the old rule
  (analysis only), `SurfCore.set_goal_model(model[, origin])` names a model, `core.goal_model`
  (23 here; also callable). `tests/python/test_goal_hull.py`. The start curtain `*16` is a plain
  box, so the clock's start was right.
* **The grader.** score.py counts a "done" only if the finishing step entered the clip hull
  (surfgym.goalhull); a box-only "done" scores as progress and shows as `box_only`. Re-scored:
  the round-1 incumbent 118.592 -> 109.692 (1/9 real, 42.57 s).
* **The user's direction (2026-10-31):** keep the checkpoints - "it's doing basically the same
  thing. It's just a question of a few more seconds. So we do not need to retrain everything ...
  Just need to train a little bit more with the fixed end box." Warm-start from the trained
  lineage; no scratch retraining. It works: uf2r2b4's x10 (its `--goal-hull`, resumed from x9's
  36.0 s box-finishing weights) flew 9/9 REAL finishes at 36.88 s (spawn clock) within ~150M
  steps. The game clock (start curtain left -> first touch of the real trigger) is ~0.78 s shorter
  than the spawn clock for the policy; Fizz's record demo reads 35.14 s on it (official 35.17 s).

## This lineage: surf_unitfarmer2 (operator, 2026-10-30)

* **Why it exists.** The user, 2026-10-30: beat unitfarmer2, "especially ... the beginning":
  the dive into the start pit that buys the speed the rest of the course needs. This map is
  the project's EXPLORATION BENCHMARK (CLAUDE.md sections 0 and 0b, read them again now): the
  deliverable is a GENERIC mechanism that finds the pit line by itself, with constants that
  would be the same on a map nobody has opened. The user's next step after this program is
  one recipe trained on hundreds of maps at once, so a trick that only works here is worth
  nothing to it.
* **What the record shows (ANALYSIS ONLY, CLAUDE.md section 0).** Times from the start
  curtain, d on `config.field`, energy height E = z + |v|^2 / 1600:
  * t+0: the curtain at 339 u/s, d 30,578.
  * t+1: it drops off the platform's south edge (z 892, d 30,121).
  * t+3: it is at the pit floor (z -857) at 1,817 u/s. It strafes during the fall: the
    October analysis measured +120 u of E there, the per-tick air-accelerate ceiling.
  * t+3.7: its potential has RISEN 3,060 u above its running minimum. The field points away
    from the pit, so the shaping charges the whole dive.
  * ~t+6.5: it is back below its t+1 potential.
  * t+7.9: it leaves the pit at 1,127 u/s (d 29,090 = 5.1% progress).
  * After that, progress is 10% at t+10, 31% at t+15, 48% at t+20, 63% at t+25, 78% at t+30,
    and the finish at t+35.2. Its E stays ~1,000-1,200 u until the last 10 s.

  These numbers are for READING results (did a run get out of the pit, how fast, where did
  it lose energy). They never enter a reward, a spawn rule, an archive key or a schedule.
* **What failed before (`docs/research-results.md`, grep `uf2` / `unitfarmer2`; read it).**
  * 2026-09-13: eight champion-free RL arms at 1B steps each, all null on the pit: ratchet,
    keys and all-heads temperature, frontier curriculum, speed reward, surf reward,
    time-penalty bias, random reachable starts.
  * 2026-10-16: Go-Explore + CEM (`tools/go_cem.py`, 192 u x energy cells) filled the pit
    (13,554 cells) and never left it. It was ENERGY-bound: it arrived 280-390 u of E below
    the record.
  * 2026-10-17: the ramp search (`tools/ramp_search.py --alloc pw`) climbed out of the pit by
    itself, with no record input, in 18.6 min. It then stopped at the next ramp
    on energy: it left the pit at 550-710 u/s against the record's 1,127.
* **Demo-trained, BANNED as bases (CLAUDE.md section 0).** Everything on this map that got
  past the pit from the map start in October used the record's 17-ramp target sequence
  (uf2SEQ_WR*, uf2R3*, uf2SC*: S20 -> S36), with stage files edited by hand after looking at
  where the policy died. That is demo-assisted and map-specific: history, not a result. Those
  checkpoints, and CLAUDE.md's list (uf2PIT*, uf2RPIT*, uf2FULL*, uf2CONTW, uf2G2), are never
  resumed, benchmarked as results, or used to build a window. The dip-speed arms
  (uf2DIPSPD*, uf2DCDIP*, uf2DIPcap*) are map-tuned analysis (0b), never a base either.
* **What transferred from princessburglar (2026-10-29, the pbf lineage: the same harness and
  recipe, champion-free; all of it is in this branch).**
  * On that map the way out of the start room also runs against the potential: pump a
    half-pipe for energy first.
  * **What found it** in ~3 min of box CPU from the true spawn was Go-Explore with the
    policy's own action frame:
    `tools/explore_phase1.py --exact-chains --cell 64 --cell-energy 32 --cell-heading 1
    --view-follow=-90,-60,-30,-15,0,15,30,60,90 --select-energy 3 --act-every 4 --decisions 100
    --maxvel 3500 --envs 4096`. It ran at `--tick-ms 8`, its policy's tick; match the tick
    of whatever will train on the line. The bursts hold a view offset from the velocity heading, and
    energy cells plus `--select-energy` keep and re-explore the energy gains. The earlier
    uniform-yaw explorer never pumped. `tools/ge_replay.py` turns a cell into an exact
    per-tick line.
  * **What taught it** was a backward curriculum on that one GE line (`--demo-file` of the
    line, launched with `box.py launch --self-states "<how it was built>"`), warm from a policy
    that already flew the rest of the course.
  * **What made it faster** was expert iteration on the policy's own beam-search lines:
    `tools/beam_tas.py` -> `tools/plan_to_bc.py` -> `--bc-file`, or the whole loop in
    `tools/exit_local.py` with `--gate-rule expected`. Keep the beam elite at its default
    0.25: 1/16 kills the population on a 7/9 policy. It took princessburglar from 64.15 s to
    62.11 s greedy, and to a 59.40 s search line.
  * Nothing in those tools names a place. The October go_cem result above is the
    counter-evidence: an energy archive alone was not enough here. That is a question for you,
    not a conclusion.
  * **The difference from princessburglar:** no policy flies this course yet. Whatever finds
    the pit line, the rest of the course must also be learned, from scratch or warm from a
    `refs/` policy.
* **Section 0b binds here more than anywhere.** No reward, spawn rule, archive key, filter or
  schedule may name the pit or a ramp, use a box around either, or carry a speed, height,
  energy or time read off the record or the map. Ask the 0b question of everything: would I
  run exactly this, with these constants, on a map I have never opened? Spawning from the
  policy's OWN states, or from a search's states rooted at the true spawn, is allowed. A
  hand-placed spawn in the pit is not.
* **The bar.** First out of the pit from the true start in a greedy eval (best progress past
  ~5% with the episode still flying), then a finish, then 9/9, then the record's 35.17 s.

## Each wakeup

You run inside `/loop` (self-paced). Every wakeup:

1. Read `AR_BRANCH.json`. If `now >= wrapup_at`, go to **Wrap-up** and do nothing else.
2. Check your box and run: `python autoresearch/box.py status --agent $AR_AGENT`, then the
   run's progress through `box.py ssh` (tail of `runs/<run>/progress.csv`, the last greedy
   eval). Score new evals after pulling them.
3. Act: analyse, code, launch, extend, or stop a losing run (a run that cannot beat the
   incumbent at matched steps is stopped, not left to burn money).
4. Write down what you learned: append to `autoresearch/NOTES.md` under a dated heading
   (take the time from `date`, never type it). Keep `RESULT.json` current (format below).
5. Schedule the next wakeup with ScheduleWakeup. While a run trains, wake every 20-30 min.
   Wake sooner only while you are actively building or a launch is in flight. One waiter,
   no polling loops, no Monitor spam: the user pays for every wakeup.

Stopping a run does not end your round. Until `wrapup_at`, a failed idea is written down and
followed by the next one (your ranking in NOTES.md, or a new idea). Release the box while you
analyse or build, but keep the loop going: an early Wrap-up leaves this branch's slot empty for
the rest of the round. The orchestrator re-arms the loop of a branch whose RESULT.json says
final before `wrapup_at`.

## Experiments: short first, long only when earned

* **First test of any idea: at most 30 min of box time.** On a 3090 the incumbent recipe
  runs roughly 150-250k env steps/s (227k on a 4080 on this map), so 30 min is about
  3e8-4e8 steps. Size `--steps` to the time, and read the box's own fps after the first
  minute.
* **Compare at MATCHED STEPS** against the incumbent run's progress.csv
  (`$AR_HOME/incumbent/`; in round 1, which has no incumbent, against your own scratch
  control on the same card type). Use the gate cleared and the step it was cleared at, never
  eval_progress alone (CLAUDE.md section 3). One seed, so a lead must be large to mean
  anything (CLAUDE.md: the gate ladder and the seed-noise floor).
* **Extend up to 6 h only if the idea is promising**: it clears a gate sooner than the
  incumbent at matched steps, or clears a gate the incumbent never cleared. Say why in
  NOTES.md when you extend.
* **Warm starts are allowed.** Push a checkpoint with `box.py deploy --ckpt ...`
  (`$AR_HOME/incumbent/ckpt.pt` from round 2 on, or a `refs/` policy) and launch with
  `box.py launch --resume ...`. This measures AT the wall (CLAUDE.md: the stuck checkpoint is
  one tool) or pushes the frontier past it. State which in NOTES.md. A recipe that only works
  warm is not yet a recipe.
* **Hard rules from CLAUDE.md that bite here:**
  * No human demo touches training (section 0).
  * No map-specific method, threshold or box in the reward, the spawns or the schedule
    (section 0b). Ask "would I run exactly this on a map I have never opened?"
  * New trainer flags default OFF and are bit-identical when off (prove it).
  * Every new flag is mirrored in `tools/record_ckpt.py` or listed TRAIN_ONLY, or the launch's
    record gate kills the run.
  * Shape-changing flags are never passed to a resume.

## What the user set for this program (2026-10-20), over CLAUDE.md's defaults

* **Ablation length.** At most 30 min first, up to 6 h when promising, as above. This replaces
  CLAUDE.md's "one hour per ablation".
* **Boxes.** One box per agent, single GPU only, and consumer RTX cards under the
  config.max_dph caps (3090 first, then 4090, then 5090; cheaper cards that do the job, like
  the 5080, are fine, and since 2026-10-25 the 3080 / 4070 / 5070 / 60 Ti class under lower
  caps: box.py takes the cheapest acceptable offer, so a box may be a slower card). No
  datacenter cards, ever. box.py and the watchdog enforce this.
* **Watching.** The user watches through Claude Remote Control and `$AR_HOME/status.md` (the
  user's view, which you do not read: see Independence). That replaces CLAUDE.md's per-box
  dashboard tunnel. A dashboard on the box is optional, and no tunnel is needed.
* **Budget.** The total is the vast credit ($64.59 at this program's start). box.py paces the
  fleet's burn at (credit - reserve) / hours left until config.budget_end_hours (+10% slack),
  ~0.75 $/h at the start: about three cheap boxes at once, so a box may be refused while the
  others run. It keeps the $10 reserve: no rent at the reserve, and no `extend` that the
  fleet's burn would pay out of it.

## Code changes

Work on YOUR branch (`ar/$AR_AGENT`, already checked out in this worktree):

1. Implement behind a flag, default off, with a unit test.
2. CPU-check on this machine before renting:
   `CUDA_VISIBLE_DEVICES=-1 nice -n 10 python -m pytest -q <your test> -p no:cacheprovider`.
   This machine has 4 vCPU and hosts the user's game server: run one CPU job at a time and
   keep it short.
3. Commit with the trailers from CLAUDE.md / the session, then `git push origin ar/$AR_AGENT`.
   Boxes clone from GitHub, so a box can only run what is pushed.
4. Deploy THAT commit: `box.py deploy --agent $AR_AGENT --branch ar/$AR_AGENT`.

**The harness is the operator's.** The files in `config.harness_files` (CLAUDE.md, this
protocol, box.py, score.py, orchestrator.py, config.json, the watchdog, deploy_box.sh,
vast_pick.py, gpu_health.py and their tests) are restored from the operator's branch at the
start of every round. An edit you make to one does not reach the next round. One exception:
`config.json`'s `lineage_keys` (`incumbent_recipe`) are the lineage's. Change the recipe there,
and if your branch wins, your value is the next round's scratch default. If a harness file
blocks you, work around it on your branch and write it in NOTES.md under a "Harness issue"
heading; the operator reads those. `autoresearch/NOTES.md`, the trainer, the tools and the tests are
yours, and they travel with your branch if it wins.

## Boxes: one per agent, and box.py is the only door

```
python autoresearch/box.py budget                                   # credit, burn, pace
python autoresearch/box.py rent    --agent $AR_AGENT --hours 1      # the deadline the watchdog enforces
python autoresearch/box.py deploy  --agent $AR_AGENT --branch ar/$AR_AGENT [--ckpt $AR_HOME/incumbent/ckpt.pt]
python autoresearch/box.py launch  --agent $AR_AGENT --run ${AR_AGENT}_<tag> --steps 4e8 -- <extra flags>
python autoresearch/box.py ssh     --agent $AR_AGENT -- "tail -3 /root/RL_Surf/runs/<run>/progress.csv"
python autoresearch/box.py pull    --agent $AR_AGENT --run <run> [--ckpt]
python autoresearch/box.py extend  --agent $AR_AGENT --hours 1
python autoresearch/box.py release --agent $AR_AGENT
```

* `launch` prepends the incumbent recipe (`config.incumbent_recipe`) to your extra flags, and
  later flags win. `--no-recipe` gives a bare scratch preset.
* A spawn-state file built from the policy's OWN states (its reservoir, its greedy line, an
  archive rooted at them) is passed with `launch --self-states "<how it was built>" --
  --spawn-states <file>`. That sets run_arm.sh's SELF_STATES=1 and records your sentence with
  the run. Never launch around `launch` (box.py ssh + run_arm.sh): the run then never reaches
  your box record.
* Never call `vastai create`/`destroy` yourself. The fleet watchdog destroys any instance that:
  * is not a single allowed RTX card under its price cap;
  * is above the box cap;
  * is past its deadline;
  * is unregistered.
* A box with no experiment running is released within 5 minutes (CLAUDE.md section 1). Do
  your analysis with the box released, unless you are about to launch again.
* If `rent` refuses on budget, the fleet is at its spending limit (`box.py budget`: burn_per_h
  against limit_per_h). Do analysis
  or coding and try later. Never work around it.
* Your box id is in `$AR_HOME/boxes/$AR_AGENT.json` and `box.py status --agent $AR_AGENT`. Keep
  it in NOTES.md too.
* A bad box (gpu_health fails, or the trainer runs far under ~150k steps/s on the incumbent
  recipe) is blocklisted with `python tools/vast_pick.py --block <id> --reason ... --detail ...`
  BEFORE you release it. That writes your branch's `tools/bad_hosts.json`. The orchestrator
  folds every branch's list into `$AR_HOME/bad_hosts.json` within 5 minutes, and `box.py`
  skips both lists, so no other agent rents that machine. `config.cpu_exclude` already drops
  the old Xeon E5/E7 hosts, which vast lists at their turbo clock.

## Harvest, score, record

* Before releasing a box, pull every run you may want to cite:
  `box.py pull --agent $AR_AGENT --run <run>`. Add `--ckpt` for the one you might hand on.
* Score: `python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/<run>/`.
* The next round's incumbent checkpoint comes from the winning eval's run, as pulled to
  `$AR_HOME/runs/` (a truncated or damaged file is skipped). It is the newest
  `ckpt_<step>.pt` AT OR BEFORE the winning eval, unless the run's newest checkpoint
  (`ckpt_final.pt`, else `ckpt_latest.pt`, else its newest `ckpt_<step>.pt`) is more than
  twice as near the eval; with no snapshot at or before the eval, it is that newest one,
  usually later than the eval. A snapshot is handed on only if you pulled it: if your best
  eval falls between snapshots, pull the one before it as well
  (`box.py pull --agent $AR_AGENT --run <run> --snapshot <step>`; operator, 2026-10-23: r7b2's
  finish came inside a transient peak, and its ckpt_latest was ~700M steps past it).
  `AR_BRANCH.json`'s incumbent gives `eval_step` and `ckpt_step`. Before you compare against
  the incumbent's score, read the checkpoint's own first eval.
* `box.py pull --ckpt` downloads to `<ckpt>.part` and replaces the copy already pulled only
  when the md5 matches the box's, so a pull that times out on a slow link loses nothing.
* `RESULT.json` (worktree root, read by the orchestrator, rewritten whenever the best moves):

```json
{"agent": "r1b2", "round": 1, "final": false,
 "best": {"score": 27.5, "eval_file": "/home/steam/ar/runs/r1b2/r1b2_x/traj_0755499008.jsonl",
          "run": "r1b2_x", "step": 755499008, "ckpt": "/home/steam/ar/runs/r1b2/r1b2_x/ckpt_latest.pt",
          "recipe": ["--obs-compass", "1"], "resume_from": null, "commit": "<sha>"},
 "summary": "<= 10 lines: what you tried, what it did at matched steps, what to try next"}
```

## Wrap-up (at `wrapup_at`, or when a message says ROUND OVER; never before)

1. Launch nothing new. Pull the best run, with its checkpoint (`--ckpt`) and the snapshot
   at or before its best eval (`--snapshot <step>`).
2. Score it. Write `RESULT.json` with `"final": true` and the summary.
3. Append the round's lesson to NOTES.md: what worked, what did not, the numbers.
4. `git add -A && git commit` (trailers) `&& git push origin ar/$AR_AGENT`.
5. `box.py release --agent $AR_AGENT`, and confirm with `box.py status --agent $AR_AGENT`.
6. Stop scheduling wakeups. The orchestrator ends this session at the deadline.

## Talking to the user

The user watches every session through Claude Remote Control and may write to you at any
time. Answer first, in a few lines:

* what is running;
* its latest eval and score against the incumbent;
* what you will do next.

ASCII only (the user's terminal is cp1251). Label every line and point of any plot (the user
is colour-blind). Times come from `date`.
