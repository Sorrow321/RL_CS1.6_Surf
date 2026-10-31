# Autoresearch protocol - branch agents (fork-and-flush on surf_za_loop)

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

Pass surf_za_loop from the true start, champion-free and generic (**This lineage** below):
first UP THE FIRST LOOP, then a finish, then finishing reliably, then faster than the human
record: krs, 491.65 s on speedrun.eu (2024-02-05), the map's only record. Its demo reads
**450.46 s on our clock** (start-curtain exit -> first finish-trigger entry; READ FIRST explains
the 41 s gap). The demo is an analysis instrument only (CLAUDE.md section 0). The record's time
is the bar a result is read against, never a constant in the reward, the spawn rule or a
schedule (CLAUDE.md 0b). Your score is
`python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/`. It is the best eval's
max-episode progress on the goal field (a finish counts 100), plus 10 x the finish fraction,
plus a time term once something finishes (10 x (1 - best finish s / 900), config.time_cap_s:
this map's runs are long). One more finishing episode of the 9 is worth 1.11 and one second off
the best finish 0.011, so the score ranks reliability before speed. Only greedy evals from the
true start count; the grader rejects anything else.

The true start. Every `info_player_start` (32, in a sealed spawn room) stands on a
trigger_teleport (*4-*11) to `info_teleport_destination` "map" (-1504, -1536, -3110.87, yaw
180). The core puts the player at (-1504, -1536, -3073.87) yaw 180, at rest, one tick after the
map's own spawn (the engine's rule: destination - hull mins + 1; verified). The zones file
declares that point as the spawn (`maps/surf_za_loop.zones.json` "spawns", source manual), so
every trainer, recorder and the grader start there. From it the player drops ~100 u to the floor
and crosses the start curtain (trigger_multiple *14 -> counter_start_button, x -1696..-1680, a
plain box) heading west. Every fall ends in *21, a trigger_teleport across the whole map at
z -4050..-4000, back to the same destination: in the game that restarts the run (the timer
restarts at the curtain), in our core it ends the episode (`--teleport-fail`). The finish is ANY
of five trigger_multiples on the stop button (*15-*19), plain boxes at z 3008-3210 at the top
of the map (READ FIRST). The server runs sv_maxvelocity 3500 (the demo's movevars;
airaccelerate 100 and gravity 800 are the core's defaults), and the record plays 7 ms frames,
so the recipe carries `--maxvel 3500 --tick-ms 7`.

* the field is `config.field` = the no-corner-cut goal graph, the field the recipe shapes on
  (`--race-no-corner-cut 1`), seeded on the five slabs; `config.start_d` is its distance at the
  spawn, 6,361.8 u.
* only the first `config.score_episodes` (9) episodes of an eval count, by header episode
  index: a bigger eval is not a better score.
* **Round 1 has no incumbent** (AR_BRANCH.json `incumbent` is null, and
  `$AR_HOME/incumbent/ckpt.pt` does not exist yet). From round 2 on it is the previous
  winner's checkpoint. `config.incumbent_recipe` is the scratch recipe (the sidistic finisher's
  flags, carried through the princessburglar and unitfarmer2 lineages; this map's tick and
  episode cap). Four policies that have never seen this map are kept as optional warm starts in
  `$AR_HOME/incumbent/refs/`, all champion-free, with the recipe's observation and action layout:
  * `uf2_x13`: the unitfarmer2 lineage's best (uf2r2b4's x13, 9/9 real finishes at 35.31 s on
    that map; from scratch on that map in round 1 - uf2r1b1's chain - then trained on spines,
    BC and spawns built from its OWN search lines); its `run.json` and best eval come with it;
  * `pb_r26`: the princessburglar expert loop's final policy, 9/9;
  * `pb_r15`: that loop's best eval, 9/9 at 62.11 s best / 65.62 s mean;
  * `sid_ref`: the sidistic lineage's finisher. It has no `run.json`; its config is inside the
    checkpoint.

  They run at a ~7.65 ms tick (7.63 requested, the [8, 8, 7] pattern), and a resume keeps it
  unless you pass `--tick-ms` (a TICK TRANSFER, allowed and logged; CLAUDE.md: a frozen line does
  not survive a tick change, a skill may). Like every checkpoint, a warm start restores its own
  shape flags; never pass them to a resume. A recipe that only works warm is not yet a recipe.

## READ FIRST - this map's finish, start and clock (operator, 2026-10-31)

* **The finish is five triggers, and the harness now knows it.** `zones.detect_zones` kept only
  the FIRST trigger on a timer button, and the five slabs' union box is two thirds open air that
  no trigger covers (a grid over it: 128k points inside a slab, 256k open and in none). Fixed in
  the harness: zones.json "end" is the union box plus its "boxes" and "models";
  `SurfCore.set_goal_box` arms the models for every caller (trainer, recorder, beam search; the
  trainer logs `finish: any of 5 trigger models`); the goal field is seeded on each slab; the
  grader counts a finish in any of them (`surfgym.goalhull.GoalSet`). A single-trigger map is
  unchanged (cannonball: 256 envs x 200 ticks bit-identical). `tests/python/test_goal_models.py`.
  The unitfarmer2 fix (a non-box trigger completes on its clip hull) rides along; every timer
  trigger here is a plain box. The record enters slab *16 at (-48, 2014, 3048).
* **The clock, and the 41 s gap.** The record's demo has 65,110 frames of 7 ms (143 fps). It
  crosses the start curtain once (leaves it at t 3.54 s), never falls, and enters *16 at
  453.99 s: 450.46 s on our clock. speedrun.eu lists 491.65 s. 491.65 / 450.46 = 1.0915, which is
  what a client that reports 7 ms per frame while its frames take ~7.64 ms (131 fps) would
  produce on a server timer; it could also be time before the demo starts. It is unresolved:
  read a result against 450.46 s (the demo on our clock, the stricter bar) and quote both.
* **A fixed latent bug.** `--respawn-front-perturb` crashed the trainer 2.6M steps into this
  map's bake: its frontier band is empty while every reservoir state is BEHIND the start (the
  drop off the destination's ledge raises d by ~90 u). Fixed on this branch (an empty band gives
  no rows; bit-identical otherwise; test in `tests/python/test_respawn_front_perturb.py`).

## READ FIRST - what the field cannot see on this map

* **The field points straight up.** The goal field is a voxel BFS that lets a player fly
  through open air in any direction, so start d (6,361.8 u) is about the height still to climb:
  6,100 u from the curtain (z -3,140) to the slabs (z 3,008). It points up everywhere and says
  nothing about HOW you get up, which is the loops: ride a ramp side to side until the speed
  carries you to the next one.
* **What the record does (ANALYSIS ONLY, CLAUDE.md section 0).** Times from the curtain;
  progress on `config.field`; energy height E = z + |v|^2 / 1600:
  * t+0..60 s: it rides the first loop. z stays within -3,100..-3,300, its progress is
    NEGATIVE (-3.6..-0.8%), and its speed goes 345 -> 1,300 u/s: E rises ~1,000 u while the
    field pays nothing.
  * t+90: 11% (z -2,464). t+150: 43%. t+150..300: deep loops between z -1,100 and +400 at
    1,900-2,100 u/s, progress swinging 33-48%, E up to ~2,200.
  * t+330: 63%. t+390: 86%. t+450.4: the finish (z 3,029).
  * Over the run: a new height record on 6.4% of its frames, the longest stretch without one
    44.7 s; on the ground 3.2%; its heading turns through 153 full circles; mean speed 1,405,
    max 2,399 u/s.

  These numbers are for READING results (did a run get up the first loop, where did it stop
  gaining energy). They never enter a reward, a spawn rule, an archive key or a schedule.
* **What that does to the recipe.** run_arm.sh SCRATCH's `--stall-secs 15` ends a training
  episode that makes no field progress for 15 s, and the ratchet pays only new progress
  records: the record's first minute is invisible to both. The recipe's `--energy-pot 3` pays
  energy gains, but only until an episode reaches the reservoir's frontier band
  (`--energy-pot-front 0.01`). The operator's smoke run (`zlop_check`, 30M steps on an RTX 5080,
  347k steps/s): its greedy episodes fell into the bottom teleport at 18-25 s with no progress,
  and 62-68% of training episodes ended by the stall kill. The training win rate (~6.5%, at
  ~0.4 s) is the recipe's backward-mix rows, which start 256-318 u from the goal (5% of d0):
  not a finish from the start (CLAUDE.md section 3: a win rate is not evidence).
* **0b binds hardest here.** This is the user's own example of what the recipe must solve
  (CLAUDE.md 0b: "you have to do some speedup on ramps that are not shown by reward vector
  field. You can't be building custom reward every time"). The way through is a GENERIC change:
  a progress or energy measure, a stall rule, an exploration mechanism or a search that would be
  right on every map, with constants set once. No reward, spawn rule, archive key, filter or
  schedule may name a loop or a ramp, use a box around one, or carry a speed, height, energy or
  time read off the record or this map. A constant re-tuned for this map (a longer stall window
  "because loops take a minute") is a map-specific method in disguise unless it is the value
  you would ship for every map; say which in NOTES.md.

## This lineage: surf_za_loop (operator, 2026-10-31)

* **Why it exists.** The user, 2026-10-31, with the map and its record demo: "This is a bit
  annoying map because it mostly consists of so-called loops. It's basically when you have a
  ramp and you're riding this ramp left and right all the time until you gain the speed, and
  then you can jump on another one. So it's quite different from normal surf maps. ... I wonder
  if they can solve this map automatically." The question is whether this lineage's generic
  machinery finds the loops by itself. A trick that only works here is worth nothing to the
  user's next step, one recipe trained on hundreds of maps at once.
* **What this branch carries.** The code is the unitfarmer2 lineage's best branch
  (ar/uf2r2b4, b93a7151) with the operator's harness. Its notes are in git history:
  `git show b93a7151:autoresearch/NOTES.md` (both unitfarmer2 rounds), and
  `git show 8a825803:autoresearch/NOTES.md` (the princessburglar lineage and its expert loop).
  What transferred, in short:
  * **unitfarmer2 (2026-10-30..31):** a policy that already finished got faster by iterating on
    its OWN search lines: beam search on the newest weights (`tools/beam_tas.py`, also
    `--pitch-hold`) -> its best real line -> a pace-gated spine pinned at the true start
    (`--demo-file` of that line with `--self-states`, `--demo-tau0 0 --demo-rate 0
    --demo-success-pace`) + BC on the line (`--bc-file`) + the lines as spawn states -> ~45 min
    of training -> repeat: 36.88 -> 35.31 s in a few hours, past the record in the simulator.
    That map had a loop of its own (60-65% of the course), and a search found it before the
    policy did.
  * **princessburglar (2026-10-29):** the way out of its start room also runs against the
    potential (pump a half-pipe for energy first), the closest thing to a loop this project has
    solved. What found it in ~3 min of box CPU from the true spawn was Go-Explore with energy
    cells and the policy's own action frame (`tools/explore_phase1.py --exact-chains --cell 64
    --cell-energy 32 --cell-heading 1 --view-follow=-90,-60,-30,-15,0,15,30,60,90
    --select-energy 3 --act-every 4 --decisions 100 --maxvel 3500 --envs 4096`, at the tick of
    whatever will train on the line); `tools/ge_replay.py` makes an exact line, and a backward
    curriculum on it taught it. Counter-evidence from unitfarmer2's October: an energy archive
    alone (`tools/go_cem.py`) filled a pit and never left it.
  * Nothing in those tools names a place. Whether they climb a loop is the question for you,
    not a conclusion.
* **The bar.** First up the first loop from the true start in a greedy eval (progress past
  ~10% with the episode still flying), then a finish, then 9/9, then the record's 450.46 s on
  our clock.

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

* **First test of any idea: at most 30 min of box time.** The incumbent recipe ran 347k env
  steps/s on an RTX 5080 on this map (the operator's zlop_check), and roughly 150-250k on a
  3090 elsewhere, so 30 min is about 3e8-6e8 steps. Size `--steps` to the time, and read the box's own fps after the first
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
* **Budget.** The total is the vast credit ($51.12 at this program's start). box.py paces the
  fleet's burn at (credit - reserve) / hours left until config.budget_end_hours (+10% slack),
  ~0.56 $/h at the start: one or two cheap boxes at once (a 5080 runs ~0.35-0.39 $/h), so a box
  will often be refused while another agent's runs. Plan CPU work (analysis, code, tests) for
  those hours. It keeps the $10 reserve: no rent at the reserve, and no `extend` that the
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
