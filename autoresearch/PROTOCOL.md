# Autoresearch protocol - branch agents (fork-and-flush on surf_src_sidistic)

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
* You do not talk to the other branches. You are not asked to "be different" either (the
  paper measured that directive: it does not help). Do the best research you can.
* At the round's end the orchestrator scores every branch with `autoresearch/score.py`. That
  score is the best greedy eval from the TRUE map start that your runs produced this round.
  The winner's branch head seeds the next round's four forks. The other branches are archived
  and pruned. So what you leave in this branch (code, NOTES.md, RESULT.json) is what the next
  round inherits if you win.

## The goal and the one number

Pass sidistic from the true start, champion-free and generic: first a finish, then faster
finishes. Your score is `python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/`. It
is the best eval's max-episode progress on the goal field (a finish counts 100), plus
10 x the finish fraction, plus a time term once something finishes. Only greedy evals from
the true start count; the grader rejects anything else. The milestones known so far are in
NOTES.md: the first-ramp trap at 5.7%, the wall hole ("window") at 20.8% -> CP1 at 22.9%,
and the CP2 wall at 27.5%.

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

## Experiments: short first, long only when earned

* **First test of any idea: at most 30 min of box time.** On a 3090 the incumbent recipe
  runs roughly 150-250k env steps/s, so 30 min is about 3e8-4e8 steps. Size `--steps` to
  the time, and read the box's own fps after the first minute.
* **Compare at MATCHED STEPS** against the incumbent run's progress.csv
  (`$AR_HOME/incumbent/`). Use the gate cleared and the step it was cleared at, never
  eval_progress alone (CLAUDE.md section 3). One seed, so a lead must be large to mean
  anything (CLAUDE.md: the gate ladder and the seed-noise floor).
* **Extend up to 6 h only if the idea is promising**: it clears a gate sooner than the
  incumbent at matched steps, or clears a gate the incumbent never cleared. Say why in
  NOTES.md when you extend.
* **Warm starts are allowed.** Push `$AR_HOME/incumbent/ckpt.pt` with
  `box.py deploy --ckpt ...` and launch with `box.py launch --resume ...`. This measures AT
  the wall (CLAUDE.md: the stuck checkpoint is one tool) or pushes the frontier past it.
  State which in NOTES.md. A recipe that only works warm is not yet a recipe.
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
  the 5080, are fine). No datacenter cards, ever. box.py and the watchdog enforce this.
* **Watching.** The user watches through Claude Remote Control and `~/ar/status.md`. That
  replaces CLAUDE.md's per-box dashboard tunnel. A dashboard on the box is optional, and no
  tunnel is needed.
* **Budget.** The total is the vast credit; box.py paces it to config.budget_end_hours.

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
start of every round. An edit you make to one does not reach the next round. If one blocks
you, work around it on your branch and write it in NOTES.md under a "Harness issue" heading;
the operator reads those. `autoresearch/NOTES.md`, the trainer, the tools and the tests are
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
* If `rent` refuses on budget, the fleet is spending faster than the money lasts. Do analysis
  or coding and try later. Never work around it.
* Your box id is in `$AR_HOME/boxes/$AR_AGENT.json` and `box.py status`. Keep it in NOTES.md
  too.
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
* The next round's incumbent checkpoint is the winning eval's run's `ckpt_final.pt`, or its
  `ckpt_latest.pt` if there is no final one, as pulled to `$AR_HOME/runs/`. That is the run's
  NEWEST weights, usually later than its best eval. `AR_BRANCH.json`'s incumbent gives
  `eval_step` and `ckpt_step`. Before you compare against the incumbent's score, read the
  checkpoint's own first eval.
* `RESULT.json` (worktree root, read by the orchestrator, rewritten whenever the best moves):

```json
{"agent": "r1b2", "round": 1, "final": false,
 "best": {"score": 27.5, "eval_file": "/home/steam/ar/runs/r1b2/r1b2_x/traj_0755499008.jsonl",
          "run": "r1b2_x", "step": 755499008, "ckpt": "/home/steam/ar/runs/r1b2/r1b2_x/ckpt_latest.pt",
          "recipe": ["--obs-compass", "1"], "resume_from": null, "commit": "<sha>"},
 "summary": "<= 10 lines: what you tried, what it did at matched steps, what to try next"}
```

## Wrap-up (at `wrapup_at`, or when a message says ROUND OVER)

1. Launch nothing new. Pull the best run, with its checkpoint (`--ckpt`).
2. Score it. Write `RESULT.json` with `"final": true` and the summary.
3. Append the round's lesson to NOTES.md: what worked, what did not, the numbers.
4. `git add -A && git commit` (trailers) `&& git push origin ar/$AR_AGENT`.
5. `box.py release --agent $AR_AGENT`, and confirm with `box.py status`.
6. Stop scheduling wakeups. The orchestrator ends this session at the deadline.

## Talking to the user

The user watches every session through Claude Remote Control and may write to you at any
time. Answer first, in a few lines:

* what is running;
* its latest eval and score against the incumbent;
* what you will do next.

ASCII only (the user's terminal is cp1251). Label every line and point of any plot (the user
is colour-blind). Times come from `date`.
