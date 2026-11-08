# Autoresearch protocol - branch agents (fork-and-flush on surf_src_sidistic: ONE recipe)

You are ONE branch agent of a fork-and-flush round (Cai et al. 2026, arXiv 2610.07447,
"Escaping Idea Basins in Autoresearch Agents"). Read this file, then `CLAUDE.md` (binding,
every rule in it applies to you), then `autoresearch/NOTES.md` (what this lineage knows),
then `AR_BRANCH.json` (who you are, your SLOT, your deadlines, the incumbent). Then start working.

## The method, and your place in it

* Each round forks K = 4 branch agents from the SAME workspace: this git branch (code,
  `autoresearch/NOTES.md`, the ledger) plus the protected incumbent (the best checkpoint and
  eval so far). Each fork starts with a FRESH context. That is the "flush", and it is the
  point: independent runs stall in different "idea basins", and a fresh context escapes the
  one the previous lineage was in.
* You do not talk to the other branches, and you do not read what they do (see
  **Independence** below). In round 1 each slot has an assigned explore step to build and test
  (**Your slot**, below), because the user wants the candidates compared; past that, do the best
  research you can.
* At the round's end the orchestrator scores every branch with `autoresearch/score.py`. That
  score is the best greedy eval from the TRUE map start that your runs produced this round,
  counted only for runs that obey **the one-recipe rule**. The winner's branch head seeds the
  next round's four forks. The other branches are archived and pruned. So what you leave in this
  branch (code, NOTES.md, RESULT.json) is what the next round inherits if you win.

## Independence: read only your own branch

Ideas cross between lineages only at the round boundary, through the winner's workspace. So
read only:

* your worktree (this branch: its code, NOTES.md, RESULT.json, AR_BRANCH.json) and its git
  history;
* your runs (`$AR_HOME/runs/$AR_AGENT/`) and your box (`box.py status --agent $AR_AGENT`,
  `box.py budget`);
* the incumbent (`$AR_HOME/incumbent/`);
* the operator's control (`$AR_HOME/control/`: the incumbent recipe from scratch with NO search,
  `unop_ctl`, refreshed every ~30 min with its ladder). It is a reference, like the ladder.

Never read, list or grep any of these:

* another branch's worktree (`$AR_HOME/wt/b<k>`), runs (`$AR_HOME/runs/<agent>`) or box
  record;
* another branch's git branch (an `ar/*` ref that is not yours);
* another branch's tmux window or Claude transcript;
* the archive of earlier rounds (`$AR_HOME/archive/`);
* the orchestrator's files: `$AR_HOME/status.md`, `state.json`, `orchestrator.log`. They are
  the user's and the operator's view of every branch.

This holds even when the goal is to avoid repeating another branch's idea. Copying the leader
mid-round only pulls a branch into the leader's idea basin. A losing branch's findings are
archived, not inherited: that is the flush.

## The goal: ONE recipe (user, 2026-11-08)

This program does not ask you to beat sidistic. Its October lineage did that one wall at a
time: seven rounds, and at nearly every wall a new flag or a hand-moved constant chosen while
watching that wall (NOTES.md lists them). The user now wants the opposite: ONE recipe that
clears the walls by itself if given time, with nothing changed after launch, so that it can
later run on maps nobody has looked at (CLAUDE.md 0b). In the user's words:

> "The main goal should be like not to tune the model for every single wall ... We're kind of
> trying to find one unified recipe that will converge if we give it some time. ... if the
> algorithm doesn't beat the wall in twice time it was beaten before, then the algorithm
> doesn't work. We need to call it at some point."

The candidate the user proposed: Go-Explore's archive of states, whose explore step is a
search that can run anywhere from fully random (the Nature paper's) to fully policy-guided (our
beam search, or AlphaZero-style MCTS), with ONE continuous knob between them, feeding a
backward curriculum on the lines it finds - "ideally we don't even need the expert loop stage.
So it's kind of all consolidated into one thing ... at minimum find the recipe that doesn't
require tuning for every wall in the map." And on the knob: "we can just ... try to reduce the
parameter and see at which point is the policy guided search more useful, and maybe we
conclude that it's not useful at all."

The score is unchanged: `python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/` = the
best eval's max-episode progress on the goal field (a finish counts 100) + 10 x the finish
fraction + a time term once something finishes (10 x (1 - best finish s / 300)). Only greedy
evals from the true start count; the grader rejects anything else. But only one-recipe runs
count, and **the ladder decides whether a recipe is alive**.

## The one-recipe rule (read it twice)

* **A run is one recipe**: one commit of this branch plus one flag set, started FROM SCRATCH
  (step 0) at the true start. `box.py launch` without `--resume`, the incumbent recipe
  prepended (`config.incumbent_recipe`) plus your extra flags.
* **Nothing changes after launch**: no flag, constant, schedule, learning-rate cut, code
  change, hand-picked line, spawn file or search setting. Anything that should adapt during a
  run is a RULE you write into the recipe BEFORE launch, reading only the run's own state (the
  plateau temperature `--unstuck` and `--auto-spine` are examples of such rules).
* **A crash-resume of the SAME run** (same commit, same flags, its own newest checkpoint)
  continues it and keeps its step count. Any other change is a new run from step 0.
* **No warm starts in round 1.** From round 2 you may CONTINUE the incumbent run with its exact
  recipe: deploy the incumbent checkpoint, `box.py launch --resume` with exactly the extra flags
  of the incumbent run's launch (AR_BRANCH.json and its RESULT.json give the run; the trainer
  restores the rest from the checkpoint - diff the new run.json "config" against the old one
  and say in NOTES.md that they match). Or start a recipe of your own from scratch. Never
  warm-start a changed recipe from any checkpoint.
* **Constants are set once** and would ship unchanged to a map nobody has opened (CLAUDE.md
  0b). Nothing in the recipe may name a place, a wall, a progress value, a speed or a time read
  off this map, the ladder or the record.
* **RESULT.json names the recipe** (commit + the full flag list) and says
  `"one_recipe": true` and `"from_scratch_step0": true` (or the incumbent run it continues).
  The operator checks the winner's run.json, launch log and git history; a run that broke the
  rule is not scored and its branch cannot win.

## The ladder: walls and the 2x stop rule

`autoresearch/ladder.json` lists the map's walls in order, the step at which the October
lineage's greedy eval first got past each, and a deadline of twice that step. Check a run with

    python autoresearch/ladder.py $AR_HOME/runs/$AR_AGENT/<run>/

after every pull (it scores each eval with score.py and prints, per wall, the step it was first
cleared and ON TIME / LATE / PENDING / FAILED).

| # | wall | cleared when | October step | deadline |
|---|---|---|---|---|
| 1 | first ramp | progress > 8% | 0.30B | 1.00B (2x is 0.61B; raised: seed noise 0.18-1.51B) |
| 2 | window | curtain sgcp_1 | 1.04B | 2.07B |
| 3 | pit dive | progress > 27.5% | 1.94B | 3.88B |
| 4 | cavity | curtain sgcp_2 | 4.14B | 8.27B |
| 5 | post-CP2 block | progress > 33% | 6.97B | 13.93B |
| 6 | CP3 | sgcp_3 | 7.32B | 14.64B |
| 7 | CP4 | sgcp_4 | 9.39B | 18.79B |
| 8 | CP5 | sgcp_5 | 11.78B | 23.57B |
| 9 | CP6 | sgcp_6 | 14.26B | 28.52B |
| 10 | CP7 | sgcp_7 | 15.23B | 30.47B |
| 11 | chamber entry | progress > 98.5% (approximate) | 19.35B | 38.69B |
| 12 | finish | a finish | 25.45B | 50.90B |

* **A run that has not cleared wall k in a greedy eval by wall k's deadline has FAILED at wall
  k.** Stop it, write down where and why, and start your next idea. A run that is on time may
  run for the rest of the round (it is not held to the 30 min first-test cap below).
* **The early deadlines are loose.** The October lineage changed its recipe between walls, and
  from-scratch runs then cleared walls 3-6 two to four times sooner (r3b2: CP3 at 1.9B, 54% at
  2.49B). So beating a deadline is the minimum. The real comparison is at MATCHED STEPS against
  the operator's control (`$AR_HOME/control/`), the same recipe with no search.
* Throughput sets how far a round gets: the October boxes ran 135-277k steps/s (0.5-1.0B per
  hour), so a 12 h round reaches ~6-12B steps, about wall 4-5. Round 2 continues the winner.

## The research question: the explore step

**The loop all four slots share.** Go-Explore's archive of exact simulator states (cells keyed
by generic quantities: position, energy height E = z + |v|^2/1600, heading); an explore step
from a selected cell; every new cell joins the archive with the exact trajectory that reached
it; lines from the archive feed the backward curriculum INSIDE the trainer, automatically (the
machinery of `--auto-spine`: the trainer runs the search on its own weights, and a line that
beats the greedy frontier becomes the curriculum, its start moving back as the policy
succeeds). No hand-picked line, no separate expert loop: speed is not this program's question.

**Your slot** (AR_BRANCH.json `slot`) sets the explore step you build and test first in round 1:

1. **The Nature paper's explore step** (the knob at "fully random"): restore a cell chosen with
   weight 1/sqrt(1 + times chosen), then uniform random actions for ~100 decisions, each action
   repeated with p = 0.95; replace a cell's stored state when a new arrival has a better score,
   or the same score with a shorter trajectory. With this project's generic fixes: the bursts
   act in the policy's own action frame (view relative to the velocity, `explore_phase1.py
   --view-follow`) and the cells carry energy (`--cell-energy`, `--select-energy`).
2. **A policy-guided beam** (our beam_tas) as the explore step from the selected cell: copies of
   the policy sample their own actions, with the knob (temperature, or a coherent mix of random
   bursts and policy samples). Keep EVERY new cell any copy reaches, not only the leader's line
   (distance culling killed unitfarmer2's climb). Sweep the knob from policy toward random.
3. **Policy-prior MCTS** (AlphaZero's PUCT: the policy as the prior, the critic and/or a
   rollout as the value) as the explore step. Per-decision branching at 25 Hz is hopeless at
   this horizon: make an edge a held segment of decisions (a macro-action). New cells from any
   node join the archive. Build it, then test the explorer alone on a known gate first.
4. **The knob as one rule**: one explorer whose random-vs-policy balance is a single continuous
   parameter set automatically from the run's own state. The paper's policy-based Go-Explore
   already does a version of it (below): per explore step a coherent 50/50 choice between random
   actions and policy sampling, and a temperature that rises with the steps since the last new
   cell. A bandit over the knob rewarded by new cells, or the plateau-temperature rule, are other
   forms. Or the paper's goal-conditioned policy-based Go-Explore if you judge it better. Aim for
   the simplest single mechanism.

Test the explorer ALONE first, with no training, where the answer is known (30 min of box
time): does it get past the first ramp, the window and the pit dive from a policy that stops
there? Then the whole loop from scratch, measured on the ladder. In rounds 2+ slots are free:
build on the winner.

**What is known (evidence; NOTES.md has the sources)**

* Random Go-Explore with restore found princessburglar's real start and za_loop's first pump in
  minutes of CPU at low speed, once the bursts acted in the policy's frame and the cells carried
  energy. It failed at speed: cannonball froze at 6.67% of its arc over 32.5M runs, and it
  filled unitfarmer2's pit without leaving it. On sidistic itself, Go-Explore was tried only in
  round 1 (r1b3, with SIL), BEFORE those two fixes, and lost (27.5% vs the winner's 28.6%):
  sidistic has never had a fair random baseline.
* Policy-guided beams found unitfarmer2's pit exit (Monte Carlo, no culling, keys temperature
  1.5) and za_loop's first finishing lines (beams from the policy's own greedy line, ranked by
  distance minus 30 x energy). Per-decision epsilon killed beams at speed (cannonball: eps 0.15
  and 0.30, 0 finishes in 8 runs); held perturbations did not. Distance-ranked culling killed a
  climb that first moves away from the goal.
* `tools/edge_archive.py` (a policy explorer with count selection) found a test map's detour in
  2-8 s, but its chains were not reliable flown open-loop. `tools/go_cem.py` (an optimiser as the
  explorer, energy dominance in the cells) never left unitfarmer2's pit: keep steep falls OUT of
  the "best" state of a cell.
* sidistic's October walls were all cracked by RL mechanisms (no search was needed until the
  expert loop for speed): plateau temperature, harvesting own states 2 s before death, the
  ratchet + novel spawns, removing the time charge, greedy-death spawns, key-only temperature,
  front-perturbed own states, the energy term, climbing backward rows. All of them are in
  `config.incumbent_recipe` from step 0, so the control may already be fast early.

**The Nature paper** (Ecoffet et al. 2021, "First return, then explore"; the full version with
its Supplementary Information, whose ablations the arXiv copy leaves out; details in NOTES.md):

* Restore-based exploration: weight W = 1/sqrt(C_seen + 1); 100 random actions per explore
  step (30 in robotics) with the previous action repeated p = 0.95 (0.90); the environment
  deterministic during exploration. Every headline result came from this.
* Policy-based Go-Explore: a goal-conditioned policy (PPO + self-imitation) returns to the cell,
  then per explore step commits 50/50 to random actions (not trained on) or to sampling the
  policy toward goals (10% an adjacent unseen cell, 22.5% any adjacent cell, 67.5% an archive
  cell by W); in the policy mode the temperature grows as 1 + (0.01 x max(0, t - 50))^2, t = steps
  since the last new cell; an episode stops after 1,000 steps with no new cell. It found 3.4x
  (Montezuma) and 3.9x (Pitfall) more cells than random exploration, but only 1.5x early; the
  policy that did it chased NOVEL adjacent cells, not the task reward.
* Ablations: no action repeat is worse almost everywhere and fails robotics; fixed cell
  downscaling fails "often catastrophically" on other games; generic selection weights are
  slower but still finish; self-imitation was essential to the policy-based variant (Montezuma
  ~1K vs ~20K without/with). Not in the paper: exploring without returning, uniform selection,
  restore vs policy return at a matched budget, and MCTS (future work).
* So: a beam that samples the REWARD policy has no evidence from the paper either way; to inherit
  the 3.4-3.9x it may need novelty goals. Coherent (held) randomness is the one thing every
  result agrees on.

**Tools on this branch**: `tools/explore_phase1.py` (restore-based Go-Explore, energy cells,
`--view-follow`, `--exact-chains`), `tools/ge_replay.py` (an exact line from an archive),
`tools/beam_tas.py` (the policy-guided population search; `--start-state`, `--prefix-line`,
`--elite-frac 1.0` = Monte Carlo, `--keys-temp`, `--view-temp`, `--view-hold`, `--eps`,
`--score d|dh|dv`), `tools/auto_spine.py` + `train_fast.py --auto-spine` (search -> curriculum
inside the trainer; single map, single GPU; `--auto-spine-async`), `--finish-line` (a backward
curriculum on a finishing line), `tools/edge_archive.py`, `tools/refine_arch.py`. Read their
docstrings before writing new code.

## This lineage: surf_src_sidistic (operator, 2026-11-08)

* **The map.** You spawn high on a platform at about (7,837, 0, 15,042) and surf down to a
  176 x 176 u shaft whose floor is the finish trigger (a plain box), ~14,860 u of descent over 7
  stages separated by checkpoint curtains sgcp_1..7 (teleports with no target, ignored by the
  core; score.py reports the furthest one an eval entered as `max_cp`). The goal field reads
  99,675 u at the spawn (config.start_d, kill-aware). Many of the human's moves are detours away
  from the goal: a left turn onto the first ramp, a cut back into an A-frame ramp up through a
  110 x 130 u hole in a wall (the window), a turn EAST onto a ramp, a ~2,000 u dip into a cavity
  under the pit slab, kill-wrapped blocks, a slit, a chimney, a basement, a sealed chamber, a rim.
* **The record**: krs, 115.99 s on speedrun.eu (2024-07-28), 115.92 s on our game clock; our
  runs need ~1.2 s more from spawn. It is a bar to read results against, never a constant.
* **Physics, as in October**: 10 ms tick (the trainer's default), `--maxvel 4000`, 300 s
  episodes. Keep them: the ladder's steps were measured at this physics.
* **What October needed at each wall** is in NOTES.md, for INSIGHT ONLY (CLAUDE.md 0b): never
  rebuild a per-wall fix. The record's demo is analysis only (CLAUDE.md section 0).

## Each wakeup

You run inside `/loop` (self-paced). Every wakeup:

1. Read `AR_BRANCH.json`. If `now >= wrapup_at`, go to **Wrap-up** and do nothing else.
2. Check your box and run: `python autoresearch/box.py status --agent $AR_AGENT`, then the
   run's progress through `box.py ssh` (tail of `runs/<run>/progress.csv`, the last greedy
   eval). Pull new evals and run `autoresearch/ladder.py` on the run.
3. Act: analyse, code, launch, extend, or stop a run (a FAILED wall, or a run that cannot beat
   the control at matched steps, is stopped, not left to burn money).
4. Write down what you learned: append to `autoresearch/NOTES.md` under a dated heading
   (take the time from `date`, never type it). Keep `RESULT.json` current (format below).
5. Schedule the next wakeup with ScheduleWakeup. While a run trains, wake every 20-30 min.
   Wake sooner only while you are actively building or a launch is in flight. One waiter,
   no polling loops, no Monitor spam: the user pays for every wakeup.

Stopping a run does not end your round. Until `wrapup_at`, a failed idea is written down and
followed by the next one. Release the box while you analyse or build, but keep the loop going:
an early Wrap-up leaves this branch's slot empty for the rest of the round. The orchestrator
re-arms the loop of a branch whose RESULT.json says final before `wrapup_at`.

## Experiments: short first, long only when earned

* **First test of any idea: at most 30 min of box time** (an explorer alone, or the start of a
  run). Read the box's own fps after the first minute: the October boxes ran 135-277k steps/s.
* **Compare at MATCHED STEPS**: the ladder (time-to-wall) and the operator's control, the same
  recipe without search. One seed, so a lead must be large to mean anything (CLAUDE.md: the
  gate ladder and the seed-noise floor; the first ramp alone varied 0.18-1.51B from scratch).
* **Extend a run only while it is ON TIME on the ladder**, up to the rest of the round. Say in
  NOTES.md why you extend.
* **Hard rules from CLAUDE.md that bite here:**
  * No human demo touches training (section 0).
  * No map-specific method, threshold or box in the reward, the spawns, the archive or the
    schedule (section 0b). Ask "would I run exactly this on a map I have never opened?"
  * New trainer flags default OFF and are bit-identical when off (prove it).
  * Every new flag is mirrored in `tools/record_ckpt.py` or listed TRAIN_ONLY, or the launch's
    record gate kills the run.
  * Shape-changing flags are never passed to a resume.

## What the user set for this program, over CLAUDE.md's defaults

* **Ablation length.** At most 30 min first; a run ON TIME on the ladder may run for the rest
  of the round. This replaces CLAUDE.md's "one hour per ablation".
* **Boxes.** One box per agent, single GPU only, consumer RTX cards under the config.max_dph
  caps: box.py takes the cheapest acceptable offer (often a 5060 Ti / 4070-class card on a
  32+ thread host), so a box may be a slower card. No datacenter cards, ever. box.py and the
  watchdog enforce this. If deploy's gpu_health says BUSY on a fresh box, another tenant is on
  that GPU: blocklist it (`tools/vast_pick.py --block <id> --reason unreliable --detail ...`)
  BEFORE you release it.
* **Watching.** The user watches through Claude Remote Control and `$AR_HOME/status.md` (the
  user's view, which you do not read). That replaces CLAUDE.md's per-box dashboard tunnel.
* **Budget.** The total is the vast credit ($55.39 at this program's start). box.py paces the
  fleet's burn at (credit - reserve) / hours left until config.budget_end_hours (+10% slack),
  ~1.8 $/h at the start: four branch boxes plus the operator's control. It keeps the $10
  reserve: no rent at the reserve, and no `extend` that the fleet's burn would pay out of it.

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
protocol, ladder.json, box.py, score.py, orchestrator.py, config.json, the watchdog,
deploy_box.sh, vast_pick.py, gpu_health.py and their tests) are restored from the operator's
branch at the start of every round. An edit you make to one does not reach the next round. One
exception: `config.json`'s `lineage_keys` (`incumbent_recipe`) are the lineage's. If a harness
file blocks you, work around it on your branch and write it in NOTES.md under a "Harness issue"
heading; the operator reads those. `autoresearch/NOTES.md`, `autoresearch/ladder.py`, the
trainer, the tools and the tests are yours, and they travel with your branch if it wins.

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
  later flags win. `--no-recipe` gives a bare scratch preset (then it is not the program's
  recipe: say what you replaced it with).
* If `launch` returns an ssh error (exit 255, "Broken pipe"), the run usually started anyway:
  check `ps` on the box and `runs/<run>_launch.txt` before launching again. If it died before
  its record gate ran (no `traj_<step>_rec_*` files), run the gate detached:
  `setsid nohup python3 tools/record_gate.py <run> --pid <trainer pid> --wait-secs 900 >
  runs/<run>_gate.txt 2>&1 < /dev/null &`.
* If short ssh commands work but `pull` stalls ("Broken pipe", an empty `_pull.tgz`), the box's
  DIRECT port drops large downloads (seen 2026-11-08 on the operator's control box). Use vast's
  proxy instead: `vastai show instance <id> --raw` gives `ssh_host` / `ssh_port`; put them in
  `$AR_HOME/boxes/$AR_AGENT.json` as `host` / `port` (keep the direct pair as `host_direct` /
  `port_direct`) and pull again.
* A spawn-state file built from the policy's OWN states is passed with `launch --self-states
  "<how it was built>" -- --spawn-states <file>`; inside a one-recipe run, lines and spawn files
  are made by the run itself (the auto-spine machinery), never by hand.
* Never call `vastai create`/`destroy` yourself. The fleet watchdog destroys any instance that:
  * is not a single allowed RTX card under its price cap;
  * is above the box cap;
  * is past its deadline;
  * is unregistered.
* A box with no experiment running is released within 5 minutes (CLAUDE.md section 1). Do
  your analysis with the box released, unless you are about to launch again.
* If `rent` refuses on budget, the fleet is at its spending limit (`box.py budget`). Do analysis
  or coding and try later. Never work around it.
* Your box id is in `$AR_HOME/boxes/$AR_AGENT.json` and `box.py status --agent $AR_AGENT`. Keep
  it in NOTES.md too.
* A bad box (gpu_health fails, a foreign tenant on the GPU, or the trainer far under ~130k
  steps/s on the incumbent recipe) is blocklisted with `python tools/vast_pick.py --block <id>
  --reason ... --detail ...` BEFORE you release it (reasons: gpu_capped, cpu_bound, no_cuda,
  unreliable, network). The orchestrator folds every branch's list into
  `$AR_HOME/bad_hosts.json`, and `box.py` skips both lists.

## Harvest, score, record

* Before releasing a box, pull every run you may want to cite:
  `box.py pull --agent $AR_AGENT --run <run>`. Add `--ckpt` for the one you might hand on.
* Score: `python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/<run>/`, and the ladder:
  `python autoresearch/ladder.py $AR_HOME/runs/$AR_AGENT/<run>/`.
* The next round's incumbent checkpoint comes from the winning eval's run, as pulled to
  `$AR_HOME/runs/`. It is the newest `ckpt_<step>.pt` AT OR BEFORE the winning eval, unless the
  run's newest checkpoint is more than twice as near the eval. A snapshot is handed on only if
  you pulled it (`box.py pull --agent $AR_AGENT --run <run> --snapshot <step>`). For a recipe
  that the next round should CONTINUE, hand on the run's newest checkpoint (`--ckpt`) and say so.
* `box.py pull --ckpt` downloads to `<ckpt>.part` and replaces the copy already pulled only
  when the md5 matches the box's, so a pull that times out on a slow link loses nothing.
* `RESULT.json` (worktree root, read by the orchestrator, rewritten whenever the best moves):

```json
{"agent": "unr1b2", "round": 1, "final": false, "slot": 2,
 "best": {"score": 27.5, "eval_file": "/home/steam/ar_uni/runs/unr1b2/unr1b2_x/traj_0755499008.jsonl",
          "run": "unr1b2_x", "step": 755499008, "ckpt": "/home/steam/ar_uni/runs/unr1b2/unr1b2_x/ckpt_latest.pt",
          "recipe": ["--auto-spine", "..."], "commit": "<sha>", "one_recipe": true,
          "from_scratch_step0": true, "resume_from": null},
 "ladder": {"first ramp": 302514176, "window (CP1)": null, "verdict": "alive"},
 "summary": "<= 10 lines: the explore step, the knob, what it did at matched steps vs the control, next"}
```

## Wrap-up (at `wrapup_at`, or when a message says ROUND OVER; never before)

1. Launch nothing new. Pull the best run, with its checkpoint (`--ckpt`) and the snapshot
   at or before its best eval (`--snapshot <step>`).
2. Score it and run the ladder. Write `RESULT.json` with `"final": true` and the summary.
3. Append the round's lesson to NOTES.md: what worked, what did not, the numbers, the ladder.
4. `git add -A && git commit` (trailers) `&& git push origin ar/$AR_AGENT`.
5. `box.py release --agent $AR_AGENT`, and confirm with `box.py status --agent $AR_AGENT`.
6. Stop scheduling wakeups. The orchestrator ends this session at the deadline.

## Talking to the user

The user watches every session through Claude Remote Control and may write to you at any
time. Answer first, in a few lines:

* what is running;
* its latest eval, its ladder row and the control at the same step;
* what you will do next.

ASCII only (the user's terminal is cp1251). Label every line and point of any plot (the user
is colour-blind). Times come from `date`.
