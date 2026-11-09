# The one-search series - slot agents (surf_src_sidistic, started 2026-11-09)

You are ONE of four slot agents. Each of you builds and runs one step of a designed series of
experiments, in parallel, on your own box. Read this file, then `CLAUDE.md` (binding, every rule
in it applies to you), then the operator's section at the top of `autoresearch/NOTES.md` (what
is known), then `AR_BRANCH.json` (who you are, your SLOT, your deadlines). Then start working.

## The question (user, 2026-11-09)

The last program (the unified-recipe program, 2026-11-08/09) ended with one recipe that
finishes sidistic from scratch: its **run 5** (`unr1b3_mcts5`, first finish at 8.23B steps,
later 9/9 finishes at 118.28 s). Run 5 is a HYBRID of two threads: PPO trains the policy, and
beside it a search (MCTS over macro-actions) finds lines that become spawn states for PPO. The
user asked whether one thread would be better. In the user's words:

> "all of the experiments follow the following pattern. We have one thread that is running
> normal PPO, and the other one is running the exploration using the search. ... MCTS mostly
> use random searches with very high temperature. That's like basically random rollout. ... the
> idea is to have just one thread. ... the way alpha 0 was trained for chess, like we didn't
> have one PPO and one search. We had just one search. And the model kind of updates of the
> policy were made based on the trees that we generated."

> "The idea in some sense is that we have one pool of spawns, not two, like we have here. We
> have maybe more balanced usage of the policy and the random actions. We can try to infer the
> optimal kind of way of doing that from the experiments that we had. And to basically update
> the policy and the value function just based on the search, just based on the tree that we
> built, or beam search as an alternative if it works better, but so far it doesn't look like
> it's looking better. Maybe it's a bad idea, I'm not sure. Maybe ... this hybrid is actually
> better, but I don't know."

So the user does not know the answer and neither does the operator. **A clean negative is a
full result here.** Your job is to make your step of the series a FAIR test of its idea and to
report honestly where it stands against run 5. Nobody is asked to beat run 5 by any means.

## What you start from: run 5, in five parts

`config.incumbent_recipe` is run 5's exact flag list (126 elements, commit 9eda7009). `box.py
launch` prepends it. Its parts:

1. **The learner.** PPO on 1,024 copies of the game, from scratch, at the true map start. The
   reward, the observation and the temperature rules are the October lineage's.
2. **The search.** When the greedy eval stops moving (`--auto-spine-patience 3e8`,
   `--auto-spine-retry 3e8`) or the curriculum has done its work (the quorum), the trainer
   starts a search on a copy of its newest weights, beside the training on the same GPU
   (`--auto-spine-async`): 6 minutes of MCTS over macro-actions on 2,048 game copies
   (`tools/mcts_search.py`, driven by `tools/beam_tas.py --mcts`). The tree's roots are states
   on the policy's own greedy line. An edge is a macro of 12 decisions. Each expansion makes 16
   children: child 0 plays the policy's own best action, and in run 5 the other 15 hold ONE
   random action for the whole macro (`--mcts-random-frac 1.0 --mcts-perturb-frac 0`).
3. **The line.** The best line of the tree, if it still beats the policy's own frontier when
   the search ends (`--auto-spine-banked`, `--auto-spine-recheck`), becomes the line of a
   backward curriculum: training starts are placed on it, first near its end, then further back
   as the policy succeeds from there.
4. **Two pools of spawns.** 25% of the training starts come from that curriculum
   (`--demo-share 0.25`). The other 75% come from October's own sources: the map start, the
   policy's states 2 s before its deaths, novel spawns, greedy-death spawns, front-perturbed
   states and backward rows (`--respawn-*`).
5. **What the tree teaches.** Nothing, directly. One line leaves each tree and it reaches the
   policy only as spawn states. The policy and the value function learn from PPO rollouts alone,
   and the tree does not use the value function at all.

Run 5's searches used 24% of its simulator ticks up to the first finish (2.57B beside 8.23B of
rollouts). Search lines opened 4 of its 12 walls (the spawn exit, the pit approach, the loop
before CP6, the finish); the policy found the other 8 itself. NOTES.md has the details.

## The series, and your slot

The four slots run in PARALLEL, so slots 1-3 each change ONE part of run 5 and slot 4 goes to
the user's endpoint. Your slot is `slot` in AR_BRANCH.json.

1. **One pool** (part 4). Run 5 with October's spawn sources off, so that every training start
   that is not the map start comes from the search's curriculum. PPO and the search are
   unchanged. This is a flag change only: launch with `--no-recipe` and run 5's list WITHOUT
   `--respawn-novel*`, `--respawn-greedy*`, `--respawn-front-perturb*`, `--respawn-backward-mix`,
   `--respawn-backward-elev` and `--demo-share` (store_true flags cannot be switched off from the
   tail, hence the explicit list). Without `--demo-share` the trainer builds every pool as map
   starts plus curriculum-window states (`demo.build_pool`, train_fast.py; read it and confirm
   the shares in the launch log). Write down two things before you launch: until the first
   search line is adopted the curriculum's "line" is the spawn state, so ALL starts are at the
   map start; and whether you keep run 5's search cadence. A changed cadence is a second change:
   say so, and prefer to run the plain version first. You can launch within the first hour.
2. **A balanced tree** (part 2). Run 5 with the tree's child mix set by a rule instead of fixed
   at all-random. Today a sampled child is random-held with probability `--mcts-random-frac`,
   else a perturbed policy sample with probability `--mcts-perturb-frac`, else a plain policy
   sample (mcts_search.py, "the knob"). Build one rule that sets those shares from the search's
   own results, with a floor so that no mode dies out. The last program's slot 4 built such a
   rule for another explorer (`git show origin/ar/unr1b4:tools/knob_ge.py`: each mode's share
   follows its yield of new cells) and in its explorer-alone tests the rule sat at 52-66% random
   and beat or tied both pure ends. What the yield should count in a tree (new cells, or lines
   that beat the best banked progress) is yours to decide and to state. The rule goes in behind
   a search flag, default off, and reaches the trainer through `--auto-spine-mcts-args "..."`
   (a value flag, so your tail overrides run 5's). Known fact to respect: run 4 (fixed 0.5 random
   / 0.25 perturbed / 0.25 policy) failed the post-CP2 block where run 5 passed it.
3. **The tree teaches, beside PPO** (part 5). Run 5 plus labels from every search tree as
   extra losses next to PPO's. Policy target: at each expanded node, push the policy toward the
   children that did better than their siblings. Value target: each node's backed-up search
   outcome. Use every expanded node, not only the best line. The labels are made by the run
   itself (the search writes them, the trainer loads them when the search ends); nobody passes a
   label file by hand. The cheapest path is the existing imitation machinery (`--bc-file`,
   `--bc-coef`, `--bc-target`, `--bc-value-coef`, `tools/plan_to_bc.py`); check first whether it
   accepts this recipe's action heads (`--keys-hold`) and fix it behind a flag if it does not.
4. **The tree only** (the user's endpoint). No PPO loss. The network learns its policy and its
   value from the trees alone, and the spawns are the tree's too. Guidance, not orders:
   * The grader scores only the greedy policy WITHOUT search from the true start, so the tree
     must be grown on states the policy itself reaches (roots on its own lines, child 0 = its own
     best action), or the policy is trained on states it never visits.
   * Targets in the style of AlphaZero / Sampled MuZero: the policy toward the children the
     search preferred (visit share, or children that beat their siblings), the value toward the
     node's backed-up outcome.
   * Start with the tree as it is (an expert that does not use the network's value), then let
     the network into the search (its policy as the prior over children, its value at the
     leaves) if time allows. Say which version each run is.
   * Your step counter must count EVERY simulator tick (see the ladder, below).
   * The trainer has no supervised-only mode today. You may add one to `train_fast.py`, or
     drive the loop from `tools/exit_local.py` / `tools/expert_dagger.py` (search, then distil,
     in rounds: the expert loop of the earlier programs, which started from a trained policy and
     must here start from a fresh network). Either way the run must write greedy evals from the
     true start as `traj_<step>.jsonl` in the format `score.py` reads, a `progress.csv` and
     checkpoints `record_ckpt.py` can play.
   * Bench first (see Benches), and keep the first version small: a run that starts in hour 6
     tells the user more than a better design that starts in hour 16.

**Slots 3 and 4, both:** the targets must be scale-free (ranks, or values normalised inside
the tree): no number in them may come from this map. The tree's value is banked progress on the
goal field, not PPO's discounted return, so do not write it into PPO's critic: give it its own
head or its own normalisation, and keep PPO's baseline intact in slot 3.

**A second recipe.** If your first run fails the ladder, write down where and why and launch
your next recipe from scratch. It may combine changes that are alive on the ladder in other
slots (for example one pool with the balanced tree). Say what you took and from where.

## Sharing: read anything, edit only your own

This is a designed series, not fork-and-flush: no winner is forked and nothing is flushed.

* You MAY read the other slots' work: their worktrees (`$AR_HOME/wt/b<k>`), their branches
  (`git fetch origin`, then `git show origin/ar/azr1b<k>:<path>`), their NOTES.md, RESULT.json and
  pulled runs (`$AR_HOME/runs/azr1b<k>/`), and `$AR_HOME/status.md`. You may take their code
  (cherry-pick or copy) once it is pushed; say in NOTES.md what you took.
* You never edit another slot's files, never touch another slot's box, tmux window or
  transcript, and never wait for another slot. If you need something another slot is building,
  build the small part you need yourself.
* The last program's branches are readable too: `origin/ar/unr1b1..4` (their NOTES.md and
  tools), and its runs under `/home/steam/ar_uni/runs/` (read-only).

## The one-recipe rule (read it twice)

* **A run is one recipe**: one commit of your branch plus one flag set, started FROM SCRATCH
  (step 0) at the true start. `box.py launch` without `--resume`.
* **Nothing changes after launch**: no flag, constant, schedule, code change, hand-picked line,
  spawn file, label file or search setting. Anything that adapts during a run is a RULE written
  into the recipe BEFORE launch that reads only the run's own state.
* **A crash-resume of the SAME run** (same commit, same flags, its own newest checkpoint) keeps
  its step count. Any other change is a new run from step 0.
* **No warm starts.** No run of this program starts from a checkpoint of another run, of run 5
  or of any earlier program. Those checkpoints are for benches only.
* **Constants are set once** and would ship unchanged to a map nobody has opened (CLAUDE.md
  0b). Nothing in the recipe may name a place, a wall, a progress value, a speed or a time read
  off this map, the ladder or the record. Ask: "would I run exactly this on a map I have never
  opened?"
* **RESULT.json names the recipe** (commit + the full flag list) and says `"one_recipe": true`
  and `"from_scratch_step0": true`. The operator audits run.json, the launch log and the git
  history of every run that is reported as a result.

## The ladder: run 5 times two, in two units

`autoresearch/ladder.json` lists the map's 12 walls in order with the time run 5's greedy eval
first got past each, and a deadline of twice that. The user's rule: "if the algorithm doesn't
beat the wall in twice time it was beaten before, then the algorithm doesn't work."

There are two units, because this series moves work between the rollouts and the search:

* **steps** = the trainer's step counter (simulator ticks spent in PPO rollouts);
* **total** = steps + the simulator ticks of every search the run had started by then.

| # | wall | cleared when | run 5 steps | deadline steps | run 5 total | deadline total |
|---|---|---|---|---|---|---|
| 1 | first ramp | progress > 8% | 0.529B | 1.600B | 0.763B | 2.000B |
| 2 | window (CP1) | curtain sgcp_1 | 0.680B | 1.600B | 0.914B | 2.000B |
| 3 | pit dive | progress > 27.5% | 1.812B | 3.625B | 2.448B | 4.897B |
| 4 | cavity (CP2) | curtain sgcp_2 | 2.114B | 4.229B | 2.887B | 5.774B |
| 5 | post-CP2 block | progress > 33% | 2.341B | 4.682B | 3.252B | 6.504B |
| 6 | CP3 | curtain sgcp_3 | 2.643B | 5.286B | 3.554B | 7.108B |
| 7 | CP4 | curtain sgcp_4 | 3.624B | 7.249B | 4.928B | 9.855B |
| 8 | CP5 | curtain sgcp_5 | 3.851B | 7.702B | 5.293B | 10.586B |
| 9 | CP6 | curtain sgcp_6 | 6.116B | 12.232B | 8.220B | 16.440B |
| 10 | CP7 | curtain sgcp_7 | 6.720B | 13.440B | 8.966B | 17.933B |
| 11 | chamber entry | progress > 98.5% | 8.230B | 16.459B | 10.801B | 21.602B |
| 12 | finish | a finish | 8.230B | 16.459B | 10.801B | 21.602B |

The two early deadlines are a floor, not 2x: the first ramp alone varied 0.18-1.51B between
from-scratch runs of one recipe.

After every pull:

    # steps (slots 1-3)
    python autoresearch/ladder.py $AR_HOME/runs/$AR_AGENT/<run>/
    # total (slots 1-3): count the searches on the box, save beside the pulled run, then
    python autoresearch/box.py ssh --agent $AR_AGENT -- \
        "cd /root/RL_Surf && python3 autoresearch/search_ticks.py runs/<run>" \
      | python autoresearch/search_ticks.py --save $AR_HOME/runs/$AR_AGENT/<run>/search_ticks.json
    python autoresearch/ladder.py --total $AR_HOME/runs/$AR_AGENT/<run>/
    # slot 4, when the trainer's step counter already counts every tick it spends
    python autoresearch/ladder.py --total-is-step $AR_HOME/runs/$AR_AGENT/<run>/

* `search_ticks.py` counts a search as `search_wall_s x env_steps_per_s` of its `summary.json`,
  from the step its launch-log line carries. A run whose searches do not go through
  `--auto-spine` writes `search_ticks.json` itself (the format is in that file's docstring).
  Label ticks count: if your recipe spends extra simulator ticks to make labels, they are search
  ticks.
* **A run that has not cleared wall k by wall k's deadline has FAILED at wall k.** For slots
  1-3 the deciding unit is steps and the total column is reported beside it; for slot 4 it is
  total. Stop a failed run, write down where and why, start your next recipe.
* **Run 5 is one seed.** A wall cleared somewhat later than run 5 is not evidence against a
  recipe; a missed 2x deadline is. Do not rank two recipes on a gap under about 30% at one wall
  (CLAUDE.md: the seed-noise floor and the gate ladder). Report time-to-wall, never a mean of
  eval progress.
* A run that is ON TIME keeps training to the end of the round. Do not stop it to try a variant.

## Benches (measurement only)

A bench answers one question in at most 30 minutes of box time, and it is never a result.

* `$AR_HOME/reference/run5/` holds run 5's `run.json`, `progress.csv`, launch log, evals,
  `search_ticks.json`, its search bundle (`auto_bundle.tgz`: every search's spine, logs and
  `summary.json`) and its checkpoints (`ckpt_<step>.pt`). Read-only. Push one to your box with
  `box.py deploy ... --ckpt $AR_HOME/reference/run5/ckpt_<step>.pt`.
* A useful place on run 5's own timeline: from about 3.9B to 5.8B steps its greedy eval crept
  at 84-86% in front of CP6. Then search 15 found a line (a loop back and a climb into a gap),
  the curriculum adopted it, and PPO learned it in 0.35B steps. `ckpt_4530372608.pt` and
  `ckpt_4900519936.pt` are inside that creep.
* A bench is PAIRED: your variant and run 5's own settings, from the same checkpoint, for the
  same minutes. Slot 2: does the rule's tree get as deep as the all-random tree from there?
  Slots 3 and 4: do labels from one tree move the policy toward the tree's line, and how fast
  beside the 0.35B steps that PPO needed?
* Run 4's checkpoints (`/home/steam/ar_uni/runs/unr1b3/unr1b3_mcts4/ckpt_*.pt`) are a policy
  stuck under the post-CP2 block from 2.7B steps on. Caveat: in run 5 the policy itself got past
  that block; no search line opened it, and 15 minutes of search alone did not either.
* These checkpoints are measurement instruments, exactly like gate boxes in CLAUDE.md 0b. No
  training run starts from one, and nothing learned from a bench may enter a recipe as a
  constant of this map.
* Keep benches short in total. The round fits ONE full from-scratch run per slot (see Time and
  money); every bench hour comes out of that run.

## Hard rules from CLAUDE.md that bite here

* **No human demo touches training** (section 0). A search line or a tree label made by the
  policy's own search from the true start is the run's own data; a record, a demo or anything
  derived from one is not. `run_arm.sh` refuses `--demo-file`, `--bc-file` and `--route-file`
  without `SELF_STATES=1`: `box.py launch --self-states "<how the file is made, 20+ characters>"`
  declares it, and NOTES.md states the provenance.
* **No map-specific method** (section 0b): no threshold, box, place or schedule read off this
  map in the reward, the spawns, the tree, the labels or the losses.
* **New flags default OFF and are bit-identical when off** (prove it with a test).
* **Every new trainer flag is mirrored in `tools/record_ckpt.py`** or listed TRAIN_ONLY in the
  same commit, or the launch's record gate kills the run.
* **Shape-changing flags are never passed to a resume.** A new head changes the checkpoint's
  shape: it exists from step 0 of the run that has it.
* **One seed. One run per recipe.**

## This map: surf_src_sidistic

* You spawn high on a platform at about (7,837, 0, 15,042) and surf down to a 176 x 176 u shaft
  whose floor is the finish trigger, ~14,860 u of descent over 7 stages separated by checkpoint
  curtains sgcp_1..7 (score.py reports the furthest one an eval entered as `max_cp`). The goal
  field reads 99,675 u at the spawn (config.start_d). Many of the moves are detours away from
  the goal; NOTES.md lists the walls.
* The record: krs, 115.99 s (115.92 s on our clock). A bar to read results against, never a
  constant.
* Physics as in run 5: 10 ms tick, `--maxvel 4000`, 300 s episodes. Keep them: the ladder was
  measured at this physics.

## Each wakeup

You run inside `/loop` (self-paced). Every wakeup:

1. Read `AR_BRANCH.json`. If `now >= wrapup_at`, go to **Wrap-up** and do nothing else.
2. Check your box and run: `python autoresearch/box.py status --agent $AR_AGENT`, then the
   run's progress through `box.py ssh` (tail of `runs/<run>/progress.csv`, the newest greedy
   eval, the newest search line in `runs/<run>_launch.txt`). Pull new evals and run the ladder
   in your units.
3. **Check the box's deadline** (`box.py status`). A box whose deadline falls before the run
   should end is extended NOW (`box.py extend`, at most 12 h each time). The watchdog destroys a
   box at its deadline whatever is running on it.
4. Act: analyse, code, launch, or stop a run that FAILED a wall.
5. Write down what you learned: append to `autoresearch/NOTES.md` under a dated heading (take
   the time from `date`, never type it). Keep `RESULT.json` current.
6. Schedule the next wakeup with ScheduleWakeup: 20-30 min while you build or while a run is in
   its first hour, 45-60 min while a run just trains. One waiter, no polling loops: the user
   pays for every wakeup.

Stopping a run does not end your round. Until `wrapup_at`, a failed recipe is written down and
followed by the next one. The orchestrator re-arms the loop of a slot whose RESULT.json says
final before `wrapup_at`.

## Time and money

* **One round of 26 h, and it is the only one.** The credit was $44.36 at the start, $34.36
  above the $10 reserve, which pays four boxes for about 26 h. There is no second round and no
  money for a fifth box.
* **What fits.** Run 5 first finished at 8.23B steps, 6.3 h on an RTX 5080 (~365k steps/s). The
  finish deadline (16.46B steps) is ~12.5 h on a 5080 and ~21 h on a 3090 (~215k steps/s). So the
  round fits one full from-scratch run per slot plus its build, NOT a failed run and then a
  second full run. If your run starts late or your box is slow, the round may end before the
  late deadlines: report the ladder as far as it got and say PENDING, never a guess.
* **Build fast, launch early.** Slot 1 launches in the first hour. Slots 2 and 3 should be
  training within about 4-6 h, slot 4 within about 8 h. A smaller first version that is
  running beats a better one that is not.
* **Boxes.** One box per agent, single GPU, consumer RTX cards under the `config.max_dph` caps;
  fast cards first (5080 / 4080 class, then 3090). `box.py` and the watchdog enforce it. The
  trainer is CPU-bound: read your box's steps/s after the first minutes and write it down. A box
  far under ~200k steps/s on run 5's recipe is blocklisted and replaced in the first minutes,
  not later.
* **Pacing.** `box.py` paces the fleet's burn at (credit - reserve) / hours left (+10%). If
  `rent` or `extend` refuses on budget, the fleet is at its limit: build or analyse, try later,
  never work around it. Release a box that has nothing running within 5 minutes.
* **Step cap.** Launch runs with `--steps 60e9` so that the cap never stops them. If anything
  in your recipe is annealed over the step budget, the cap is part of the recipe: say so.
* **Watching.** The user watches every session through Claude Remote Control and
  `$AR_HOME/status.md`. That replaces CLAUDE.md's per-box dashboard tunnel.

## Code changes

Work on YOUR branch (`ar/$AR_AGENT`, already checked out in this worktree):

1. Implement behind a flag, default off, with a unit test.
2. CPU-check on this machine before renting:
   `CUDA_VISIBLE_DEVICES=-1 nice -n 10 python -m pytest -q <your test> -p no:cacheprovider`.
   This machine has 4 vCPU, no GPU, and hosts the user's game server: run one CPU job at a time
   and keep it short. Four agents share it. Never run a trainer or a search here.
3. Commit with the trailers from CLAUDE.md / the session, then `git push origin ar/$AR_AGENT`.
   Boxes clone from GitHub, so a box can only run what is pushed.
4. Deploy THAT commit: `box.py deploy --agent $AR_AGENT --branch ar/$AR_AGENT`.

**The harness is the operator's.** The files in `config.harness_files` (CLAUDE.md, this
protocol, ladder.json, ladder.py, search_ticks.py, box.py, score.py, orchestrator.py,
config.json, the watchdog, deploy_box.sh, vast_pick.py, gpu_health.py and their tests) are not
yours to change. If one blocks you, work around it on your branch and write it in NOTES.md
under a "Harness issue" heading; the operator reads those. The trainer, the tools, the tests
and NOTES.md are yours.

## Boxes: one per agent, and box.py is the only door

```
python autoresearch/box.py budget                                   # credit, burn, pace
python autoresearch/box.py rent    --agent $AR_AGENT --hours 8      # the deadline the watchdog enforces (<= 12)
python autoresearch/box.py deploy  --agent $AR_AGENT --branch ar/$AR_AGENT [--ckpt <bench checkpoint>]
python autoresearch/box.py launch  --agent $AR_AGENT --run ${AR_AGENT}_<tag> --steps 60e9 -- <extra flags>
python autoresearch/box.py ssh     --agent $AR_AGENT -- "tail -3 /root/RL_Surf/runs/<run>/progress.csv"
python autoresearch/box.py pull    --agent $AR_AGENT --run <run> [--ckpt] [--snapshot <step>]
python autoresearch/box.py extend  --agent $AR_AGENT --hours 8
python autoresearch/box.py release --agent $AR_AGENT
```

* `launch` prepends run 5's recipe (`config.incumbent_recipe`) to your extra flags, and a later
  VALUE flag wins. `--no-recipe` gives the bare scratch preset; then pass the whole list
  yourself and write it in NOTES.md (slot 1 needs this). Run names start with `<agent>_`.
* Every launch goes through `tools/run_arm.sh` (that is what `launch` runs). The one exception
  is a slot 4 run whose driver is not `train_fast.py`: start it with `box.py ssh` under `nohup`,
  with its full command line saved to `runs/<run>_launch.txt` on the box and in NOTES.md, a pid
  file, and `tools/record_ckpt.py` run by you on its first checkpoint (the record gate's job).
* If `launch` returns an ssh error (exit 255, "Broken pipe"), the run usually started anyway:
  check `ps` on the box and `runs/<run>_launch.txt` before launching again.
* Never call `vastai create` / `destroy` yourself. The fleet watchdog destroys any instance
  that is not a single allowed RTX card under its price cap, is above the box cap, is past its
  deadline, or is unregistered.
* Your box id is in `$AR_HOME/boxes/$AR_AGENT.json` and `box.py status --agent $AR_AGENT`. Keep
  it in NOTES.md too.
* A bad box (gpu_health fails or says BUSY, a foreign tenant on the GPU, or the trainer far
  under ~200k steps/s on run 5's recipe) is blocklisted with `python tools/vast_pick.py --block
  <id> --reason ... --detail ...` BEFORE you release it (reasons: gpu_capped, cpu_bound, no_cuda,
  unreliable, network).
* Never unpickle a checkpoint on this machine. Read data pulled from a box with `python -I`.

## Harvest, score, record

* `box.py pull` brings run.json, progress.csv, the launch log and the evals. It does NOT bring
  the run's `auto/` directory. Before you release a box, also save `search_ticks.json` (above)
  and a bundle of the search summaries and logs (tar them on the box WITHOUT the per-search
  checkpoint copies, fetch the tar, check its md5). A released box is gone with everything on it.
* Pull with `--ckpt` the run you report, and `--snapshot <step>` at or before its best eval.
  `ckpt_latest.pt` is rewritten every ~60 s; if the md5 check fails, pull again.
* Score: `python autoresearch/score.py --best $AR_HOME/runs/$AR_AGENT/<run>/` = the best eval's
  max-episode progress (a finish counts 100) + 10 x the finish fraction + a time term once
  something finishes. Only greedy evals from the true start count. The score is a diagnostic
  here; **the ladder is the verdict**.
* `RESULT.json` (worktree root, read by the orchestrator, rewritten whenever something moves):

```json
{"agent": "azr1b2", "round": 1, "final": false, "slot": 2,
 "best": {"score": 27.5, "eval_file": "/home/steam/ar_az/runs/azr1b2/azr1b2_x/traj_0755499008.jsonl",
          "run": "azr1b2_x", "step": 755499008, "ckpt": "/home/steam/ar_az/runs/azr1b2/azr1b2_x/ckpt_latest.pt",
          "recipe": ["--auto-spine-mcts-args", "..."], "commit": "<sha>", "one_recipe": true,
          "from_scratch_step0": true, "resume_from": null},
 "ladder": {"unit": "steps", "first ramp": 302514176, "window (CP1)": null, "verdict": "alive",
            "total": {"first ramp": 421000000}},
 "summary": "<= 10 lines: what this slot changed in run 5, where the run stands on the ladder against run 5 in both units, what it means for the user's question, next"}
```

## Wrap-up (at `wrapup_at`, or when a message says ROUND OVER; never before)

1. Launch nothing new. Pull the best run with its checkpoint (`--ckpt`) and the snapshot at or
   before its best eval; save `search_ticks.json` and the search bundle.
2. Score it and run the ladder in both units. Write `RESULT.json` with `"final": true`.
3. Append the round's lesson to NOTES.md: what you changed, the ladder beside run 5's, what
   the result says about the user's question (one thread or two; what the tree should teach),
   what you would run next, and every caveat (one seed, a slower box, a late start).
4. `git add -A && git commit` (trailers) `&& git push origin ar/$AR_AGENT`.
5. `box.py release --agent $AR_AGENT`, and confirm with `box.py status --agent $AR_AGENT`.
6. Stop scheduling wakeups. The orchestrator ends this session at the deadline.

## Talking to the user

The user watches every session through Claude Remote Control and may write to you at any
time. Answer first, in a few plain sentences:

* what is running, and what it changes in run 5;
* its newest eval and its ladder row beside run 5's;
* what you will do next.

Plain words: the user asked for "less LLM language and more human language", and for
explanations built from the major components rather than a pile of numbers. ASCII only (the
user's terminal is cp1251). Label every line and point of any plot (the user is colour-blind).
Times come from `date`.
