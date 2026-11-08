# autoresearch: fork-and-flush on surf_src_sidistic - ONE recipe

An unattended research loop on the agent host (the droplet, no GPU). It runs the method of
Cai et al. 2026, "Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents"
(arXiv 2610.07447), toward a goal that differs from the earlier programs': **ONE training
recipe, frozen at launch, that clears surf_src_sidistic's walls from scratch with no per-wall
tuning** (user, 2026-11-08). Each wall must fall no later than twice the steps the October
lineage needed (autoresearch/ladder.json, autoresearch/ladder.py). The question is the explore
step of a Go-Explore -> backward-curriculum loop: the Nature paper's random bursts, a
policy-guided beam, or policy-prior MCTS, and where on one continuous knob between random and
policy-guided it should sit; in round 1 each slot builds one of them (PROTOCOL.md "Your slot").
This is the harness's fifth program. sidistic itself was first finished by the first program
(2026-10-20..23, ~/ar), one wall at a time; then princessburglar (~/ar_pb), unitfarmer2
(~/ar_uf2) and za_loop (~/ar_loop). State lives in ~/ar_uni, the checkout is ~/RL_Surf_uni
(branch uni/main), the tmux session is `ar_uni`, and agents and branches are named
unr<round>b<k> / ar/unr<round>b<k>. The operator runs a control on its own box (`unop_ctl`, the
recipe with no search), mirrored to ~/ar_uni/control/.

## How it works

* **Rounds.** Each round forks **4 branch agents** from one workspace: a git commit plus the
  protected incumbent checkpoint and eval. Every branch is a fresh Claude Code session
  ("flush") in its own slot (`$AR_HOME/wt/b1..b4`) on its own branch
  (`ar/<agent_prefix>r<round>b<k>`; config agent_prefix).
* **Schedule.** Rounds last **12 h** (user, 2026-10-20: 24 h is too long for one
  lineage), **2** of them (what the credit buys at four boxes; a top-up and a larger `rounds`
  extend it). The schedule is fixed, as the paper advises, with K = 4. Change `rounds` / `round_hours` in `config.json` to extend, and move
  `budget_end_hours` to the new end with them.
* **Selection.** At a round's end every branch is scored by `autoresearch/score.py`. A
  branch's score is its best greedy eval from the true start. The winner's branch head seeds
  the next round. The losers are archived (`~/ar_uni/archive/r<n>/`) and pruned.
* **Inside a round.** Each agent follows `autoresearch/PROTOCOL.md`:
  * it rents at most one GPU box through `autoresearch/box.py`;
  * it runs ablations of at most 30 min first and up to 6 h when promising;
  * it pushes its code to its branch so boxes can clone it;
  * it harvests to `~/ar_uni/runs/<agent>/`;
  * it writes `RESULT.json` and appends to `autoresearch/NOTES.md`.
* **The lead.** A fifth session, `ar-lead`, has no box. It watches everything and answers
  your questions.

## Hard limits (config.json, enforced in code)

* **Cards.** RTX 3090 / 3090 Ti / 4080 / 5080 and, since 2026-10-25, the cheaper 3080 / 4070
  / 5070 / 60 Ti class cards (>= 12 GB) first, cheapest of those; then 4090; then 5090.
* **Price caps** ($/h for the whole box, its disk included; the cheaper cards' caps are below
  the good cards', user 2026-10-25):

  | Card | Max $/h |
  |---|---|
  | 3090, 3090 Ti, 4080, 4080 Super, 5080 | 0.65 |
  | 4070 Ti Super, 5070 Ti | 0.30 |
  | 3080, 3080 Ti, 4070, 4070 Super, 4070 Ti, 5070 | 0.25 |
  | 4060 Ti 16 GB, 5060 Ti 16 GB | 0.20 |
  | 4090 | 0.45 |
  | 5090 | 0.60 |

* **Boxes.** Single GPU only, at most 4 live boxes, one per agent.
* **Enforcement.** `box.py` refuses any rental outside these limits. The fleet watchdog
  destroys any instance on the account that breaks them, whoever rented it. It needs two
  consecutive sightings before acting.
* **Budget.** No new box once the credit falls to the $10 reserve. This program has no flat
  rate (config.budget_rate_usd_h is absent): the fleet's burn (live boxes plus the new one) may
  not exceed (credit - reserve) / hours left until budget_end_hours, +10%, ~1.8 $/h at the
  start ($55.39 of credit on 2026-11-08): four cheap branch boxes plus the operator's control.
  `box.py extend` is refused when the fleet's burn over the extension would come out of the
  reserve.
* **Deadlines.** Every box has a watchdog deadline (`box.py rent --hours`, `box.py extend`).
* **Readiness.** A box must be running within 5.5 min of creation and reachable by ssh within
  90 s after that; otherwise it is blocklisted and destroyed.

## Watching it

* **Claude app or claude.ai/code -> Code.** Every session connects to Remote Control at
  start-up: `ar-lead`, and `ar-r<round>b<k>` for the branches. Open one and write to it.
  Ask `ar-lead` for the overall picture.
* `ssh steam@<host>`, then `tmux attach -t ar_uni`. Ctrl-b w lists the windows (orch, b1..b4;
  the fleet watchdog daemon is the first program's, shared); Ctrl-b d detaches.
* `cat ~/ar_uni/status.md`, which is refreshed every 5 min. `tail ~/ar_uni/orchestrator.log`.
* `python autoresearch/box.py status` shows the boxes, their cost and the budget.

## Controlling it

| Command | Effect |
|---|---|
| `python autoresearch/orchestrator.py wrapup` | end the current round early, gracefully |
| `python autoresearch/orchestrator.py stop` | release every box, end every branch session, stop |
| `rm ~/ar_uni/STOP && bash ~/ar_uni/start_orch.sh` (in tmux `ar_uni`, window orch) | resume after a stop (state.json keeps the round) |

## One-time setup on the agent host (done by the setup session, 2026-10-20)

* **Python.**
  * The venv `~/ar/venv` holds numpy, CPU torch, scipy, numba, opencv-headless, pytest and
    vastai.
  * The C core is built (`build.sh`).
* **Maps.** `maps/<config map stem>.*` (bsp + baked caches + zones.json) is copied into the
  main checkout, and `$AR_HOME/maps/<config maps_tarball>` holds the copy that boxes receive.
* **Incumbent.** None in round 1: every run starts from scratch (PROTOCOL.md, the one-recipe
  rule). From round 2 on, `~/ar_uni/incumbent/ckpt.pt` is the last winner's, to be CONTINUED
  with its exact recipe or ignored.
* **Claude Code** (`~/.claude/settings.json`): `permissions.defaultMode: "auto"` and
  `remoteControlAtStartup: true`. Log in once with `claude auth login` (a claude.ai account;
  Remote Control does not work with an API key). The five working directories (main checkout,
  b1..b4) are trusted once.
* **vast.** The API key is in `~/.config/vastai/vast_api_key`. This machine's ssh key is on
  the vast account.
* **One daemon per account.** Run the fleet watchdog daemon only here while this runs.
  Another daemon with a different registry destroys this fleet's boxes.
