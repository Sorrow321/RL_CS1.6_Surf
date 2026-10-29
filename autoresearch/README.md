# autoresearch: fork-and-flush on surf_princessburglar_b26

An unattended research loop on the agent host (the droplet, no GPU). It runs the method of
Cai et al. 2026, "Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents"
(arXiv 2610.07447), toward one goal: **pass surf_princessburglar_b26 from the true start**
(the start_run teleport's destination, declared in the map's zones file), champion-free and
generic (CLAUDE.md sections 0 and 0b), then reliably, then faster than 75.18 s, the human world
record (a bar to read results against, never a reward constant). The second program of this
harness: the first, on surf_src_sidistic (2026-10-20..23, 7 rounds, ~/ar), finished that map
from the true start; this one (user, 2026-10-23) asks whether its recipe converges quickly on
the map the surf community calls the hardest. State lives in ~/ar_pb, agents and branches are
named pbr<round>b<k> / ar/pbr<round>b<k>. **Since 2026-10-29 a second lineage runs on the
corrected core** (6bd7f931: every finish of the first one climbed through a doorway the old
core made 3 u too low). Its agents are pbfr<round>b<k>, and it has its own state.json; the
first lineage's state is `~/ar_pb/state_pbr_oldcore.json`. PROTOCOL.md "This lineage" says why
and what it starts from.

## How it works

* **Rounds.** Each round forks **4 branch agents** from one workspace: a git commit plus the
  protected incumbent checkpoint and eval. Every branch is a fresh Claude Code session
  ("flush") in its own slot (`$AR_HOME/wt/b1..b4`) on its own branch
  (`ar/<agent_prefix>r<round>b<k>`; config agent_prefix).
* **Schedule.** Rounds last **12 h** (user, 2026-10-20: 24 h is too long for one
  lineage). There were **6** (72 h); the user added a 7th and an 8th on 2026-10-23, so 8
  rounds, ~98 h in all. The schedule is fixed, as the paper advises, with K = 4. Change
  `rounds` / `round_hours` in `config.json` to extend, and move `budget_end_hours` to the new
  end with them.
* **Selection.** At a round's end every branch is scored by `autoresearch/score.py`. A
  branch's score is its best greedy eval from the true start. The winner's branch head seeds
  the next round. The losers are archived (`~/ar/archive/r<n>/`) and pruned.
* **Inside a round.** Each agent follows `autoresearch/PROTOCOL.md`:
  * it rents at most one GPU box through `autoresearch/box.py`;
  * it runs ablations of at most 30 min first and up to 6 h when promising;
  * it pushes its code to its branch so boxes can clone it;
  * it harvests to `~/ar/runs/<agent>/`;
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
* **Budget.** No new box once the credit falls to the $10 reserve. The fleet's burn (live
  boxes plus the new one) may not exceed config.budget_rate_usd_h, $1.2/h since 2026-10-25
  (the user); without that key, (credit - reserve) / hours left until budget_end_hours.
  `box.py extend` is refused when the fleet's burn over the extension would come out of the
  reserve.
* **Deadlines.** Every box has a watchdog deadline (`box.py rent --hours`, `box.py extend`).
* **Readiness.** A box must be running within 5.5 min of creation and reachable by ssh within
  90 s after that; otherwise it is blocklisted and destroyed.

## Watching it

* **Claude app or claude.ai/code -> Code.** Every session connects to Remote Control at
  start-up: `ar-lead`, and `ar-r<round>b<k>` for the branches. Open one and write to it.
  Ask `ar-lead` for the overall picture.
* `ssh steam@<host>`, then `tmux attach -t ar`. Ctrl-b w lists the windows (watchdog, orch,
  lead, b1..b4); Ctrl-b d detaches.
* `cat ~/ar/status.md`, which is refreshed every 5 min. `tail ~/ar/orchestrator.log`.
* `python autoresearch/box.py status` shows the boxes, their cost and the budget.

## Controlling it

| Command | Effect |
|---|---|
| `python autoresearch/orchestrator.py wrapup` | end the current round early, gracefully |
| `python autoresearch/orchestrator.py stop` | release every box, end every branch session, stop |
| `rm ~/ar/STOP && bash autoresearch/start.sh` | resume after a stop (state.json keeps the round) |

## One-time setup on the agent host (done by the setup session, 2026-10-20)

* **Python.**
  * The venv `~/ar/venv` holds numpy, CPU torch, scipy, numba, opencv-headless, pytest and
    vastai.
  * The C core is built (`build.sh`).
* **Maps.** `maps/<config map stem>.*` (bsp + baked caches + zones.json) is copied into the
  main checkout, and `$AR_HOME/maps/<config maps_tarball>` holds the copy that boxes receive.
* **Incumbent.** `~/ar/incumbent/` is round 0: sidEM2cap2surfGR at 749.7M, plus the surf
  run's progress.csv as a reference.
* **Claude Code** (`~/.claude/settings.json`): `permissions.defaultMode: "auto"` and
  `remoteControlAtStartup: true`. Log in once with `claude auth login` (a claude.ai account;
  Remote Control does not work with an API key). The five working directories (main checkout,
  b1..b4) are trusted once.
* **vast.** The API key is in `~/.config/vastai/vast_api_key`. This machine's ssh key is on
  the vast account.
* **One daemon per account.** Run the fleet watchdog daemon only here while this runs.
  Another daemon with a different registry destroys this fleet's boxes.
