# `--respawn-rare F`: Go-Explore's return over the reservoir, per cell

Autoresearch branch r2b3 (2026-10-20). Default OFF; with the flag off the trainer is
bit-identical to the trainer before it (config, `progress.csv`, eval trajectory bytes and
weights of the CPU smoke; `tests/python/test_respawn_rare.py`).

## Why

The reservoir draw that shipped is uniform over its states, and the reservoir stores states in
proportion to the time real episodes spend in a place, so almost every return lands on the
beaten line. Go-Explore (Ecoffet et al. 2021) returns to a CELL chosen with weight
`1 / sqrt(1 + N)`, N the cell's visit count, so exploration restarts where the policy has
rarely been. On sidistic the field routes through a window the agent cannot fly to from the
CP2 slot (r2b3 notes, 2026-10-20), and the human leaves on a detour the field charges for: the
way on starts somewhere the policy's own episodes rarely go, and the uniform draw almost
never returns there.

## What it does (`surfgym.respawn.RareCellReturn`, wired in `train_fast.py`)

* **Counts.** Once per decision every REAL env (map start or reservoir spawn) that entered a
  new `--respawn-rare-cell` (256 u) position cell adds 1 to that cell's N. An env flying a
  `--respawn-novel` episode is not counted: "rare" means where the policy's own episodes
  rarely go, not where a novel row put it. A respawn that lands in a new cell counts too, so a
  cell that is returned to often stops being rare (Go-Explore's "times chosen").
* **The draw.** A share F of the reservoir draw goes to cells by `1 / sqrt(1 + N)`, split
  evenly over the reservoir states standing in each cell (Go-Explore selects a cell; a
  per-state weight would let the most visited cells draw the most spawns - the
  `--plan-return 1` density bias, ledger 2026-09-26). The rest of the draw stays uniform over
  states, so the main line keeps its practice. The map-start share and the novel rows are
  untouched (novel rows are cut from the reservoir rows after the draw, as before).
* **Nothing is moved and nothing is invented.** Every spawn is a state a real episode reached
  and survived `--respawn-margin` past, exactly as without the flag; only which of them is
  drawn changes. The ceiling of `--respawn-novel` and the harvest rules are unchanged.
* One cell size and one share, the same on every map; no map quantity.

## `--respawn-rare-arch 1`: Go-Explore's archive (no pre-death margin)

The reservoir only holds states harvested `--respawn-margin` (2 s) before an episode's end, so
where every real episode dies within 2 s of arriving - sidistic's shaft past the CP2 slot - it
holds nothing, and no return can start there. With the archive on:

* every REAL env that flies into a cell (not a respawn: a jump over 512 u between two
  decisions is not flight) writes its state into that cell's ring (4 states, oldest
  overwritten), with no margin;
* the share F of the spawn rows that are neither map starts nor novel rows is drawn from the
  archive: a cell by `1 / sqrt(1 + N)` - half of those draws among the deepest quarter of the
  archived cells by field distance (the frontier, as the novel sampler's deep shell), the rest
  among all archived cells - then one state of its ring, speed scaled like a reservoir row. The
  reservoir draw itself stays uniform;
* a return lands in its cell, so N grows with every return: a cell that leads nowhere (a doomed
  state) stops being drawn, which is what replaces the margin.

A simulator check of the idea (analysis only, `autoresearch/analysis/sid_goexplore_probe.py`):
from the policy's own greedy states in the CP2 slot, an archive of (128 u cell x heading x speed
x climb) with `1 / sqrt(1 + chosen)` returns and 1 s of random held actions found the cavity
behind the slot's end wall - the first piece of the human's way on, which no uniform-draw
training episode had ever reached - after ~10,000 returns (~0.6M ticks).

## `--respawn-rare-own 1`: archive returns do not use up the novel rows

With `--respawn-novel`, a novel row may only be placed in a cell no REAL episode entered
recently ("where the policy does not go on its own"). An archive return IS real, so once the
archive works a frontier region (sidistic: the cavity and the space under the shaft past the
CP2 slot) every cell there counts as visited and the novel rows - the mechanism that taught the
incumbent policy the segment after that region - are pushed out of it. With this flag an episode
that started from an archive row (tagged by spawn origin, the novel sampler's own rule) no longer
marks the novel visit grid; it still lowers the novel ceiling (it is real).

## Read-out

* Step line: `rare eff E/C N n` - the draw spread over E effective cells of the C cells the
  reservoir stands in, and n the draw-weighted mean entry count of the drawn cells.
* Every 100 iterations: the cells counted so far and the entries.
* `progress.csv`, last and only when on: `rare/cells`, `rare/res_cells`, `rare/eff`,
  `rare/n_draw`.
* Checkpoint: `respawn_rare` holds the counts per map; a bare resume restores the flag, the
  cell size and the counts. `record_ckpt.py` lists both keys as TRAIN_ONLY.

Refused without the reservoir (`--respawn-frac 0`) and with `--respawn-binned`,
`--respawn-mode` other than uniform, `--plan-return`, `--respawn-random`, `--goals` and DDP.

## On a resume

A bare resume restores the share, the cell size, `-arch`, `-speed`, `-own` and the counts (and the
archive). `--respawn-rare 0` on a resume turns the whole mechanism off: the sub-flags are restored
only while the share is > 0.

## Flags

| flag | default | meaning |
|---|---|---|
| `--respawn-rare` | 0 (off) | share F of the reservoir draw that goes by cell rarity |
| `--respawn-rare-cell` | 256 | the position cell, u |
| `--respawn-rare-arch` | 0 | 1 = draw the share from the margin-free per-cell archive |
| `--respawn-rare-speed` | 0 | archive cells keyed by B horizontal-speed buckets too (the --int-speed binning) |
| `--respawn-rare-own` | 0 | 1 = archive returns do not mark the novel visit grid |
