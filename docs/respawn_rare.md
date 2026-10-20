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

## Flags

| flag | default | meaning |
|---|---|---|
| `--respawn-rare` | 0 (off) | share F of the reservoir draw that goes by cell rarity |
| `--respawn-rare-cell` | 256 | the position cell, u |
