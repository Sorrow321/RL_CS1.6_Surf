# `--int-mode edge` and the predecessor archive

**Status (user rule 0b, 2026-09-14):** edge novelty and the count-gated
archive (`--int-rare`) are GENERIC and stay candidates. The archive's speed
gate (`--int-rare-speed 1200`) and the dip-speed bonus family
(`--dip-speed-coef/-cap/-margin`, documented in the ledger) were tuned to
unitfarmer2's pit and are ANALYSIS instruments only - never recipe
components. UNITFARMER IS A BENCHMARK, NOT THE GOAL. See `CLAUDE.md` 0b.

Built 2026-09-13 from the cross-review (`docs/gpt-cross-review.md`, "New,
simple mechanisms" 1 and 2) for the unitfarmer2 exploration benchmark
(`docs/uf2-exploration-review.md`). Both are default-off; with the flags
absent the trainer's config dump and behaviour are byte-identical.

## Why

The count-based novelty that produced the first champion-free pit entries
(`--int-coef 2.5`, batch 10) keys its counts on (256 u cell, 8 yaw sectors,
3 speed buckets) and pays every env entering a key from the same
pre-increment count. Turning the camera or crossing a speed bin inside one
cell is therefore paid as new, which is exactly the crawling-in-the-pit the
campers did, and the temperature's count decay re-novelised exhausted cells
every period. And once an entrant found the pit, nothing kept the run-up to
that decision in the spawn distribution: the T-conditioned family's
explorers entered while its T = 0 member never did.

## 1. Directed-transition novelty: `--int-mode edge`

* Counts are kept on **directed edges of the position-cell graph**
  (previous cell -> cell, no yaw, no speed), hashed into a `2**--int-edge-bits`
  table (default 22; collisions only merge two edges' counts).
* Paid only when the position cell changes; north and south exits from the
  platform cell are different edges; A -> B and B -> A are different edges.
* **Batch-safe**: envs entering the same edge on one tick are ranked, the
  k-th is paid as the k-th visit (`c / sqrt(count + rank + 1)`), and the
  table counts each of them.
* **Never decayed**: `decay_counts` (the `--unstuck` count decay) leaves the
  edge table alone - an exhausted loop stays exhausted.
* `--int-rare K`: an entry whose count (rank included) is below K raises
  `RaceReward.rare_entry[env]` for that tick - the signal the archive gates
  on. Works in cell mode too.
* Under `--curiosity-cond` the bonus still scales with the member's T.

## 2. The survivor-gated predecessor archive: `--archive-frac`

`python/surfgym/archive.py`, `PredecessorArchive`.

* Every env keeps a ring of its own recent states (`--archive-window` seconds,
  default 3, one snapshot per decision).
* When an env takes a **rare transition** (`rare_entry`), its ring - the
  run-up to the decision - is frozen as a pending prefix.
* If the env is still alive `--archive-hold` seconds later (default 3), or
  ends by timeout/finish, the prefix is **committed** to a FIFO archive
  (`--archive-cap` rows, default 20,000). A death inside the hold discards it:
  the archive holds the run-ups to survivable discoveries only.
* Each iteration, after the spawn source (reservoir, own window, frontier)
  has built the pool, a `--archive-frac` share of the pool's rows is replaced
  by uniformly drawn archive rows. The map-start share shrinks by the same
  fraction; nothing else changes.
* Logged as `archive/rows`, `archive/commits`, `archive/rare`,
  `archive/discards`; the step line prints `arch <rows> +<commits> (rare,
  lost, pending)`. Checkpointed (`ck["archive"]`), restored on resume.
* Needs `--int-coef > 0` and `--int-rare > 0`; single-map; refused with
  `--respawn-random`.

This is the manual "find the gate, cut a window of the policy's own states
before it, spawn from it" step of the method made automatic, with no map
boxes and no demo: the gate is wherever a rare transition was survived, and
the window is the states before it.

## Pinned

`tests/python/test_edge_novelty.py` (8 tests): edge mode ignores view and
speed changes in place while cell mode pays for them; edges are directed;
same-tick entries are ranked and counted; `--int-rare` flags the first K;
the edge table is never decayed; the archive commits on survival, discards
on death, commits on timeout, replaces exactly the requested pool share,
round-trips its state; a CPU trainer smoke logs the columns and checkpoints
the archive.

## First arms (batch 14, unitfarmer2, champion-free)

`uf2EDGE` (edge novelty 4x + archive 20%), `uf2EDGE0` (edge novelty alone),
`uf2EDGE10` (edge novelty 10x + archive), each 1B, ratchet + keys T with the
true-start alive reach; verdict = the start-line bench with the pit-speed
rung.
