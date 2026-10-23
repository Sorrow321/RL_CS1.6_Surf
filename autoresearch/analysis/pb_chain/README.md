# Go-Explore spines on surf_princessburglar_b26 (pbr1b2, round 1)

STATE_DTYPE rows for `train_fast.py --demo-file ... --demo-reach R` (a Salimans-Chen backward
curriculum over a partial spine; launch through `box.py launch --self-states "<provenance>"`).

Provenance: Go-Explore phase 1 from the map start (`tools/explore_phase1.py --view-velocity 1
--dump-paths 1 --maxvel 3500 --cell 64 --cell-speed 8`, 8 min, 4096 envs, seed 0): random
held-key bursts in the simulator - no demo, no policy, no reward, no route. The dense chain to
the archive's deepest cell, every state kept once (`tools/archive_chain.py`).

* `chain_dedup_full.npy` - 714 rows, start -> 3.84% (the deepest archived cell, south of the
  pocket under the platform). Rows around the pocket move (old raw rows 594-626) are a move the
  policy has not repeated (its band sat at 0% success).
* `chain_dedup_378.npy` - 333 rows, start -> NW/SW-corner climb -> south ledge (1.44%). The
  spine of pbr1b2_ge13 (score 3.234: greedy from the true start into the pocket, 3.2%).
  The climb's run-up is rows ~125-156 (the band that took longest).

Resuming a checkpoint trained on these (e.g. pbr1b2_ge13): its config carries
`demo_file = runs/pbr1b2_ge3/chain_dedup_378.npy`, which a fresh box does not have. Either copy
this file there, or pass `--demo-file ''` (an empty string switches the curriculum off on a
resume) together with the recipe's spawn features if you want them back.

## Resuming the round-1 handoff checkpoints (both tied at 3.235)

* pbr1b2_ge14 / ckpt_853016576.pt (= pbr1b2_ge13's final weights): its config restores
  `demo_file = runs/pbr1b2_ge3/chain_dedup_378.npy` (and demo_window/rate/min_ep/grow).
* pbr1b2_ge16 / ckpt_1260388352.pt: its config restores
  `spawn_states = runs/pbr1b2_ge3/spine378_spawn.npy` (frac 0.15) and `demo_file = ''`.

Both paths are byte-identical copies of `chain_dedup_378.npy` here. On a fresh box either

    mkdir -p runs/pbr1b2_ge3
    cp autoresearch/analysis/pb_chain/chain_dedup_378.npy runs/pbr1b2_ge3/chain_dedup_378.npy
    cp autoresearch/analysis/pb_chain/chain_dedup_378.npy runs/pbr1b2_ge3/spine378_spawn.npy

or switch them off on the resume: `--demo-file '' --demo-front-frac 0 --demo-grow 0` and/or
`--spawn-states ''`. Both checkpoints carry --time-pen 0 and --ent 0.002 in their config.
