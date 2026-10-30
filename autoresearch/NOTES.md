# NOTES - the surf_unitfarmer2 lineage

The lineage's memory: every round's winner appends what it learned (what worked, what did not,
the numbers). Read it before you start; append to it before you wrap up.

## Round 0 (operator, 2026-10-30)

* A new lineage on a new map. The two earlier fork-and-flush programs on this harness -
  surf_src_sidistic (2026-10-20..23) and surf_princessburglar_b26 (2026-10-23..30) - left their
  code and tools in this branch's history. Their notes are NOT copied here; they are in this
  branch's own git history (`git show 8a825803:autoresearch/NOTES.md`: the first
  princessburglar lineage, rounds 1-9, and the operator's expert loop after it). The second
  princessburglar lineage (pbf*, the corrected core) is not in this history; what transferred
  from it is summarised in PROTOCOL.md **This lineage**, with its exact flags.
* unitfarmer2's own history (2026-09..10-17, before this harness existed: the curriculum, novelty,
  frontier-loop and search arms) is in `docs/research-results.md` (grep `uf2` / `unitfarmer2`).
  Read it before you plan: it lists what failed, why, and which checkpoints are demo-trained
  (CLAUDE.md section 0: never resume, benchmark or build on those).
* The map files came from the user on 2026-10-30 (the BSP, md5 9834c24c; its WR demo, analysis
  only). The operator baked the caches on one box (smoke launch uf2op_bake, 40M steps from
  scratch at 227k steps/s on an RTX 4080). That launch passed the record gate (greedy, stoch,
  mixed, POV), and its 0.5M-step eval scored 2.831 (best progress 2.8%, 0/9).
* The WR demo, parsed, is in `$AR_HOME/wr/`: `surf_unitfarmer2_wr.jsonl` at 10 ms, the raw
  frames, and the movevars. It is for ANALYSIS ONLY (CLAUDE.md section 0). It never becomes a
  spawn file, a spine, a target sequence, a BC set or a reward constant.

## Round log

## uf2r1b2 (round 1, slot 2) - 2026-10-30 21:40 UTC start

Box 58455373 (RTX 4080 SUPER, EPYC 7K62, 96 threads visible but a 30.7-CPU cgroup quota),
0.41 $/h, rented 21:44 UTC.

### The start, measured (22:00 UTC; the record as a YARDSTICK only)
* The record on config.field (d0 30,647.5): 1.7% at the platform's south edge (t 2.5 s from the
  spawn), down to **-7.3%** at the pit bottom (t 5.5 s, z ~-800, |v| 1,800, E ~1,200), back to
  1.3% at t 8.0, 3.7% at t 9.0, 5.3% at t 9.5 (z 342, 1,123 u/s), 7.5% at t 10.5 (-2969, -1306,
  483). E = z + |v|^2/1600 stays 1,050-1,360 through the pit.
* **The floor under the start is a teleport**: trigger_teleport *18 -> recomeco is a sheet at
  z 160-161 over x -3456..-896, y -2400..-448. Everything that leaves the platform north or east
  without the pit's speed falls onto it (the September "north slide" ends there too). The
  record crosses that region at z ~340-580. The field cannot see it (it lets the player float
  over the sheet), so the field's arrows point down the start ramp, into the sheet.
* trigger_teleport *7 (bbox x -2880..-1472, y -3936..192, z -1024..577) is the pit's catch; the
  pit floor kills at z ~-990.

### Control: the incumbent recipe from scratch (uf2r1b2_ctl, 4e8 steps, evals every 50M)
* Evals 2.85 (0.5M) / 5.26 (50M) / 5.66 (101M) / **5.69 (151M)**. At 151M all 9 greedy episodes
  are one line: down the start ramp at ~1,130 u/s (Emax 1,340: it strafes), into sheet *18 at
  t 4.4 s at (-1600, -2022, 181), 5.7%. At 50M half of them still died on a ledge at z ~540-590
  (2.7-3.2%). Never the pit.

### Go-Explore from the true spawn (pbf flags, 10 ms) never leaves the pit (uf2r1b2_ge1)
* `explore_phase1 --exact-chains --cell 64 --cell-energy 32 --cell-heading 1
  --view-follow=-90,-60,-30,-15,0,15,30,60,90 --select-energy 3 --act-every 4 --decisions 100
  --maxvel 3500 --envs 4096`, 15 min (0.7M ticks/s: the archive bookkeeping is single-thread
  Python, ~11 cores busy of 30): 631k cells, 10.3M runs, no goal. 89% of the archive is in the
  pit (progress < -1%).
* Deepest cell 5.81% (and 28,002 u on the corner-cutting field at the end): a slow walk east of
  the platform -> a ledge at z 548 -> a floor at z ~300 -> north at 40-320 u/s, falling into
  sheet *18 at (-1919, -1847). The north-slide family again, at E ~350.
* Chains that dove below -5% and came back above -3%: max E **26**. The deep pit cells (progress
  -12..-5%) are at z ~-990 with |v| 1,370-1,500, E 170-410: ~800 u of energy height below the
  record's ~1,200 in the bowl. Random search is energy-bound in the pit, as in October.
* WHY (core physics, tests/python/test_ge_nobrake.py): under --view-follow the held keys keep
  one wish angle beta to the velocity for the whole hold (0.8 s on average at 95% repeat).
  GoldSrc's air acceleration adds min(250, 30 - v.wishdir) u/s per tick along the wish: beta <=
  ~88 deg does nothing, beta = 90 is the strafe (a gain), beta > 90 is a brake of up to 250 u/s
  PER TICK. Uniform keys x the offset list wish backward about half the time.
* New: `explore_phase1 --view-follow-nobrake` (commit 19ae2856): such a draw presses the
  opposite keys (beta -> beta - 180); no extra random draw, so off is byte-identical (tested
  against HEAD's explorer). Physics only: no map constant.

### Probes (22:05-22:15 UTC)
* Control evals 5.62 / 5.65 / 5.59 at 202-302M: plateaued on the same line (sheet *18 at
  4.5 s, 5.4-5.7%). 240k steps/s on the 4080S once the explorer stopped competing.
* `--view-follow-nobrake` GE (uf2r1b2_ge2, 20 min): NOT better in the pit. Chains that dove below
  -5% come back with E <= -116; the deep cells sit on the pit floor (z ~-990, E ~0). So the
  energy is lost to impacts in the bowl, not to held brake keys. Its deepest cells (8.1%) are
  slow north-slide variants that walked west for 25-30 s at z ~180-220 and fall onto *18.
* `pb_r26` (princessburglar 9/9, never saw this map; record_ckpt from the true start, 3 greedy
  + 9 stochastic): no dive. All 12 episodes drop south-west onto a ledge at z ~510-590 near
  (-2350, -2600) and die there (2.1-3.2%, 6.5-15 s) - the spot the control's 50M eval died on,
  right under the record's pit-exit corridor (the record passes (-2488, -2603) at z 584; the
  pit's catch *7 reaches z 577). A warm surfer is no shortcut through the start.

### Arm uf2r1b2_egt (launched 22:20 UTC): per-call energy shaping + Go-Explore energy-top spawns
* Why: (1) the recipe's energy term (`--energy-pot 3 --energy-pot-front 0.01
  --energy-pot-front-bonus 1`) is paid once at the frontier band, so nothing charges the north
  slide's energy loss (~-505 u of z + |v|^2/1600 at the sheet) or pays energy kept in the pit.
  Per call at K 12 (pb_r26's own setting, carried unchanged from princessburglar) the slide nets
  ~+5.5 - 6.1 = -0.6, the record's pit loop ~+5.6 by t 10; a crash in the bowl costs ~-11. (2) The
  policy needs practice in the pit with the energy to leave it: GE's RETURN draw is 87% pit but
  mostly doomed floor states (median E < 0); the explorer's own energy-top set (the most
  energetic cell of each 64 u place, the set --select-energy boosts) is 19,228 rows, 25.6% of them
  pit states with E > 1,000 (the -5..-2% band: median 1,655 u/s at z -719, E 1,032).
* New tool: `tools/ge_return_states.py` (`--tops`; tests/python/test_ge_return_states.py).
* Flags: the incumbent recipe + `--energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0
  --spawn-states runs/uf2r1b2_ge1/tops.npy --spawn-states-frac 0.3`, 4e8 steps from scratch,
  evals every 50M. Self-states: the branch's own GE archive rooted at the true spawn (md5
  4d3f8d2d). Matched-step reference: uf2r1b2_ctl (5.26 / 5.66 / 5.69 / 5.62 / 5.65 / 5.59 / 5.66
  at 50-352M, all on sheet *18).
* Diagnostic to watch in training: `dip/max_survived_depth` (the deepest setback an episode
  climbed back out of, % of the map): the record's pit is a ~9% dip; the control sat at ~0.7
  after its first 100M.
* Ops: run_arm.sh SCRATCH pins `--ckpt-every 1e9` (no milestone snapshots in a 4e8 run): pass
  `--ckpt-every 5e7` from the next launch on, so the hand-off is next to the best eval.
