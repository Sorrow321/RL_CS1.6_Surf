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

## Round 1, branch uf2r1b4 (2026-10-30 21:57 UTC, first wakeup)

* **The bar on the grader's field (analysis of the WR line, ruler only).** On goalc_32 the
  September RL "north dead end" (-1340, -990, 183), where the plain recipe's greedy line died at
  6.2 s, reads d 27,807 = **9.27 %**. The record's own line only passes 9.27 % about 11.3 s after
  the demo start (1.76 % best until 8.5 s, 6.2 % at 10 s, 8.4 % at 11 s, 12.0 % at 12 s). So a
  scratch policy that takes the dead end scores ~9.3, and a pit line only beats it once it is
  past the pit exit AND the next ramps. The pit itself is all NEGATIVE progress (d up to 32.8k).
* **Kill geometry** (BSP trigger_teleport bboxes, every one fatal in our core): recomeco over the
  pit (x -2880..-1472, y -3936..192, z -1024..577; a union of brushes), a slab at z 160-161
  (x -3456..-896, y -2400..-448) just under the dead end, a slab at z -768 to the north-west,
  and seven part2 catches further on.
* **Local GE probe, the pbf explorer flags** (`explore_phase1 --exact-chains --cell 64
  --cell-energy 32 --cell-heading 1 --view-follow=-90..90 --select-energy 3 --act-every 4
  --decisions 100 --maxvel 3500`, 10 ms, 1024 envs, 2 threads, 6 min): 395k cells, 377k of them
  in the pit; best 5.3 % (east of the start at z ~190, falling); 0 pit exits. Same-motion
  energy against the record's positions (ruler): 1,206 vs 1,294 at the record's 4.0 s, then
  628 vs 1,208 at 4.5 s and nothing usable after. It repeats October's go_cem verdict: random
  bursts lose the bowl's energy, so a GE line through the pit is not on offer from this
  explorer. Script: arch_probe.py (scratchpad; reads an archive against goalc_32 and the ruler).
* **First arm (pending a box): uf2r1b4_pbw1**, a warm start from refs/pb_r26 (never saw this map;
  it learned to pump princessburglar's start half-pipe, the closest generic skill to this
  bowl). Checkpoint `runs/uf2r1b4/_init/pb26_uf2_init.pt` (md5 e2b2315f): pb_r26's weights,
  optimizer and config with map -> surf_unitfarmer2 and its reservoir, novelty counts, unstuck
  state, demo spine and BC set dropped (all princessburglar run state). Resume flags: the
  recipe's training flags minus the shape flags, `--lr 1e-4`, and the checkpoint's own energy
  shaping (per-call, K = 12, restored) instead of the recipe's front bonus; tick 7.64 ms as the
  checkpoint carries (the record's ~131 fps; +43 % air-strafe energy per second over 10 ms).
  What it measures: does a skilled surfer find the pit line where scratch arms never did.
* Box: none. `box.py rent` refused on the fleet's budget pace (burn 0.79 of 0.82 $/h, two boxes
  live, cheapest offer 0.467 $/h) from 21:46 UTC; a retrier (scratchpad auto_box.sh) rents
  through box.py every 2 min and then deploys and launches.
* **The "north dead end" is not a topological dead end (geometry, 22:05 UTC; occ_32 + goalc_32
  slices, scratchpad).** The pit column (x -2832..-1560) opens at its NORTH end (y -1320..-1064,
  z ~200-700) both WEST and EAST into two side regions that are mirror images about x ~ -2176
  (the spawn's x). Both side regions are closed to the north by a wall at y ~ -424 that is open
  only below z ~ 50, and the z 160 kill slab spans x -3456..-896, y -2400..-448 between them. The
  record takes the WEST side and threads the gap at its outer edge: (-3465, -1294, 273) at 11 s,
  just west of the slab, then (-3272, -213, -493) at 12 s, under the wall into the big north room
  (d 24.5-26.7k = 13-20 %). The September policy's 9.3 % death at (-1340, -990, 183) is the EAST
  side, ~450 u west of the slab's east edge (x -896), 23 u above the slab: it descended too far
  from the outer edge. So the field's way out of the pit's north end exists on both sides; what
  kills a field-follower there is energy/precision at the side region (reach x > -896 above
  z 160, then drop under the wall), not topology. Read evals with that in mind: 9.3 % = fell on
  the east slab, ~13-15 % = under the wall into the north room.
* The local explorer probe never went east of x -1460 in 6 min (0 cells): random bursts do not
  carry enough speed out of the pit column's north end.
* 22:25 UTC: the start region is 99.46 % mirror-symmetric about x = -2176 (occ_32, x -3700..-650,
  y -4000..1600, z -1000..1200; the next-best axis 98.2 %), the kill slab's x range -3456..-896
  included: the east exit is the record's west exit mirrored, so it is flyable at the record's
  energy. CPU smoke of pbw1's exact resume (16 envs, --no-compile --fp32): starts, restores the
  checkpoint's shape flags, lr 2e-5 -> 1e-4, every recipe spawn mechanism armed; zero-shot greedy
  peaks at 104 u/s and dies at 9-14 s. The checkpoint carries --yaw-entropy 0 and a yaw sigma of
  0.017 (z units): view exploration on this map comes only from --front-temp (x2) and the
  frontier perturbations (heading +-90 deg, speed x1-1.3).

### Harness issue (uf2r1b4, 2026-10-30 22:50 UTC): one hour without a box on the budget pace
* `box.py rent` has refused every 2 min since 21:46 UTC: "burn 0.79 $/h, limit 0.822 $/h", 2 boxes
  live, cheapest acceptable offers 0.268 (4080) / 0.281 (5070 Ti) / 0.467 (3090). With four
  agents and boxes at ~0.4 $/h the pace fits two boxes, so half the branches wait for the other
  half's runs to end (which the protocol lets run up to 6 h). Not worked around: the retrier only
  calls `box.py rent`. Meanwhile this branch does CPU analysis.
* **THE WALL IS AN ENERGY BUDGET OF ~100 u (23:35 UTC, ANALYSIS BENCH, CLAUDE.md section 0: the
  record's state is a bench start only, its inputs are never read).**
  `autoresearch/analysis/uf2_mpc_exit_bench.py`: tools/mpc_bench.py's receding-horizon CEM
  (256 x 3, 1.2 s plan, 8 ticks per replan) with a FIELD-PROGRESS objective (min d over the
  plan, alive, + 0.25 x energy), started at the record's 10.0 s state (-2500, -1356, 493), the
  pit column's north end turning west, with its speed rescaled to an energy height E.
  Reaching the north room (y > -400, 16-18 %) in 3 s, 3 seeds each:

  | E at the north end | 8 ms tick | 10 ms tick |
  |---|---|---|
  | 1,125 (the record's) | 3/3 | 3/3 |
  | 1,100 | 3/3 | 3/3 |
  | 1,060 | 2/3 | 1/3 |
  | 1,030 | 0/3 | 1/3 |
  | 1,000 / 900 / 800 / 700 / 600 | 0/2 each | - |

  Failures either die on the z 160 slab at 7.6-8.9 % (the September "north dead end" exactly)
  or stall on the north end's ledge at 6.3-7.6 %. So: a line must reach the north end with
  E >= ~1,060-1,100. From the spawn's 1,188 that is a NET loss of at most ~100 u over the whole
  start; the record nets -63 (+190 of fall strafe at 131 fps, -86 landing on S19, the bowl's
  losses). A field-follower's short turn-around arrives with hundreds less, which is the 9 %
  dead end. What must be learned is energy management (air-strafe gains, tangential landings,
  a wide turn-around), so: per-call energy shaping is aimed at the right quantity, and the 10 ms
  tick costs ~30-50 u of the fall-strafe gain against the record's 7.6 ms (56 vs ~78 u/s
  ceiling) - a third to a half of the whole slack.
* 23:56 UTC, still no box (refused every 2 min since 21:46). The retrier now launches TWO arms on
  the one box when it comes (a hedge: this branch may get one box window this round), then the
  weaker is stopped at its first evals:
  * `uf2r1b4_pbw1`: the warm start above (resume of pb26_uf2_init.pt; --lr 1e-4; the
    checkpoint's per-call K = 12 energy shaping and 7.64 ms tick).
  * `uf2r1b4_sk12`: SCRATCH, the incumbent recipe + `--tick-ms 7.63 --energy-pot 12
    --energy-pot-front 0 --energy-pot-front-bonus 0` - the same two energy levers from nothing
    (the record's 131 fps physics, the bench's ~100 u budget; per-call shaping charges every
    landing and pays every strafe). CPU smoke (16 envs): starts, [8, 8, 7] ms pattern, 230 s
    episode cap, gamma/time-pen/stall-eps converted per tick.
  Both are compared on gates cleared (the 9.3 % east slab, the north room at 13-18 %) and the
  greedy line's energy at the pit's north end (uf2_eval_lines.py prints it).
* 01:50 UTC, more nulls from searching on CPU (no box yet):
  * The 6-min GE archive holds NO state climbing north out of the bowl's south end (y < -2700,
    vy > 300, vz > 0) with E >= 900 (the record climbs S19 at 7.9 s with 1,127). Its
    "north-heading" high-E states are steep falls onto the kill layer at y ~-2550..-2750, z ~560.
  * Geometry of the loop: the z 160 kill slab covers the pit column's whole north part
    (y -2400..-448), so the bowl under it connects to the flight level above it ONLY at the south
    end: a line must go down, lap the bowl (the field inside the bowl correctly points SOUTH,
    d 31k at the south end -> 33k at the north end under the central block), climb out at the
    south end heading north and fly north above the slab to the exits.
  * The progress(+energy) MPC from the TRUE spawn (A/D + view only, no W, no jump; 8 s)
    dithers on the platform at ~100 u/s and drops onto a ledge at z 548 (3.9 %) - no dive.
* Harness issue, continued (03:52 UTC): still refused every 60 s, burn 0.79 / limit 0.818, the
  same two boxes for 6 h; this branch has had no GPU minute this round. A pace that admits two
  ~0.4 $/h boxes for four branches, with 6-h arms allowed, can starve half the branches for a
  whole 12-h round.

### Round 1 lesson, uf2r1b4 (wrap-up 2026-10-31 09:00 UTC): no GPU minute; the wall is measured
* **No box all round.** `box.py rent` was refused every 1-2 min from 21:46 to 08:15 UTC on the fleet
  budget pace (burn 0.79 $/h against a limit of 0.815-0.823, the same two boxes the whole night;
  cheapest acceptable offers 0.196-0.467 $/h). No run, no eval, no score. See "Harness issue"
  above: four branches, a pace for two ~0.4 $/h boxes, 6-h arms allowed.
* **What this branch measured instead (CPU, analysis only; the record as a ruler/bench start):**
  1. On the grader's field the September "north dead end" reads 9.27 %, and the record passes
     9.27 % only ~11.3 s in. The pit itself is negative progress.
  2. The start region is 99.46 % mirror-symmetric about x = -2176. The 9.3 % death is the EAST
     side exit falling on the z 160 kill slab (x -3456..-896, y -2400..-448); the record takes the
     WEST exit at the slab's outer edge and drops under the y ~-424 wall into the north room.
  3. `autoresearch/analysis/uf2_mpc_exit_bench.py`: from the record's north-end state (10.0 s),
     a progress MPC reaches the north room (16-18 %) at E 1,100-1,125 (3/3 at 8 and 10 ms),
     1/3-2/3 at 1,060, 0-1/3 at 1,030, 0 at <= 1,000 (falls on the slab at 7.6-8.9 % or stalls
     at 6-7.6 %). **The start is a ~100 u net energy budget** (spawn 1,188; record nets -63).
  4. The bowl under the slab reaches the flight level only at its SOUTH end: the line laps the
     bowl, climbs out heading north and flies north over the slab (the field inside the bowl
     points south, correctly).
  5. GE with the pbf flags (6 min, 2 threads) fills the pit (377k cells) but loses the bowl's
     energy after ~4 s and holds no south-end climb with E >= 900; a progress MPC from the true
     spawn never dives. refs/pb_r26 zero-shot on this map dithers at ~100 u/s and dies at 2.4 %.
* **What the next round should run first** (prepared and CPU-smoke-tested here, never on a GPU):
  energy-management arms, because the gate is energy, not route:
  * `uf2r1b4_sk12`: scratch, the recipe + `--tick-ms 7.63 --energy-pot 12 --energy-pot-front 0
    --energy-pot-front-bonus 0` (the record's 131 fps: ~+40 % air-strafe energy per second; per-call
    shaping charges every landing and pays every strafe);
  * `uf2r1b4_pbw1`: refs/pb_r26 warm (`runs/uf2r1b4/_init/pb26_uf2_init.pt`, md5 e2b2315f: pb_r26 with
    princessburglar's run state and curricula dropped), `--resume`, `--map maps/surf_unitfarmer2.bsp
    --lr 1e-4 --ep-ticks 30000` + the recipe's training flags (its own K = 12 per-call shaping and
    7.64 ms tick restored). Its yaw sigma is 0.017 (yaw entropy 0 in the checkpoint).
  Read every eval with `autoresearch/analysis/uf2_eval_lines.py` (E at the pit's north end, side,
  north room): an arm is improving when that energy rises toward ~1,060-1,100 even before the
  score moves off ~9.3 %.
