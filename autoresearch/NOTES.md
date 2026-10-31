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

### uf2r1b3, round 1 - 2026-10-30 21:55 UTC: first look (no box yet; the fleet was at its burn limit)

* **What kills the potential-following line (measurement, from the BSP).** trigger_teleport *18
  (-> recomeco, the spawn) is a slab at z 160..161 over x -3456..-896, y -2400..-448: the floor of
  the whole region between the start pit and the course north of it. The goal field (goalc_32)
  is built over free voxels and routes THROUGH it: its steepest descent from the spawn runs east
  of the pit, (-1520,-2128,336) -> (-1168,-1616,176) -> (-1072,-1104,80) -> north. A policy that
  follows it reaches (-1340,-990,183) = d 27,807 = **9.27 % on the grader** and is teleported.
  That is the "north dead end" of September's stage-1 arms, and every scratch run's first local
  optimum here: a branch scoring ~9.3 has learned the field, not the map. The record dives past
  the slab's far-west edge (x < -3,456) with ~1,200 u/s after the pit pump. *7 (recomeco) is the
  pit's own fall catch (brush-shaped; the core tests brush containment, not the AABB).
* **The field inside the pit points the right way.** Descents from the record's pit points go up
  the pit's south ramp to the upper level (z ~528, d ~29,700) and then west/north-west, i.e. a
  ratchet episode SPAWNED in the pit (b_0 = its own d, ~32-34k) is paid for climbing out along
  the record's exit. Only the dive itself (start -> pit) is unpaid.
* **explore_phase1, the pbf config, locally (3 min, 2 threads, 0.13M ticks/s ~ 1/10 of a box):**
  300k cells, best 5.1 % (the east route, depth 6.8 s), z min -998. Along the record's path the
  best same-motion energy height is at the record's level to t 2.5 s (1,169 vs 1,295) and then
  collapses in the bowl: 306 at t 3.0 (record 1,204), -140..-399 at t 3.5-5.0, nothing within
  200 u of the record from t 7.5 on. The direct drop from the platform reaches the upper level
  (z 485-587) at |v| 394-644 (E 645-707) against the record's 920-1,100: the pit pump is the
  missing speed. Same failure as October's go_cem. Box-scale compute is the open question.
* **tools/ge_spawns.py** (new, test tests/python/test_ge_spawns.py): archive -> --spawn-states,
  the most energetic state per 64 u place (14,083 places from the 300k-cell smoke archive),
  --watch re-converts atomically when the archive changes (the trainer re-reads the file on an
  mtime change every 20 iterations), so a search can run beside the trainer on the same box.
* **Plan for the first box:** explore_phase1 (pbf config, 10 ms, ~12 threads) from the true spawn
  + ge_spawns --watch + the scratch incumbent recipe with --spawn-states (frac 0.3). Read: does
  the archive leave the pit; does training from pit states climb out; does the start line follow.

### uf2r1b3 - 2026-10-30 22:52 UTC: the explorer and the bowl (local CPU; still no box)

* **More search does not cross the bowl.** explore_phase1 (pbf config + --no-brake), 12 min
  local (~4x the 3-min smoke): along the record's path the best same-motion energy height is now
  AT the record's level right up to the bowl - 1,252 at the record's t 2.5 s point (record
  1,295, |v| 1,715 vs 1,686) - and still collapses within the next 0.5 s: 276 at t 3.0 (record
  1,204), 16 at t 3.5. Nothing within 200 u of the record from t 8.0 on. 430k cells, best 5.6 %.
* **--no-brake** (new, explore_phase1): bursts never hold keys whose wish direction points
  against the velocity (GoldSrc air accel brakes up to 250 u/s per tick there; the record's first
  15 s never do, 57-72 % of its ticks sit in the 89-91 deg strafe window). Small gain before the
  bowl (3-min A/B: 1,302/1,278/1,252/1,190/1,151 vs 1,286/1,289/1,239/1,119/1,169 at t
  0.5-2.5), no effect in it.
* **--surf-hold P** (new): a surfing burst presses the strafe key INTO the surfable plane it
  touched, view on the velocity. Fires on only ~4 % of decisions here; no effect on the bowl
  (3-min A/B: 157 at t 3.0). Kept, default off.
* **Why (bench from the record's own states, MEASUREMENT ONLY, scratchpad bowl_bench.py):** from
  its t 2.5 s state, 0.05 s before the bowl contact, "no keys" crashes to E 128 within 0.1 s,
  "hold A" to 193, "hold D" keeps 1,301-1,312 (and dies ~0.4 s later without the next
  manoeuvre); the record's own keys + yaw replayed open loop keep ~1,150-1,200 for 0.5 s, then
  diverge (10 ms vs its 7-8 ms frames). D here steers AWAY from the bowl wall it is about to hit,
  so it meets the curve tangentially: an anticipation, not "press into what you touch". The pit
  pump is a chain of such 0.1 s knife-edges - closed-loop control. A random burst picks the right
  key at the right tick rarely, and a chain of them almost never, which is why every archive
  (October's, this one) arrives at the bowl at record energy and stops there.
* **Consequence for the plan:** use the archive for what it does well - coverage up to the bowl
  at near-record energy - and let RL learn the closed-loop part from those spawns, with per-call
  energy shaping (--energy-pot K --energy-pot-front 0 --energy-pot-front-bonus 0) so a crash in
  the bowl costs at the tick it happens (K 8: a 1,000 u crash = -8, the progress reward of 2,450 u).
* **Correction (23:28 UTC): the archive is NOT at record energy before the bowl.** With a
  strict same-motion match (within 150 u, velocity cos > 0.95; the 0.8 test let free falls pass
  as surfing), the 12-min archive's best is 1,259 at the record's t 1.25 (record 1,374: the
  record's fall air-strafe, ~+120 u at its 7-8 ms frames, at most ~+60 at our 10 ms), 1,008 at
  t 1.5 (record 1,295: the first ramp landing costs the archive ~250 u, the record ~80), 759 at
  t 2.0, 524 at t 2.25, 16 at t 3.0. The "1,250 at t 2.5" states are doomed free falls into the
  pit's floor teleport (every constant control from them ends in it within 0.2-0.7 s). So the
  deficit begins at the FIRST ramp, 1.5 s from the start - October's "280-390 u short" again.
  The search never rides the south curve the way the record does; RL has to learn it from the
  true start's own fall, not from an archive state.

### Harness issue (uf2r1b3, 2026-10-31 00:02 UTC): two boxes hold the whole budget pace for hours
* From ~21:45 UTC on 2026-10-30, two live boxes (0.79 $/h together, ~0.39-0.40 each) sat against
  the pacer's limit (0.82 $/h), so every `box.py rent` from this branch was refused for 2h15m+
  (a retry every 100 s; the cheapest passing offer then was a 0.268 4080). With K = 4 branches and
  a ~0.75 $/h pace, a 0.4 $/h box is half the fleet: two branches can work and two wait. Not
  worked around (PROTOCOL: never). A per-branch share (pace / K) or a lower per-box cap while
  others wait would let all four branches run one cheap box each.
* **--strafe-share P** (new, explore_phase1; 01:52 UTC): P of fresh burst draws air-strafe (offset 0, no
  forward key, A or D). Local 3-min A/Bs, strict same-motion energy at the record's t 1.25 / 1.5 s
  (record 1,374 / 1,295): no-brake alone 1,259 / 1,008 (the 12-min run from the same seed: the
  SAME numbers - the record line's coverage saturates in the first minutes), strafe-share 0.3 seed
  0: 1,272 / 1,239 (3 states within 150 u of the record's energy), seed 1: 1,290 / 1,082,
  + surf-hold 0.5: 1,281 / 1,016. So +15-30 u before the first ramp and a lucky line now and then;
  the first-ramp landing (~250 u lost against the record's ~80) and everything after it are not
  fixed by a better random prior. Kept, default off (flag off byte-identical, checked).
* **Warm start from refs/pb_r26, what a resume on this map would take (02:26 UTC, not run).** The
  checkpoint's config restores, on a bare resume: map surf_princessburglar_b26 (run_arm's resume
  branch passes no --map, and its PRE-LAUNCH record gate records the source checkpoint on ITS map,
  which the box does not have), demo_file + demo_front_frac 0.8 (its own pb spine; with the file
  blanked, --demo-front-frac 0 is needed or the trainer exits), lr 2e-5, energy_pot 12 (per call),
  tick 7.64 ms (the finished 10 -> 7.63 ramp), obs_potential_d0 for pb. bc_file is NOT restored;
  the pb reservoir is dropped on the map mismatch (respawn.py); novelty tables of the wrong length
  are re-zeroed. So a warm start needs a retargeted checkpoint (config map -> surf_unitfarmer2,
  demo/spawn files cleared, demo_front_frac 0, run state dropped), pushed with deploy --ckpt.
  Kept as the second option; the first box goes to the scratch arm.
* **The warm start is ready and goes first (03:04 UTC).** tools/ckpt_retarget.py (new, test
  tests/python/test_ckpt_retarget.py) turns refs/pb_r26 into a uf2 warm start (config map ->
  surf_unitfarmer2; demo/BC/spawn files, demo_front_frac, obs_potential_d0 cleared; reservoir,
  novelty table, unstuck tracker dropped; weights, optimizer, step 80.08B and shape flags kept;
  md5 b2426d19). Local checks: the record gate passes on it; a 32-env CPU resume runs at 7.64 ms
  ([8,8,7]), energy_pot 12 per call (its own), episode cap 320.9 s; its yaw sigma is 0.017 (a
  near-deterministic policy). Zero-shot from the true start (greedy 2, stochastic 3, 6.5-19 s) it
  hops slowly on the platform (|v| < 300) and steps off into the fall catch at z ~510-550, best
  2.2-2.4 %: the upper-level floor south of the platform (z 490) lies inside trigger *7's brush,
  so it is a death floor too (the operator's scratch smoke dies at the same place). Why warm
  first: the wall here is low-level energy control (landings, ramp riding, a pump), and pb_r26
  learned exactly that with per-call energy shaping on a map whose start is a half-pipe pump.
  The scratch arm (same additions, 8 ms) stays the next one; a recipe that only works warm is not
  yet a recipe.

### uf2r1b3 round 1 lesson (draft 06:42 UTC; final at wrap-up)
* **No box all round** (every rent refused by the burn pace from 21:45 on: two 0.39-0.40 $/h boxes
  held 0.79 of the 0.82 $/h limit). Everything below is analysis and code, measured on CPU.
* **What this map is, for the next agent (measurements):** a field-following line dies at 9.27 %
  on the kill slab *18 (z 160); the z 490 floor south of the platform is inside fall catch *7; the
  record pumps the start pit (flying UNDER the slab at z -870), comes back up, surfs a central ramp
  (z ~465, x -2,250..-2,100) north over the slab and dives past its far-west edge (x < -3,456).
  A score above ~9.3 needs that whole chain from the true start.
* **Search alone does not do it:** explore_phase1 (pbf config, + --no-brake / --surf-hold /
  --strafe-share, 3-12 min local) never rides the south curve the record rides: strict same-motion
  energy is 250-290 u short at the first ramp (t 1.5 s) and the archive's coverage of the record's
  line saturates within minutes. Benches from the record's own states show the bowl is a 0.1 s
  closed-loop choice of strafe key. Use RL for the control, search for coverage.
* **Ready to run, in order:** (1) warm: refs/pb_r26 retargeted (tools/ckpt_retarget.py) +
  --spawn-states from the GE archive (tools/ge_spawns.py, autoresearch/spines/uf2r1b3_ge12_seed.npy)
  + its own per-call energy shaping, lr 1e-4; (2) scratch: the recipe + --tick-ms 8 + per-call
  --energy-pot 8 + the same spawns. Both flag sets pass a local CPU smoke run and the record gate.
