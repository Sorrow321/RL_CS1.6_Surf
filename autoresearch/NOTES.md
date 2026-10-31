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
* **egt result (4e8, 22:20-22:44 UTC): NULL, worse than the control.** Evals 2.96 / 2.11 / 2.84 /
  2.62 / 1.34 / 2.58 / 0.84 / 1.95 at 0.5-352M (control 2.85 / 5.26 / 5.66 / 5.69 / 5.62 / 5.65 /
  5.59 / 5.66). The greedy CAMPS on the platform: 37-60 s of bunny hops at 280-430 u/s (Emax
  ~1,250, minz 1,059-1,181) until the eval's 30 s patience ends it. Per-call energy shaping makes
  every way off the platform a charged energy loss (the slide, the ledge, a crash in the bowl) and
  standing still free: the deception is not solved, the policy just stops trying. Per-call K 12
  helped princessburglar's late gates, where the policy already flew; here it kills exploration.
  `dip/max_survived_depth` read 14.5-19.7 in training, but the control showed the identical 14.48
  at 20-83M with no pit spawns: an artifact of some respawn mode, not a pit exit.

### The archive's "high-energy pit states" are free falls into the floor (bench, 22:45 UTC)
* Bench (measurement only): the 4,927 tops rows with z < 100 and E > 1,000, 24 stochastic episodes
  each from egt's and the control's final weights (record_ckpt --spawn-states). **0/24 and 0/24
  climb out; every episode dies within 0.0-0.4 s at z ~-970..-998.** Those states are falls at
  1,550-1,825 u/s into the pit floor: E is high only because the impact has not happened yet
  (October's "energy dominance is optimistic about contact losses still to come"). The pit's
  energy-top cells are the doomed ones by construction; a spawn file of them teaches nothing.
* So arm uf2r1b2_gts (the control's reward + 40% of those spawns), launched 22:42, was stopped at
  25.7M steps (22:44) before it spent its budget on doomed falls.
* If GE states are used again: select by what happens AFTER the state (its subtree survives, or
  a contact with a surface), never by the state's own energy.

### Arm uf2r1b2_dcn (launched 22:45 UTC) and GE run 3 with an admission margin (22:58 UTC)
* dcn = the incumbent recipe + `--death-charge 1.0 --int-coef 2.5 --int-speed 8 --int-view 0
  --int-speed-weight 3 --int-speed-weight-up --ckpt-every 5e7`, 4e8 from scratch: September's
  closest generic pit recipe (uf2DCspdw10v0up: 100% pit entry, 100% take-off attempts at ~1,000
  u/s, 0 exits) on top of today's frontier machinery (greedy-death respawns, front perturbation,
  unstuck temperature), which is what cracked princessburglar's gates once a policy reached them.
  The point is a policy that LIVES in the pit, so a search from its own lip states can find the
  exit. Caveat for the score: a greedy line that dives and dies scores ~2% (< the slide's 5.7).
* New `explore_phase1 --admit-margin SECS` (f5e3c375; tests/python/test_ge_admit_margin.py, off =
  byte-identical): a visit opens a new cell only if its run lived SECS past it (or was alive at
  the chunk's end) - the trainer's --respawn-margin rule applied to the archive, so a doomed fall
  never becomes a cell and --select-energy cannot prefer one. GE3 = GE1's flags + `--admit-margin
  2` (the recipe's own margin, carried unchanged), 20 min beside the dcn trainer.
* **dcn result (4e8, 22:45-23:10 UTC): NULL.** Evals 1.71 / 3.10 / 0.94 / 2.68 / 1.25 / 2.84 /
  2.20 / 2.89 (control 5.3-5.7). The death charge removes the slide (it nets 0) but nothing
  replaces it: the greedy waits on the platform up to ~20 s and drops onto the south-west ledge
  region (z ~510-596, (-1800..-2630, -2560..-2715)), dying there at 2.1-2.9% - the place pb_r26
  died; the record flies over it at z 584 (the pit catch *7 tops out at 577 there). 16
  stochastic episodes from the true start: 7 truncated at the 300 s cap ON THE PLATFORM, the rest
  that ledge. **0 dives.** September's DC entries came with the scratch preset's 0.005/tick time
  penalty (a cheap death + novel cells = dive for novelty); at the recipe's 0.001 the charged
  death makes the agent camp.
* **Bench: no policy survives the archive's "viable" top pit states either.** GE3 (admit margin 2)
  top pit cells with E > 1,000 (2,515): dcn 0/24, control 0/24, pb_r26 0/24 - all three die at the
  SAME instants (0.0-0.6 s), i.e. independent of the policy: the states were doomed. Cause: the
  first --admit-margin exempted runs still alive at the chunk's end, so the last 2 s of every
  surviving burst were admitted unchecked, and an energy ranking picks exactly the falls that die
  just after. Fixed (77d4b6bb): a visit needs SEEN survival of the margin. GE4 = GE3 with the fix,
  25 min. Lesson for anyone ranking states by energy: rank only states with an observed future.
* Ramp search (October's tools/ramp_search.py, the one sequence-free pit exit): needs a mesh
  export + ramps_mesh vocabulary that this checkout does not have for uf2, and the allocation that
  found the exit (run 2, an optimistic piece-pair prior) was replaced; runs 3, 5, 6 with the
  current code found none. Not run.

### Arm uf2r1b2_seq (launched 23:21 UTC): the recipe + `--speed-equiv 1.0`
* The shaping potential becomes d_eff = d - |v|^2/2g (beta 1: a unit of speed height is worth a
  unit of distance), paid as its change, with the speed credit refunded at death. Under the
  ratchet the dive's speed gain PAYS (+6.7 by the record's t 4.5) and a crash charges, and a policy
  that keeps its speed in the bowl keeps the credit; the slide still nets only its progress
  (its speed credit is refunded when it hits *18). The aim is a policy that LIVES in the pit,
  which the exit search needs. Known costs: the exit climb converts speed into height and is
  charged until the ratchet pays progress; lapping the bowl is credited (September's
  dip-speed arms lapped at 1,740 u/s). Generic: no map constant (beta 1 is the unit).

### Harness issue: `box.py pull --snapshot STEP` never pulls (zero-padded names)
`--snapshot` is `type=int` and asks for `ckpt_{STEP}.pt`, but the trainer writes milestones as
`ckpt_%010d.pt` (`ckpt_0150994944.pt`): the md5 lookup comes back empty and the pull skips the
file without a word. Workaround: scp the padded file myself, md5-checked against the box, into
`$AR_HOME/runs/<agent>/<run>/` (done for uf2r1b2_seq's ckpt_0150994944.pt, md5 3fef2861).

### seq result and a route the potential hides: the record's post-pit corridor WITHOUT the pit
* **seq (speed-equiv 1, 4e8): null for the pit, best score so far.** Evals 2.04 / 2.88 / 5.15 /
  **5.86** / 5.69 / 5.65 / 5.70 / 5.61 at 0.5-352M (= the control within noise). The greedy slides
  into *18 like the control; 16 stochastic episodes: 7/9 slide, 2/9 the south-west drop, 0 dives.
  The dive's speed credit is refunded when it crashes, so it never pays.
* **The trough.** GE2's deepest western chain (cell 466306, replayed exactly) shows a route the
  field does not: drop off the platform's SOUTH-WEST CORNER (x ~-2,590), land on a floor at z 548
  (t 3, a hard landing: E 1,152 -> 603), then a north-south TROUGH under the platform: bottom at
  x ~-2,350 / z ~310-375, the east side up to z ~600-650 at x ~-2,170, from y ~-2,350 to ~-1,070.
  That is exactly the corridor the record flies after the pit (t 9-10: (-2381, -2221, 518) ->
  (-2352, -1755, 342) -> (-2493, -1356, 492)), and it leaves it WEST at z ~480 and ~1,000 u/s to
  clear sheet *18's west edge (x -3,456). GE's random strafes PUMP the trough: E 470 -> 678 over
  25 s; it leaves west at ~700 u/s and falls onto *18 at 7.9%.
* Earliest chains to each progress level over GE1-4 (measurement): >= 7.0% in 10.1 s (GE1, west
  via the trough) / 9.1 s (GE2, a fall north-east); >= 7.8% in 12.4 s (GE2); >= 8.0% in 22.8 s.
  Every cell above ~6.5% is a fall that ends on *18: they score, they do not continue.
* So the speed the course needs may also be bought by pumping the trough (princessburglar's
  start was a half-pipe pump too) - a generic skill on a route the potential does not show.

### Arm uf2r1b2_tc (launched 23:50 UTC): backward curriculum on GE1's 10 s trough line
* Line: GE1 cell 613541's chain (12 links, exact): walk SW on the platform (0-3 s), drop off the
  SW corner (3-4.5 s, 926 u/s), land on the z-548 floor (5.0 s, E 1,172 -> 592), the trough north
  (6-9.5 s), exit west (9.5 s) and fall to 7.05% (10.1 s). Spine = tools/stitch_spine.py
  spine:<the spawn> ge:<archive>:<chain> (views along the velocity, keys translated):
  runs/uf2r1b2_lines/w7_spine.npy (+ .keys.npy), 1,012 states, md5 d2a75be9 / keys 6b6d82c5,
  end d 28,488. Self-states: the branch's own GE line from the true spawn.
* Flags: recipe + `--demo-file w7_spine.npy --demo-keys w7_spine.keys.npy --demo-window 5
  --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5 --demo-rate 0.3 --demo-front-frac 0.8
  --demo-success-d 100 --demo-success-bonus 50 --keys-hold-spawn-duck --yaw-entropy 0
  --demo-front-temp 0.5 --demo-front-ticks 40 --demo-front-view 6 --demo-match-band
  --no-respawn-novel --respawn-backward-mix 0 --int-real-only 0 --ckpt-every 5e7`, 4e8 from
  scratch (princessburglar's curriculum set; the curriculum refuses novel / backward-mix rows).
* **tc result (4e8, 23:50-00:14 UTC): 7.425, the round's best so far.** Evals 2.69 / 3.76 /
  **7.43 (101M)** / 7.36 / 7.42 / 7.31 / 7.28 / 7.33 at 0.5-352M - every eval from 101M on has
  9/9 episodes at 7.1-7.4%. The curriculum walked from the spine's end to its start in ~100M
  steps (`demo starts [0,1011] ... success 99.3% over 500 eps`). Hand-off snapshot
  ckpt_0100663296.pt (md5 3360a70e) pulled by hand (the harness issue above).
* The greedy line at 352M, 5.4 s (faster than the 10 s search line): runs SW on the platform
  strafing (E 1,192 -> 1,325), drops off the corner, **lands on the z-548 floor at t 2.75 (E
  1,177 -> 623: half its energy in one impact)**, rides the trough bottom north at ~745 u/s, leaves
  west at z ~300 and falls onto *18 at (-2,725, -1,355) at 7.3%. The record leaves the trough at
  z ~490 and ~1,000 u/s and clears *18's west edge (x -3,456); from z 300 that edge needs ~1,700
  u/s. So the next gate is ENERGY, lost in one landing: princessburglar's energy-gate playbook
  (per-call energy shaping on a flown route; a GE energy search from the policy's own states
  before the loss; pick by energy; teach with --demo-success-energy).

### The energy gate: tce (resume + per-call energy K 8) and GE5 from the policy's own fall (00:20 UTC)
* tce = tc's ckpt_final (400M) resumed with the same spine and flags + `--demo-tau0 0
  --energy-pot 8 --energy-pot-front 0 --energy-pot-front-bonus 0` (every loss of energy height
  charged when it happens: the floor landing's -560 u costs ~-4.5), 4e8 more steps. On a flown
  route the energy term is a price on the landing, not a reason to camp (egt camped because no
  route paid yet).
* GE5 = explore_phase1 (pbf flags) rooted at 29 states of tc's own greedy line BEFORE the landing
  (ticks 100-268: the platform run and the fall; `--roots-spine --roots-keep-all`, no boost), 12
  min: looking for a fall into the trough that keeps its energy (the line holds E ~640-660 there,
  the record ~1,100). Then tools/energy_pick.py -> stitch_spine -> --demo-success-energy.
* tc's greedy line (record_ckpt --dump-states, runs/uf2r1b2_lines/tc_greedy_states.npz): tick 270
  (-2547, -2504, 533) at 1,077 u/s E 1,258 -> tick 275 on the floor (-2549, -2477, 548) at 487 u/s
  E 696. One impact, -560 E.
* **tce result (400M -> 800M, 00:16-00:42 UTC): 7.699 (702.5M), +0.3 over tc.** Evals 7.35
  (the resumed weights) / 7.62 / 7.60 / 7.45 / 7.67 / 7.60 / **7.70** / 7.63. Same route; the
  floor still costs ~500 E (t 2.75: 1,138 -> 622) and an earlier lip ~220 (t 1.5-1.75); the gain
  is a higher trough exit (z ~340 vs ~300) that falls further west (x -2,812 vs -2,725).
  Hand-off ckpt_0702021632.pt (md5 7a479c04), pulled by hand.
* **GE5 (rooted before the landing): no energy to find.** energy_pick against tc's line: +10 E at
  3.5-4.5%, -175 at 4.5-5.5%, nothing at 5.5-6.5%. Physically: a 600 u fall onto a flat floor
  loses its vertical energy (vz^2/1600 ~ 600 u) whatever the line, and the record never lands
  there - it crosses that floor 36 u up, moving north at 919 u/s out of the pit. On this route
  the only source of the missing ~450 E is PUMPING THE TROUGH (~56 u of E per second of ideal
  strafing at ~700 u/s: ~8 s).
* The training frontier (reservoir min-depth 93.58% = 6.4%) sits behind the greedy's 7.7%: the
  west exit falls onto *18 in < 2 s, inside the reservoir's 2 s margin, so the boosted frontier
  rows (x1.0-1.3) spawn at the exit, not past it.
* Next (00:45 UTC): GE6 rooted at 29 states of tce's own trough ride (after the landing; E
  613-684), 15 min - does random strafing on the half-pipe find energy and a west exit past 7.7%?
  And tce2 = tce's final + `--energy-pot 16` + `--demo-success-bonus 0` (the curriculum's spawns
  kept): the +50 bonus is paid at the line's end, which the direct route already reaches, so it
  PENALISES an 8 s pump by delaying the bonus (~-17 in discounted value); energy at K 16 pays the
  pump ~+7.
* **tce2 (K 16 per call, bonus 0; 800M -> 1.2B): COLLAPSED.** Evals **7.854** (800.6M: tce's final
  weights after one iteration - the round's best) / 2.98 / 2.98 / 2.96 / 3.46 / 3.49 / 3.51 / 3.52.
  The late policy strafes on the platform (Emax ~1,357) and drops into the south-east catch at
  2.9 s. The energy term "telescopes per episode, no death refund", so dying rich pays (+2.6 at
  K 16 for +165 E), while the trough route pays the floor's -600 E (-9.6). Per-call energy is
  safe only at a K where the route's progress outweighs its unavoidable losses (K 8 here).
* Hand-off for the 7.854 eval: tce2 had no snapshot at or before 800.6M, so the hand-off rule would
  take its newest (collapsed) checkpoint. I copied the exact checkpoint tce2 resumed from (tce's
  ckpt_final, its own step 800,063,488, md5 d1ce2797) into $AR_HOME/runs/uf2r1b2/uf2r1b2_tce2/ as
  ckpt_0800063488.pt - the weights that eval ran on, one PPO iteration earlier.
* **GE6 (15 min, rooted at 29 states of tce's own trough ride): no pump.** energy_pick against
  tce's line: +59 E at 5-6%, +49 at 6-7%, -193 at 7-8%; the deepest cells reach 8.0% (falls onto
  *18 at (-2,901, -1,227)). Random bursts lose at the trough walls what they gain by strafing.
* Arm tcf (01:10 UTC): tce's final weights + the same curriculum and bonus + `--energy-pot 12
  --energy-pot-front 0.01 --energy-pot-front-bonus 1` (the recipe's own mode at K 12: K x the
  kinetic energy height, paid ONCE when the record first enters the frontier band - here the
  trough's west exit - and forfeited by dying earlier, so nothing to farm and no charge on the
  floor). Leaving the trough at 1,100 u/s instead of 700 is worth ~+5.4: a direct price on a pump.
* **GE4 (strict admission margin 2 s) holds VIABLE high-energy pit states**: its energy-top pit
  cells reach E 939-954 at progress -4..-1% (GE1-3's pit cells were doomed or E <= 413). 487 rows
  with z < 100 and E > 800 (median E 870, z -465, 1,460 u/s). tce's policy from them (24 stoch,
  bench only): 0/24 climb out, but they live 0.3-1.4 s (GE's random actions survived 2 s from the
  same states: the policy has never practised the pit). An exit at E ~870 is physically possible:
  over the catch *7 (top z 577) at ~680 u/s, into the trough with ~870 E (the route has ~650).
* Next arm (after tcf): the corridor curriculum + ~20% spawns from GE4's viable pit states (the
  branch's own search from the true spawn), so the policy that already values the trough learns
  to survive the pit and climb toward it - the pit's exit is the corridor's south end.
* **tcf (frontier energy bonus K 12, 800M -> 1.2B): REGRESSED** 7.84 (the resumed weights) ->
  7.32 / 7.13 / 7.11 / 7.12 / 7.06 / 7.09 / 7.10. A once-per-episode bonus for the speed brought
  to the frontier band also pays for reaching the band early and fast, not for going further.
* **GE7 (rooted at GE4's 487 viable pit states, strict margin 2 s, 15 min): no exit** (88k cells;
  nothing deeper than the start's distance).
* **beam_tas on tce's weights from the true start** (4,096 envs, resample 25, --score d,
  --save-frontier): every lineage dies by 6 s, so its frontier is the last resample before the
  fall (6.93%), below the greedy's in-fall peak (7.85). On a route that ENDS in a fall, a d-ranked
  beam cannot see past the fall; the trough route is at its ceiling (~8% from E ~690, by GE5/GE6
  and ballistics: from z ~490 at ~570 u/s the fall lands near x -3,000, short of *18's edge).
* Arm tcp (01:36 UTC): tce's final weights + the corridor curriculum (bonus 50) + per-call energy
  K 8 + `--spawn-states runs/bench_pit/pit_viable_ge4.npy --spawn-states-frac 0.2` (md5 01b9efbb):
  the policy that already values the trough practises the pit from viable states and is paid by
  the ratchet for climbing toward the trough (the pit's exit is the corridor's south end).
  Measure: the pit bench (0/24 climb-outs for tce) and the greedy evals vs 7.85.

### The pump exists: GE from the policy's own trough ride (02:00-02:25 UTC)
* **tcp (20% viable pit spawns, 800M -> 1.2B): no pit skill, route held.** Evals 7.35 / 7.78 /
  7.65 / 7.74 / 7.84 / 7.60 / 7.69 / 7.77; pit bench 0/24 climb-outs with the SAME death instants
  as every earlier policy. Restored by core.set_state or by the spawn pool, those GE4 states die at
  identical ticks (32-108) with neutral input: falls at 1,400+ u/s toward the floor that survive
  only with exact, immediate steering onto a ramp (GE's random inputs found it, no trained
  policy does). Pit practice from search states: dead end, three ways (GE1 doomed, GE3 leaky,
  GE4 viable-but-unsteerable).
* **GE8 (40 min, rooted at 29 states of tce's own greedy trough ride, --admit-margin 1): PUMPS THE
  TROUGH.** energy_pick against tce's line: **+345 E at 5.74% (h 1,004 vs 659, 1,022 u/s), +340 at
  6.05% (1,005, 1,039 u/s)** - chains of ~26 s of random oscillation along the half-pipe, from E
  ~650 to ~1,005 (the record carries ~1,100 there). Its deepest cell is only 7.0%: the admission
  margin keeps the final falls out of the archive.
* **Bench: tce's policy from the pumped trough states** (3,766 rows, 4.5-6.6% and E >= 900 -
  a MEASUREMENT window, never a training file): best 8.93% at (-3,049, -940) - the round's
  furthest - median 6.7%, all onto *18 within 1.5 s (it was never trained from there). So energy
  in the trough is what the next ~1% needs.
* New: `ge_return_states.py --not-behind-roots FIELD` (274f6bbd): only rows not behind the
  search's own roots (GE8 fell back into the pit from the trough: 69% of its tops). Training file
  = GE8 `--tops --not-behind-roots`: 2,188 rows, all in the trough, E median 970 (md5 5b1e73ca).
* Arm tcq (02:27 UTC): tce's final + the corridor curriculum + per-call energy K 8 +
  `--spawn-states runs/uf2r1b2_ge8/tops_fwd.npy --spawn-states-frac 0.3`: the west exit from
  high trough energy, and a critic that learns what trough energy is worth (then: does the
  policy start pumping from the true start?).
* **tcq result (800M -> 1.2B): 7.953 (901M), the round's best.** Evals 7.37 / 7.80 / **7.95** / 7.84
  / 7.82 / 7.83 / 7.81 / 7.70 (medians up to 7.8%): practising the west exit from pumped trough
  states lifts the greedy a little; it still does not pump from the true start. Hand-off
  ckpt_0900726784.pt (md5 48c49d2a), pulled by hand.
* Arm tcpump (02:36 UTC), princessburglar's energy-gate playbook: spine = tce's own greedy line to
  its trough row 447 + GE8's chain to cell 230626 (26.5 s of north-south pumping, E 644 -> 1,005,
  ends at 6.05%) - tools/stitch_spine.py --no-cut, 3,103 states, md5 028b28ce. Curriculum from
  its end with the ENERGY rule: success = crossing d <= 28,894 with energy height >= 855
  (`--demo-success-energy 150`); `--demo-milestones 28588` keeps the trough route's old success
  line paying 50, so a pump first is +50 on top of the route. Resumed from tcq's 900.7M
  snapshot, with GE8's trough spawns (30%) and per-call energy K 8 kept.
* **tcpump result (900.7M -> 1.3B): 8.199 (1,153M), the round's best.** Evals 8.01 / 7.82 / 7.66
  / 7.87 / 7.89 / **8.20** / 8.06 / 7.96. The energy-rule curriculum walked from the spine's end
  (row 3,102) to row 1,692 at 96-98% success (the policy pumps from mid-pump states to E >= 855).
  The greedy from the true start does NOT pump yet (6 s runs; the curriculum has not reached the
  pump's start at row 447): its gain is a further-west exit (x ~-2,900 vs -2,800) learned from
  high-energy states. Hand-off ckpt_1152385024.pt (md5 865bf7f9), pulled by hand.
* tcpump2 (03:02 UTC): tcpump's final (1.3B) + the same flags + `--demo-tau0 1692` (the
  curriculum position is not in the checkpoint), 4e8.
* **tcpump2 (tau0 1692, 1.3B -> 1.7B): STALLED, no new best.** Evals 1.87 (the first eval after
  the resume) / 8.01 / 8.02 / 7.96 / 7.97 / 7.72 / 7.78 / 7.61. The curriculum oscillated the whole
  run: band [1692,1696] 98-99% -> advance to [1687,1691] -> **exactly 0.0% over 500 eps** ->
  back off (already at the end of tcpump). Spine rows 1680-1699 are a smooth surf south up the
  trough's side (E ~822-824, the same keys, ducked); nothing visible explains a 99% -> 0% step in
  5 ticks - likely a bookkeeping edge, not a skill (princessburglar's lesson: check a stuck band's
  outcomes before calling it difficulty). Not resolved; the next spine avoids it.
* **GE9 (20 min, rooted at GE8's 2,188 pumped trough states): the NORTH-EAST exit.** Deepest cells
  **10.23%** at (-1,143, -762, 183) - the trough's east side leads north-east, onto *18's NE part,
  much deeper than the west exit (8-9%). By the root's energy: E 697 -> 8.25%, **E 747 -> 9.25%**
  (a 0.77 s chain from (-1,850, -1,247, 381)), E 861 -> 10.23% (2.4 s), E 1,010 -> 10.07%. The
  fastest route past 8.8% found: tce's line + GE8's chain to cell 192980 (15.7 s in the trough, E
  692 -> ~826) + GE9's chain to cell 63710 (the NE exit, 9.25%).
* Arm tcne (03:43 UTC): spine = tce line [0, 447] + GE8 chain to 192980 + GE9 chain to 63710
  (stitch_spine, 2,098 states, md5 88b6c7ce, end 9.25%); depth rule (success d <= 27,912 = 8.92%),
  bonus 50, milestone 28588, GE8 trough spawns 30%, per-call energy K 8; resumed from tcpump's
  8.2 snapshot (ckpt_1152385024).
* **tcne result (1.152B -> 1.552B): 9.647 (1,455M), the round's best.** Evals 8.15 / 8.53 / 8.72 /
  8.78 / 8.85 / 8.84 / **9.65** / 9.53 (medians 8.4-9.5%). The depth-rule curriculum walked the
  whole 2,098-state spine back to the true start (tau 0 at ~98.7% success) in ~400M steps - no
  stall this time. The greedy line (10.4-10.6 s): platform -> SW drop -> floor (E 599) -> trough
  north -> climbs the east side at the north end -> ONE pump swing back south -> north again at
  812 u/s (E ~750) -> **exits north-east** -> onto *18 at (-1,253, -924) at 9.3-9.65%. It learned a
  one-cycle version of GE8's 15.7 s wander. Hand-off ckpt_1454374912.pt (md5 9c473627), pulled
  by hand.
* Next cycle (04:00 UTC): GE10 rooted at 47 states of tcne's own greedy line's last 3 s (ticks
  752-1032, before the fall; runs/uf2r1b2_lines/ne_tail_roots.npy), 15 min; tcne2 = tcne's final
  + the same spine at tau 0 (consolidation) on the GPU meanwhile.
