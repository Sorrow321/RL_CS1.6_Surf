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

## uf2r1b1 (round 1, slot 1) - 2026-10-30 21:40 UTC start

Box 58455319: RTX 5080 + i9-14900K (32 threads, cgroup quota 30.7 CPUs), 0.38 $/h, Utah.

### The grader's field along the record (ANALYSIS ONLY, to read results)
`autoresearch/analysis/uf2_ge_report.py` reads a GE archive on config.field; the record on the
same field (goalc_32, d0 30,647.5), t from the start curtain: running-min progress 1.76% at t+1.5
(the platform's edge), then the pit: progress -7.56% at t+3.5 (d 32,965), back to 0.85% at t+6,
**5.48% at t+8.0 (out of the pit, 1,104 u/s, z 354)**, 8.7% at t+9.5, 10.2% at t+10, 15% at t+11,
31% at t+15, 48% at t+20. E = z + |v|^2/1600: 1,306 at the curtain, 1,361 at t+1 (air strafe off
the edge), 1,204 at t+3, 1,047-1,145 from t+4.5 to t+8.

### The start, in our core (analysis, 21:50-22:00 UTC)
* Kill geometry (BSP triggers, all fatal under --teleport-fail): `*18` recomeco is a 1-u SHEET at
  z 160-161 over x -3,456..-896, y -2,400..-448 (north of the platform, under the record's pit-exit
  flight, which passes its WEST edge at x < -3,456 before dropping below it at t+10); `*7`
  recomeco's AABB covers the pit and the space under the platform's edges up to z 577 (the pit
  floor's kill and the catches under the platform); `*8` a sheet at z -768 further north.
* **The trap**: the field (free-space geodesic) runs over the `*18` sheet. The scratch control's
  greedy at 76M (`uf2r1b1_ctl`, score 4.910) walks off the platform's east side and dies on the
  sheet at (-1,767, -2,190, 198) at 4.7 s, 4.9% progress, 9/9 identical; E 1,228 -> 575.
* pb_r26 zero-shot (greedy, 9 eps, recorded on the box): drops off the platform's south side
  and dies in `*7`'s catch under it at z ~505-590 after 6-13 s, 2.2-2.9% (score 2.860).
* Go-Explore from the true spawn, pbf config at 10 ms (`--exact-chains --cell 64 --cell-energy 32
  --cell-heading 1 --view-follow=-90..90 --select-energy 3 --act-every 4 --decisions 100 --envs
  4096`, 12 threads beside the control; `runs/ge/uf2a` on the box): 672k cells in 10 min. It
  matches the record's energy down to the south ramp (same-motion E at t+1 / t+2: 1,294 / 1,182
  vs the record's 1,361 / 1,272), then its pit states are FREE FALLS (z -1,000..-800 with E up
  to 1,233 but vz -1,808; same-motion E at t+3 / t+3.5 / t+4.5: 705 / 189 / 38 vs 1,204 /
  1,201 / 1,113): uniform bursts do not surf the bowl. Its deepest cells (8.5%) are falls onto
  the `*18` sheet north of the platform (E 250-900). No pit exit, nothing past the sheet.
