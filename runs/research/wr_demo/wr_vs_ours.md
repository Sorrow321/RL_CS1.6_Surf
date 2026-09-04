# WR demo vs our best finish - surf_src_cannonball

Demo `surf_src_cannonball_world_record.dem`: HLDEMO protocol 5 / net 48, map
`surf_src_cannonball_b5`, 72.45 s playback, 9,518 usercmd frames at 7-8 ms
(131 fps). Tools: `tools/demo/parse_hldemo.py` (dumps our jsonl format),
`tools/demo/compare_wr.py` (this analysis; all tables in `tables.md`; plots
`speed_vs_arc.png`, `map_views.png`, `ramps.png`, `control.png`). Ours =
xQR32 traj_7405830144 ep 0; champ = sISV_par2 traj_8454144000 ep 8.

**Physics of the record** (demo movevars): gravity 800, airaccelerate 100,
accelerate 5, friction 4, edgefriction 2, stopspeed 75, maxspeed 320, stepsize
18, bounce 1 - ours. The block says `sv_maxvelocity 3500`, but the recorded
velocity saturates per component at 4,000 (124 frames above 3,999), so the cap
in force was 4000 = ours. Geometry matches our BSP: 0 of 1,803 sampled origins
in solid; all 20 ramp contacts >= 50 ms find a surface 24-26 u along the
implied normal with the same normal within 5 deg.

## 1. Timing

run | start curtain | finish curtain | zone clock | native clock
---|---|---|---|---
WR | 1.810 s | 70.411 s | **68.60 s** | 72.45 s of playback
ours | 0.984 s | 77.862 s | **76.88 s** | 77.86 s from spawn
champ | 1.033 s | 81.373 s | **80.34 s** | 81.37 s from spawn

Our recorder drops the terminal state; the finish is the last row extrapolated
12-13 ms at its own velocity. Gap **8.28 s**. Path 226.7 ku (WR) vs 232.0 ku.

## 2. Where the 8.28 s goes (route deciles, zone clock)

deciles | WR | ours | gap | WR v | ours v
---|---|---|---|---|---
1 (0-23 ku) | 12.14 | 12.44 | 0.30 | 1907 | 1862
2-3 | 14.37 | 15.23 | 0.86 | 3224 | 3043
4-7 | 25.49 | 28.99 | 3.50 | 3643 | 3204
8-9 | 12.64 | 13.64 | 1.00 | 3675 | 3400
10 (to finish) | 5.75 | 7.56 | **1.81** | - | -

Faster everywhere (0.3-0.9 s per decile; mean 3,304 vs 3,019 u/s, max 4,641
vs 4,209) plus a 1.8 s block in the finish room.

## 3. Ramps

Same ramps, same time on them: WR 20 contacts (19.1% of the run), ours 24
(20.5%); free flight 79.4% vs 79.5%. What differs is the cost per contact
(energy destroyed by PM_ClipVelocity, M = 1e6 u^2/s^2):

arc (ku) | ours v_in->v_out, loss | WR v_in->v_out, loss
---|---|---
49.0 | 3667->3389, 0.69 | 3903->3650, 0.50
139.6 | 3567->3644, 0.20 in 0.58 s | 4067->4076, 0.15 in 0.18 s
181.6 (pit) | 4209->3975, **1.04**, z_min -5809 | 4641->3984, 0.27, z_min -5362
212 (finish room) | 3708->3720, 0.41 + 0.44 second touch | 3699->3251, 1.10
whole run | **6.09** | **4.14**

The WR arrives 300-500 u/s faster at most ramps and touches them shorter and
higher.

## 4. Finish room - the "last ramp"

Nobody uses ramp 2 (under the finish wall, y 6500-7400, up to the platform
at z -1824): all three fly 1 -> finish; the WR clears the platform by 66 u.

run | dive | ramp-1 phase | exit v / vz | flight | finish z
---|---|---|---|---|---
WR | 2.57 s | **0.73 s**, 1 touch, x -9657 -> -9727 | 3251 / +2208 | 4.00 s | -1758
ours | 2.37 s | **3.00 s**, 3 touches, x -9666 -> -13024 | 3171 / +2099 | 3.84 s | -1546
champ | 2.39 s | 2.59 s, 2 touches, x -9567 -> -13202 | 3306 / +2362 | 4.25 s | -1323

The WR bottoms out at z -4995 and the quarter-pipe turns it around on first
contact; ours bottoms out 275 u deeper, slides 3,360 u along the pipe and
launches at x -13024 with the same exit speed 2.3 s later. The curtain spans
x -14720..-8064, so the WR's x -9679 crossing is inside our corridor; the
champion line has the same slide.

## 5. Control (timed window; pitch negative = down)

quantity | WR | ours
---|---|---
pitch mean / p10 / p90 | -16.4 / -33.5 / +0.7 | -25.5 / -54.6 / -1.2 (champ -34.5 / -66.9 / -0.5)
yaw rate median / p90 / max (deg/s) | 51 / 117 / 612 | 59 / 156 / 1000
A/D held; W/S held | 97.7%; **0%** | 92.1%; 11.4%
strafe hold median / mean; flips per s | 0.42 / 1.06 s; 0.42 | 0.09 / 0.32 s; 1.50
wishdir-velocity angle in 89.5-90.5 deg | **82.6%** | 53.4%
angle < 85 (no gain) / > 90.5 (braking) | 0.2% / 6.0% | 15.5% / 13.4%
energy per braking step (u^2/s^2) | -691 | -2467
jump presses (held) / duck presses (held) | 0 / 0 | 283 (36%) / 176 (29%)

Energy budget (M, WR / ours): gravity 9.82 / 9.66; air-strafe gain +2.36 /
+1.22, braking -1.22 / -1.86 (net **+1.15 / -0.65**); ramps 4.14 / 6.09;
maxvelocity clamp **3.55 / 0.11**; KE at finish 3.47 / 3.30 (closes to
0.2-0.4 M, the gravity discretisation).

## 6. Where our agent loses time

1. **Strafe geometry (reachable, largest).** The human keeps the wish
   direction within +-0.5 deg of perpendicular to velocity on 83% of
   free-flight steps; ours 53%, with 13% of steps past perpendicular (braking
   at 3.6x the human's cost per step) and 16% under 85 deg (zero gain). Net
   air-strafe energy +1.15 vs -0.65 M; the 1.8 M swing is twice the kinetic
   energy behind the 9.4% mean-speed gap.
2. **W key (reachable).** Held on 11.4% of our steps, 7.2% with A/D (a 45 deg
   wishdir accelerates nothing at these speeds). Never by the human.
3. **Key cadence (reachable).** Human: one strafe key 0.42 s median / 1.06 s
   mean, 0.42 flips per s. Ours: 0.09 / 0.32 s, 1.5 flips per s.
4. **Jump/duck (reachable).** Zero presses by the human in 68 s; ours 283
   jump and 176 duck. Duck shifts the hull 18 u at every ramp contact.
5. **Ramp entries (partly reachable).** 47% more energy destroyed on the same
   ramps; the pit at 181.6 ku costs 1.04 vs 0.27 M because ours hits it 219 u
   lower. The WR arrives faster and higher, so this compounds with 1-3.
6. **Finish-room line (line rebuild, inside our corridor).** Turn up on the
   first contact at x ~ -9650 instead of sliding to -13024: 2.3 s, the whole
   last-decile gap, bought with 0.7 M more energy at that ramp.
7. **Pitch (sensor only).** Ours looks 25 deg down on average (p10 -55), the
   human 16 (p10 -34). Pitch moves nobody (our pm_tick gets pitch 0; GoldSrc
   sidemove uses the pitch-free right vector); it only changes what the lidar
   sees.
8. **Not reachable.** The WR is pinned at the 4,000 u/s component clamp 1.3%
   of its time and throws 3.55 M away there; we are 3.8 M short of reaching
   it. Its 131 fps also gives 31% more air-accelerate steps per second than
   our 100 Hz sim (the 30 u/s cap is per usercmd).

## Correction (2026-09-04 evening, route-level search)

Measured against the planner's best line at 131 Hz, two claims above do
not hold: the record takes the SAME ramps (21 matched contacts, only
entry speed and height differ) and does NOT skip the last ramp (neither
line touches finish-room ramp 2). The lines are within 1,309 u over the
whole route. The gap is 3.70 s of strafe cadence with no line separation
plus 1.42 s in the finish-room bowl, where the record rides the near wall
(-140 deg of bearing in 0.38 s, one touch) and ours slides the floor
(3,357 u, 3 touches). See docs/research-results.md, "there is NO route
alternative on cannonball".
