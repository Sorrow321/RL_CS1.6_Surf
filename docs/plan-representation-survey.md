# How to pass a goal / plan to the low-level policy - literature survey (2026-09-24)

Two survey agents, launched after the blue200 result that a fan-conditioned executor converges only
~15% faster than a plan-free policy rewarded for progress along the same fixed line (ledger
2026-09-24). Scope 1: robotics, driving, drones, racing games, legged / marine robots. Scope 2:
goal-conditioned RL, hierarchical RL, path/trajectory-conditioned policies. Numbers were read from the
papers; [inferred] marks the agents' own reading. Most fusion head-to-heads are IMITATION learning;
the only RL one is Chaplot 2018.

## Our representation at the time

`--goal-obs fan`: the nearest plan point + 8 points at arc offsets speed x {0.25..2.0} s, in the view
frame (forward, left, up), each divided by its nominal distance, clipped: 27 scalars concatenated to
the MLP input. `fanline`: + the next 12 plan vertices as ~1.5 px dots in an extra depth channel.

## Findings

1. **A plan input pays off when the plan VARIES; on one fixed route a plan-free policy learns the
   route.**
   - Song et al. IROS 2021, drone racing, next N = 1/2/3 gates:
     - fixed track: lap 8.20 / 8.14 / 8.16 s;
     - 1,000 random tracks: crashes 23.0 / 2.5 / 2.3%.
   - Vasco et al. RLC 2024 (GT7): an actor with no course points matched GT Sophy's lap times (its
     critic got them).
   - Codevilla et al. ICRA 2018: a goal vector barely beat no conditioning (24/30 vs 20/26%
     success).
   - => our fixed-route comparison could not show much.
2. **Dense, long, time-indexed point sequences are the best-evidenced track encoding.**
   - GT Sophy (Nature 2022): 3 lines (left edge, centre, right edge) x 60 ego-frame 3-D points,
     span ~6 s at current speed. Maggiore lap: points 114.47 s vs wall-lidar+curvature 117.11 vs
     projected position >130.
   - Horizon:
     - Vasco: a 2 s span was "unreliable", 4 s at least as good as 6 s;
     - DATT (CoRL 2023, quadrotor): tracking error 0.240 m with a 0.3 s horizon, 0.055 at 0.6 s,
       0.101 at 1.2 s; 1 point fails; a world-frame reference fails, body frame works.
   - Our fan: 9 points over 2 s - the short end.
3. **How the plan is FUSED matters more than how it is encoded.**

   | study | fusion comparison | result |
   |---|---|---|
   | Chaplot AAAI 2018 (A3C, first-person ViZDoom) | gated attention (instruction gates conv channels) vs concat | 0.83 vs 0.24 hard, 0.73 vs 0.12 zero-shot |
   | FGPrompt NeurIPS 2023 (image-goal nav, RL) | late / FiLM / early | 13.0 / 77.3 / 78.9% |
   | Monaci 2025 | late vs channel-stacked | 13.8 vs 83.2% |
   | NoMaD | attention + goal mask / early CNN / late CNN | 98 / 68 / 52% |
   | GNM | joint obs+goal encoder vs channel-stacked vs siamese | 1.0/0.95 vs 0.52/0.72 vs 0.73/0.26 |
   | ViNT | late fusion | "often ignoring the goal entirely" |
   | Codevilla | command-selected output branches / command as input | 88/64 / 78/52% |
   | Haro 2026 (legged nav) | learned-query attention over 15 waypoint tokens vs concat | attention learned fastest |

4. **Paths in images: a connected, time-coded curve in its own channel; the sparse-dot version is
   the weak one.**
   - RT-Trajectory (ICLR 2024, IL; path drawn on a blank canvas, red = normalised time, green =
     height, channel-stacked, new first-conv weights zero-initialised): 2.5D 67% / 2D 50% / goal
     image 26% / language 17%.
   - HAMSTER (ICLR 2025): separate path channels 1.00/0.98 vs drawn over RGB 0.83/0.73 (10 trials).
   - CoordConv (NeurIPS 2018): plain convs are bad at reading coordinates from sparse pixels; x,y
     coordinate channels fix it.
   - But with the SAME content, vectors match or beat rasters at far lower compute:
     - PlanT: driving score 81.4 vs 77.5;
     - VectorNet: 3.67 vs 4.49 m endpoint error at 0.04 vs 10.6 GFLOPs.
   - No RL-control study compares a path drawn into the ego image with vector points.
5. **The objective matters more than the representation.**
   - Progress / goal rewards beat tracking rewards:
     - Song 2023, drone racing: RL success 100% vs 44% / 0% for MPC tracking;
     - Rudin IROS 2022: final-position reward crossed 1.2 m gaps vs 0.65 m for continuous tracking;
     - Haro: a path-following reward collapses on degraded paths.
   - We already pay progress along the plan.
6. **Force the plan to matter.**
   - Train on MANY varied plans: HER / DFP - DFP with a random goal each episode lost little on the
     fixed goal (31.5 vs 33.6 frags) and won 77.8 vs 2.7 on a new one.
   - Plan dropout / masking: NoMaD p = 0.5, AlphaStar zeroed its plan 10% of the time.
   - Auxiliary predictions of plan-relative futures: DFP 22.6 vs 5.0 frags; Ye et al. CoRL 2020
     aux tasks reached PPO PointNav's baseline 5.5x faster.
   - Train with PERTURBED plans before a learned planner feeds imperfect ones:
     - Haro: 0.70-0.89 success vs 0.18-0.29 without noise;
     - Yu RA-L 2025: 80 vs 54%.
7. **For the future learned planner: raw relative 3-D subgoal positions (+ a time budget)**, not
   learned latents.
   - HIRO (NeurIPS 2018): raw relative positions 0.99 / 0.92 / 0.66 on AntMaze / Push / Fall vs
     0 / 0 / 0 with FuN-style latent goals.
   - Nachum ICLR 2019: learned goal spaces "completely fail" beyond the point mass.
   - Rudin / HiTS: position + time budget.
   - The measured benefit of hierarchy is mostly EXPLORATION (Nachum et al. 2019, "Why does
     hierarchy (sometimes) work") - CLAUDE.md 0c's mission.

## Candidate arms (generic: constants in seconds, identical on every map)

- **P0, the protocol.** Compare representations on VARYING plans:
  - the stage-1 random-target mix, or planner subgoals;
  - plus zero-shot on a held-out map.
  - The fixed start->finish route cannot separate them (finding 1).
  - Prerequisite: a varied-plan setup that converges. srR200f's (80% random / 20% finish, arc scale
    100 per 90,000 u) did not; the untested candidate is the same mix at the finish-plan scale
    (`--goal-kcap` ~6.66).
- **P1, "Sophy fan".**
  - ~30 points every 0.2 s out to 4-6 s + the nearest point, in the VELOCITY frame (the view frame
    differs while air-strafing [inferred]).
  - Each point as [unit direction, log(1 + d/d0)], plus the plan tangent.
  - A shared per-point MLP + learned-query attention pooling. No rendering.
  - Tensor shapes change: from scratch only.
- **P2, fusion.** FiLM / gating of the last conv blocks by the fan embedding, zero-initialised (the
  arm starts as today), keeping the concat.
- **P3, force use.** Plan dropout (mask 25-50% of episodes) + an auxiliary head predicting arc
  progress at the fan offsets.
- **P4 (only if the plan must stay in the image).**
  - A connected 2-3 px polyline, depth-tested, in two channels: time-to-reach and height relative to
    the agent.
  - CoordConv channels; zero-initialised new first-layer weights.
  - ~10-19% throughput, like the potential channel.

## Sources

Scope 1 (robotics, driving, drones, racing):
- GT Sophy <https://www.nature.com/articles/s41586-021-04357-7>
- Vasco 2024 <https://arxiv.org/abs/2406.12563>
- Fuchs 2021 <https://arxiv.org/abs/2008.07971>
- Imamura 2021 <https://arxiv.org/abs/2111.06449>
- Linesight <https://github.com/Linesight-RL/linesight>
- Remonda 2019 <https://arxiv.org/abs/2104.11106>
- TC-Driver <https://arxiv.org/abs/2205.09370>
- Cai 2020 <https://arxiv.org/abs/2001.01377>
- Swift <https://www.nature.com/articles/s41586-023-06419-4>
- Song 2021 <https://arxiv.org/abs/2103.08624>
- Penicka 2022 <https://arxiv.org/abs/2203.15052>
- DATT <https://arxiv.org/abs/2310.09053>
- Loquercio 2021 <https://arxiv.org/abs/2110.05113>
- Yu 2025 <https://arxiv.org/abs/2512.09571>
- DD-PPO <https://arxiv.org/abs/1911.00357>
- ViNT <https://arxiv.org/abs/2306.14846>
- FGPrompt <https://arxiv.org/abs/2310.07473>
- Monaci 2025 <https://arxiv.org/abs/2507.01667>
- Haro 2026 <https://arxiv.org/abs/2603.13888>
- Rudin 2022 <https://arxiv.org/abs/2209.12827>
- CIL <https://arxiv.org/abs/1710.02410>
- Hidden Biases <https://arxiv.org/abs/2306.07957>
- Hecker 2018 <https://arxiv.org/abs/1803.10158>
- Roach <https://arxiv.org/abs/2108.08265>
- ChauffeurNet <https://arxiv.org/abs/1812.03079>
- PlanT <https://arxiv.org/abs/2210.14222>
- VectorNet <https://arxiv.org/abs/2005.04259>
- Meyer 2020 <https://arxiv.org/abs/1912.08578>
- Havenstrom 2021 <https://arxiv.org/abs/2006.09792>
- RT-Trajectory <https://arxiv.org/abs/2311.01977>
- HAMSTER <https://arxiv.org/abs/2502.05485>
- Geles 2024 <https://arxiv.org/abs/2406.12505>
- Song 2023 <https://arxiv.org/abs/2310.10943>
- TraceVLA <https://arxiv.org/abs/2412.10345>

Scope 2 (goal-conditioned and hierarchical RL):
- UVFA <http://proceedings.mlr.press/v37/schaul15.html>
- HER <https://arxiv.org/abs/1707.01495>
- BVN <https://arxiv.org/abs/2204.13695>
- Contrastive RL <https://arxiv.org/abs/2206.07568>
- DFP <https://arxiv.org/abs/1611.01779>
- HIRO <https://arxiv.org/abs/1805.08296>
- Nachum 2019 <https://arxiv.org/abs/1810.01257>
- FuN <https://arxiv.org/abs/1703.01161>
- HAC <https://arxiv.org/abs/1712.00948>
- HRAC <https://arxiv.org/abs/2006.11485>
- LESSON <https://openreview.net/forum?id=wxRwhSdORKG>
- Director <https://arxiv.org/abs/2206.04114>
- HIQL <https://arxiv.org/abs/2307.11949>
- HiTS <https://arxiv.org/abs/2112.03100>
- DTSIL <https://arxiv.org/abs/1907.10247>
- Peng 2020 <https://arxiv.org/abs/2004.00784>
- CoordConv <https://arxiv.org/abs/1807.03247>
- Gated-Attention <https://arxiv.org/abs/1706.07230>
- GNM <https://arxiv.org/abs/2210.03370>
- NoMaD <https://arxiv.org/abs/2310.07896>
- GEECO <https://arxiv.org/abs/2003.08854>
- Ye 2020 <https://proceedings.mlr.press/v155/ye21a.html>
- AlphaStar <https://www.nature.com/articles/s41586-019-1724-z>
- Why hierarchy works <https://arxiv.org/abs/1909.10618>
- InfoBot <https://arxiv.org/abs/1901.10902>
