# 09 — Training Basis & First Task (speed-on-ramp)

## The extensibility basis (what makes future rewards cheap)

1. **Python reward hook** (`SurfVecEnv(reward_fn=...)`): rewards are computed
   in Python from batched arrays — `fn(prev_obs, obs, terminal_obs,
   base_rewards, done, trunc, core) -> (N,) f32`. The C core's spline-progress
   reward arrives as `base_rewards` (zero when no waypoints are set), so any
   mix of C-progress + Python terms composes without touching C. Contract
   detail that matters: on ended envs `obs` is already the next episode's
   first obs (same-step autoreset) — final-tick values come from
   `terminal_obs`, and `prev_obs` never leaks across episodes.
2. **Spawn pool** (`spawn_mode=2` + `surf_set_spawn_pool`): resets copy a
   random entry from a Python-authored array of full `SurfState`s (origin,
   yaw, velocity), with the config yaw jitter applied. Any curriculum is now
   "build a different pool": ramp faces today, ramp-approach states, spline
   progress points, or replayed human checkpoints later.
3. **`ramp_spawn_pool`** scans the BSP for surfable faces (0.35 < n_z < 0.68,
   80u+ of air above) and — the important part — **auditions** every
   candidate by simulating 80 no-input ticks and keeping only spots that
   actually slide (≥120 u/s). surf_ski_2 yields 568 audited spawns.

## Task 1: maximize speed (this doc's run)

- Spawn: pool of 568 ramp faces, 30u above, facing down-slope, zero velocity,
  ±8° yaw jitter. Episode: 500 ticks (5 s), `water_fail=1`.
- Reward: `SpeedReward` — per-tick horizontal-speed delta ×0.01. Telescopes
  to (final − spawn) speed: literally "maximize speed at the horizon", but
  dense. Passive baseline is ~0 (a no-input slide reaches the trough in ~2 s
  and friction erases it), so all reward is skill.
- PPO (SB3): 512 envs, n_steps 128, batch 16k, γ 0.995, net 256×256,
  ent 0.005. ~78k steps/s end-to-end on the 5090 (torch cu128 build —
  the stock cu121 wheel predates sm_120 and silently falls back to CPU).
- `train_speed.py` records a greedy trajectory every N steps into
  `runs/<run>/traj_*.jsonl` — drag into the viewer to literally watch it
  learn. `eval/final_speed` in TensorBoard is the headline metric.

## Next rungs (in order, each a small delta on this basis)

1. **Speed + survive**: same reward, longer horizon (15–30 s) — forces ramp
   transfers instead of one-ramp milking.
2. **Route**: author waypoints in the viewer → `ProgressPlusSpeedReward`
   (C progress + speed term), spawn pool from spline points
   (`spawn_mode=1` already does this) — the full surf-run task.
3. **Completion bonus shaping**, jail-avoidance margins, style terms
   (e.g. penalize ground contact) — all pure `reward_fn` edits.
4. Multi-map pools; harder maps; eventually the ReHLDS deployment gate
   (docs/05 tier 2).
