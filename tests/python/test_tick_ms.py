"""--tick-ms: the physics tick as a variable (surfgym.tick, core.py, the
trainer's conversions, the recorder's time base, the honesty tools).

Why: GoldSrc's air-accelerate impulse saturates at 30 u/s PER FRAME (pm.c
PM_AirAccelerate), so a strafer's acceleration is proportional to the frame
rate; the cannonball WR demo runs 7/8 ms frames (7.63 ms = 131 fps), 31%
more air-accelerate steps per second than our 10 ms tick. What is pinned:

1. ``tick_ms=10.0`` is BYTE-IDENTICAL to today: the config dump, a 200-tick
   scripted rollout (obs rows AND states) and the recorded trajectory file.
2. 7.63 ms is realised as the shortest integer pattern within 0.05 ms
   (``[8, 8, 7]`` = 7.667 ms), the core cycles it, and the same scripted
   3-second air strafe collects ~1.30x the impulses and MORE speed than at
   10 ms (the ratio is printed; the v^2 gain tracks the impulse count, the
   speed gain is smaller because each impulse adds 30^2/(2v)).
3. gamma / stall / respawn margin / goal kmin,kmax / snap cadence / time
   penalty land on the SAME SECONDS at both ticks, and reduce to the legacy
   ``* 100.0`` / ``/ 100.0`` bit for bit at 10 ms.
4. A trajectory recorded at the pattern is timed correctly by
   tools/eval_honesty.py (a synthetic 131-fps episode reads 3.0 s, not
   3.9 s), and a header without ``tick_ms`` is refused, never assumed.
5. A checkpoint trained at 10 ms resumed with ``--tick-ms 7.63`` prints the
   TICK TRANSFER notice, keeps the episode cap in seconds, runs, and writes
   both ticks into run.json (CPU, tiny nets; ~minutes).

    python -m pytest tests/python/test_tick_ms.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.tick import (TickClock, episode_seconds, header_fields,  # noqa: E402
                          header_tick_ms, step_seconds, tick_pattern)

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
ROUTE = ROOT / "maps" / "surf_src_cannonball.route.npz"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")

needs_core = pytest.mark.skipif(not (CANNONBALL.exists() and DLL.exists()),
                                reason="needs the built core + cannonball")


def _core(tick_ms=None, **over):
    from surfgym import SurfCore, default_config
    cfg = default_config(**over)
    return SurfCore(str(CANNONBALL), cfg, tick_ms=tick_ms) if tick_ms is not None \
        else SurfCore(str(CANNONBALL), cfg)


# ==========================================================================
# the pattern
# ==========================================================================
def test_pattern_is_the_shortest_within_tolerance():
    assert tick_pattern(10) == [10]
    assert tick_pattern(10.0) == [10]
    assert tick_pattern(8) == [8]
    assert tick_pattern(7.5) == [8, 7]
    assert tick_pattern(7.63) == [8, 8, 7]          # the WR demo's mean
    assert tick_pattern(7.6) == [8, 8, 7, 8, 7]
    tc = TickClock(7.63)
    assert abs(tc.ms - 7.63) <= 0.05 and abs(tc.ms - 23.0 / 3.0) < 1e-12
    assert abs(tc.hz - 130.43) < 0.01
    # n=1 (7 or 8) misses by >= 0.37, n=2 (7.5) by 0.13: 3 is the shortest
    assert abs(7.5 - 7.63) > 0.05 and abs(8 - 7.63) > 0.05
    with pytest.raises(ValueError):
        tick_pattern(0.5)
    with pytest.raises(ValueError):
        tick_pattern(60)


def test_reference_clock_is_the_legacy_arithmetic_bit_for_bit():
    tc = TickClock(10.0)
    assert tc.is_reference and tc.pattern == [10] and tc.ms == 10.0
    for s in (15.0, 30.0, 10.0, 2.0, 0.25, 1.0, 5.0, 0.07, 12.345):
        assert tc.secs_to_ticks(s) == int(s * 100.0)
        assert tc.secs_to_ticks(s, "round") == int(round(s * 100.0))
    for n in (1500, 12000, 7, 3):
        assert tc.ticks_to_secs(n) == n / 100.0
    assert tc.gamma(0.9995) == 0.9995
    assert tc.per_tick(0.005) == 0.005
    assert tc.per_tick(32.0) == 32.0


# ==========================================================================
# 3. the conversions land on the same seconds
# ==========================================================================
def test_conversions_keep_seconds_at_7_63():
    ref, tc = TickClock(10.0), TickClock(7.63)
    # stall window: 15 s = 1500 ticks at 10 ms, 1956 ticks at 7.667 ms
    assert ref.secs_to_ticks(15.0) == 1500
    st = tc.secs_to_ticks(15.0)
    assert st == 1956 and abs(tc.ticks_to_secs(st) - 15.0) < tc.ms / 1000.0
    # respawn margin 10 s, goal band 1 s / 5 s, snapshot cadence 1 s / 0.25 s
    for secs, legacy in ((10.0, 1000), (2.0, 200), (1.0, 100), (5.0, 500),
                         (0.25, 25)):
        assert ref.secs_to_ticks(secs, "round") == legacy
        n = tc.secs_to_ticks(secs, "round")
        assert abs(tc.ticks_to_secs(n) - secs) <= 0.5 * tc.ms / 1000.0
        assert abs(n / legacy - 10.0 / tc.ms) < 0.02   # 1.304x the ticks
    # gamma: the same horizon in seconds -> the same discount over 20 s
    g10, g131 = ref.gamma(0.9995), tc.gamma(0.9995)
    assert g10 == 0.9995 and 0.9995 < g131 < 1.0
    n10, n131 = 20.0 / 0.010, 20.0 / (tc.ms / 1000.0)
    assert abs(g10 ** n10 - g131 ** n131) < 1e-9
    assert abs(g131 - 0.9995 ** (tc.ms / 10.0)) < 1e-15
    # per-tick amounts: the same per second
    assert abs(tc.per_tick(0.005) * 1000.0 / tc.ms - 0.005 * 100.0) < 1e-12
    assert abs(tc.per_tick(32.0) - 32.0 * tc.ms / 10.0) < 1e-12
    # the finish clock inside RaceReward: ticks -> seconds at its tick
    from surfgym.rewards import RaceReward
    assert RaceReward.stats_from_vector(
        np.array([1, 0, 0, 0, 1500.0, 1.0]))["finish_s"] == 15.0
    assert abs(RaceReward.stats_from_vector(
        np.array([1, 0, 0, 0, 1956.0, 1.0]), tick_ms=tc.ms)["finish_s"]
        - 15.0) < 0.01


# ==========================================================================
# the recorder's time base
# ==========================================================================
def test_episode_seconds_is_exact_under_a_pattern_and_refuses_without_tick():
    assert header_fields(10.0, [10], 0) == {"tick_ms": 10}
    h = header_fields(23.0 / 3.0, [8, 8, 7], 0)
    assert h == {"tick_ms": 7.666667, "tick_pattern_ms": [8, 8, 7],
                 "tick_phase": 0}
    assert episode_seconds({"tick_ms": 10}, 300) == 3.0
    assert episode_seconds(h, 391) == (130 * 23 + 8) / 1000.0
    h1 = dict(h, tick_phase=1)
    assert episode_seconds(h1, 391) == (130 * 23 + 8) / 1000.0
    h2 = dict(h, tick_phase=2)
    assert episode_seconds(h2, 391) == (130 * 23 + 7) / 1000.0
    assert step_seconds(h2, 4) == [0.007, 0.008, 0.008, 0.007]
    assert header_tick_ms({"tick_ms": 7.666667}) == 7.666667
    for bad in (None, {}, {"map": "x"}):
        with pytest.raises(ValueError, match="tick_ms"):
            episode_seconds(bad, 10)
        with pytest.raises(ValueError, match="tick_ms"):
            header_tick_ms(bad)


# ==========================================================================
# 1. tick_ms 10 is byte-identical
# ==========================================================================
def _scripted_rollout(core, n=200, seed=3):
    from surfgym import ScriptedStrafer
    pol = ScriptedStrafer(core, period_ticks=37)
    obs = core.reset(seed).copy()
    rows, sts = [obs], [core.get_states()]
    for _ in range(n):
        obs, *_ = core.step(pol.act(obs))
        rows.append(obs.copy())
        sts.append(core.get_states())
    return np.stack(rows), sts


@needs_core
def test_tick_10_is_byte_identical(tmp_path):
    from surfgym import config_to_dict
    from surfgym.record import record_rollout
    from surfgym import ScriptedStrafer
    kw = dict(num_envs=4, spawn_mode=0, max_episode_ticks=10000,
              lidar_w=8, lidar_h=4)
    plain, tick = _core(None, **kw), _core(10.0, **kw)
    assert config_to_dict(plain.config) == config_to_dict(tick.config)
    assert tick.tick_pattern == (10,) and tick.tick_ms == 10.0
    assert tick._tick_pat is None            # the per-step setter is never armed
    o0, s0 = _scripted_rollout(plain)
    o1, s1 = _scripted_rollout(tick)
    assert np.array_equal(o0, o1)
    assert all(np.array_equal(a, b) for a, b in zip(s0, s1))
    # the recorded file, byte for byte (header "tick_ms": 10 as always)
    files = []
    for c in (plain, tick):
        out = tmp_path / f"r{len(files)}.traj.jsonl"
        record_rollout(c, ScriptedStrafer(c, period_ticks=37), out,
                       episodes=1, max_ticks=120, seed=0)
        files.append(out.read_bytes())
    assert files[0] == files[1]
    hdr = json.loads(files[0].splitlines()[0])
    assert hdr["tick_ms"] == 10 and "tick_pattern_ms" not in hdr


# ==========================================================================
# 2. the pattern runs, and the strafe gains more at 131 Hz
# ==========================================================================
AIR_POINT = (1305.057, 11282.656, -12140.515)    # measured free-air point


def _free_air_point(core, clear=2500.0, seed=0):
    mins, maxs = core.map_bounds()
    rng = np.random.default_rng(seed)
    cands = [np.asarray(AIR_POINT, np.float32)]
    cands += [rng.uniform(mins + clear, maxs - clear).astype(np.float32)
              for _ in range(4000)]
    dirs = [np.array(d, np.float32) * clear for d in
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    for p in cands:
        if core.point_contents(p) != -1:
            continue
        if all(core.trace(p, p + d, 0).fraction >= 1.0 for d in dirs):
            return p
    pytest.skip("no free air point found on the map")


def _strafe(core, p, n_ticks, v0=250.0):
    """Optimal air strafe: --yaw-adaptive bin 4 (k = -1 -> turn right at
    atan(30/|v|) per tick) with D held; sv_gravity 0 keeps it airborne."""
    from surfgym.core import STATE_DTYPE
    st = np.zeros(1, STATE_DTYPE)[0]
    st["origin"] = p
    st["velocity"] = (v0, 0.0, 0.0)
    st["onground"] = -1
    core.reset(0)
    core.set_state(0, st)
    a = np.array([[4, 3, 1, 2, 0, 0]], np.int32)
    ms = 0
    for _ in range(n_ticks):
        ms += core.tick_pattern[core.tick_phase]
        core.step(a)
    v = core.states_view[0]["velocity"]
    assert int(core.states_view[0]["onground"]) == -1
    return ms, float(np.hypot(v[0], v[1]))


@needs_core
def test_pattern_core_cycles_and_gains_more_speed(capsys):
    kw = dict(num_envs=1, spawn_mode=0, max_episode_ticks=100000, lidar_w=0,
              lidar_h=0, sv_gravity=0.0, yaw_adaptive=1, sv_maxvelocity=4000.0)
    c10, c131 = _core(10.0, **kw), _core(7.63, **kw)
    assert c131.tick_pattern == (8, 8, 7)
    assert abs(c131.tick_ms - 23.0 / 3.0) < 1e-12
    assert c131.config.phys.msec == 8 and c131.tick_phase == 0
    # the core cycles 8, 8, 7, 8, 8, 7, ... one element per batch step
    c131.reset(1)
    seen = []
    for _ in range(7):
        c131.step(np.array([[7, 3, 1, 1, 0, 0]], np.int32))
        seen.append(int(c131.config.phys.msec))
    assert seen == [8, 8, 7, 8, 8, 7, 8] and c131.tick_phase == 1
    c131.reset(1)
    assert c131.tick_phase == 0

    p = _free_air_point(c10)
    n10, n131 = 300, TickClock(7.63).secs_to_ticks(3.0, "round")
    assert n131 == 391
    t10, v10 = _strafe(c10, p, n10)
    t131, v131 = _strafe(c131, p, n131)
    assert t10 == 3000 and abs(t131 - 3000) <= 8
    v0 = 250.0
    gain10, gain131 = v10 ** 2 - v0 ** 2, v131 ** 2 - v0 ** 2
    print(f"\nair strafe 3 s from {v0:g} u/s: 10 ms {n10} impulses -> "
          f"{v10:.2f} u/s | 7.63 ms ({c131.tick_ms:.4f}) {n131} impulses -> "
          f"{v131:.2f} u/s | impulse ratio {n131 / n10:.3f}, speed ratio "
          f"{v131 / v10:.3f}, v^2-gain ratio {gain131 / gain10:.3f} "
          f"(ideal 30^2 per impulse: {900 * n10:.0f} vs {900 * n131:.0f})")
    assert v131 > v10                               # MORE speed per second
    assert abs(n131 / n10 - 1.303) < 0.005          # the impulse count
    assert 1.28 < gain131 / gain10 < 1.33           # v^2 tracks the count
    assert 1.05 < v131 / v10 < 1.20                 # speed: 30^2/(2v) each
    assert gain10 > 0.95 * 900 * n10                # saturated 30 u/s impulses


@needs_core
def test_pattern_recording_header_and_seconds(tmp_path):
    from surfgym import ScriptedStrafer
    from surfgym.record import record_rollout
    from surfgym.route import episodes_from_traj
    c = _core(7.63, num_envs=1, spawn_mode=0, max_episode_ticks=100000,
              lidar_w=0, lidar_h=0)
    out = tmp_path / "p.traj.jsonl"
    record_rollout(c, ScriptedStrafer(c, period_ticks=37), out,
                   episodes=None, max_ticks=391, seed=0)
    lines = out.read_text(encoding="utf-8").splitlines()
    hdr = json.loads(lines[0])
    assert hdr["tick_ms"] == 7.666667
    assert hdr["tick_pattern_ms"] == [8, 8, 7] and hdr["tick_phase"] == 0
    eps, hdrs = episodes_from_traj(out, with_headers=True)
    assert len(eps) == 1 and hdrs[0]["tick_pattern_ms"] == [8, 8, 7]
    assert episode_seconds(hdrs[0], len(eps[0])) == 2.998   # 391 ticks
    eps_plain = episodes_from_traj(out)
    assert len(eps_plain) == 1 and np.array_equal(eps_plain[0], eps[0])


# ==========================================================================
# 4. eval_honesty times a 131-fps trajectory from its header
# ==========================================================================
def _synthetic_traj(path, header, n_rows):
    """n_rows ticks flying along the reference route at ~300 u/s."""
    z = np.load(ROUTE)
    pts = np.asarray(z["route"], np.float64)
    spacing = float(z["spacing"]) if "spacing" in z.files else 128.0
    dt = episode_seconds(header, 1) if "tick_ms" in header else 0.01
    per_tick = 300.0 * dt / spacing      # route vertices advanced per tick
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for t in range(n_rows):
            i = min(int(t * per_tick), len(pts) - 1)
            x, y, zz = (float(v) for v in pts[i])
            f.write(json.dumps([t, x, y, zz, 300.0, 0.0, 0.0, 0.0, 0, 0, 0.0,
                                0.0, 0.0, 1, 1], separators=(",", ":")) + "\n")
        f.write(json.dumps({"end": "fail", "ticks": n_rows,
                            "best_progress": 0.0}) + "\n")


@pytest.mark.skipif(not ROUTE.exists(), reason="needs the cannonball route")
def test_eval_honesty_times_from_the_header(tmp_path):
    base = {"map": "surf_src_cannonball", "phys": {"msec": 8}, "episode": 0}
    h131 = {**base, **header_fields(23.0 / 3.0, [8, 8, 7], 0)}
    _synthetic_traj(tmp_path / "t131.jsonl", h131, 391)
    h10 = {**base, "tick_ms": 10}
    _synthetic_traj(tmp_path / "t10.jsonl", h10, 300)
    cmd = [sys.executable, str(ROOT / "tools" / "eval_honesty.py"),
           "--route", str(ROUTE), "--map", "surf_src_cannonball"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd + [str(tmp_path / "t131.jsonl"),
                              str(tmp_path / "t10.jsonl")],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "tick 7.66667 ms" in out and "tick 10 ms" in out
    # 391 ticks at the pattern = 2.998 s -> "3.0s"; a 10 ms reading would
    # have said 3.9s
    assert "ep0:    3.0s" in out and "3.9s" not in out
    # a header without tick_ms is refused loudly, never timed at 10 ms
    _synthetic_traj(tmp_path / "bad.jsonl", base, 300)
    r = subprocess.run(cmd + [str(tmp_path / "bad.jsonl")],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert r.returncode != 0 and "tick_ms" in (r.stdout + r.stderr)


# ==========================================================================
# 5. resume a 10 ms checkpoint at 7.63 ms (CPU, tiny)
# ==========================================================================
SMOKE_FLAGS = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "64",
               "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
               "--lidar-cell", "32", "--lidar-range", "11500",
               "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
               "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
               "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
               "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
               "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
               "--ep-ticks", "3000", "--time-pen", "0.005",
               "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
               "--race-dist", "geodesic", "--maxvel", "4000",
               "--train-stride", "1", "--yaw-adaptive",
               "--respawn-frac", "0.9", "--respawn-margin", "10",
               "--respawn-reservoir", "1000", "--int-coef", "0.25",
               "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
               "--record-every", "1e12", "--eval-eps", "1",
               "--eval-greedy-only", "--no-eval-at-start"]


def _train(run, extra, timeout=900, drop=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="8", NUMBA_NUM_THREADS="8")
    flags = list(SMOKE_FLAGS)
    for d in drop:                       # a flag + its value
        i = flags.index(d)
        del flags[i:i + 2]
    cmd = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
           "--run", run] + flags + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=timeout)


@needs_core
@pytest.mark.skipif(not (ROOT / "maps" / "surf_src_cannonball.goal_32.npz").exists(),
                    reason="needs the prebaked cannonball goal field")
def test_resume_10ms_checkpoint_at_7_63_prints_the_notice_and_runs():
    import shutil
    run10, run131 = "tick_test_10", "tick_test_131"
    for r in (run10, run131):
        shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)
    r = _train(run10, ["--steps", "4096"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "tick 10 ms (100 Hz, the reference" in r.stdout
    ck = ROOT / "runs" / run10 / "ckpt_latest.pt"
    assert ck.exists()
    meta = json.loads((ROOT / "runs" / run10 / "run.json").read_text())
    assert meta["config"]["tick_ms"] == 10.0
    assert meta["config"]["tick_pattern_ms"] == [10]
    assert meta["config"]["tick_ms_ckpt"] is None
    assert meta["config"]["gamma_tick"] == 0.9995
    assert meta["config"]["ep_secs"] == 30.0

    # no explicit --ep-ticks: the cap is carried over in SECONDS (3000 ticks
    # at 10 ms = 30 s -> 3913 ticks), which is what the notice reports
    r = _train(run131, ["--ckpt", str(ck), "--tick-ms", "7.63",
                        "--steps", "8192"], drop=("--ep-ticks",))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = r.stdout
    assert "!! TICK TRANSFER: this checkpoint was trained at 10 ms" in out
    assert "pattern [8,8,7] ms = 7.6667 ms (130.4 Hz)" in out
    assert "ep_ticks 3000 at 10 ms = 30 s -> 3913 ticks" in out
    assert "decisions every 4 tick(s) = 30.7 ms" in out
    assert "resumed" in out and "step " in out
    meta = json.loads((ROOT / "runs" / run131 / "run.json").read_text())
    c = meta["config"]
    assert c["tick_ms"] == 7.63 and c["tick_ms_ckpt"] == 10.0
    assert c["tick_pattern_ms"] == [8, 8, 7]
    assert abs(c["tick_ms_eff"] - 23.0 / 3.0) < 1e-9
    assert abs(c["gamma_tick"] - 0.9995 ** (23.0 / 30.0)) < 1e-12
    assert abs(c["time_pen_tick"] - 0.005 * 23.0 / 30.0) < 1e-12
    assert c["ep_ticks"] == 3913 and abs(c["ep_secs"] - 30.0) < 0.01
    assert c["gamma"] == 0.9995 and c["time_pen"] == 0.005   # flag values
    for r_ in (run10, run131):
        shutil.rmtree(ROOT / "runs" / r_, ignore_errors=True)


# ==========================================================================
# 6. the planner (tools/beam_tas.py) and the readers of its output time
#    every tick count at the tick the file says
# ==========================================================================
def test_ticks_to_secs_is_exact_under_the_pattern_and_legacy_at_10():
    from surfgym.tick import ticks_to_secs
    # 10 ms: the legacy ticks / 100.0, bit for bit (the planner's
    # summary.json best_s / greedy_s / gain_s were that expression)
    for n in (0, 1, 7395, 7566, 7761, 12000):
        assert ticks_to_secs(n) == n / 100.0
        assert ticks_to_secs(n, 10.0, [10]) == n / 100.0
    # the [8, 8, 7] pattern from phase 0: 9,645 ticks = 3,215 x 23 ms
    pat = [8, 8, 7]
    ms = 23.0 / 3.0
    assert abs(ticks_to_secs(9645, ms, pat) - 73.945) < 1e-9
    assert abs(ticks_to_secs(9646, ms, pat) - 73.953) < 1e-9     # + 8
    assert abs(ticks_to_secs(9647, ms, pat) - 73.961) < 1e-9     # + 8
    assert abs(ticks_to_secs(9648, ms, pat) - 73.968) < 1e-9     # + 7
    # from phase 2 the first tick is the 7
    assert abs(ticks_to_secs(1, ms, pat, phase=2) - 0.007) < 1e-12
    # a mean-only reading (no pattern) is off by up to a tick; the
    # readers pass the pattern the planner saved
    assert abs(ticks_to_secs(9646, ms) - 9646 * ms / 1000.0) < 1e-12
    assert abs(ticks_to_secs(9646, ms) - 73.953) > 1e-6
    # the same count read as 10 ms would be a 30% lie
    assert abs(ticks_to_secs(9645) - 96.45) < 1e-9


def test_planner_stamps_the_tick_and_a_synthetic_result_converts(tmp_path):
    """beam_tas: a 10 ms core's header keys are the legacy ones (so the
    10 ms file is byte-identical), a pattern adds tick_pattern_ms + phase,
    the gravity step is the MEAN tick's, and a synthetic 7.63 ms planner
    result converts to seconds correctly wherever it is read (the planner's
    own stamp, expert_loop.tick_secs, plan_to_bc.load_plans)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import beam_tas
    import expert_loop
    import plan_to_bc

    class _Phys:
        msec, sv_gravity = 10, 800.0

    class _Cfg:
        phys = _Phys()

    class _Core10:              # a 10 ms core: no pattern attributes
        config = _Cfg()

    class _Core763(_Core10):    # the [8, 8, 7] core, mid-pattern
        tick_pattern, tick_phase, tick_ms = (8, 8, 7), 1, 23.0 / 3.0

    assert beam_tas.tick_header(_Core10()) == {"tick_ms": 10}
    assert beam_tas.tick_header(_Core763()) == {
        "tick_ms": round(23.0 / 3.0, 6), "tick_pattern_ms": [8, 8, 7],
        "tick_phase": 1}

    # the header's phys block states the NOMINAL tick, not the phase the
    # core happens to be in (config.phys.msec mirrors the last tick run)
    from surfgym import default_config

    class _Real10:
        config = default_config()                 # msec 10, no pattern

    class _RealMid:                               # a [8,8,7] core after a 7
        config = default_config()
        config.phys.msec = 7
        tick_pattern = (8, 8, 7)

    class _RealNominal(_RealMid):                 # branch tick-consumers' API
        nominal_msec = 8

    assert beam_tas.phys_header(_Real10()) == beam_tas.phys_to_dict(
        _Real10.config.phys)
    assert beam_tas.phys_header(_Real10())["msec"] == 10
    assert beam_tas.phys_header(_RealMid())["msec"] == 8
    assert beam_tas.phys_header(_RealNominal())["msec"] == 8
    assert list(beam_tas.phys_header(_RealMid())) == list(
        beam_tas.phys_to_dict(_RealMid.config.phys))     # key order kept
    # the gravity step: -8 u/tick at 10 ms; the mean tick's -6.133 under
    # the pattern, which free flight (-6.4 / -5.6 per real tick) never
    # departs from by more than the 1 u contact tolerance
    assert beam_tas.gravity_step(_Core10()) == -8.0
    g = beam_tas.gravity_step(_Core763())
    assert abs(g - (-800.0 * 23.0 / 3.0 / 1000.0)) < 1e-12
    assert max(abs(-6.4 - g), abs(-5.6 - g)) < 1.0

    # the summary.json / npz stamp
    tc = TickClock(7.63)
    st = beam_tas.tick_stamp(tc, 10.0)
    assert st["tick_pattern_ms"] == [8, 8, 7] and st["tick_ms_requested"] == 7.63
    assert st["tick_ms_ckpt"] == 10.0 and abs(st["tick_ms"] - 23.0 / 3.0) < 1e-12
    assert beam_tas.tick_stamp(TickClock(10.0), 10.0) == {
        "tick_ms": 10.0, "tick_pattern_ms": [10], "tick_ms_requested": 10.0}
    npz = beam_tas.tick_npz(tc, 10.0)
    assert npz["tick_pattern_ms"].dtype == np.int32
    assert list(npz["tick_pattern_ms"]) == [8, 8, 7]

    # a synthetic planner result: 9,645 ticks at the pattern is 73.945 s
    # (the same count at 10 ms would read 96.45 s); 7,395 ticks in a
    # summary that predates the flag is the old 73.95 s
    s763 = {"best_ticks": 9645, **st}
    assert abs(expert_loop.tick_secs(s763["best_ticks"], s763["tick_ms"],
                                     s763["tick_pattern_ms"]) - 73.945) < 1e-9
    s10 = {"best_ticks": 7395}
    assert expert_loop.tick_secs(s10["best_ticks"], s10.get("tick_ms"),
                                 s10.get("tick_pattern_ms")) == 73.95
    assert expert_loop.tick_secs(7395, 10.0, [10]) == 7395 / 100.0
    assert expert_loop.tick_secs(None) == 0.0

    # plan_to_bc reads the plan's tick (10 when the npz predates the flag)
    # and refuses to pool plans searched at different ticks
    from surfgym.core import STATE_DTYPE
    base = dict(spawn_state=np.zeros(1, STATE_DTYPE),
                obs_start=np.zeros(15, np.float32), gate_seed=np.int32(0),
                act_every=np.int32(3), map=np.str_("m"),
                greedy_ticks=np.int32(0), acts=np.zeros((4, 6), np.int8),
                finish_ticks=np.int32(400))
    np.savez(tmp_path / "old.npz", **base)
    np.savez(tmp_path / "p763.npz", **base, **npz)
    old = plan_to_bc.load_plans([tmp_path / "old.npz"])
    assert (old["tick_ms"], old["tick_pattern_ms"], old["tick_ms_requested"]) \
        == (10.0, [10], 10.0)
    new = plan_to_bc.load_plans([tmp_path / "p763.npz"])
    assert new["tick_pattern_ms"] == [8, 8, 7] and new["tick_ms_requested"] == 7.63
    assert abs(new["tick_ms"] - 23.0 / 3.0) < 1e-12
    with pytest.raises(SystemExit):
        plan_to_bc.load_plans([tmp_path / "old.npz", tmp_path / "p763.npz"])


@needs_core
def test_set_tick_phase_re_phases_the_pattern_and_is_a_no_op_at_10():
    """beam_tas --commit re-centres its population on a state captured
    mid-window; the core's pattern phase must follow the committed tick
    count or the open-loop replay (phase 0 from tick 0) runs different ms."""
    a = np.zeros((1, 6), np.int32)
    core = _core(tick_ms=7.63, num_envs=1)
    core.reset(0)
    assert core.tick_phase == 0
    core.step(a)
    assert core.config.phys.msec == 8 and core.tick_phase == 1
    core.set_tick_phase(2)
    core.step(a)
    assert core.config.phys.msec == 7 and core.tick_phase == 0
    core.set_tick_phase(4)                    # modulo the pattern length
    assert core.tick_phase == 1
    core.step(a)
    assert core.config.phys.msec == 8
    c10 = _core(num_envs=1)
    c10.reset(0)
    c10.set_tick_phase(2)
    assert c10.tick_phase == 0
    c10.step(a)
    assert c10.config.phys.msec == 10
# 6. review-tick regressions: constants the first --tick-ms build got wrong.
#    Every one of them is identity at 10 ms.
# ==========================================================================
def _tick_env(tick_ms, yaw_adaptive):
    """What train_fast.py / record_ckpt.py hand default_config() for a run
    at ``tick_ms``. Stated once here so both call sites are checked against
    the same rule."""
    tc = TickClock(tick_ms)
    if tc.is_reference or yaw_adaptive:
        return {}
    return {"yaw_rate_max_deg": tc.per_tick(10.0)}


@needs_core
def test_yaw_adaptive_keeps_the_reference_ceiling():
    """--yaw-adaptive redefines a yaw bin as K_BINS * atan(30/|v|) - the
    optimal-strafe angle per FRAME, which does NOT depend on the tick. The
    ceiling is then only a clamp AND the divisor of obs column 10
    (env.c: last_yd / yaw_rate_max_deg), so scaling it with the tick buys
    no constant deg/s and silently multiplies the action-echo observation
    by 10/tick. Measured below. The fixed-bin mode is the opposite case
    and must keep scaling."""
    from surfgym import SurfCore, default_config
    from surfgym.core import STATE_DTYPE

    def probe(tick_ms, yaw_adaptive, ceiling=None, v0=2000.0, bin_=11):
        kw = dict(num_envs=1, spawn_mode=0, max_episode_ticks=100000,
                  lidar_w=0, lidar_h=0, sv_gravity=0.0,
                  yaw_adaptive=1 if yaw_adaptive else 0, sv_maxvelocity=4000.0)
        kw.update(_tick_env(tick_ms, yaw_adaptive) if ceiling is None
                  else {"yaw_rate_max_deg": ceiling})
        c = SurfCore(str(CANNONBALL), default_config(**kw), tick_ms=tick_ms)
        st = np.zeros(1, STATE_DTYPE)[0]
        st["origin"] = AIR_POINT
        st["velocity"] = (v0, 0.0, 0.0)
        st["onground"] = -1
        c.reset(0)
        c.set_state(0, st)
        obs, *_ = c.step(np.array([[bin_, 3, 1, 2, 0, 0]], np.int32))
        return float(c.states_view[0]["yaw"]), float(obs[0, 10])

    scale = TickClock(7.63).ms / 10.0                     # 0.76667
    bad_ceiling = TickClock(7.63).per_tick(10.0)          # the pre-fix value
    for v0 in (800.0, 2000.0):
        y10, o10 = probe(10.0, True, v0=v0)
        y131, o131 = probe(7.63, True, v0=v0)
        # tick-free bins: the same action turns the same amount and the
        # policy's own action echo is unchanged
        assert y131 == y10 and o131 == o10
        # pre-fix: a scaled ceiling leaves the DELTA alone (the clamp does
        # not bind above ~223 u/s) and inflates the observation 1.304x
        _, o_bad = probe(7.63, True, ceiling=bad_ceiling, v0=v0)
        assert abs(o_bad / o10 - 1.0 / scale) < 1e-4

    # the clamp regime (below ~223 u/s): a scaled ceiling instead REMOVES
    # per-tick turn authority the weights were trained with
    y_lo, _ = probe(10.0, True, v0=100.0)
    y_lo_bad, _ = probe(7.63, True, ceiling=bad_ceiling, v0=100.0)
    assert abs(y_lo_bad / y_lo - scale) < 1e-4

    # FIXED bins are the opposite case and must keep scaling: the delta is
    # deg/tick, so tick/10 holds deg/SECOND, and obs column 10 (delta over
    # that same ceiling) comes out unchanged
    assert _tick_env(7.63, False) == {"yaw_rate_max_deg": bad_ceiling}
    y10f, o10f = probe(10.0, False, v0=2000.0)
    y131f, o131f = probe(7.63, False, v0=2000.0)
    assert abs(y131f / y10f - scale) < 1e-4 and abs(o131f - o10f) < 1e-6
    # and 10 ms hands the core nothing extra, in either mode
    assert _tick_env(10.0, True) == {} and _tick_env(10.0, False) == {}


def test_episode_cap_default_is_a_duration_not_a_tick_count():
    """--ep-ticks' DEFAULT is a backstop in SECONDS (120 s race / 7 s), so
    it converts at the run's tick. A literal 12000 would be 92.0 s at
    --tick-ms 7.63, and cannonball's own finishers take 77-81 s."""
    ref, tc = TickClock(10.0), TickClock(7.63)
    assert ref.secs_to_ticks(120.0, "round") == 12000     # bit-identical
    assert ref.secs_to_ticks(7.0, "round") == 700
    assert tc.secs_to_ticks(120.0, "round") == 15652
    assert abs(tc.ticks_to_secs(15652) - 120.0) < 0.01
    assert abs(tc.ticks_to_secs(12000) - 92.0) < 0.01     # the bug
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "args.ep_ticks = 12000 if args.reward" not in src


def _assign_sources(path, names, func="main"):
    """Source text of EVERY assignment to each of ``names`` inside ``func``.
    Static, for the same reason record_ckpt's own audit_cfg is: these live
    inside ``if cfg.get("obs_reward")`` / the non-reference tick branch,
    which no test can reach without a real obs-reward checkpoint and its
    baked goal field."""
    import ast
    import warnings
    text = Path(path).read_text(encoding="utf-8")
    with warnings.catch_warnings():          # \\m in a Windows path docstring
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func)
    out = {n: [] for n in names}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id].append(ast.get_source_segment(text, node.value))
    return out


def test_both_call_sites_gate_the_yaw_ceiling_on_yaw_adaptive():
    """The rule measured above has to hold where the cores are actually
    built - the trainer AND the recorder, which must agree or a recording
    is not the run it claims to mirror - and the planner (beam_tas
    build_sim), whose SEARCH runs in that action space: with the ceiling
    scaled, the +-20 / +-8 yaw bins it proposes turn 0.767x at surf
    speeds and its 7.63 ms finish time is a different physics."""
    for f, fn in (("python/train_fast.py", "main"),
                  ("tools/record_ckpt.py", "main"),
                  ("tools/beam_tas.py", "build_sim")):
        srcs = _assign_sources(ROOT / f, {"_tick_env"}, func=fn)["_tick_env"]
        assert srcs, f
        assert any("yaw_adaptive" in (s or "") for s in srcs), (f, srcs)


def test_record_ckpt_rescales_the_obs_reward_mirror_for_the_tick():
    """train_fast feeds its eval mirror RaceReward's time_pen (already
    scaled by tick/10) and GAMMA_T ** K. record_ckpt has to do the same, or
    a --tick-ms recording feeds these weights a slot-12 value they were
    never trained on - the column CLAUDE.md says made sOBSR's evals
    meaningless."""
    got = _assign_sources(ROOT / "tools" / "record_ckpt.py", {"tp", "ng_g"})
    assert any("TICK.per_tick(" in (s or "") for s in got["tp"]), got["tp"]
    assert any("TICK.gamma(" in (s or "") for s in got["ng_g"]), got["ng_g"]
    # the numbers: xQR32 (time_pen 0.01, gamma 0.9995, K=3 at 10 ms)
    # recorded at --tick-ms 7.63 --act-every 4
    tc = TickClock(7.63)
    trained = 0.01 * 3                        # what the policy read at 10 ms
    fixed = tc.per_tick(0.01) * 4             # 30.7 ms per decision
    broken = 0.01 * 4                         # the un-rescaled value
    assert abs(fixed / trained - 1.0) < 0.03
    assert abs(broken / trained - 1.0) > 0.30
    # slot 12 is tanh(r / 0.1), so under --race-latch (delta == 0) the
    # floor the policy reads moves by a tenth of full scale
    assert abs(np.tanh(-broken / 0.1) - np.tanh(-trained / 0.1)) > 0.08
    assert abs(np.tanh(-fixed / 0.1) - np.tanh(-trained / 0.1)) < 0.01
    # gamma is stored at the 10 ms reference, so the tax mirror raises the
    # PER-TICK gamma; identity at the reference
    assert TickClock(10.0).gamma(0.9995) ** 3 == 0.9995 ** 3
    assert tc.gamma(0.9995) ** 4 > 0.9995 ** 4


def test_record_ckpt_carries_the_cap_and_stamps_the_act_every_override():
    """An override that changes what a recording MEANS goes into the
    trajectory header, like maxvel / tick_ms_ckpt - two rows of the round-30
    table differ only in --act-every. And the episode cap is a duration:
    12,000 ticks is 120 s at 10 ms and 92 s at 7.667."""
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert 'header_extra["act_every_ckpt"] = act_every_ckpt' in src
    assert 'header_extra["act_every"]' in src
    assert "ep_ticks = TICK.secs_to_ticks(_cap_s" in src
    # the conversion is an exact round trip when the tick does not change
    for tick in (10.0, 7.63):
        tc = TickClock(tick)
        for n in (700, 3000, 12000):
            assert tc.secs_to_ticks(tc.ticks_to_secs(n), "round") == n
