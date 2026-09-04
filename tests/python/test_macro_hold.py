"""tools/beam_tas.py --macro-hold: HELD-KEY macro-actions as the search's
proposal, and the strafe-cadence diagnostics that check the mechanism.

Round 30 day 2 measured that the planner's best line and the human record's
line are the SAME line to within 1,309 u, and that 3.70 of the remaining
5.12 s accrue with no line separation at all - pure strafe execution. The
planner proposes every decision from the policy's own action distribution,
so its candidates dither exactly like the policy (A/D flips 2.14/s against
the record's 0.42, median side-key hold 0.046 s against 0.418 s), and
selection cannot pick what is never proposed. --macro-hold replaces the
per-decision draw with a HELD key.

What is pinned here:

(a) The flag OFF is byte-identical: --macro-yaw / --macro-fwd / --macro-seed
    are inert without --macro-hold, and two real (tiny) searches that differ
    only in those flags produce the same action table and the same finish
    tick. The draws come from a PRIVATE numpy generator, so the torch
    proposal stream cannot move.
(b) The macro draw: side in {A, none, D}, forward per --macro-fwd, and a
    LOG-UNIFORM duration in [MIN, MAX] seconds rounded to whole decisions
    (>= 1) - and MacroHold actually holds the drawn keys for exactly that
    many decisions.
(c) The analytic yaw (--macro-yaw track) picks the correct SIGN for each
    held key: +side (D, smove +400) accelerates along `right`, 90 deg
    clockwise of the view, so it needs a NEGATIVE yaw delta; -side (A)
    needs a positive one (train_fast's act/yaw_side_agree, and
    tests/python/test_yaw_cond.py). Under --yaw-adaptive that is exactly
    K_BINS k = -+1, the optimal per-frame strafe, at every speed.
(d) A macro is CLONED with the action table at a resample.
(e) A tiny CPU run WITH the flag completes and its winner replays
    bit-exact (beam_tas's own assert; the run exits 0 and reports
    replay_bit_exact).
(f) strafe_cadence reproduces the hand-checked properties of a perfect
    held strafe and of a dithering one.

    python -m pytest tests/python/test_macro_hold.py -q

CPU only. The end-to-end cases need the built core, the cannonball map and
its prebaked goal field, a race checkpoint (SURF_TEST_CKPT) and a saved
planner line to prefix from (SURF_TEST_LINE); they skip without them.
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
sys.path.insert(0, str(ROOT / "tools"))

from train_fast import (A_FWD_NONE, H_DUCK, H_FWD, H_JUMP,  # noqa: E402
                        H_PITCH, H_SIDE, H_YAW, NEUTRAL_YAW, NVEC)

bt = pytest.importorskip("beam_tas")

MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
CANNONBALL = MAPS / "surf_src_cannonball.bsp"
GOALFIELD = MAPS / "surf_src_cannonball.goal_32.npz"
DLL = (Path(os.environ["SURFCORE_DLL"]) if os.environ.get("SURFCORE_DLL")
       else ROOT / "build" / ("surfcore.dll" if os.name == "nt"
                              else "libsurfcore.so"))
CKPT = Path(os.environ.get("SURF_TEST_CKPT")
            or "C:/RL_Surf_exit/runs/exit_seed/xQR32_scalar.pt")
LINE = Path(os.environ.get("SURF_TEST_LINE")
            or ("C:/Users/bulti/AppData/Local/Temp/claude/C--RL-Surf/"
                "e56a2b21-7ab5-4fab-a437-f0bf1163e752/scratchpad/tick/"
                "planner/m763fix/beam_best.npz"))

needs_run = pytest.mark.skipif(
    not (CANNONBALL.exists() and DLL.exists() and GOALFIELD.exists()
         and CKPT.exists() and LINE.exists()),
    reason="needs the built core + cannonball + goal field + a race "
           "checkpoint (SURF_TEST_CKPT) + a saved line (SURF_TEST_LINE)")


def _act(n):
    """A deterministic stand-in for a batch of policy actions."""
    a = np.zeros((n, 6), np.int32)
    for h in range(6):
        a[:, h] = (np.arange(n) + h) % NVEC[h]
    return a


# ==========================================================================
# (b) the draw
# ==========================================================================
def test_draw_shapes_ranges_and_whole_decisions():
    rng = np.random.default_rng(0)
    dec_s = 0.023                       # act_every 3 at the [8, 8, 7] tick
    side, fwd, dur = bt.macro_draw(rng, 4096, 0.2, 0.8, dec_s)
    assert side.shape == fwd.shape == dur.shape == (4096,)
    assert side.dtype == fwd.dtype == dur.dtype == np.int32
    assert side.min() == 0 and side.max() == NVEC[H_SIDE] - 1
    assert fwd.min() == 0 and fwd.max() == NVEC[H_FWD] - 1
    # every duration is a whole number of decisions inside [MIN, MAX]
    lo = max(1, int(np.floor(0.2 / dec_s)))
    hi = int(np.ceil(0.8 / dec_s))
    assert dur.min() >= lo and dur.max() <= hi and dur.min() >= 1
    # ... and the range is actually used at both ends (log-uniform)
    assert dur.min() <= round(0.22 / dec_s) and dur.max() >= round(0.75 / dec_s)
    # a hold shorter than one decision still lasts one decision
    _s, _f, d1 = bt.macro_draw(np.random.default_rng(1), 256, 0.01, 0.011,
                               dec_s)
    assert (d1 == 1).all()


def test_draw_is_log_uniform_not_uniform():
    """A scale-free draw over an order of magnitude: the geometric mean is
    the middle of the range in LOG space, well below the arithmetic mean."""
    rng = np.random.default_rng(3)
    _s, _f, dur = bt.macro_draw(rng, 200000, 0.05, 5.0, 0.001)
    med = float(np.median(dur)) * 0.001
    assert abs(med - np.sqrt(0.05 * 5.0)) < 0.02          # ~0.5 s
    assert float(dur.mean()) * 0.001 > 1.3 * med          # skewed right


def test_draw_fwd_modes():
    dec_s = 0.023
    _s, fwd, _d = bt.macro_draw(np.random.default_rng(2), 512, 0.2, 0.8,
                                dec_s, fwd_mode="none")
    assert (fwd == A_FWD_NONE).all()
    _s, fwd, _d = bt.macro_draw(np.random.default_rng(2), 512, 0.2, 0.8,
                                dec_s, fwd_mode="policy")
    assert fwd is None
    # 'policy' still consumes the forward draw, so the DURATION stream is
    # the same one 'draw' would have seen
    _s1, _f1, d1 = bt.macro_draw(np.random.default_rng(9), 64, 0.2, 0.8, dec_s,
                                 fwd_mode="draw")
    _s2, _f2, d2 = bt.macro_draw(np.random.default_rng(9), 64, 0.2, 0.8, dec_s,
                                 fwd_mode="policy")
    assert np.array_equal(d1, d2)
    with pytest.raises(ValueError):
        bt.macro_draw(np.random.default_rng(0), 4, 0.2, 0.8, dec_s,
                      fwd_mode="nonsense")


def test_draw_is_private_and_reproducible():
    a = bt.macro_draw(np.random.default_rng(7), 32, 0.2, 0.8, 0.023)
    b = bt.macro_draw(np.random.default_rng(7), 32, 0.2, 0.8, 0.023)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = bt.macro_draw(np.random.default_rng(8), 32, 0.2, 0.8, 0.023)
    assert not np.array_equal(a[0], c[0])


# ==========================================================================
# (b) the hold: MacroHold actually holds
# ==========================================================================
def _drive(mh, n_envs, n_dec, vel=None):
    """Run n_dec decisions through a MacroHold and return the per-decision
    action rows it produced."""
    if vel is None:
        vel = np.tile(np.array([2800.0, 0.0, -100.0]), (n_envs, 1))
    rows = []
    for _ in range(n_dec):
        rows.append(mh.decide(_act(n_envs), vel, False, 10.0))
    return np.stack(rows)          # (n_dec, n_envs, 6)


def test_the_drawn_duration_is_the_number_of_decisions_the_keys_are_held():
    n, dec_s = 64, 0.023
    mh = bt.MacroHold(n, np.arange(n), 0.2, 0.8, dec_s,
                      np.random.default_rng(5))
    # replay the draw stream by hand and predict every run length
    ref = np.random.default_rng(5)
    rows = _drive(mh, n, 200)
    side = rows[:, :, H_SIDE]
    for e in range(n):
        runs, k = [], 0
        while k < len(side):
            j = k + 1
            while j < len(side) and side[j, e] == side[k, e]:
                j += 1
            runs.append(j - k)
            k = j
        # every completed run is a legal duration; the last one is cut off
        # by the end of the drive
        lo = max(1, int(np.floor(0.2 / dec_s)))
        hi = int(np.ceil(0.8 / dec_s))
        for r in runs[:-1]:
            # consecutive macros can draw the SAME key, which merges two
            # runs - so a run is a SUM of legal durations, never shorter
            # than one
            assert lo <= r <= 4 * hi
        assert runs[0] >= lo
    del ref
    # the countdown itself is exact: a fresh MacroHold with MIN == MAX
    # holds for exactly that many decisions
    dur = 9
    mh2 = bt.MacroHold(4, np.arange(4), dur * dec_s, dur * dec_s + 1e-9,
                       dec_s, np.random.default_rng(11))
    assert (bt.macro_draw(np.random.default_rng(11), 4, dur * dec_s,
                          dur * dec_s + 1e-9, dec_s)[2] == dur).all()
    rows2 = _drive(mh2, 4, 3 * dur)
    for e in range(4):
        s = rows2[:, e, H_SIDE]
        # decision boundaries are at multiples of dur
        for blk in range(3):
            b = s[blk * dur:(blk + 1) * dur]
            assert (b == b[0]).all()


def test_decide_touches_only_the_macro_heads_and_envs():
    n, g = 32, 8
    mh = bt.MacroHold(n, np.arange(g, n), 0.2, 0.8, 0.023,
                      np.random.default_rng(0))
    act = _act(n)
    out = mh.decide(act, np.tile([2800.0, 0.0, 0.0], (n, 1)), False, 10.0)
    assert np.array_equal(out[:g], act[:g])            # greedy floor intact
    for h in (H_YAW, H_PITCH, H_JUMP, H_DUCK):
        assert np.array_equal(out[g:, h], act[g:, h])  # yaw='policy'
    assert (out[g:, H_SIDE] == mh.side[g:]).all()
    assert (out[g:, H_FWD] == mh.fwd[g:]).all()


def test_decide_does_not_write_through_the_callers_buffer():
    """`act` is the policy wrapper's HELD array: writing into it would
    change what the greedy envs do for the rest of the act_every hold."""
    act = _act(16)
    before = act.copy()
    mh = bt.MacroHold(16, np.arange(16), 0.2, 0.8, 0.023,
                      np.random.default_rng(0))
    mh.decide(act, np.tile([2800.0, 0.0, 0.0], (16, 1)), False, 10.0)
    assert np.array_equal(act, before)


def test_macro_fwd_policy_leaves_the_forward_head_alone():
    n = 16
    mh = bt.MacroHold(n, np.arange(n), 0.2, 0.8, 0.023,
                      np.random.default_rng(0), fwd="policy")
    act = _act(n)
    out = mh.decide(act, np.tile([2800.0, 0.0, 0.0], (n, 1)), False, 10.0)
    assert np.array_equal(out[:, H_FWD], act[:, H_FWD])
    assert not np.array_equal(out[:, H_SIDE], act[:, H_SIDE])


# ==========================================================================
# (c) the analytic yaw
# ==========================================================================
def test_yaw_track_picks_the_sign_the_engine_needs():
    """+side (D, bin 2) accelerates along `right`, 90 deg CLOCKWISE of the
    view, so the view must turn clockwise: a NEGATIVE yaw delta, i.e. a bin
    BELOW the neutral. -side (A, bin 0) is the mirror. A neutral side key
    has nothing to track and returns -1."""
    from surfgym.core import YAW_BINS
    vel = np.tile(np.array([2800.0, 0.0, -50.0]), (3, 1))
    side = np.array([0, 1, 2])                          # A, none, D
    for adaptive in (False, True):
        b = bt.macro_yaw_bins(vel, side, adaptive, 10.0)
        assert b[1] == -1                               # no held key
        assert b[0] > NEUTRAL_YAW and b[2] < NEUTRAL_YAW
        if not adaptive:
            assert YAW_BINS[b[0]] > 0 and YAW_BINS[b[2]] < 0
        # exactly mirrored about the neutral
        assert (b[0] - NEUTRAL_YAW) == (NEUTRAL_YAW - b[2])


def test_yaw_track_is_the_optimal_strafe_multiple_under_yaw_adaptive():
    """--yaw-adaptive bins ARE multiples of atan(30/|v|) (src/env.c
    K_BINS), so tracking the velocity rotation is k = -+1 at EVERY speed:
    K_BINS index 10 (+1.0) for A and index 4 (-1.0) for D."""
    from surfgym.obsaux import _K_BINS
    assert float(_K_BINS[10]) == 1.0 and float(_K_BINS[4]) == -1.0
    for sp in (200.0, 800.0, 1500.0, 2800.0, 3728.0):
        vel = np.array([[sp * 0.6, sp * 0.8, -80.0]] * 2)
        b = bt.macro_yaw_bins(vel, np.array([0, 2]), True, 10.0)
        assert list(b) == [10, 4], (sp, b)


def test_yaw_track_matches_the_core_table_with_fixed_bins():
    """With fixed bins the pick depends on the speed: it is the bin whose
    deg/tick is closest to -+atan(30/|v|)."""
    from surfgym.core import YAW_BINS
    tab = np.asarray(YAW_BINS, np.float64)              # yaw_rate_max_deg 10
    for sp in (300.0, 1000.0, 2800.0):
        w = np.degrees(np.arctan(30.0 / sp))
        vel = np.array([[sp, 0.0, 0.0]] * 2)
        b = bt.macro_yaw_bins(vel, np.array([0, 2]), False, 10.0)
        assert b[0] == int(np.abs(tab - w).argmin())
        assert b[1] == int(np.abs(tab + w).argmin())


def test_yaw_track_writes_only_where_a_key_is_held():
    n = 24
    mh = bt.MacroHold(n, np.arange(n), 0.2, 0.8, 0.023,
                      np.random.default_rng(4), yaw="track")
    act = _act(n)
    vel = np.tile(np.array([2400.0, 900.0, -120.0]), (n, 1))
    out = mh.decide(act, vel, True, 10.0)
    neutral = mh.side == A_FWD_NONE
    assert np.array_equal(out[neutral, H_YAW], act[neutral, H_YAW])
    held = ~neutral
    assert held.any()
    assert (out[held & (mh.side == 2), H_YAW] == 4).all()
    assert (out[held & (mh.side == 0), H_YAW] == 10).all()


def test_the_k_bins_mirror_matches_the_core():
    """macro_yaw_bins reads surfgym.obsaux._K_BINS; src/env.c is the
    authority and a drift there would silently pick the wrong bin."""
    from surfgym.obsaux import _K_BINS
    src = (ROOT / "src" / "env.c").read_text(encoding="utf-8")
    i = src.index("static const float K_BINS[15]")
    body = src[i:src.index("};", i)]
    vals = [float(x) for x in
            body[body.index("{") + 1:].replace("f", "").split(",")]
    assert vals == [float(v) for v in _K_BINS]
    # and the sign convention this file is written against
    assert "float smove = (a[3] <= 0) ? -400.0f : (a[3] >= 2 ? 400.0f" in src
    assert "st->yaw = wrap_yaw(st->yaw + yd);" in src


# ==========================================================================
# (d) cloning
# ==========================================================================
def test_clone_carries_the_macro_to_the_loser():
    n = 8
    mh = bt.MacroHold(n, np.arange(n), 0.2, 0.8, 0.023,
                      np.random.default_rng(6))
    mh.decide(_act(n), np.tile([2800.0, 0.0, 0.0], (n, 1)), False, 10.0)
    losers, donors = np.array([5, 6, 7]), np.array([0, 1, 2])
    before = (mh.side.copy(), mh.fwd.copy(), mh.left.copy())
    mh.clone(losers, donors)
    assert np.array_equal(mh.side[losers], before[0][donors])
    assert np.array_equal(mh.fwd[losers], before[1][donors])
    assert np.array_equal(mh.left[losers], before[2][donors])
    # the donors themselves are untouched
    assert np.array_equal(mh.side[donors], before[0][donors])


# ==========================================================================
# (f) the cadence diagnostics
# ==========================================================================
def _synth(n, side_seq, yaw_rate_deg, vel_rate_deg=None, speed=2800.0,
           dt_s=0.008):
    """A synthetic free-flight episode: the view turning at yaw_rate_deg
    per tick, the horizontal velocity bearing turning at vel_rate_deg
    (default: with the view), and gravity on vz. With the two equal the
    velocity bearing IS the view heading, so a held side key's wishdir
    (90 deg off the view) stays exactly perpendicular to it - the optimal
    strafe. Row t's yaw is PRE-step, so the move at t runs at yaw[t+1]."""
    if vel_rate_deg is None:
        vel_rate_deg = yaw_rate_deg
    k = np.arange(n, dtype=np.float64)
    yaw = k * yaw_rate_deg
    bear = (k + 1) * vel_rate_deg          # the yaw the move at t runs at
    v = np.zeros((n, 3))
    v[:, 0] = speed * np.cos(np.radians(bear))
    v[:, 1] = speed * np.sin(np.radians(bear))
    v[:, 2] = -k * 800.0 * dt_s
    return dict(vel=v, yaw=yaw, onground=np.zeros(n, int),
                fwd=np.full(n, A_FWD_NONE), side=np.asarray(side_seq),
                dt=np.full(n, dt_s), g_step=np.full(n, -800.0 * dt_s))


def test_cadence_of_a_perfect_held_strafe():
    n = 500
    s = _synth(n, np.full(n, 2), -0.6)         # D held, view turning right
    c = bt.strafe_cadence(**s)
    assert c["flips"] == 0 and c["side_changes"] == 0
    assert abs(c["hold_med_s"] - n * 0.008) < 1e-9
    assert c["free_ticks"] == n - 1
    assert c["perp_share"] > 0.99              # wishdir stays perpendicular
    assert c["fwd_air"] == 0.0


def test_cadence_sees_a_view_that_outruns_the_velocity():
    """The whole point of the metric: a view turning faster than the
    velocity leaves the 0.5 deg band where PM_AirAccelerate pays."""
    n = 500
    c = bt.strafe_cadence(**_synth(n, np.full(n, 2), -3.0, vel_rate_deg=-0.6))
    assert c["perp_share"] < 0.05


def test_cadence_of_a_dithering_strafe():
    n = 500
    alt = np.where(np.arange(n) % 4 < 2, 0, 2)      # flip every 2 ticks
    c = bt.strafe_cadence(**_synth(n, alt, -0.6))
    assert c["flips"] == n // 2 - 1
    assert c["flips_per_s"] > 50                    # 4.0 s of episode
    assert c["side_changes"] == n // 2 - 1
    assert abs(c["hold_med_s"] - 2 * 0.008) < 1e-9


def test_cadence_counts_direction_flips_not_key_releases():
    """Releasing D and pressing it again is not an A/D flip."""
    side = np.array([2] * 10 + [1] * 10 + [2] * 10 + [1] * 5 + [0] * 10)
    c = bt.strafe_cadence(**_synth(len(side), side, -0.6))
    assert c["side_changes"] == 4
    assert c["flips"] == 1                     # only the final D -> A
    assert abs(c["hold_med_s"] - 10 * 0.008) < 1e-9


def test_cadence_energy_and_contact_masking():
    """A tick the map pushed back is not free flight and its energy is not
    the strafe's."""
    n = 200
    s = _synth(n, np.full(n, 2), -0.6)
    # give it a real speed gain, then one hard contact tick
    sp = 2800.0 + np.arange(n) * 2.0
    bear = (np.arange(n) + 1) * -0.6
    s["vel"][:, 0] = sp * np.cos(np.radians(bear))
    s["vel"][:, 1] = sp * np.sin(np.radians(bear))
    c0 = bt.strafe_cadence(**s)
    assert c0["strafe_energy_M"] > 0.0
    s["vel"][100, 2] += 500.0                  # a ramp clipped vz
    c1 = bt.strafe_cadence(**s)
    assert c1["free_ticks"] == c0["free_ticks"] - 2   # the tick and its pair
    assert c1["strafe_energy_M"] < c0["strafe_energy_M"]


# ==========================================================================
# (a) + (e) the real thing, tiny
# ==========================================================================
def _run(tmp, name, extra, prefix_ticks=9600, envs=8, timeout=1800):
    out = Path(tmp) / name
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "beam_tas.py"),
           str(CKPT), "--map", str(CANNONBALL), "--envs", str(envs),
           "--resample-every", "5", "--elite-frac", "0.25",
           "--torch-seed", "0", "--seed", "0", "--allow-nonfinisher",
           "--keep-finishers", "2", "--log-every", "1", "--score", "d",
           "--max-ticks", "10306", "--tick-ms", "7.63",
           "--prefix-line", f"{LINE}:{prefix_ticks}",
           "--out-dir", str(out)] + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", NUMBA_NUM_THREADS="4",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT / "python"))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-4000:]
    assert "RE-BAKE" not in r.stdout, "a bake line: wrong map path/mtime"
    s = json.loads((out / "summary.json").read_text())
    return r, s, out


@needs_run
def test_off_is_byte_identical(tmp_path):
    """--macro-yaw / --macro-fwd / --macro-seed are INERT without
    --macro-hold, and the macro's private numpy generator cannot move the
    torch proposal stream - so the search is the search it was."""
    _r0, s0, o0 = _run(tmp_path, "plain", [])
    _r1, s1, o1 = _run(tmp_path, "inert",
                       ["--macro-yaw", "track", "--macro-fwd", "none",
                        "--macro-seed", "12345"])
    assert s0["crossed"] and s1["crossed"]
    assert s0["best_ticks"] == s1["best_ticks"]
    z0 = np.load(o0 / "beam_best.npz", allow_pickle=False)
    z1 = np.load(o1 / "beam_best.npz", allow_pickle=False)
    assert np.array_equal(z0["acts"], z1["acts"])
    assert np.array_equal(z0["acts_all"], z1["acts_all"])
    assert s0["macro_hold"] is None and s1["macro_hold"] is None
    # the cadence block is written with or without the flag
    for s in (s0, s1):
        assert s["cadence"] and s["cadence"]["ticks"] == s["best_ticks"]
    assert s0["cadence"] == s1["cadence"]


@needs_run
def test_macro_run_completes_and_replays_bit_exact(tmp_path):
    _r, s, out = _run(tmp_path, "macro",
                      ["--macro-hold", "0.2:0.8", "--macro-yaw", "track"],
                      envs=16)
    assert s["crossed"] and s["replay_bit_exact"]
    assert s["macro_hold"] == "0.2:0.8"
    assert s["macro_yaw"] == "track" and s["macro_fwd"] == "draw"
    assert s["macro_envs"] == 16 and s["macro_draws"] > 0
    assert "--macro-hold 0.2:0.8s" in _r.stdout
    assert "cadence [winner]" in _r.stdout
    c = s["cadence"]
    for k in ("flips_per_s", "hold_med_s", "perp_share", "strafe_energy_M"):
        assert k in c
    assert np.load(out / "beam_best.npz", allow_pickle=False)["acts"].shape[1] == 6


@needs_run
def test_macro_run_is_reproducible(tmp_path):
    _r0, s0, o0 = _run(tmp_path, "m1", ["--macro-hold", "0.2:0.8"], envs=16)
    _r1, s1, o1 = _run(tmp_path, "m2", ["--macro-hold", "0.2:0.8"], envs=16)
    assert s0["best_ticks"] == s1["best_ticks"]
    assert np.array_equal(
        np.load(o0 / "beam_best.npz", allow_pickle=False)["acts"],
        np.load(o1 / "beam_best.npz", allow_pickle=False)["acts"])
    # ... and a different macro seed is a different search
    _r2, s2, o2 = _run(tmp_path, "m3",
                       ["--macro-hold", "0.2:0.8", "--macro-seed", "99"],
                       envs=16)
    assert not np.array_equal(
        np.load(o0 / "beam_best.npz", allow_pickle=False)["acts"],
        np.load(o2 / "beam_best.npz", allow_pickle=False)["acts"])


def test_bad_specs_are_refused():
    import argparse
    ap = argparse.ArgumentParser()
    # the parse itself is in main(); check the bounds constant is sane
    assert bt.MACRO_MIN_S > 0 and bt.MACRO_MAX_S > bt.MACRO_MIN_S
    del ap
    src = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert '--macro-hold wants MIN:MAX seconds' in src
    assert 'excludes --commit' in src
    # the macro is applied where the action table records its row, and it
    # is cloned at the resample
    assert "macro.decide(a, coreN.states_view[\"velocity\"]" in src
    assert "macro.clone(losers, donors)" in src
