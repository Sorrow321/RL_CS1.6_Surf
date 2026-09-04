"""tools/beam_tas.py --branch-grid: the DETERMINISTIC junction fork.

--branch-at fills the forked envs with a random correlated burst; on the
cannonball finish room that produced 0 record-like continuations out of
1,536 draws. --branch-grid fills them with an ENUMERATED grid of held-key
macro plans instead - yaw-bin offset x side key x hold duration x
{macro, macro + mirrored counter-macro} - replicated round-robin over the
population, with the policy continuing after each plan's macro ends.

What is pinned here:

1. OFF is byte-identical: with the flag absent nothing is parsed, nothing
   is applied, and (the end-to-end half) a tiny search writes the same
   action table as one given an --branch-grid whose trigger never fires.
2. The grid ENUMERATES the stated plans, deterministically and with no
   randomness at all: same spec -> same plan list -> same env assignment.
3. The macro touches only the two heads that steer a surf flight, never
   writes through the policy wrapper's held action buffer, expires per
   plan, and mirrors the second segment.
4. A tiny CPU run with the flag completes and replays bit-exact (beam_tas
   asserts the replay itself; the test asserts the exit code and the line).

    python -m pytest tests/python/test_branch_grid.py -q
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

from train_fast import H_SIDE, H_YAW, NVEC   # noqa: E402

bt = pytest.importorskip("beam_tas")

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
CKPT = Path("C:/RL_Surf_exit/runs/exit_seed/xQR32_scalar.pt")


def _act(n):
    """A deterministic stand-in for a batch of policy actions."""
    a = np.zeros((n, 6), np.int32)
    for h in range(6):
        a[:, h] = (np.arange(n) + h) % NVEC[h]
    return a


# ---------------------------------------------------------------- parsing

def test_default_spec_is_the_168_plan_grid():
    plans, meta = bt.branch_grid_parse("", 3)
    assert meta["yaw"] == [-9, -6, -3, 0, 3, 6, 9]
    assert meta["side"] == [0, 1, 2]
    assert meta["hold"] == [21, 42, 84, 168]
    assert meta["seg"] == 2
    assert meta["plans"] == len(plans) == 7 * 3 * 4 * 2 == 168
    assert meta["max_ticks"] == 336            # the longest hold, doubled


def test_plan_order_is_deterministic_and_nested_yaw_side_hold_mirror():
    plans, _ = bt.branch_grid_parse("yaw=-2,0,2:side=0,2:hold=6,12:seg=2", 3)
    assert plans == [
        (-2, 0, 6, False), (-2, 0, 6, True),
        (-2, 0, 12, False), (-2, 0, 12, True),
        (-2, 2, 6, False), (-2, 2, 6, True),
        (-2, 2, 12, False), (-2, 2, 12, True),
        (0, 0, 6, False), (0, 0, 6, True),
        (0, 0, 12, False), (0, 0, 12, True),
        (0, 2, 6, False), (0, 2, 6, True),
        (0, 2, 12, False), (0, 2, 12, True),
        (2, 0, 6, False), (2, 0, 6, True),
        (2, 0, 12, False), (2, 0, 12, True),
        (2, 2, 6, False), (2, 2, 6, True),
        (2, 2, 12, False), (2, 2, 12, True)]
    again, _ = bt.branch_grid_parse("yaw=-2,0,2:side=0,2:hold=6,12:seg=2", 3)
    assert again == plans


def test_holds_round_up_to_whole_decisions_and_dedupe():
    plans, meta = bt.branch_grid_parse("yaw=0:side=1:hold=20,21,22:seg=1", 3)
    # 20 -> 21, 21 -> 21 (already whole), 22 -> 24; the duplicate collapses
    assert meta["hold"] == [21, 24]
    assert [p[2] for p in plans] == [21, 24]


def test_seg1_has_no_mirror_and_halves_the_macro_window():
    p1, m1 = bt.branch_grid_parse("hold=30:seg=1", 3)
    p2, m2 = bt.branch_grid_parse("hold=30:seg=2", 3)
    assert all(not p[3] for p in p1)
    assert len(p2) == 2 * len(p1)
    assert m1["max_ticks"] == 30 and m2["max_ticks"] == 60


def test_side_p_keeps_the_policys_own_key():
    plans, meta = bt.branch_grid_parse("yaw=1:side=p:hold=3:seg=1", 3)
    assert meta["side"] == [-1] and plans == [(1, -1, 3, False)]
    act = _act(8)
    out = bt.branch_grid_apply(act, np.arange(8), plans, np.zeros(8, int), 0)
    assert np.array_equal(out[:, H_SIDE], act[:, H_SIDE])
    assert np.array_equal(out[:, H_YAW],
                          np.clip(act[:, H_YAW] + 1, 0, NVEC[H_YAW] - 1))


@pytest.mark.parametrize("spec", ["yaw=0:bogus=1", "seg=3", "hold=0",
                                  "side=9"])
def test_bad_specs_are_refused(spec):
    with pytest.raises(SystemExit):
        bt.branch_grid_parse(spec, 3)


# ------------------------------------------------------------ application

def test_apply_touches_only_yaw_and_side_of_the_forked_envs():
    n, k = 32, 8
    act = _act(n)
    idx = np.arange(n - k, n)
    plans, _ = bt.branch_grid_parse("yaw=-3,3:side=0,2:hold=9:seg=1", 3)
    assign = np.arange(k) % len(plans)
    out = bt.branch_grid_apply(act, idx, plans, assign, 0)
    assert np.array_equal(out[: n - k], act[: n - k])
    for h in range(6):
        if h in (H_YAW, H_SIDE):
            continue
        assert np.array_equal(out[idx, h], act[idx, h])
    for j, p in zip(idx, assign):
        y_off, side, _h, _m = plans[p]
        assert out[j, H_YAW] == np.clip(act[j, H_YAW] + y_off, 0,
                                        NVEC[H_YAW] - 1)
        assert out[j, H_SIDE] == side


def test_apply_does_not_write_through_the_callers_buffer():
    act = _act(16)
    before = act.copy()
    plans, _ = bt.branch_grid_parse("yaw=4:side=0:hold=3:seg=1", 3)
    bt.branch_grid_apply(act, np.arange(8, 16), plans, np.zeros(8, int), 0)
    assert np.array_equal(act, before)


def test_macro_expires_per_plan_and_the_policy_takes_over():
    act = _act(4)
    plans, _ = bt.branch_grid_parse("yaw=3:side=0:hold=6,12:seg=1", 3)
    assign = np.array([0, 0, 1, 1])          # hold 6, hold 6, 12, 12
    idx = np.arange(4)
    for k, expect in ((0, [1, 1, 1, 1]), (5, [1, 1, 1, 1]),
                      (6, [0, 0, 1, 1]), (12, [0, 0, 0, 0])):
        out = bt.branch_grid_apply(act, idx, plans, assign, k)
        touched = [int(not np.array_equal(out[i], act[i])) for i in idx]
        assert touched == expect, (k, touched)


def test_mirror_flips_the_yaw_offset_and_the_side_key():
    act = _act(2)
    plans, _ = bt.branch_grid_parse("yaw=4:side=0:hold=6:seg=2", 3)
    assert plans == [(4, 0, 6, False), (4, 0, 6, True)]
    idx, assign = np.arange(2), np.array([0, 1])
    seg1 = bt.branch_grid_apply(act, idx, plans, assign, 0)
    assert seg1[0, H_SIDE] == 0 and seg1[1, H_SIDE] == 0
    seg2 = bt.branch_grid_apply(act, idx, plans, assign, 6)
    assert np.array_equal(seg2[0], act[0])            # no mirror: expired
    assert seg2[1, H_SIDE] == 2                       # mirrored key
    assert seg2[1, H_YAW] == np.clip(act[1, H_YAW] - 4, 0, NVEC[H_YAW] - 1)
    assert np.array_equal(bt.branch_grid_apply(act, idx, plans, assign, 12),
                          act)


def test_yaw_offset_clips_into_range_and_never_leaves_a_legal_bin():
    n = NVEC[H_YAW]
    act = np.zeros((n, 6), np.int32)
    act[:, H_YAW] = np.arange(n)
    plans, _ = bt.branch_grid_parse("yaw=9:side=1:hold=3:seg=1", 3)
    out = bt.branch_grid_apply(act, np.arange(n), plans, np.zeros(n, int), 0)
    assert out[:, H_YAW].min() >= 0 and out[:, H_YAW].max() < n
    assert np.array_equal(out[:, H_YAW], np.clip(np.arange(n) + 9, 0, n - 1))


def test_grid_draws_no_randomness():
    """The whole point against --branch-at: two calls with no generator in
    sight are identical, so the torch stream and every unbranched env's
    proposal are untouched by construction rather than by convention."""
    act = _act(64)
    plans, _ = bt.branch_grid_parse("", 3)
    assign = np.arange(64) % len(plans)
    a = bt.branch_grid_apply(act, np.arange(64), plans, assign, 5)
    b = bt.branch_grid_apply(act, np.arange(64), plans, assign, 5)
    assert np.array_equal(a, b)


# ------------------------------------------------------------- end to end

SMOKE = ["--map", str(CANNONBALL), "--envs", "32", "--resample-every", "25",
         "--elite-frac", "0.25", "--greedy-envs", "4", "--score", "d",
         "--torch-seed", "0", "--seed", "0", "--greedy-eps", "1",
         "--keep-finishers", "2", "--allow-nonfinisher", "--skip-gate",
         "--log-every", "50", "--max-ticks", "300", "--objective", "finish"]


def _beam(out_dir, extra, timeout=1800):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
               NUMBA_NUM_THREADS="4")
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "beam_tas.py"),
           str(CKPT), *SMOKE, "--out-dir", str(out_dir), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=timeout)


needs_run = pytest.mark.skipif(
    not (CANNONBALL.exists() and CKPT.exists()),
    reason="needs the cannonball map and the round-30 planner checkpoint")


@needs_run
def test_off_is_byte_identical_to_a_grid_that_never_fires(tmp_path):
    """The flag's OFF path must not perturb the search at all: an inert
    --branch-grid (a trigger tick past --max-ticks) parses the grid, sizes
    the population and then never fires, and the run has to come out with
    the same action table, tick for tick, as a plain one."""
    a = _beam(tmp_path / "plain", [])
    assert a.returncode == 0, a.stdout[-4000:] + a.stderr[-4000:]
    b = _beam(tmp_path / "grid",
              ["--branch-grid", "t999999:yaw=-3,0,3:side=0,1,2:hold=21:"
                                "seg=2"])
    assert b.returncode == 0, b.stdout[-4000:] + b.stderr[-4000:]
    assert "BRANCH NEVER FIRED" in b.stdout
    za = np.load(tmp_path / "plain" / "beam_best.npz", allow_pickle=False)
    zb = np.load(tmp_path / "grid" / "beam_best.npz", allow_pickle=False)
    assert np.array_equal(za["acts"], zb["acts"])
    assert np.array_equal(za["acts_all"], zb["acts_all"])
    sa = json.loads((tmp_path / "plain" / "summary.json").read_text())
    sb = json.loads((tmp_path / "grid" / "summary.json").read_text())
    assert sa["branch_grid"] is None and sb["branch_grid"] is not None
    assert sb["branch_grid_spec"]["plans"] == 18
    for k in ("finishes", "finish_ticks", "generations", "best_arc"):
        assert sa.get(k) == sb.get(k), k


@needs_run
def test_a_firing_grid_runs_and_replays_bit_exact(tmp_path):
    r = _beam(tmp_path / "fire",
              ["--branch-grid", "t60:yaw=-3,0,3:side=0,1,2:hold=21:seg=1",
               "--branch-protect", "2"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--branch-grid: 9 plans" in r.stdout
    assert "BRANCH: fork" in r.stdout
    assert "bit-exact" in r.stdout
    s = json.loads((tmp_path / "fire" / "summary.json").read_text())
    assert s["branch_grid_spec"]["plans"] == 9
    assert len(s["branch_grid_plans"]) == 9
    assert s["branch_fired_tick"] is not None
