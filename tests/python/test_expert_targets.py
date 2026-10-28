"""The expert loop's BC targets are the policy's OWN choices (pbx, 2026-10-28).

On pbr9b1_c9 (princessburglar: --keys-hold, --view-absolute velocity, a 10 -> 7.63 ms tick ramp)
BC on the policy's own greedy line read head-acc 0.654, joint-acc 0.163, view-mse 304 where it
must read ~1 / ~1 / ~0. Four causes, each pinned here:

  1. **held-key redundancy** - beam_tas stored ENGINE rows and plan_to_bc rebuilt policy rows
     with the keep rule (KEEP wherever a key is unchanged), while the policy as often re-presses
     the held key (the same engine action). The planner now records the row the policy chose
     before the keep bins resolve, plan_to_bc resolves it back to the engine rows as a check
     (surfgym.bc.keys_resolve_rows) and takes the survivors by ENGINE rows + view
     (survivor_probs' prefix_tables);
  2. **the drawn view z** - a target rebuilt from the physical view is clipped where tanh
     saturates; the planner records the z its heads drew and BCDataset prefers it (view_z);
  3. **the pitch ceiling** - the trainer anchors the view ceilings to the tick ramp's START
     (VIEW_TICK), beam_tas.build_sim and record_ckpt scaled them to the current tick
     (surfgym.tick.view_tick);
  4. **the critic probe** - --score dv's value_fn ran _obs on the decision tick and the next
     decision reset every env's held keys to NEUTRAL (beam_tas keys_snapshot / keys_restore);

plus the spawn's held keys (--keys-hold-spawn-duck: surfgym.bc.spawn_held) and the spine's held
keys for --demo-keys (surfgym.bc.line_held_keys). Everything is off / byte-identical for a plan,
a file or a checkpoint without the new columns or settings.

    python -m pytest tests/python/test_expert_targets.py -q
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.bc import (keys_policy_rows, keys_resolve_rows,  # noqa: E402
                        line_held_keys, spawn_held, survivor_probs)
from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.keyshold import HELD_HEADS, KEEP, NEUTRAL, KeysHold  # noqa: E402
from surfgym.tick import TickClock, view_tick  # noqa: E402

SKI = ROOT / "maps" / "surf_ski_2.bsp"
LIB = ROOT / "build" / ("libsurfcore.so" if sys.platform != "win32" else "surfcore.dll")
needs_core = pytest.mark.skipif(not (SKI.exists() and LIB.exists()),
                                reason="needs the built core and maps/surf_ski_2.bsp")


def _policy_line(n=80, seed=0):
    """Policy-space rows the way a keys-hold policy emits them: mostly KEEP, now and then an
    explicit bin - often the one already held (the same engine action as KEEP)."""
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 6), np.int64)
    p[:, 0] = rng.integers(0, 15, n)
    p[:, 1] = rng.integers(0, 7, n)
    p[:, 4] = rng.integers(0, 2, n)
    for h, hi in ((2, 4), (3, 4), (5, 3)):
        r = rng.random(n)
        p[:, h] = np.where(r < 0.6, KEEP, rng.integers(1, hi, n))
    return p


# --------------------------------------------------------------------------
# 1. policy rows: the resolver, the start state, the spine's held keys
# --------------------------------------------------------------------------
@pytest.mark.parametrize("start", [None, (1, 1, 0), (0, 2, 1)])
def test_keys_resolve_rows_is_keysholds_own_resolve_from_the_start(start):
    pol = _policy_line()
    eng, held, feat = keys_resolve_rows(pol, start)
    k = KeysHold(1)
    if start is not None:
        k.state[0] = start
    for t in range(len(pol)):
        np.testing.assert_array_equal(held[t], k.state[0])           # what the decision SAW
        np.testing.assert_array_equal(feat[t], k.features()[0])
        row = pol[t:t + 1].astype(np.int32).copy()
        k.resolve(row)
        np.testing.assert_array_equal(eng[t], row[0])                # what the engine got


def test_the_keep_rule_and_the_policys_rows_are_one_engine_line_but_not_one_target():
    pol = _policy_line(seed=4)
    eng, _h, _f = keys_resolve_rows(pol)
    rebuilt, _h2, _f2 = keys_policy_rows(eng)
    # the rebuilt rows resolve to the same engine line ...
    np.testing.assert_array_equal(keys_resolve_rows(rebuilt)[0], eng)
    # ... but are not the rows the policy chose wherever it re-pressed the held key
    explicit_same = np.zeros(len(pol), bool)
    for c, h in enumerate(HELD_HEADS):
        explicit_same |= (pol[:, h] != KEEP) & (pol[:, h] - 1 == _h[:, c])
    assert explicit_same.any()
    assert (rebuilt != pol).any(axis=1)[explicit_same].all()


def test_keys_policy_rows_defaults_to_neutral_and_takes_a_start():
    eng = keys_resolve_rows(_policy_line(seed=2))[0]
    a = keys_policy_rows(eng)
    b = keys_policy_rows(eng, NEUTRAL)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    _p, held, _f = keys_policy_rows(eng, (1, 1, 1))
    np.testing.assert_array_equal(held[0], (1, 1, 1))


def test_an_unrecorded_or_out_of_range_policy_row_is_refused():
    pol = _policy_line(n=5)
    bad = pol.copy()
    bad[2] = -1                                    # beam_tas hist_p: not the policy's own row
    with pytest.raises(ValueError):
        keys_resolve_rows(bad)
    bad = pol.copy()
    bad[1, 5] = 3                                  # duck has 3 policy bins (0, 1, 2)
    with pytest.raises(ValueError):
        keys_resolve_rows(bad)


def test_spawn_held_ducks_only_a_crouched_spawn_and_only_with_the_flag():
    st = np.zeros(1, STATE_DTYPE)
    np.testing.assert_array_equal(spawn_held(st, True), NEUTRAL)     # standing: map start
    st["ducked"] = 1
    np.testing.assert_array_equal(spawn_held(st, False), NEUTRAL)
    np.testing.assert_array_equal(spawn_held(st, True), (1, 1, 1))


def test_line_held_keys_is_roll_keys_convention():
    """state t was reached under tick t-1's engine keys; state 0 holds the start."""
    K = 4
    eng = keys_resolve_rows(_policy_line(n=30, seed=7))[0]
    n = len(eng) * K
    keys = line_held_keys(eng, K, n, (1, 1, 1))
    np.testing.assert_array_equal(keys[0], (1, 1, 1))
    ticks = np.repeat(eng, K, axis=0)
    for t in range(1, n):
        np.testing.assert_array_equal(keys[t], ticks[t - 1][list(HELD_HEADS)])
    assert keys.dtype == np.int32 and keys.shape == (n, 3)


# --------------------------------------------------------------------------
# 2. who stood at a state: the engine rows and the view, not the policy rows
# --------------------------------------------------------------------------
def test_survivor_prefix_defaults_to_the_action_tables():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, (12, 6))
    tabs = []
    for i in range(5):
        t = base.copy()
        t[3 + i:, 2] = rng.integers(0, 3, len(t) - 3 - i)
        tabs.append(t)
    w = rng.random(5)
    for j in range(5):
        np.testing.assert_array_equal(survivor_probs(tabs, w, j),
                                      survivor_probs(tabs, w, j, prefix_tables=tabs))


def test_keep_and_an_explicit_press_of_the_held_key_stand_in_one_state():
    eng = np.zeros((4, 6), np.int64)
    eng[:, 3] = 0                                   # side A held from decision 0
    pa = keys_policy_rows(eng)[0]                   # KEEP after the first press
    pb = pa.copy()
    pb[1:, 3] = 1                                   # the same line, re-pressing A explicitly
    np.testing.assert_array_equal(keys_resolve_rows(pb)[0], eng)
    p_eng_only = survivor_probs([pa, pb], None, 0)  # prefix on the POLICY rows: they split
    p = survivor_probs([pa, pb], None, 0, prefix_tables=[eng, eng])
    assert p_eng_only[2, 3, KEEP] == 1.0            # decision 2: b already "left" the state
    np.testing.assert_allclose(p[2, 3, [KEEP, 1]], [0.5, 0.5])   # one state, two choices


def test_lines_that_agree_in_bins_but_not_in_view_are_not_survivors():
    acts = np.zeros((5, 6), np.int64)
    acts[:, 2] = 1
    va = np.zeros((5, 2), np.float32)
    vb = va.copy()
    vb[1, 0] = 7.5                                  # diverged in the view at decision 1
    b2 = acts.copy()
    b2[3, 2] = 2
    pre_a = np.concatenate([acts, va], 1)
    pre_b = np.concatenate([b2, vb], 1)
    p = survivor_probs([acts, b2], None, 0, prefix_tables=[pre_a, pre_b])
    assert p[3, 2, 1] == 1.0                        # b is elsewhere by decision 2
    assert survivor_probs([acts, b2], None, 0)[3, 2, 1] == 0.5      # the bins-only rule


# --------------------------------------------------------------------------
# 3. the drawn view z: a BC column, preferred over the rebuilt one
# --------------------------------------------------------------------------
def _bc_args(n, keys=True):
    rng = np.random.default_rng(1)
    st = np.zeros(n, STATE_DTYPE)
    st["origin"] = rng.normal(0, 100, (n, 3))
    act = keys_policy_rows(keys_resolve_rows(_policy_line(n=n))[0])[0]
    act[:, 0], act[:, 1] = 7, 3                     # the view columns are NEUTRAL bins
    feat = keys_policy_rows(keys_resolve_rows(_policy_line(n=n))[0])[2]
    meta = {"obs_reward": False, "n_latch": 0, "act_every": 4, "keys_hold": 1,
            "view_absolute": "velocity", "view_mode": 1}
    view = np.stack([rng.uniform(-170, 170, n), rng.uniform(-70, 30, n)], 1).astype(np.float32)
    view[0] = (180.0, 30.0)                         # saturated: |u| = 1 on both heads
    return (st, np.zeros((n, 15), np.float32), np.zeros(n, np.float32), act,
            np.ones(n, np.float32), np.zeros(n, np.int32), meta), view, feat


def _pre_change_bc(tmp_path):
    """surfgym/bc.py as it was before the view_z column, imported as a sibling package."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=300", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    src = None
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/surfgym/bc.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"def keys_resolve_rows" not in s.stdout:
            src = s.stdout
            break
    if src is None:
        pytest.skip("no pre-change bc.py in the history")
    pkg = tmp_path / "surfgym_pre"
    shutil.copytree(ROOT / "python" / "surfgym", pkg,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "bc.py").write_bytes(src)
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module("surfgym_pre.bc")
    finally:
        sys.path.remove(str(tmp_path))


def test_a_file_without_view_z_is_byte_identical_to_the_pre_change_writer(tmp_path):
    from surfgym.bc import save_bc_dataset
    old = _pre_change_bc(tmp_path)
    args, view, feat = _bc_args(9)
    a, b = tmp_path / "new.npz", tmp_path / "old.npz"
    save_bc_dataset(a, *args, view=view, keys=feat)
    old.save_bc_dataset(b, *args, view=view, keys=feat)
    assert a.read_bytes() == b.read_bytes()


def test_view_z_round_trips_merges_and_is_preferred_by_bcdataset(tmp_path):
    import torch
    from surfgym.bc import BCDataset, load_bc_arrays, merge_bc_files, save_bc_dataset
    from surfgym.view import z_from_view_any
    args, view, feat = _bc_args(9)
    vz = z_from_view_any(view, "velocity", 1.33)
    vz[0] = (5.25, -4.5)                            # beyond atanh(0.999) = 3.8: unrebuildable
    f1, f2, f3 = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "ab.npz"
    save_bc_dataset(f1, *args, view=view, keys=feat, view_z=vz)
    save_bc_dataset(f2, *args, view=view, keys=feat, view_z=vz)
    np.testing.assert_array_equal(load_bc_arrays(f1)["view_z"], vz)
    merge_bc_files([f1, f2], f3)
    np.testing.assert_array_equal(load_bc_arrays(f3)["view_z"], np.concatenate([vz, vz]))
    kw = dict(n_latch=0, obs_reward=False, view_continuous=True, yaw_adaptive=True,
              pitch_rate_max_deg=1.33, view_absolute="velocity", keys_hold=True)
    ds = BCDataset(f1, torch.device("cpu"), **kw)
    np.testing.assert_array_equal(ds.vz.numpy(), vz)
    assert "drawn z" in ds.view_note
    f4 = tmp_path / "plain.npz"
    save_bc_dataset(f4, *args, view=view, keys=feat)
    ds4 = BCDataset(f4, torch.device("cpu"), **kw)
    assert abs(float(ds4.vz[0, 0]) - float(np.arctanh(0.999))) < 1e-5   # the clipped rebuild
    with pytest.raises(ValueError):
        save_bc_dataset(tmp_path / "x.npz", *args, keys=feat, view_z=vz)  # z without a view


# --------------------------------------------------------------------------
# 4. the view ceilings: train_fast's VIEW_TICK
# --------------------------------------------------------------------------
RAMP = {"from_ms": 10.0, "to_ms": 7.63, "steps": 800000000, "origin_step": 14364966912}


def test_view_tick_is_the_ramp_start_and_otherwise_the_tick_itself():
    t = TickClock(7.64224)
    assert view_tick({}, t) is t and view_tick(None, None) is None
    assert view_tick({"tick_schedule": RAMP}, t).is_reference
    assert view_tick({"tick_schedule": "8:7.63:1e8"}, t).requested_ms == 8.0


def test_the_trainer_record_ckpt_and_beam_tas_agree_on_the_anchor():
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "VIEW_TICK = TICK if tick_sched is None else TickClock(tick_sched.from_ms)" in tf
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert "VTICK = view_tick(cfg, TICK)" in rc
    bt = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert "vtick = view_tick(cfg, tick)" in bt


@needs_core
@pytest.mark.parametrize("adaptive", [True, False])
def test_build_sim_anchors_the_view_ceilings_to_the_ramp_start(adaptive):
    import beam_tas
    tick = TickClock(7.64224)
    cfg = {"pitch_rate": 1.33, "maxvel": 3500.0, "yaw_adaptive": adaptive,
           "view_absolute": "velocity"}
    plain = beam_tas.build_sim(cfg, str(SKI), 1, 100, tick=tick)
    ramped = beam_tas.build_sim(dict(cfg, tick_schedule=RAMP), str(SKI), 1, 100, tick=tick)
    # without a schedule: scaled to the tick the core runs at, exactly as before
    assert abs(plain.config.pitch_rate_max_deg - tick.per_tick(1.33)) < 1e-6
    # with the ramp: the trainer's 1.33 deg/tick (VIEW_TICK = 10 ms), the physics still 7.667
    assert abs(ramped.config.pitch_rate_max_deg - 1.33) < 1e-6
    assert abs(ramped.config.yaw_rate_max_deg - 10.0) < 1e-6
    if not adaptive:
        assert abs(plain.config.yaw_rate_max_deg - tick.per_tick(10.0)) < 1e-6
    assert list(ramped.tick_pattern) == list(plain.tick_pattern) == [8, 8, 7]


# --------------------------------------------------------------------------
# 5. the wrapper's record, and the critic probe on a decision tick
# --------------------------------------------------------------------------
class _Scripted:
    """A keys-hold eval wrapper whose policy is a script of POLICY-space rows; _obs runs as in
    every real _decide (it is where the held keys sync to episode starts)."""

    @staticmethod
    def make(core, rows, k):
        import torch
        import train_fast
        train_fast.set_keys_hold(True)
        stub = types.SimpleNamespace(view_continuous=False, view_absolute=None,
                                     yaw_side=None, gru=None)

        class W(train_fast._TorchPolicyBase):
            def _decide(self, obs):
                self._obs(obs)
                r = np.asarray(rows[min(self.i, len(rows) - 1)], np.int32)[None].repeat(
                    core.num_envs, 0).copy()
                self.i += 1
                return r
        w = W(stub, train_fast.HeadPacker(torch.device("cpu")), torch.device("cpu"),
              None, core, k, 1, keys_hold=True)
        w.i = 0
        return w


@pytest.fixture
def _nvec():
    import train_fast
    yield
    train_fast.set_keys_hold(False)


@needs_core
def test_the_wrapper_publishes_the_policy_row_only_when_asked(_nvec):
    from surfgym import SurfCore, default_config
    core = SurfCore(str(SKI), default_config(num_envs=2, spawn_mode=2, max_episode_ticks=500))
    obs = core.reset(0)
    rows = [[7, 3, 1, 1, 0, 0], [7, 3, KEEP, 1, 0, KEEP]]   # press A, then RE-press A
    w = _Scripted.make(core, rows, 4)
    w.act(obs)
    assert w.policy_act is None                               # off: nothing copied
    w2 = _Scripted.make(core, rows, 4)
    w2.record_policy_act = True
    eng = w2.act(obs)
    np.testing.assert_array_equal(w2.policy_act[0], rows[0])  # the policy's own row ...
    assert eng[0, 2] == 0 and eng[0, 3] == 0                  # ... and the resolved engine row
    for _ in range(3):
        eng = w2.act(obs)
    eng = w2.act(obs)                                          # decision 2
    np.testing.assert_array_equal(w2.policy_act[0], rows[1])  # the explicit re-press is kept
    assert eng[0, 3] == 0


@needs_core
@pytest.mark.parametrize("restore", [False, True])
def test_a_critic_probe_on_the_decision_tick_keeps_the_held_keys(_nvec, restore):
    import beam_tas
    from surfgym import SurfCore, default_config
    K = 4
    core = SurfCore(str(SKI), default_config(num_envs=2, spawn_mode=2, max_episode_ticks=5000))
    obs = core.reset(0)
    rows = [[7, 3, KEEP, 1, 0, KEEP]] + [[7, 3, KEEP, KEEP, 0, KEEP]] * 20   # A, then keep
    w = _Scripted.make(core, rows, K)
    for t in range(3 * K):                         # three decisions, A held throughout
        a = w.act(obs)
        obs, _r, _d, _tr, _tm = core.step(a)
    assert (w.keys.state[:, 1] == 0).all()
    snap = beam_tas.keys_snapshot(w)
    w._obs(obs)                                    # value_fn: the critic reads this tick's row
    if restore:
        beam_tas.keys_restore(w, snap)
    a = w.act(obs)                                 # the next decision, on the SAME tick
    if restore:
        assert (a[:, 3] == 0).all(), "the probe must not reset the held strafe"
    else:
        assert (a[:, 3] == 1).all(), "without the restore the probe resets it (the old bug)"


# --------------------------------------------------------------------------
# 6. the drivers: beam_tas records, plan_to_bc uses, exit_local passes the spine keys
# --------------------------------------------------------------------------
def test_the_planner_records_and_the_distiller_uses_the_policys_rows():
    bt = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    for frag in ("_w.record_policy_act = True", "_w.spawn_duck = SPAWN_DUCK",
                 "gpol.spawn_duck = SPAWN_DUCK", "hist_p[:d + 1, losers] = hist_p[:d + 1, donors]",
                 "hist_z[:d + 1, losers] = hist_z[:d + 1, donors]", '"acts_pol_all"',
                 '"view_z_all"', "ksaved = keys_snapshot(spol)", "keys_restore(spol, ksaved)"):
        assert frag in bt, frag
    pb = (ROOT / "tools" / "plan_to_bc.py").read_text(encoding="utf-8")
    for frag in ("keys_resolve_rows(_pol, KSTART)", "prefix_tables=prefix_tabs",
                 "line_held_keys(best_acts, K, len(best_states), KSTART)[sel]",
                 'with_suffix(".keys.npy")'):
        assert frag in pb, frag
    ex = (ROOT / "tools" / "exit_local.py").read_text(encoding="utf-8")
    assert '"--demo-grow", "0", *demo_keys, *args.extra]' in ex
    assert 'args.demo_keys == "auto" and keys_train.exists()' in ex
