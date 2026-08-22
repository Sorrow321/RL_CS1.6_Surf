"""--race-gravity: the gravity-directional goal graph and its plumbing.

The plain goal graph is an UNDIRECTED BFS over free space, so a voxel counts
as "close to the finish" even when the only path there is a one-way fall.
Measured on surf_src_cannonball that makes the final descent read as ~8.3k
units of NEGATIVE progress along the champion's own winning line, and
turning back at route vertex 1600 locally optimal - which is exactly where
234 greedy episodes of three control arms stopped.

``build_goal_field(..., gravity_dir=True)`` rebuilds the same field over a
directional graph (fall and air-strafe freely, climb only along geometry).
These tests pin the two halves that can break silently:

  * the graph semantics - a one-way shaft is one-way, a climb along
    geometry still works, and falling/strafing weights are BIT-identical to
    the plain graph (the directional field must still be metres-of-track,
    or the shaping stops telescoping);
  * the plumbing - ``--race-gravity 0`` must reuse the control's own cached
    field verbatim (same file, same signature, no rebake), the directional
    bake must live in its own cache, and the flag must restore from the
    checkpoint, land in the saved config, and be mirrored by
    ``tools/record_ckpt.py`` (whose audit refuses to record under semantics
    it was never taught).

CPU only - no GPU, no map, no bake.

    python -m pytest tests/python/test_race_gravity.py -q
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import goalfield                                   # noqa: E402
from surfgym.goalfield import _bfs_geodesic, build_goal_field   # noqa: E402

CELL = 32.0


# ------------------------------------------------------------ graph semantics

def _solve(occ, seed, gravity_dir):
    d, _reach, _it = _bfs_geodesic(occ, seed, CELL, gravity_dir=gravity_dir,
                                   device="cpu", verbose=False)
    return d.numpy()


def _sealed_pit():
    """Floor everywhere, a 1-voxel shaft walled in to z=5, open air above.

    Nothing solid reaches from the shaft to the goal plane, so the only way
    from the pit to the goal is UP through empty air."""
    occ = np.zeros((20, 5, 7), np.uint8)
    occ[0, :, :] = 1                       # floor
    occ[1:6, 1:4, 2] = 1                   # shaft wall -x
    occ[1:6, 1:4, 4] = 1                   # shaft wall +x
    occ[1:6, 1, 3] = 1                     # shaft wall -y
    occ[1:6, 3, 3] = 1                     # shaft wall +y
    seed = np.zeros(occ.shape, bool)
    seed[18] = True                        # goal plane, 18 cells overhead
    return occ, seed


def test_undirected_graph_pays_a_one_way_fall_as_a_path_back_up():
    # the defect itself, in one number: the pit reads 17 cells from the
    # finish although the player can only ever fall into it
    occ, seed = _sealed_pit()
    d = _solve(occ, seed, gravity_dir=False)
    assert d[1, 2, 3] == pytest.approx(17.0 * CELL)


def test_gravity_dir_cuts_the_climb_but_never_the_fall():
    occ, seed = _sealed_pit()
    d = _solve(occ, seed, gravity_dir=True)
    assert not np.isfinite(d[1, 2, 3])           # pit bottom: no way out
    assert not np.isfinite(d[1, 2, 0])           # floor away from the shaft
    assert d[19, 2, 3] == pytest.approx(CELL)    # above it: still falls in
    assert d[18, 2, 3] == pytest.approx(0.0)     # the goal plane itself
    assert not np.isfinite(d[0, 2, 0])           # solids stay unreachable


def test_gravity_dir_allows_a_climb_along_geometry():
    # a solid column up to the goal plane: every voxel beside it is
    # surface-adjacent, so the wavefront may climb the whole way
    occ = np.zeros((20, 5, 7), np.uint8)
    occ[0, :, :] = 1
    occ[1:18, 2, 5] = 1
    seed = np.zeros(occ.shape, bool)
    seed[18] = True
    d = _solve(occ, seed, gravity_dir=True)
    assert np.isfinite(d[1, 2, 4])                     # hugging the column
    assert d[1, 2, 4] >= 17.0 * CELL - 1e-6            # never cheaper than h
    assert np.isfinite(d[1, 2, 0])                     # walk over, climb up
    occ[1:18, 2, 5] = 0                                # delete the column
    assert not np.isfinite(_solve(occ, seed, True)[1, 2, 4])
    # ...and the undirected graph cannot see the difference: why this hid
    assert np.isfinite(_solve(occ, seed, False)[1, 2, 4])


def test_falling_and_strafing_are_bit_identical_to_the_plain_graph():
    # open air, single seed on the floor: every path is a fall plus strafe,
    # so the directional field must equal the plain one EXACTLY - the
    # euclidean step weights are untouched and the potential still
    # telescopes into metres-of-track
    occ = np.zeros((20, 5, 7), np.uint8)
    seed = np.zeros(occ.shape, bool)
    seed[0, 2, 3] = True
    du = _solve(occ, seed, gravity_dir=False)
    dg = _solve(occ, seed, gravity_dir=True)
    assert np.array_equal(du, dg)
    assert dg[5, 2, 3] == pytest.approx(5.0 * CELL)
    assert dg[5, 2, 0] == pytest.approx((3.0 * np.sqrt(2.0) + 2.0) * CELL)


def test_nothing_below_an_airborne_goal_claims_to_reach_it():
    occ = np.zeros((20, 5, 7), np.uint8)
    seed = np.zeros(occ.shape, bool)
    seed[19] = True
    assert np.isfinite(_solve(occ, seed, gravity_dir=False)).all()
    assert not np.isfinite(_solve(occ, seed, gravity_dir=True)[:19]).any()


# ------------------------------------------------------------------- the cache

class _StubCore:
    def __init__(self, bsp_path):
        self.bsp_path = str(bsp_path)


def _tiny_world(tmp_path, monkeypatch):
    """A 1-room map whose occupancy never touches the DLL or a real bsp."""
    bsp = tmp_path / "tiny.bsp"
    bsp.write_bytes(b"not a bsp, but _map_sig only stats it")
    occ = np.zeros((12, 5, 7), np.uint8)
    occ[0, :, :] = 1                                   # floor
    occ[1:11, 2, 5] = 1                                # climbable column
    mins = np.zeros(3, np.float64)
    monkeypatch.setattr(goalfield, "goal_occupancy",
                        lambda core, cell, cache_dir=None: (occ, mins))
    zone = {"mins": [0.0, 0.0, 10.0 * CELL], "maxs": [6.0 * CELL,
                                                      4.0 * CELL,
                                                      11.0 * CELL]}
    return _StubCore(bsp), zone


def test_flag_off_reuses_the_control_cache_and_never_rebakes(tmp_path,
                                                             monkeypatch):
    core, zone = _tiny_world(tmp_path, monkeypatch)
    plain = build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                             device="cpu", gravity_dir=False)
    assert (tmp_path / f"tiny.goal_{CELL:g}.npz").exists()

    def _explode(*a, **k):
        raise AssertionError("rebaked a field that was already cached")

    monkeypatch.setattr(goalfield, "goal_occupancy", _explode)
    # the control's call is the no-keyword one; gravity_dir=False must hit
    # the very same cache file with the very same signature
    again = build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                             device="cpu")
    off = build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                           device="cpu", gravity_dir=False)
    assert np.array_equal(plain.grid, again.grid)
    assert np.array_equal(plain.grid, off.grid)
    assert off.reach_max == plain.reach_max


def test_the_directional_bake_gets_its_own_cache_file(tmp_path, monkeypatch):
    core, zone = _tiny_world(tmp_path, monkeypatch)
    plain = build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                             device="cpu")
    grav = build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                            device="cpu", gravity_dir=True)
    assert (tmp_path / f"tiny.goalg_{CELL:g}.npz").exists()
    # different graph, different field: the floor away from the column can
    # only be reached by falling, so the directional bake must not match
    assert not np.array_equal(plain.grid, grav.grid)
    # and a directional request must never be served the plain cache
    (tmp_path / f"tiny.goalg_{CELL:g}.npz").unlink()

    def _explode(*a, **k):
        raise AssertionError("served the PLAIN cache to a directional build")

    monkeypatch.setattr(goalfield, "goal_occupancy", _explode)
    with pytest.raises(AssertionError):
        build_goal_field(core, zone, cell=CELL, cache_dir=tmp_path,
                         device="cpu", gravity_dir=True)


# ---------------------------------------------------------------- the plumbing

def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _add_argument(tree, flag):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            and n.args and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == flag]


def _assigns_to_args(tree, attr):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == attr
                    and isinstance(t.value, ast.Name) and t.value.id == "args"
                    for t in n.targets)]


def _calls_named(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_trainer_registers_the_flag_and_defaults_it_off():
    tree = _tree(ROOT / "python" / "train_fast.py")
    got = _add_argument(tree, "--race-gravity")
    assert len(got) == 1
    kw = {k.arg: k.value for k in got[0].keywords}
    # default=None is what makes a checkpoint restore possible at all: a
    # hard 0 default is indistinguishable from "the user asked for off"
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None
    assert {c.value for c in kw["choices"].elts} == {0, 1}
    # restore + default: two assignments, and the OFF default is 0
    assigns = _assigns_to_args(tree, "race_gravity")
    assert len(assigns) == 2
    consts = [a.value.value for a in assigns
              if isinstance(a.value, ast.Constant)]
    assert consts == [0]


def test_trainer_restores_the_flag_from_the_checkpoint_config():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    # the same shape every other obs/reward flag uses: read it off the
    # checkpoint when it was not passed, and SAY so in the restored list
    assert 'ck_cfg.get("race_gravity")' in src
    assert 'restored.append(f"race_gravity={args.race_gravity}")' in src


def test_trainer_saves_the_flag_into_the_run_config():
    tree = _tree(ROOT / "python" / "train_fast.py")
    hits = [d for d in ast.walk(tree) if isinstance(d, ast.Dict)
            for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and k.value == "race_gravity"
            and isinstance(v, ast.Attribute) and v.attr == "race_gravity"]
    assert hits, "race_gravity must land in the saved config"


def test_every_trainer_field_build_threads_the_flag():
    # the eval-progress field, the kill-aware shaping field and the respawn
    # binning field all have to agree about gravity, or the arm trains on
    # one potential and is scored on another
    tree = _tree(ROOT / "python" / "train_fast.py")
    calls = _calls_named(tree, "build_goal_field")
    assert calls
    for c in calls:
        assert any(k.arg == "gravity_dir" for k in c.keywords), \
            f"build_goal_field at line {c.lineno} ignores --race-gravity"


def test_record_ckpt_mirrors_the_flag():
    tree = _tree(ROOT / "tools" / "record_ckpt.py")
    calls = _calls_named(tree, "build_goal_field")
    assert calls
    for c in calls:
        assert any(k.arg == "gravity_dir" for k in c.keywords), \
            "record_ckpt would feed --obs-reward from the WRONG field"


def test_record_ckpt_audit_accepts_a_gravity_config():
    from record_ckpt import audit_cfg
    cfg = {"reward": "race", "map": "surf_src_cannonball", "race_gravity": 1,
           "race_kill_aware": 0, "obs_reward": True}
    audit_cfg(cfg, strict=True)          # must not raise
    with pytest.raises(SystemExit):      # the guard still works
        audit_cfg(dict(cfg, race_antigravity=1), strict=True)
