"""Reverse Curriculum Generation from the goal (Florensa, Held, Wulfmeier,
Zhang, Abbeel, CoRL 2017, arXiv 1707.05300) - the three pieces that build
and consume the goal-rooted start pool:

(a) ``explore_phase1 --roots-goal``: ``goal_roots`` stands the roots on the
    floor inside the finish box footprint, at rest (stub geometry and the
    real labyrinth_left100); a box with no floor in reach falls back to its
    centre; a short REAL goal-rooted search never "finishes", keeps its
    roots at depth 0 inside the box and grows depth away from the goal.
(b) ``tools/goal_curriculum_pool.py``: rows in DESCENDING depth with the
    goal-nearest state outside the hull-inflated finish box LAST; the three
    velocity modes; tick / stuck zeroed; 0.1 u dedup; the provenance json;
    a forward (map-start rooted) archive is refused.
(c) ``DemoCurriculum(front_frac=)`` / ``--demo-front-frac``: off draws the
    same indices bit for bit; on puts that share of the draws on the
    frontier band; refused without grow; the trainer / recorder wiring.
(d) a CPU trainer smoke on labyrinth_left100 with a pool from a real
    goal-rooted search: the curriculum loads it, realized spawns are matched
    to pool rows, tau moves toward the start, and record_ckpt records the
    checkpoint (the record gate's backend) in greedy, stochastic and
    drop-spawn modes.

(a)-real, (b)-integration and (d) need the built core and the labyrinth map
(SURFCORE_DLL; SURF_TEST_MAPS_POOL = the directory holding
labyrinth_left100.bsp and its caches, e.g. C:/RL_Surf/maps_pool from a
worktree).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                            # noqa: E402
from surfgym.respawn import DemoCurriculum                      # noqa: E402

import explore_phase1 as ep                                     # noqa: E402
import goal_curriculum_pool as gcp                              # noqa: E402

_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))


def _find_map(stem: str):
    dirs = [os.environ.get("SURF_TEST_MAPS_POOL"), ROOT / "maps_pool", ROOT / "maps"]
    for d in dirs:
        if d and (Path(d) / f"{stem}.bsp").is_file():
            return Path(d) / f"{stem}.bsp"
    return None


LAB100 = _find_map("labyrinth_left100")
needs_lab = pytest.mark.skipif(
    LAB100 is None or not DLL.exists()
    or not (LAB100 is not None and LAB100.with_suffix(".zones.json").exists()),
    reason="needs the built core + labyrinth_left100 with its zones "
           "(SURFCORE_DLL, SURF_TEST_MAPS_POOL)")


# ==========================================================================
# (a) goal roots
# ==========================================================================
def _stub(floor_z=0.0):
    fc = ep.FakeCore("stub.bsp", SimpleNamespace(num_envs=1, max_episode_ticks=10))
    fc.floor_z = floor_z
    return fc


BOX = {"mins": [100.0, -40.0, -10.0], "maxs": [300.0, 60.0, 80.0]}


def test_goal_roots_stand_on_the_floor_inside_the_footprint_at_rest():
    roots, info = ep.goal_roots(_stub(0.0), BOX, 16, np.random.default_rng(0))
    assert len(roots) == 16 and info == {"floor": 16, "fallback": False,
                                         "samples": 16}
    assert roots.dtype == STATE_DTYPE
    o = roots["origin"].astype(np.float64)
    # inside the footprint, inset by the hull half-width
    assert np.all(o[:, 0] >= 116.0) and np.all(o[:, 0] <= 284.0)
    assert np.all(o[:, 1] >= -24.0) and np.all(o[:, 1] <= 44.0)
    # on the floor: floor z + the standing half height
    assert np.allclose(o[:, 2], 36.0)
    assert not roots["velocity"].any() and not roots["basevelocity"].any()
    assert np.all((roots["yaw"] >= 0.0) & (roots["yaw"] < 360.0))
    assert len(set(np.round(roots["yaw"], 3).tolist())) == 16       # random
    assert np.all(roots["onground"] == -1) and not roots["tick"].any()
    # the grid spans the footprint (4 distinct x, 4 distinct y)
    assert len(set(o[:, 0].tolist())) == 4 and len(set(o[:, 1].tolist())) == 4


def test_goal_roots_fall_back_to_the_box_centre_without_a_floor():
    for floor in (None, -500.0):          # no floor at all / one out of reach
        roots, info = ep.goal_roots(_stub(floor), BOX, 16,
                                    np.random.default_rng(0))
        assert info["fallback"] and info["floor"] == 0 and len(roots) == 1
        assert np.allclose(roots["origin"][0], [200.0, 10.0, 35.0])
        assert not roots["velocity"].any()


def test_goal_roots_narrow_box_uses_its_centre_line():
    box = {"mins": [0.0, 0.0, 0.0], "maxs": [20.0, 400.0, 100.0]}   # < 1 hull wide
    roots, info = ep.goal_roots(_stub(0.0), box, 9, np.random.default_rng(0))
    assert not info["fallback"] and len(roots) == 3 == info["samples"]
    assert np.allclose(roots["origin"][:, 0], 10.0)                 # the centre line
    assert np.allclose(roots["origin"][:, 1], [16.0, 200.0, 384.0])


@needs_lab
def test_goal_roots_on_the_real_labyrinth():
    from surfgym.core import SurfCore, default_config
    from surfgym.zones import load_zones
    core = SurfCore(str(LAB100), default_config(num_envs=1, spawn_mode=2,
                                                lidar_w=0, lidar_h=0))
    try:
        box = load_zones(str(LAB100), create=False)["end"]
        roots, info = ep.goal_roots(core, box, 16, np.random.default_rng(0))
        assert len(roots) == 16 and info["floor"] == 16 and not info["fallback"]
        o = roots["origin"].astype(np.float64)
        lo, hi = np.asarray(box["mins"]), np.asarray(box["maxs"])
        assert np.all((o[:, :2] >= lo[:2]) & (o[:, :2] <= hi[:2]))
        assert not roots["velocity"].any()
        for p in o:
            t0 = core.trace(tuple(p), tuple(p), hull=0)
            assert not t0.startsolid                    # the hull fits
            t = core.trace(tuple(p), (p[0], p[1], p[2] - 2.0), hull=0)
            assert t.fraction < 1.0 and float(t.normal[2]) >= 0.7   # standing
        # a root IS at the goal: the core would complete it on tick one
        assert gcp.in_goal(roots, box).all()
    finally:
        core.close()


# ==========================================================================
# (b) the exporter
# ==========================================================================
GBOX = {"mins": [0.0, 0.0, 0.0], "maxs": [100.0, 100.0, 100.0]}
SPAWN = [2000.0, 50.0, 36.0]


def _archive(tmp_path: Path, roots="goal", extra_rows=()):
    """A synthetic goal-rooted archive: roots in the box, a row inside the
    hull-inflated box only, rows walking away along +x with depth, a
    duplicate origin, a ducked row. Returns (npz path, rows' depths)."""
    rows = [  # (origin, depth, ducked)
        ((50.0, 50.0, 36.0), 0, 0),            # root in the box
        ((20.0, 80.0, 36.0), 0, 0),            # root in the box
        ((110.0, 50.0, 36.0), 3, 0),           # inside the INFLATED box (+16)
        ((130.0, 50.0, 36.0), 7, 0),           # just outside: the goal-nearest row
        ((400.0, 50.0, 36.0), 40, 0),
        ((400.04, 50.0, 36.0), 55, 0),         # duplicate at 0.1 u, deeper
        ((800.0, 50.0, 60.0), 90, 1),          # ducked, far
        ((1990.0, 50.0, 36.0), 250, 0),        # next to the spawn
        ((1500.0, 300.0, 36.0), 200, 0),
    ] + list(extra_rows)
    n = len(rows)
    st = np.zeros(n, STATE_DTYPE)
    for i, (o, d, dk) in enumerate(rows):
        st[i]["origin"] = o
        st[i]["velocity"] = (100.0 + i, -20.0, 5.0)
        st[i]["basevelocity"] = (1.0, 2.0, 3.0)
        st[i]["yaw"] = 10.0 * i
        st[i]["ducked"] = dk
        st[i]["tick"] = 77
        st[i]["stuck_ticks"] = 3
        st[i]["progress"] = 5.0
    depth = np.array([r[1] for r in rows], np.int64)
    key = np.arange(n, dtype=np.int64) * 7 + 3
    parent = np.array([-1, -1] + [0] * (n - 2), np.int64)
    dist = depth.astype(np.float64) * 2.5
    p = tmp_path / "arch" / "archive.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, key=key, state=st, seen=np.ones(n, np.int64),
             chosen=np.zeros(n, np.int64), parent=parent, depth=depth,
             dist=dist)
    meta = {"map": "toy", "tool": "explore_phase1", "cell": 64.0, "seed": 0,
            "tick_ms": 10, "goal_box": GBOX, "spawns": [SPAWN]}
    if roots is not None:
        meta["roots"] = roots
    (p.parent / "archive.meta.json").write_text(json.dumps(meta), "utf-8")
    return p, depth


def test_export_orders_by_descending_depth_and_drops_goal_states(tmp_path):
    arch, _ = _archive(tmp_path)
    out = tmp_path / "pools" / "toy_goalpool.npy"
    prov = gcp.export(arch, out, "zero", argv=["test"])
    pool = np.load(out, allow_pickle=False)
    assert pool.dtype == STATE_DTYPE
    x = pool["origin"][:, 0].astype(np.float64)
    # 9 rows - 2 roots - 1 in the inflated box - 1 duplicate = 5
    assert len(pool) == 5 == prov["rows"]
    assert prov["excluded_in_goal"] == 3 and prov["dedup_dropped"] == 1
    # descending depth: 250 (by the spawn), 200, 90, 40 (the shallower twin
    # of the duplicate), 7 (the goal-nearest row outside the box) LAST
    assert np.allclose(x, [1990.0, 1500.0, 800.0, 400.0, 130.0])
    assert prov["first_row_depth"] == 250 and prov["last_row_depth"] == 7
    assert prov["depth_ticks"] == [7, 250]
    assert prov["depth_seconds"] == [0.07, 2.5]
    assert not gcp.in_goal(pool, GBOX).any()
    assert not pool["tick"].any() and not pool["stuck_ticks"].any()
    assert not pool["progress"].any() and not pool["seg_hint"].any()
    # the start report: row 0 sits 10 u from the spawn, inside one 64 u cell
    assert prov["start_reached"] and prov["start_row"] == 0
    assert prov["start_gap"] == pytest.approx(10.0)
    assert prov["start_row_depth"] == 250
    assert prov["rank_corr_depth_vs_goal_dist"] == pytest.approx(1.0)
    js = json.loads(out.with_suffix(".json").read_text("utf-8"))
    assert js["provenance"].startswith("machine-generated")
    assert js["velocity"] == "zero" and js["argv"] == ["test"]
    assert js["archive_meta"]["roots"] == "goal" and js["git"]
    assert js["order"].startswith("DESCENDING depth")
    # --keep-goal-states keeps the box rows, still goal-nearest LAST
    pk, _, ck = gcp.build_pool(*_cols(arch), GBOX, "zero", keep_goal_states=True)
    assert ck["excluded_in_goal"] == 0 and len(pk) == 8
    assert pk["origin"][-1][0] in (50.0, 20.0)           # a depth-0 root last


def _cols(arch):
    with np.load(arch, allow_pickle=False) as z:
        return z["state"], z["depth"], z["key"]


def test_export_velocity_modes(tmp_path):
    arch, _ = _archive(tmp_path)
    st, depth, key = _cols(arch)
    z, src, _ = gcp.build_pool(st, depth, key, GBOX, "zero")
    k, src_k, _ = gcp.build_pool(st, depth, key, GBOX, "keep")
    r, src_r, _ = gcp.build_pool(st, depth, key, GBOX, "reverse")
    assert np.array_equal(src, src_k) and np.array_equal(src, src_r)
    assert not z["velocity"].any() and not z["basevelocity"].any()
    assert np.array_equal(k["velocity"], st["velocity"][src])
    assert np.array_equal(k["basevelocity"], st["basevelocity"][src])
    assert np.array_equal(r["velocity"], -st["velocity"][src])
    assert not r["basevelocity"].any()
    want_yaw = np.mod(st["yaw"][src].astype(np.float64) + 180.0, 360.0)
    assert np.allclose(r["yaw"], want_yaw)
    assert np.array_equal(z["yaw"], st["yaw"][src])      # zero keeps the yaw
    for p in (z, k, r):                                  # same rows, same order
        assert np.array_equal(p["origin"], st["origin"][src])
        assert not p["tick"].any() and not p["stuck_ticks"].any()
    with pytest.raises(ValueError, match="velocity"):
        gcp.build_pool(st, depth, key, GBOX, "backwards")


def test_export_in_goal_uses_the_ducked_hull():
    s = np.zeros(2, STATE_DTYPE)
    s["origin"] = [[50.0, 50.0, 125.0], [50.0, 50.0, 125.0]]   # 25 u above the top
    s["ducked"] = [0, 1]
    # standing hull reaches 36 u below the origin (overlaps), ducked only 18
    assert gcp.in_goal(s, GBOX).tolist() == [True, False]


def test_export_refuses_a_forward_archive(tmp_path):
    arch, _ = _archive(tmp_path, roots=None)
    with pytest.raises(SystemExit, match="roots-goal"):
        gcp.export(arch, tmp_path / "x.npy")


def test_exported_pool_is_a_demo_curriculum_spine(tmp_path):
    extra = [((130.0 + 37.0 * i, 900.0 + 3.0 * i, 36.0), 300 + i, 0)
             for i in range(200)]
    arch, _ = _archive(tmp_path, extra_rows=extra)
    out = tmp_path / "p.npy"
    gcp.export(arch, out)
    pool = np.load(out, allow_pickle=False)
    dc = DemoCurriculum(pool, window=16, rate=0.1, min_ep=50.0, grow=16,
                        front_frac=2.0 / 3.0)
    # every row is creditable: its rounded origin is unique
    assert np.array_equal(dc.match(pool["origin"]), np.arange(len(pool)))
    start = np.zeros(1, STATE_DTYPE)
    start["origin"] = [SPAWN]
    p = dc.build_pool(start, pool_size=512, fresh_frac=0.1)
    idx = dc.match(p["origin"])
    # before any outcome: every curriculum start is the goal-nearest row
    assert set(idx[idx >= 0].tolist()) == {len(pool) - 1}
    assert int((idx < 0).sum()) == round(512 * 0.1)


# ==========================================================================
# (c) --demo-front-frac
# ==========================================================================
def _spine(n=1000):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 10.0
    return s


def _start_pool():
    s = np.zeros(4, STATE_DTYPE)
    s["origin"][:, 0] = -1000.0
    return s


def _drive(dc, n_iter, win_rate, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        pool = dc.build_pool(_start_pool(), pool_size=512, fresh_frac=0.1)
        idx = dc.match(pool["origin"])
        idx = idx[idx >= 0]
        dc.note_outcomes(idx, rng.random(len(idx)) < win_rate)


def test_front_frac_zero_draws_the_same_indices_bit_for_bit():
    S = _spine()
    a = DemoCurriculum(S, window=32, rate=0.1, min_ep=50.0, seed=7, grow=32)
    b = DemoCurriculum(S, window=32, rate=0.1, min_ep=50.0, seed=7, grow=32,
                       front_frac=0.0)
    rng = np.random.default_rng(1)
    for _ in range(300):
        pa = a.build_pool(_start_pool(), pool_size=512, fresh_frac=0.1)
        pb = b.build_pool(_start_pool(), pool_size=512, fresh_frac=0.1)
        assert pa.tobytes() == pb.tobytes()
        ia = a.match(pa["origin"])
        w = rng.random(len(ia)) < 0.6
        a.note_outcomes(ia[ia >= 0], w[ia >= 0])
        b.note_outcomes(ia[ia >= 0], w[ia >= 0])
        assert a.tau == b.tau
    assert a.tau < len(S) - 1                     # the curriculum did move


def test_front_frac_puts_its_share_on_the_frontier_band():
    S = _spine()
    D = 32
    dc = DemoCurriculum(S, window=D, rate=0.1, min_ep=50.0, seed=7, grow=D,
                        front_frac=2.0 / 3.0)
    _drive(dc, 400, win_rate=0.5)
    tau = dc.tau
    assert tau < len(S) - 1 - 4 * D               # well off the end
    pool = dc.build_pool(_start_pool(), pool_size=8192, fresh_frac=0.1)
    idx = dc.match(pool["origin"])
    demo = idx[idx >= 0]
    lo, hi = dc._band()
    assert (lo, hi) == (dc.tau, dc.tau + D - 1)
    assert demo.min() >= dc.tau and demo.max() <= len(S) - 1
    share = float(((demo >= lo) & (demo <= hi)).mean())
    # 2/3 on the band + the uniform third's own band share
    want = 2.0 / 3.0 + (1.0 / 3.0) * D / (len(S) - dc.tau)
    assert abs(share - want) < 0.03, (share, want)
    assert "of demo draws on the band" in dc.last_info
    # without it the band's share is only D / (n - tau)
    dc0 = DemoCurriculum(S, window=D, rate=0.1, min_ep=50.0, seed=7, grow=D)
    dc0.tau = dc.tau
    p0 = dc0.build_pool(_start_pool(), pool_size=8192, fresh_frac=0.1)
    i0 = dc0.match(p0["origin"])
    i0 = i0[i0 >= 0]
    lo0, hi0 = dc0._band()
    s0 = float(((i0 >= lo0) & (i0 <= hi0)).mean())
    assert s0 < 0.25 < share


def test_front_frac_is_refused_without_grow_or_out_of_range():
    with pytest.raises(ValueError, match="grow"):
        DemoCurriculum(_spine(), window=10, front_frac=0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DemoCurriculum(_spine(), window=10, grow=10, front_frac=1.5)


def test_front_frac_wiring_trainer_and_recorder():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-front-frac"' in src
    # dumped ONLY when on (flag-off run.json unchanged), restored on resume
    i = src.index('meta["config"]["demo_front_frac"]')
    assert "float(args.demo_front_frac) > 0.0" in src[i - 200:i]
    assert 'ck_cfg.get("demo_front_frac")' in src
    assert "front_frac=float(args.demo_front_frac)" in src
    import record_ckpt
    assert "demo_front_frac" in record_ckpt.TRAIN_ONLY


# ==========================================================================
# (b)+(a) integration and (d) the trainer smoke, on the real labyrinth
# ==========================================================================
def _search(out: Path, iters: int = 30, envs: int = 64):
    args = ep.build_parser().parse_args([
        "--map", str(LAB100), "--out", str(out), "--roots-goal",
        "--cell", "64", "--act-every", "4", "--ep-ticks", "3000",
        "--envs", str(envs), "--max-iters", str(iters), "--print-every", "1000",
        "--snapshot-every", "0", "--seed", "0"])
    assert ep.explore(args) == 0
    return out / "archive.npz"


@needs_lab
def test_real_goal_rooted_search_and_export(tmp_path):
    arch = _search(tmp_path / "search")
    with np.load(arch, allow_pickle=False) as z:
        st, par, dep, dist = z["state"], z["parent"], z["depth"], z["dist"]
    meta = json.loads((arch.parent / "archive.meta.json").read_text("utf-8"))
    assert meta["roots"] == "goal" and meta["goals"] == 0
    assert not list(arch.parent.glob("win_*"))           # nothing "finished"
    box = meta["goal_box"]
    roots = par < 0
    assert roots.any() and not dep[roots].any()
    assert gcp.in_goal(st[roots], box).all()
    inner = par >= 0
    assert np.all(dep[inner] > dep[par[inner]])
    assert len(dep) > 60                                  # it left the goal room
    fin = np.isfinite(dist)
    assert gcp.rank_corr(dep[fin], dist[fin]) > 0.9       # depth ~ geodesic order
    assert meta["start_gap"] is not None and meta["depth_max"] == int(dep.max())
    out = tmp_path / "pool.npy"
    prov = gcp.export(arch, out)
    pool = np.load(out, allow_pickle=False)
    assert len(pool) == prov["rows"] > 20
    assert not gcp.in_goal(pool, box).any()
    assert not pool["velocity"].any()
    dd = dep[np.array([int(np.flatnonzero(
        np.all(st["origin"] == r["origin"], axis=1))[0]) for r in pool])]
    assert np.all(np.diff(dd) <= 0)                       # descending depth
    assert dd[-1] == dep[~gcp.in_goal(st, box)].min()     # nearest row last


@needs_lab
def test_trainer_smoke_with_a_goal_pool_and_the_recorder(tmp_path):
    """CPU: train_fast on labyrinth_left100 with a pool from a real goal-
    rooted search under the sparse reward. The curriculum must load it,
    match realized spawns to rows (demo-tracked episodes > 0), see finishes
    from the goal end, and move tau toward the start; record_ckpt (the
    record gate's backend) must record the checkpoint in all three gate
    modes."""
    from test_unstuck import SMOKE_FLAGS, TRAIN, _run
    arch = _search(tmp_path / "search", iters=40, envs=128)
    pool = tmp_path / "labyrinth_left100_goalpool.npy"
    gcp.export(arch, pool)
    n = len(np.load(pool, allow_pickle=False))
    run = "goal_curriculum_smoke"
    rdir = ROOT / "runs" / run
    shutil.rmtree(rdir, ignore_errors=True)
    flags = [a for a in SMOKE_FLAGS if a != "--map" and not a.endswith(".bsp")
             and not a.startswith("--obs-potential")]
    extra = ["--map", str(LAB100), "--steps", "240000",
             "--respawn-frac", "0.9", "--goal-cell", "32",
             "--respawn-margin", "1", "--seed", "0", "--race-dist", "euclid",
             "--race-shaping", "0", "--time-pen", "0", "--int-coef", "0",
             "--demo-file", str(pool), "--demo-window", "8",
             "--demo-grow", "8", "--demo-rate", "0.1", "--demo-min-ep", "20",
             "--demo-front-frac", "0.667"]
    env_keep = os.environ.get("SELF_STATES")
    os.environ["SELF_STATES"] = "1"
    try:
        r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + flags + extra,
                 timeout=3600)
    finally:
        if env_keep is None:
            os.environ.pop("SELF_STATES", None)
        else:
            os.environ["SELF_STATES"] = env_keep
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-4000:]
    assert f"demo curriculum: {n} states from {pool}" in log
    assert "--demo-front-frac 0.667" in log
    assert "machine-generated" in log                     # the sidecar echoed
    assert "demo curriculum: tau <- earlier" in log       # it moved back
    tracked = [ln for ln in log.splitlines() if "demo-tracked eps" in ln]
    assert tracked
    eps = float(tracked[-1].split("demo-tracked eps")[1].split()[0].replace(",", ""))
    wins = float(tracked[-1].split("wins")[1].split()[0].replace(",", ""))
    assert eps > 0 and wins > 0                           # matched, and finished
    cfg = json.loads((rdir / "run.json").read_text("utf-8"))["config"]
    assert cfg["demo_front_frac"] == pytest.approx(0.667)
    assert cfg["demo_grow"] == 8 and cfg["demo_file"] == str(pool)
    ck = rdir / "ckpt_latest.pt"
    if not ck.exists():
        ck = rdir / "ckpt_final.pt"
    rec = ROOT / "tools" / "record_ckpt.py"
    for mode in ([], ["--stochastic"], ["--spawn", "mixed"]):
        out = rdir / f"rec{'_'.join(m.strip('-') for m in mode)}.jsonl"
        rr = _run([sys.executable, "-u", str(rec), str(ck), "--map", str(LAB100),
                   "--episodes", "1", "--ep-ticks", "200", "--out", str(out)] + mode,
                  timeout=900)
        assert rr.returncode == 0, (mode, rr.stdout[-3000:] + rr.stderr[-3000:])
        assert out.exists()
    shutil.rmtree(rdir, ignore_errors=True)
