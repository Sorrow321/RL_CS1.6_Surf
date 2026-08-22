"""build_route.py's three non-champion line sources, and what they preserve.

Round 19's xARC finished surf_src_cannonball by paying arc length along a
reference line - but that line was the CHAMPION's own winning trajectory, so
the finishes proved a monotone progress coordinate works, not that the agent
solved the map unaided. These tests cover the machinery that answers "does the
line have to be a champion's?":

* ``archive_chain`` - a line from a reward-free Go-Explore phase-1 archive,
  which contains no champion and no reward;
* ``decimate`` / ``quantize`` - degrading a line until it carries the CORRIDOR
  and not the racing line;
* ``--allow-unfinished`` - the deliberate, visible relaxation of "a route must
  reach the finish", which those lines need and the champion path must not
  silently get.

Everything here is CPU-only geometry: no map, no DLL, no GPU, no torch.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.route import ArcProgress, resample_polyline  # noqa: E402


def _load_build_route():
    spec = importlib.util.spec_from_file_location(
        "build_route_mod", ROOT / "tools" / "build_route.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BR = _load_build_route()


def _l_route(n=200, leg=4000.0):
    """An L-shaped polyline resampled at 128 u, like the tool's own selftest."""
    t = np.linspace(0, leg, n)
    line = np.concatenate([
        np.stack([t, np.zeros_like(t), np.zeros_like(t)], 1),
        np.stack([np.full_like(t, leg), t, np.zeros_like(t)], 1)])
    pts, _ = resample_polyline(line, 128.0)
    return pts.astype(np.float64)


def _fake_archive(path):
    """A 6-cell archive: a spine 0->1->2->3->4 plus a shallow branch 1->5."""
    n = 6
    st = np.zeros(n, dtype=[("origin", np.float32, 3)])
    st["origin"] = np.stack([np.arange(n) * 100.0, np.zeros(n), np.zeros(n)], 1)
    st["origin"][5] = [50.0, 900.0, 0.0]
    np.savez(path,
             state=st,
             parent=np.array([-1, 0, 1, 2, 3, 1], np.int64),
             depth=np.array([0, 10, 20, 30, 40, 15], np.int64),
             dist=np.array([9.0, 8.0, 7.0, 6.0, 5.0, 99.0]))
    return path


# --------------------------------------------------------------------- chain

def test_archive_chain_walks_parents_root_first(tmp_path):
    xyz, leaf, info = BR.archive_chain(_fake_archive(tmp_path / "a.npz"), "depth")
    assert leaf == 4 and info["chain"] == 5
    assert np.allclose(xyz[:, 0], [0, 100, 200, 300, 400])
    assert info["leaf_depth_ticks"] == 40


def test_archive_chain_leaf_rules_can_disagree(tmp_path):
    """The two rules are different questions and must be able to differ."""
    p = tmp_path / "b.npz"
    st = np.zeros(4, dtype=[("origin", np.float32, 3)])
    st["origin"] = np.stack([np.arange(4) * 100.0, np.zeros(4), np.zeros(4)], 1)
    np.savez(p, state=st, parent=np.array([-1, 0, 1, 0], np.int64),
             depth=np.array([0, 10, 40, 5], np.int64),
             dist=np.array([9.0, 8.0, 7.0, 0.5]))
    _, by_depth, _ = BR.archive_chain(p, "depth")
    _, by_dist, _ = BR.archive_chain(p, "dist")
    assert by_depth == 2 and by_dist == 3


def test_archive_chain_survives_a_parent_cycle(tmp_path):
    """A corrupted snapshot must truncate, not hang."""
    p = tmp_path / "c.npz"
    st = np.zeros(3, dtype=[("origin", np.float32, 3)])
    np.savez(p, state=st, parent=np.array([1, 2, 1], np.int64),
             depth=np.array([0, 1, 2], np.int64), dist=np.zeros(3))
    xyz, _, info = BR.archive_chain(p, "depth")
    assert 0 < info["chain"] <= 3 and len(xyz) == info["chain"]


def test_archive_chain_rejects_an_unknown_leaf_rule(tmp_path):
    with pytest.raises(ValueError):
        BR.archive_chain(_fake_archive(tmp_path / "d.npz"), "champion")


# ----------------------------------------------------------------- degrading

def test_decimate_keeps_the_ends_and_drops_the_middle():
    pts = _l_route()
    d = BR.decimate(pts, 32)
    assert np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])
    assert len(pts) // 32 <= len(d) <= len(pts) // 32 + 2


def test_decimate_below_two_is_the_identity():
    pts = _l_route()
    assert BR.decimate(pts, 1) is pts and BR.decimate(pts, 0) is pts


def test_decimation_shortens_the_path_because_chords_cut_corners():
    pts = _l_route()
    _, full = resample_polyline(pts, 128.0)
    _, cut = resample_polyline(BR.decimate(pts, 32), 128.0)
    assert cut < full


def test_quantize_snaps_to_lattice_centres_and_dedups():
    pts = _l_route()
    q = BR.quantize(pts, 1024.0)
    assert np.allclose((q + 512.0) % 1024.0, 0.0)
    assert len(q) < len(pts)
    assert np.abs(np.diff(q, axis=0)).sum(1).min() > 0  # no repeats


def test_quantize_error_is_bounded_by_the_lattice_half_diagonal():
    pts = _l_route()
    c = 1024.0
    q = BR.quantize(pts, c)
    d = np.sqrt(((pts[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(1)
    assert d.max() <= np.sqrt(3) * c / 2 + 1e-6


# ------------------------------------------------------- functional property

def test_a_degraded_line_is_still_a_usable_progress_coordinate():
    """The point of the whole exercise: an agent riding the SOURCE line must
    still advance monotonically to the end of a decimated copy of it."""
    src = _l_route()
    coarse, length = resample_polyline(BR.decimate(src, 32), 128.0)
    arc = ArcProgress(coarse.astype(np.float64), 128.0, corridor=1500.0,
                      window=16)
    arc.reset(src[:1])
    seen = [float(arc.arc[0])]
    for t in range(1, len(src)):
        arc.advance(src[t:t + 1])
        seen.append(float(arc.arc[0]))
    seen = np.asarray(seen)
    assert np.all(np.diff(seen) >= -1e-6), "arc went backwards on a forward run"
    assert seen[-1] > 0.98 * length


def test_arc_on_a_degraded_line_is_still_a_potential():
    """Out and back nets zero - the anti-farming property must survive."""
    src = _l_route()
    coarse, _ = resample_polyline(BR.decimate(src, 32), 128.0)
    arc = ArcProgress(coarse.astype(np.float64), 128.0, corridor=1500.0,
                      window=16)
    path = np.concatenate([src[:120], src[:120][::-1]])
    arc.reset(path[:1])
    total = 0.0
    for t in range(1, len(path)):
        d, _ = arc.advance(path[t:t + 1])
        total += float(d[0])
    assert abs(total) < 1.0, total


# ------------------------------------------------------------------ end gap

def test_end_gap_is_zero_inside_the_box_and_the_miss_outside_it():
    assert BR.end_gap(np.array([[0.0, 0.0, 0.0]]), [-1, -1, -1], [1, 1, 1]) == 0.0
    assert abs(BR.end_gap(np.array([[5.0, 0.0, 0.0]]), [-1, -1, -1], [1, 1, 1])
               - 4.0) < 1e-6


# --------------------------------------------------------------------- CLI

def _run(args, **kw):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "build_route.py")]
                          + args, capture_output=True, text=True, **kw)


def test_cli_selftest_passes():
    r = _run(["--selftest"])
    assert r.returncode == 0 and "selftest OK" in r.stdout, r.stderr


def test_archive_without_allow_unfinished_is_refused(tmp_path):
    a = _fake_archive(tmp_path / "a.npz")
    r = _run(["--archive", str(a), "--out", str(tmp_path / "o.npz")])
    assert r.returncode != 0
    assert "allow-unfinished" in (r.stderr + r.stdout)


def test_a_truncated_line_is_written_as_truncated(tmp_path):
    """--allow-unfinished must MARK the file, not just permit it."""
    a = _fake_archive(tmp_path / "a.npz")
    out = tmp_path / "o.npz"
    r = _run(["--archive", str(a), "--allow-unfinished", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    z = np.load(out)
    assert bool(z["truncated"]) is True
    assert float(z["end_gap"]) > 64.0
    assert "go-explore archive chain" in str(z["derivation"])
    assert "TRUNCATED" in r.stdout


def _fake_finisher(path, map_name="surf_src_cannonball", n=400):
    """A recording that starts anywhere and ends inside the map's real end box."""
    end = json.loads((ROOT / "maps" / f"{map_name}.zones.json")
                     .read_text(encoding="utf-8"))["end"]
    goal = (np.asarray(end["mins"], float) + np.asarray(end["maxs"], float)) / 2
    start = goal + np.array([9000.0, -6000.0, 4000.0])
    t = np.linspace(0.0, 1.0, n)[:, None]
    xyz = start + (goal - start) * t + np.stack(
        [400 * np.sin(6 * t[:, 0]), 300 * np.cos(5 * t[:, 0]),
         200 * np.sin(9 * t[:, 0])], 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"map": map_name, "tick_ms": 10, "episode": 0}) + "\n")
        for i in range(n):
            f.write(json.dumps([i, *[round(float(v), 2) for v in xyz[i]],
                                0.0, 0.0, 0.0, 0.0]) + "\n")
        f.write(json.dumps({"end": "done", "ticks": n}) + "\n")
    return path


def test_flag_off_is_bit_identical_to_the_branch_point(tmp_path):
    """The champion path must not have moved. Runs the PRE-CHANGE tool from
    git and this one over the same recording and compares the route arrays.

    Skips (rather than fails) where git or the branch point is unavailable, so
    the suite still runs on a rented box that cloned with --depth 1.
    """
    old_src = subprocess.run(
        ["git", "show", "origin/arclen:tools/build_route.py"],
        capture_output=True, text=True, cwd=str(ROOT))
    if old_src.returncode != 0 or "def pick_route" not in old_src.stdout:
        pytest.skip("origin/arclen:tools/build_route.py not available")
    # it must live in tools/ - the tool resolves the repo root, and so the
    # map's zones.json, from its own __file__
    old = ROOT / "tools" / "_build_route_branchpoint.py"
    old.write_text(old_src.stdout, encoding="utf-8")
    traj = _fake_finisher(tmp_path / "traj.jsonl")
    a, b = tmp_path / "new.npz", tmp_path / "old.npz"
    try:
        r_new = _run(["--out", str(a), str(traj)])
        r_old = subprocess.run(
            [sys.executable, str(old), "--out", str(b), str(traj)],
            capture_output=True, text=True)
    finally:
        old.unlink(missing_ok=True)
    assert r_new.returncode == 0, r_new.stderr
    assert r_old.returncode == 0, r_old.stderr
    za, zb = np.load(a), np.load(b)
    assert np.array_equal(za["route"], zb["route"])
    assert float(za["spacing"]) == float(zb["spacing"])
    assert float(za["seconds"]) == float(zb["seconds"])
    assert not bool(za["truncated"])


def test_the_committed_coarse_line_is_what_it_claims_to_be():
    """The line this round actually ran: 58 waypoints, no racing line left."""
    coarse = ROOT / "maps" / "surf_src_cannonball.coarse32.route.npz"
    champ = ROOT / "maps" / "surf_src_cannonball.route.npz"
    if not (coarse.exists() and champ.exists()):
        pytest.skip("routes not present")
    c = np.asarray(np.load(coarse)["route"], np.float64)
    p = np.asarray(np.load(champ)["route"], np.float64)
    d = np.sqrt(((p[:, None, :] - c[None, :, :]) ** 2).sum(-1)).min(1)
    # far enough off the champion line to carry none of its precision ...
    assert d.max() > 500.0
    # ... and still inside the 1500 u corridor the reward gates income on
    assert d.max() < 1500.0
    assert not bool(np.load(coarse)["truncated"])
