"""no_corner_cut (surfgym.goalfield._bfs_geodesic / build_goal_field, 2026-10-22, r5b4): a
diagonal wavefront step may not squeeze past a solid voxel of its bounding box.

* a one-voxel-thick SLANTED wall leaks under the 26-neighbour graph and holds without it;
* a straight passage one voxel wide stays open (face steps are untouched);
* off: the same call without the keyword returns exactly the shipped distances;
* build_goal_field caches it apart (goalc_) and signs it apart (c1_).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import _bfs_geodesic  # noqa: E402


def _slanted_wall(n=12):
    """(1, n, n) slab: a diagonal wall of single voxels (i, i) splitting the plane; the
    goal seed sits on one side at (0, n-1), the probe on the other at (n-1, 0)."""
    occ = np.zeros((1, n, n), np.uint8)
    for i in range(n):
        occ[0, i, i] = 1
    seed = np.zeros_like(occ, bool)
    seed[0, 0, n - 1] = True          # (y=0, x=n-1): above the diagonal
    return occ, seed


def test_a_slanted_one_voxel_wall_leaks_with_corners_and_holds_without():
    occ, seed = _slanted_wall()
    d26, _, _ = _bfs_geodesic(occ, seed, 1.0, device="cpu", verbose=False)
    dnc, _, _ = _bfs_geodesic(occ, seed, 1.0, device="cpu", verbose=False,
                              no_corner_cut=True)
    probe = (0, 11, 0)                # (y=11, x=0): below the diagonal
    assert np.isfinite(float(d26[probe]))            # the leak: a diagonal hop across
    assert not np.isfinite(float(dnc[probe]))        # sealed (no other way round here)


def test_a_one_voxel_corridor_stays_open():
    occ = np.ones((1, 5, 9), np.uint8)
    occ[0, 2, :] = 0                  # a straight corridor one voxel wide
    seed = np.zeros_like(occ, bool)
    seed[0, 2, 8] = True
    dnc, _, _ = _bfs_geodesic(occ, seed, 1.0, device="cpu", verbose=False,
                              no_corner_cut=True)
    assert float(dnc[0, 2, 0]) == 8.0


def test_off_is_the_shipped_graph():
    rng = np.random.default_rng(3)
    occ = (rng.random((6, 7, 8)) < 0.3).astype(np.uint8)
    seed = np.zeros_like(occ, bool)
    seed[3, 3, 3] = True
    occ[3, 3, 3] = 0
    a, ra, _ = _bfs_geodesic(occ, seed, 2.0, device="cpu", verbose=False)
    b, rb, _ = _bfs_geodesic(occ, seed, 2.0, device="cpu", verbose=False,
                             no_corner_cut=False)
    assert ra == rb
    assert bool((a == b).all() | ((a != a) & (b != b)).all()) or np.array_equal(
        a.numpy(), b.numpy())
    # and no-corner-cut never SHORTENS a distance (it only removes edges)
    c, _, _ = _bfs_geodesic(occ, seed, 2.0, device="cpu", verbose=False, no_corner_cut=True)
    fa, fc = a.numpy(), c.numpy()
    both = np.isfinite(fa) & np.isfinite(fc)
    assert (fc[both] >= fa[both] - 1e-4).all()
    assert (np.isfinite(fc) <= np.isfinite(fa)).all()


def test_build_goal_field_caches_and_signs_it_apart():
    src = (ROOT / "python" / "surfgym" / "goalfield.py").read_text(encoding="utf-8")
    assert "{'c1_' if no_corner_cut else ''}" in src
    assert "{'c' if no_corner_cut else ''}" in src
