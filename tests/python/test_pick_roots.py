"""tools/pick_roots.py (pbr2b1, 2026-10-24): Go-Explore roots = the policy's own states in a
window of seconds before each episode's deepest tick on a field."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from surfgym.core import STATE_DTYPE  # noqa: E402
import pick_roots  # noqa: E402


def test_window_before_the_deepest_tick():
    def ep(n, best):
        s = np.zeros(n, STATE_DTYPE)
        s["origin"][:, 0] = np.abs(np.arange(n) - best)    # deepest (x = 0) at tick `best`
        s["tick"] = np.arange(n)
        s["stuck_ticks"] = 3
        return s
    dist = lambda o: o[:, 0]                                  # noqa: E731
    r = pick_roots.pick([ep(1000, 600), ep(500, 100)], dist, window=(0.25, 2.0), n=10_000)
    # episode 1: ticks 400..575; episode 2: ticks 0..75 (the window clipped at the start)
    assert len(r) == 176 + 76
    assert (r["tick"] == 0).all() and (r["stuck_ticks"] == 0).all()
    x = r["origin"][:, 0]
    assert x.min() == 25 and x.max() == 200
    few = pick_roots.pick([ep(1000, 600)], dist, window=(0.25, 2.0), n=8)
    assert len(few) == 8 and few["origin"][0, 0] == 200 and few["origin"][-1, 0] == 25


def test_max_d_keeps_only_frontier_episodes():
    def ep(n, best, depth):
        s = np.zeros(n, STATE_DTYPE)
        s["origin"][:, 0] = np.abs(np.arange(n) - best) + depth
        return s
    dist = lambda o: o[:, 0]                                  # noqa: E731
    r = pick_roots.pick([ep(1000, 600, 0), ep(1000, 600, 500)], dist, window=(0.25, 2.0),
                        n=10_000, max_d=100.0)
    assert len(r) == 176 and r["origin"][:, 0].max() == 200


def test_first_within_anchors_the_first_approach():
    s = np.zeros(1000, STATE_DTYPE)
    x = np.full(1000, 500.0)
    x[300:400] = 10.0                 # first approach: within 20 of the minimum
    x[800] = 0.0                      # the deepest tick, after a wander
    s["origin"][:, 0] = x
    dist = lambda o: o[:, 0]                                  # noqa: E731
    late = pick_roots.pick([s], dist, window=(0.0, 1.0), n=10_000)
    first = pick_roots.pick([s], dist, window=(0.0, 1.0), n=10_000, first_within=20.0)
    assert len(late) == 101 and late["origin"][-1, 0] == 0.0          # ticks 700..800
    assert len(first) == 101 and (first["origin"][-1, 0] == 10.0)     # ticks 200..300
